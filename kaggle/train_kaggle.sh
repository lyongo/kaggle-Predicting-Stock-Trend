#!/bin/bash
# ============================================================================
# Kronos Kaggle 股票趋势预测模型微调启动脚本
# ============================================================================
# 使用方法:
#   单GPU训练: bash train_kaggle.sh
#   多GPU训练: bash train_kaggle.sh --multi-gpu --num-gpus 8
#   跳过tokenizer训练: bash train_kaggle.sh --skip-tokenizer
#   跳过basemodel训练: bash train_kaggle.sh --skip-basemodel
#   跳过已存在的模型: bash train_kaggle.sh --skip-existing
# ============================================================================

set -e  # 遇到错误立即退出

# ============================================================================
# 配置参数
# ============================================================================
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
KRONOS_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"
# 默认配置文件（如果未指定，使用标准配置）
DEFAULT_CONFIG_FILE="${SCRIPT_DIR}/kaggle_config.yaml"
CONFIG_FILE="${DEFAULT_CONFIG_FILE}"
FINETUNE_DIR="${KRONOS_DIR}/finetune_csv"
OUTPUT_DIR="${KRONOS_DIR}/kaggle_finetuned"
LOG_DIR="${OUTPUT_DIR}/logs"

# 默认参数
NUM_GPUS=1
SKIP_TOKENIZER=false
SKIP_BASEMODEL=false
SKIP_EXISTING=false
DIST_BACKEND="nccl"

# ============================================================================
# 解析命令行参数
# ============================================================================
while [[ $# -gt 0 ]]; do
    case $1 in
        --multi-gpu)
            NUM_GPUS="$2"
            shift 2
            ;;
        --num-gpus)
            NUM_GPUS="$2"
            shift 2
            ;;
        --skip-tokenizer)
            SKIP_TOKENIZER=true
            shift
            ;;
        --skip-basemodel)
            SKIP_BASEMODEL=true
            shift
            ;;
        --skip-existing)
            SKIP_EXISTING=true
            shift
            ;;
        --config)
            # 如果路径是绝对路径，直接使用
            if [[ "$2" == /* ]]; then
                CONFIG_FILE="$2"
            else
                # 相对路径，转换为相对于脚本目录的绝对路径
                # 处理相对路径（如 ../kaggle/kaggle_config.yaml 或 kaggle_config.yaml）
                if [[ "$2" == */* ]]; then
                    # 包含路径分隔符，需要解析
                    CONFIG_FILE="$(cd "${SCRIPT_DIR}" && cd "$(dirname "$2")" && pwd)/$(basename "$2")"
                else
                    # 只有文件名，在脚本目录中查找
                    CONFIG_FILE="${SCRIPT_DIR}/$2"
                fi
            fi
            shift 2
            ;;
        --dist-backend)
            DIST_BACKEND="$2"
            shift 2
            ;;
        -h|--help)
            echo "使用方法: $0 [选项]"
            echo ""
            echo "选项:"
            echo "  --multi-gpu NUM        启用多GPU训练，指定GPU数量"
            echo "  --num-gpus NUM         指定GPU数量（同--multi-gpu）"
            echo "  --skip-tokenizer       跳过tokenizer训练"
            echo "  --skip-basemodel       跳过basemodel训练"
            echo "  --skip-existing        跳过已存在的模型训练"
            echo "  --config PATH          指定配置文件路径"
            echo "  --dist-backend BACKEND 指定分布式后端 (nccl/gloo)"
            echo "  -h, --help             显示此帮助信息"
            echo ""
            echo "示例:"
            echo "  $0                                    # 单GPU训练"
            echo "  $0 --multi-gpu 8                      # 8 GPU训练"
            echo "  $0 --skip-tokenizer                   # 只训练basemodel"
            echo "  $0 --skip-existing --multi-gpu 4      # 4 GPU训练，跳过已存在的模型"
            exit 0
            ;;
        *)
            echo "未知参数: $1"
            echo "使用 --help 查看帮助信息"
            exit 1
            ;;
    esac
done

# ============================================================================
# 环境检查
# ============================================================================
echo "============================================================================"
echo "Kronos Kaggle 模型微调启动脚本"
echo "============================================================================"
echo "配置信息:"
echo "  脚本目录: ${SCRIPT_DIR}"
echo "  Kronos目录: ${KRONOS_DIR}"
echo "  配置文件: ${CONFIG_FILE}"
echo "  训练目录: ${FINETUNE_DIR}"
echo "  输出目录: ${OUTPUT_DIR}"
echo "  日志目录: ${LOG_DIR}"
echo "  GPU数量: ${NUM_GPUS}"
echo "============================================================================"

# 检查配置文件
if [ ! -f "${CONFIG_FILE}" ]; then
    echo "❌ 错误: 配置文件不存在: ${CONFIG_FILE}"
    exit 1
fi
echo "✓ 配置文件存在"

# 检查训练脚本
if [ ! -f "${FINETUNE_DIR}/train_sequential.py" ]; then
    echo "❌ 错误: 训练脚本不存在: ${FINETUNE_DIR}/train_sequential.py"
    exit 1
fi
echo "✓ 训练脚本存在"

# 检查数据文件
DATA_PATH=$(grep "data_path:" "${CONFIG_FILE}" | awk '{print $2}' | tr -d '"')
if [ ! -f "${DATA_PATH}" ]; then
    echo "❌ 错误: 训练数据文件不存在: ${DATA_PATH}"
    echo "   请先运行 kaggle_data_preprocess.py 生成训练数据"
    exit 1
fi
echo "✓ 训练数据文件存在: ${DATA_PATH}"

# 检查预训练模型
TOKENIZER_PATH=$(grep "pretrained_tokenizer:" "${CONFIG_FILE}" | awk '{print $2}' | tr -d '"')
PREDICTOR_PATH=$(grep "pretrained_predictor:" "${CONFIG_FILE}" | awk '{print $2}' | tr -d '"')

if [ ! -d "${TOKENIZER_PATH}" ]; then
    echo "⚠ 警告: Tokenizer模型路径不存在: ${TOKENIZER_PATH}"
fi

if [ ! -d "${PREDICTOR_PATH}" ]; then
    echo "⚠ 警告: Predictor模型路径不存在: ${PREDICTOR_PATH}"
fi

# 检查CUDA
if command -v nvidia-smi &> /dev/null; then
    GPU_COUNT=$(nvidia-smi --list-gpus | wc -l)
    echo "✓ 检测到 ${GPU_COUNT} 个GPU"
    
    if [ "${NUM_GPUS}" -gt "${GPU_COUNT}" ]; then
        echo "⚠ 警告: 请求的GPU数量 (${NUM_GPUS}) 超过可用GPU数量 (${GPU_COUNT})"
        NUM_GPUS=${GPU_COUNT}
        echo "   已调整为使用 ${NUM_GPUS} 个GPU"
    fi
else
    echo "⚠ 警告: 未检测到nvidia-smi，可能无法使用GPU"
fi

# 创建输出目录
mkdir -p "${OUTPUT_DIR}"
mkdir -p "${LOG_DIR}"

# ============================================================================
# 准备训练命令参数（将在切换到训练目录后更新为绝对路径）
# ============================================================================
# 注意：配置文件路径将在切换到训练目录后转换为绝对路径
TRAIN_ARGS=()

if [ "${SKIP_TOKENIZER}" = true ]; then
    echo "✓ 将跳过tokenizer训练"
fi

if [ "${SKIP_BASEMODEL}" = true ]; then
    echo "✓ 将跳过basemodel训练"
fi

if [ "${SKIP_EXISTING}" = true ]; then
    echo "✓ 将跳过已存在的模型"
fi

# ============================================================================
# 启动训练
# ============================================================================
echo ""
echo "============================================================================"
echo "开始训练..."
echo "============================================================================"
echo "训练时间: $(date '+%Y-%m-%d %H:%M:%S')"
echo ""

# 确保配置文件路径是绝对路径（在切换目录前）
if [[ "${CONFIG_FILE}" != /* ]]; then
    # 如果还不是绝对路径，转换为绝对路径（相对于脚本目录）
    CONFIG_FILE="${SCRIPT_DIR}/${CONFIG_FILE}"
fi

# 验证配置文件是否存在
if [ ! -f "${CONFIG_FILE}" ]; then
    echo "❌ 错误: 配置文件不存在: ${CONFIG_FILE}"
    exit 1
fi

# 切换到训练目录
cd "${FINETUNE_DIR}"

# 配置文件路径现在应该是绝对路径，可以直接使用
TRAIN_ARGS=("--config" "${CONFIG_FILE}")

if [ "${SKIP_TOKENIZER}" = true ]; then
    TRAIN_ARGS+=("--skip-tokenizer")
fi

if [ "${SKIP_BASEMODEL}" = true ]; then
    TRAIN_ARGS+=("--skip-basemodel")
fi

if [ "${SKIP_EXISTING}" = true ]; then
    TRAIN_ARGS+=("--skip-existing")
fi

# 生成日志文件名
TIMESTAMP=$(date '+%Y%m%d_%H%M%S')
LOG_FILE="${LOG_DIR}/train_${TIMESTAMP}.log"

if [ "${NUM_GPUS}" -eq 1 ]; then
    # 单GPU训练
    echo "模式: 单GPU训练"
    echo "配置文件: ${CONFIG_FILE}"
    echo "日志文件: ${LOG_FILE}"
    echo ""
    
    python train_sequential.py "${TRAIN_ARGS[@]}" 2>&1 | tee "${LOG_FILE}"
    
    TRAIN_EXIT_CODE=${PIPESTATUS[0]}
else
    # 多GPU训练 (DDP)
    echo "模式: 多GPU分布式训练 (${NUM_GPUS} GPUs)"
    echo "配置文件: ${CONFIG_FILE}"
    echo "分布式后端: ${DIST_BACKEND}"
    echo "日志文件: ${LOG_FILE}"
    echo ""
    
    export DIST_BACKEND="${DIST_BACKEND}"
    
    torchrun \
        --standalone \
        --nproc_per_node="${NUM_GPUS}" \
        train_sequential.py \
        "${TRAIN_ARGS[@]}" \
        2>&1 | tee "${LOG_FILE}"
    
    TRAIN_EXIT_CODE=${PIPESTATUS[0]}
fi

# ============================================================================
# 训练结果
# ============================================================================
echo ""
echo "============================================================================"
if [ ${TRAIN_EXIT_CODE} -eq 0 ]; then
    echo "✅ 训练完成！"
    echo ""
    echo "模型保存位置:"
    EXP_NAME=$(grep "exp_name:" "${CONFIG_FILE}" | awk '{print $2}' | tr -d '"')
    BASE_PATH=$(grep "base_path:" "${CONFIG_FILE}" | awk '{print $2}' | tr -d '"')
    
    if [ "${SKIP_TOKENIZER}" != true ]; then
        TOKENIZER_SAVE_PATH="${BASE_PATH}${EXP_NAME}/tokenizer/best_model"
        if [ -d "${TOKENIZER_SAVE_PATH}" ]; then
            echo "  Tokenizer: ${TOKENIZER_SAVE_PATH}"
        fi
    fi
    
    if [ "${SKIP_BASEMODEL}" != true ]; then
        BASEMODEL_SAVE_PATH="${BASE_PATH}${EXP_NAME}/basemodel/best_model"
        if [ -d "${BASEMODEL_SAVE_PATH}" ]; then
            echo "  Basemodel: ${BASEMODEL_SAVE_PATH}"
        fi
    fi
    
    echo ""
    echo "日志文件: ${LOG_FILE}"
else
    echo "❌ 训练失败，退出代码: ${TRAIN_EXIT_CODE}"
    echo "请查看日志文件: ${LOG_FILE}"
fi
echo "============================================================================"
echo "训练结束时间: $(date '+%Y-%m-%d %H:%M:%S')"
echo "============================================================================"

exit ${TRAIN_EXIT_CODE}

