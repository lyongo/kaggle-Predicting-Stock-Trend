#!/bin/bash
# ============================================================================
# 使用预训练Kronos模型进行推理并生成Kaggle提交文件
# ============================================================================

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
KRONOS_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"

echo "============================================================================"
echo "使用预训练Kronos模型进行Kaggle预测"
echo "============================================================================"

# 路径配置
PRETRAINED_TOKENIZER="/mnt/shared-storage-user/zhaoliangliang/model/Kronos-Tokenizer-base"
PRETRAINED_MODEL="/mnt/shared-storage-user/zhaoliangliang/model/Kronos-base"
TEST_INFO_PATH="${KRONOS_DIR}/kaggle_data/test_ticker_info.csv"
PREDICTIONS_PATH="${KRONOS_DIR}/kaggle_predictions_pretrained.csv"
TEST_CSV_PATH="/mnt/shared-storage-user/zhaoliangliang/dataset/predicting-stock-trends-rise-or-fall/test.csv"
SUBMISSION_PATH="${KRONOS_DIR}/kaggle_submission_pretrained.csv"

# 检查路径
echo "检查路径..."
if [ ! -d "${PRETRAINED_TOKENIZER}" ]; then
    echo "❌ 错误: Tokenizer路径不存在: ${PRETRAINED_TOKENIZER}"
    exit 1
fi

if [ ! -d "${PRETRAINED_MODEL}" ]; then
    echo "❌ 错误: 模型路径不存在: ${PRETRAINED_MODEL}"
    exit 1
fi

if [ ! -f "${TEST_INFO_PATH}" ]; then
    echo "❌ 错误: 测试集信息文件不存在: ${TEST_INFO_PATH}"
    echo "   请先运行 kaggle_data_preprocess.py"
    exit 1
fi

echo "✓ 所有路径检查通过"
echo ""

# 步骤1: 运行推理
echo "============================================================================"
echo "步骤1: 运行推理"
echo "============================================================================"
cd "${SCRIPT_DIR}"

python kaggle_inference_pretrained.py

if [ $? -ne 0 ]; then
    echo "❌ 推理失败"
    exit 1
fi

echo ""
echo "✓ 推理完成"
echo ""

# 步骤2: 生成提交文件
echo "============================================================================"
echo "步骤2: 生成Kaggle提交文件"
echo "============================================================================"

python generate_submission.py \
    --predictions "${PREDICTIONS_PATH}" \
    --test_csv "${TEST_CSV_PATH}" \
    --output "${SUBMISSION_PATH}"

if [ $? -ne 0 ]; then
    echo "❌ 提交文件生成失败"
    exit 1
fi

echo ""
echo "============================================================================"
echo "✅ 完成！"
echo "============================================================================"
echo "预测结果: ${PREDICTIONS_PATH}"
echo "提交文件: ${SUBMISSION_PATH}"
echo ""
echo "可以将 ${SUBMISSION_PATH} 提交到Kaggle"
echo "============================================================================"

