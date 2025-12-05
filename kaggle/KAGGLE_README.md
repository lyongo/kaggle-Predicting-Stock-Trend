# Kaggle股票趋势预测比赛 - Kronos模型实现

本项目使用Kronos模型参加Kaggle "Predicting Stock Trends: Rise or Fall"比赛。

## 项目结构

```
Kronos/
├── kaggle/                        # Kaggle比赛相关脚本
│   ├── kaggle_data_preprocess.py  # 数据预处理脚本
│   ├── kaggle_config.yaml         # 微调配置文件（标准配置）
│   ├── kaggle_config_stage1.yaml  # 阶段1配置（快速验证）
│   ├── kaggle_config_stage3.yaml  # 阶段3配置（精细微调）
│   ├── kaggle_inference.py         # 推理脚本
│   ├── generate_submission.py     # 提交文件生成脚本
│   ├── train_kaggle.sh            # 训练启动脚本
│   ├── validate_data_format.py    # 数据格式验证脚本
│   ├── KAGGLE_README.md           # 本文件
│   ├── FINETUNING_GUIDE.md        # 通用微调指南
│   └── KAGGLE_FINETUNING_GUIDE.md # Kaggle比赛专门微调指南（推荐阅读）
├── kaggle_data/                   # 处理后的数据目录
│   ├── train/                     # 训练数据（按ticker分文件）
│   ├── test/                      # 测试数据
│   ├── train_combined.csv         # 合并的训练数据（用于微调）
│   ├── train_ticker_info.csv      # 训练集元数据
│   └── test_ticker_info.csv       # 测试集元数据
└── kaggle_finetuned/              # 微调后的模型保存目录
    └── kaggle_stock_trends/
        ├── tokenizer/
        └── basemodel/
```

## 使用步骤

### 1. 数据预处理

首先运行数据预处理脚本，将Kaggle数据集转换为Kronos格式：

```bash
cd /mnt/shared-storage-user/zhaoliangliang/code/Kronos/kaggle
python kaggle_data_preprocess.py
```

这个脚本会：
- 读取Kaggle的train.csv和test.csv
- 将数据转换为Kronos所需的格式（timestamps, open, high, low, close, volume, amount）
- 按ticker分组保存训练数据
- 为每个测试样本准备历史数据
- 合并所有ticker的训练数据用于模型微调

### 2. 配置微调参数

编辑 `kaggle_config.yaml` 文件，设置：
- 预训练模型路径（pretrained_tokenizer 和 pretrained_predictor）
- 数据路径（data_path 指向合并的训练数据）
- 训练超参数（epochs, batch_size, learning_rate等）

### 3. 微调模型

#### 方式1: 使用训练启动脚本（推荐）

```bash
cd /mnt/shared-storage-user/zhaoliangliang/code/Kronos/kaggle

# 单GPU训练
bash train_kaggle.sh

# 多GPU训练（推荐）
bash train_kaggle.sh --multi-gpu 8

# 快速验证（阶段1）
bash train_kaggle.sh --config kaggle_config_stage1.yaml

# 精细微调（阶段3）
bash train_kaggle.sh --config kaggle_config_stage3.yaml --multi-gpu 8
```

#### 方式2: 直接使用训练脚本

```bash
cd /mnt/shared-storage-user/zhaoliangliang/code/Kronos/finetune_csv
python train_sequential.py --config ../kaggle/kaggle_config.yaml
```

如果需要多GPU训练：
```bash
torchrun --standalone --nproc_per_node=8 train_sequential.py --config ../kaggle/kaggle_config.yaml
```

**详细微调建议请参考**: `KAGGLE_FINETUNING_GUIDE.md`

### 4. 运行推理

微调完成后，使用推理脚本对测试集进行预测：

```bash
cd /mnt/shared-storage-user/zhaoliangliang/code/Kronos/kaggle
python kaggle_inference.py
```

推理脚本会：
- 加载微调后的模型和tokenizer
- 对每个测试样本进行价格预测
- 将价格预测转换为涨跌分类（预测价格 > 当前价格 = 涨(1)，否则 = 跌(0)）
- 保存预测结果到 kaggle_predictions.csv

### 5. 生成提交文件

最后，生成Kaggle提交格式的文件：

```bash
cd /mnt/shared-storage-user/zhaoliangliang/code/Kronos/kaggle
python generate_submission.py
```

这个脚本会：
- 读取预测结果
- 格式化为Kaggle要求的格式（ID, Pred列）
- 保存为 kaggle_submission.csv

### 6. 提交到Kaggle

将生成的 `kaggle_submission.csv` 文件上传到Kaggle比赛页面进行提交。

## 注意事项

1. **数据格式**：确保Kaggle数据集路径正确，train.csv和test.csv在指定目录下
   - 运行 `python validate_data_format.py` 验证数据格式
2. **模型路径**：需要先下载Kronos预训练模型，并更新config.yaml中的路径
3. **GPU内存**：如果GPU内存不足，可以减小batch_size或lookback_window
4. **预测逻辑**：当前实现使用预测价格与当前价格比较来判断涨跌，可以根据实际情况调整
5. **微调策略**：建议先使用 `kaggle_config_stage1.yaml` 快速验证，再使用标准配置训练
6. **多GPU训练**：推荐使用多GPU训练以加快速度，使用 `train_kaggle.sh --multi-gpu 8`

## 推荐阅读顺序

1. **快速开始**: 阅读本文件（KAGGLE_README.md）
2. **完整指南**: **强烈推荐阅读 `COMPLETE_GUIDE.md`**，包含完整的训练、推理、提交流程和微调建议
3. **数据预处理**: 运行 `kaggle_data_preprocess.py` 并验证数据格式
4. **微调指南**: 阅读 `KAGGLE_FINETUNING_GUIDE.md`，了解针对Kaggle比赛的专门建议
5. **开始训练**: 使用 `train_kaggle.sh` 启动训练
6. **推理提交**: 训练完成后运行推理和提交脚本

## 文件说明

### 核心脚本
- `kaggle_data_preprocess.py`: 数据预处理，将Kaggle格式转换为Kronos格式
- `kaggle_inference.py`: 推理脚本，使用微调模型进行预测
- `generate_submission.py`: 提交文件生成脚本
- `train_kaggle.sh`: 训练启动脚本（支持单GPU/多GPU训练）
- `validate_data_format.py`: 数据格式验证脚本

### 配置文件
- `kaggle_config.yaml`: 标准微调配置文件（推荐用于正式训练）
- `kaggle_config_stage1.yaml`: 阶段1配置（快速验证，减少epoch）
- `kaggle_config_stage3.yaml`: 阶段3配置（精细微调，降低学习率）

### 文档
- `KAGGLE_README.md`: 本文件，快速开始指南
- `COMPLETE_GUIDE.md`: **完整指南（强烈推荐阅读）**
  - 包含完整的训练、推理、提交流程
  - 不同配置的训练方法
  - 详细的微调建议
  - 常见问题解答
  - 最佳实践总结
- `KAGGLE_FINETUNING_GUIDE.md`: Kaggle比赛专门微调指南
  - 包含针对二分类任务的特殊建议
  - 多ticker数据处理策略
  - 参数调整详细说明
  - 训练策略和最佳实践
- `FINETUNING_GUIDE.md`: 通用微调指南
- `PRETRAINED_INFERENCE_README.md`: 预训练模型推理指南

## 依赖

- pandas
- numpy
- torch
- Kronos模型代码（需要从Kronos项目导入）

## 问题排查

如果遇到问题：
1. 检查数据路径是否正确
2. 检查模型路径是否存在
3. 检查GPU是否可用（如果使用CUDA）
4. 查看错误日志定位具体问题

