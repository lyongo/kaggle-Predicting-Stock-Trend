# Kaggle股票趋势预测 - Kronos模型实现

本项目使用[Kronos](https://github.com/shiyu-coder/Kronos)模型参加Kaggle比赛 [Predicting Stock Trends: Rise or Fall](https://www.kaggle.com/competitions/predicting-stock-trends-rise-or-fall)。

## 📋 目录

- [项目简介](#项目简介)
- [快速开始](#快速开始)
- [完整流程](#完整流程)
- [详细文档](#详细文档)
- [项目结构](#项目结构)

## 🎯 项目简介

本项目使用Kronos（一个金融时间序列基础模型）来预测股票价格的涨跌趋势。Kronos是一个专门为金融K线数据设计的大语言模型，能够有效捕捉时间序列中的长期依赖关系。

### 比赛任务
- **任务类型**: 二分类（涨/跌预测）
- **数据规模**: 5000个ticker，20M+行历史数据
- **预测目标**: 预测下一个时间点的价格，转换为涨(1)/跌(0)分类

## 🚀 快速开始

### 方式1: 使用预训练模型（最快，无需训练）

```bash
cd kaggle
python run_pretrained_inference.py
```

这会自动完成推理并生成提交文件，约需10-30分钟。

### 方式2: 微调后使用（推荐，效果更好）

```bash
# 1. 数据预处理
cd kaggle
python kaggle_data_preprocess.py

# 2. 训练模型
bash train_kaggle.sh --multi-gpu 8

# 3. 推理和提交
python kaggle_inference.py
python generate_submission.py
```

## 📝 完整流程

### 步骤1: 下载Kaggle数据

1. 访问 [Kaggle比赛页面](https://www.kaggle.com/competitions/predicting-stock-trends-rise-or-fall/data)
2. 下载以下文件：
   - `train.csv` - 训练数据
   - `test.csv` - 测试数据
3. 将数据保存到指定目录（默认：`/mnt/shared-storage-user/zhaoliangliang/dataset/predicting-stock-trends-rise-or-fall/`）

**数据格式**:
- `train.csv`: 包含 `Ticker`, `Date`, `Open`, `High`, `Low`, `Close`, `Volume`, `Dividends`, `Stock Splits`
- `test.csv`: 包含 `ID`, `Date`

### 步骤2: 数据预处理

将Kaggle数据转换为Kronos模型所需的格式：

```bash
cd kaggle
python kaggle_data_preprocess.py
```

**输出**:
- `kaggle_data/train/` - 按ticker分组的训练数据（每个ticker一个CSV文件）
- `kaggle_data/test/` - 测试数据
- `kaggle_data/train_combined.csv` - 合并的训练数据（用于微调）
- `kaggle_data/test_ticker_info.csv` - 测试集元数据

**数据格式验证**:
```bash
python validate_data_format.py
```

### 步骤3: 模型训练（可选）

#### 3.1 快速验证（1-2小时）

用于验证流程是否正确：

```bash
bash train_kaggle.sh --config kaggle_config_stage1.yaml
```

#### 3.2 标准训练（4-8小时，单GPU）

```bash
bash train_kaggle.sh
```

或使用标准配置文件：

```bash
bash train_kaggle.sh --config kaggle_config.yaml
```

#### 3.3 多GPU训练（推荐，30-60分钟）

```bash
bash train_kaggle.sh --multi-gpu 8
```

#### 3.4 精细微调（如果标准训练效果不理想）

```bash
bash train_kaggle.sh --config kaggle_config_stage3.yaml --multi-gpu 8
```

**训练配置说明**:
- `kaggle_config_stage1.yaml`: 快速验证（减少epoch）
- `kaggle_config.yaml`: 标准训练（推荐）
- `kaggle_config_stage3.yaml`: 精细微调（降低学习率，增加epoch）

### 步骤4: 推理

#### 使用预训练模型

```bash
cd kaggle
python kaggle_inference_pretrained.py
```

**输出**: `kaggle_predictions_pretrained.csv`

#### 使用微调模型

```bash
python kaggle_inference.py
```

**输出**: `kaggle_predictions.csv`

### 步骤5: 生成提交文件

#### 使用预训练模型结果

```bash
python generate_submission.py \
    --predictions ../kaggle_predictions_pretrained.csv \
    --test_csv /path/to/test.csv \
    --output ../kaggle_submission_pretrained.csv
```

#### 使用微调模型结果

```bash
python generate_submission.py \
    --predictions ../kaggle_predictions.csv \
    --test_csv /path/to/test.csv \
    --output ../kaggle_submission.csv
```

### 步骤6: 提交到Kaggle

将生成的提交文件（`kaggle_submission.csv` 或 `kaggle_submission_pretrained.csv`）上传到 [Kaggle比赛页面](https://www.kaggle.com/competitions/predicting-stock-trends-rise-or-fall/submit) 进行提交。

## 📚 详细文档

### 核心文档

- **[COMPLETE_GUIDE.md](kaggle/COMPLETE_GUIDE.md)** - 完整指南（强烈推荐）
  - 包含完整的训练、推理、提交流程
  - 不同配置的训练方法
  - 详细的微调建议
  - 常见问题解答

- **[KAGGLE_FINETUNING_GUIDE.md](kaggle/KAGGLE_FINETUNING_GUIDE.md)** - Kaggle比赛专门微调指南
  - 针对二分类任务的特殊建议
  - 多ticker数据处理策略
  - 参数调整详细说明

- **[KAGGLE_README.md](kaggle/KAGGLE_README.md)** - 快速开始指南

### 其他文档

- `FINETUNING_GUIDE.md` - 通用微调指南
- `PRETRAINED_INFERENCE_README.md` - 预训练模型推理指南

## 📁 项目结构

```
Kronos/
├── kaggle/                        # Kaggle比赛相关脚本
│   ├── kaggle_data_preprocess.py  # 数据预处理脚本
│   ├── kaggle_config.yaml         # 标准微调配置
│   ├── kaggle_config_stage1.yaml  # 快速验证配置
│   ├── kaggle_config_stage3.yaml  # 精细微调配置
│   ├── kaggle_inference.py        # 微调模型推理脚本
│   ├── kaggle_inference_pretrained.py  # 预训练模型推理脚本
│   ├── generate_submission.py     # 提交文件生成脚本
│   ├── train_kaggle.sh            # 训练启动脚本
│   ├── validate_data_format.py    # 数据格式验证脚本
│   ├── run_pretrained_inference.py # 一键推理+提交脚本
│   ├── COMPLETE_GUIDE.md          # 完整指南
│   ├── KAGGLE_FINETUNING_GUIDE.md # 微调指南
│   └── KAGGLE_README.md           # 快速开始指南
├── kaggle_data/                   # 处理后的数据（不包含在仓库中）
│   ├── train/                     # 训练数据（按ticker分文件）
│   ├── test/                      # 测试数据
│   ├── train_combined.csv         # 合并的训练数据
│   └── test_ticker_info.csv       # 测试集元数据
├── kaggle_finetuned/              # 微调后的模型（不包含在仓库中）
│   └── kaggle_stock_trends/
│       ├── tokenizer/
│       └── basemodel/
├── finetune_csv/                  # Kronos微调框架
└── model/                         # Kronos模型代码
```

## ⚙️ 环境要求

### 依赖

```bash
pip install -r requirements.txt
```

主要依赖：
- `torch` - PyTorch
- `pandas` - 数据处理
- `numpy` - 数值计算
- `tqdm` - 进度条

### 预训练模型

需要下载Kronos预训练模型：
- Tokenizer: `/path/to/Kronos-Tokenizer-base`
- Model: `/path/to/Kronos-base`

在配置文件中更新模型路径（`kaggle/kaggle_config.yaml`）。

### 硬件要求

- **训练**: 推荐使用GPU（单GPU或多GPU）
  - 单GPU: 至少8GB显存
  - 多GPU: 支持DDP训练
- **推理**: GPU或CPU（GPU更快）

## 🔧 配置说明

### 训练配置

编辑 `kaggle/kaggle_config.yaml` 设置：

```yaml
data:
  data_path: "/path/to/train_combined.csv"
  lookback_window: 256      # 历史窗口长度
  predict_window: 1         # 预测窗口
  train_ratio: 0.9          # 训练集比例
  val_ratio: 0.1            # 验证集比例

training:
  tokenizer_epochs: 20      # Tokenizer训练轮数
  basemodel_epochs: 15      # Basemodel训练轮数
  batch_size: 32             # 批次大小
  tokenizer_learning_rate: 0.0002
  predictor_learning_rate: 0.00004

model_paths:
  pretrained_tokenizer: "/path/to/Kronos-Tokenizer-base"
  pretrained_predictor: "/path/to/Kronos-base"
```

详细配置说明请参考 [COMPLETE_GUIDE.md](kaggle/COMPLETE_GUIDE.md)。

## 💡 使用建议

### 首次使用

1. **快速验证**: 使用预训练模型快速提交，获得baseline分数
2. **数据验证**: 运行 `validate_data_format.py` 确保数据格式正确
3. **标准训练**: 使用标准配置进行训练
4. **精细调优**: 根据结果调整超参数

### 训练策略

- **渐进式训练**: 快速验证 → 标准训练 → 精细微调
- **多GPU训练**: 显著加快训练速度
- **早停策略**: 监控验证损失，避免过拟合

详细建议请参考 [KAGGLE_FINETUNING_GUIDE.md](kaggle/KAGGLE_FINETUNING_GUIDE.md)。

## 🐛 常见问题

### Q1: 数据预处理很慢
**A**: 已优化使用 `groupby()` 和 `itertuples()`，处理5000个ticker约需4-5分钟。

### Q2: GPU内存不足
**A**: 减小 `batch_size` 或 `lookback_window`，或使用梯度累积。

### Q3: 训练速度太慢
**A**: 使用多GPU训练：`bash train_kaggle.sh --multi-gpu 8`

### Q4: 验证损失不下降
**A**: 降低学习率或增加训练轮数。

更多问题请参考 [COMPLETE_GUIDE.md](kaggle/COMPLETE_GUIDE.md) 中的常见问题部分。

## 📊 性能参考

### 训练时间估算

| 配置 | 单GPU | 8 GPU |
|------|-------|-------|
| 快速验证 | 1-2小时 | 10-20分钟 |
| 标准训练 | 4-8小时 | 30-60分钟 |
| 精细微调 | 6-12小时 | 1-2小时 |

### 推理时间

- 约5000个样本，单GPU预计10-30分钟

## 📄 许可证

本项目基于Kronos项目，请参考 [LICENSE](LICENSE) 文件。

## 🙏 致谢

- [Kronos](https://github.com/shiyu-coder/Kronos) - 金融时间序列基础模型
- [Kaggle](https://www.kaggle.com/competitions/predicting-stock-trends-rise-or-fall) - 比赛平台

## 📮 联系方式

如有问题或建议，请提交Issue或Pull Request。

---

**祝你在Kaggle比赛中取得好成绩！** 🏆

