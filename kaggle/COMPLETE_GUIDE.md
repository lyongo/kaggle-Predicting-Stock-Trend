# Kaggle股票趋势预测比赛 - 完整指南

## 📋 目录

1. [快速开始](#快速开始)
2. [完整流程](#完整流程)
3. [训练配置说明](#训练配置说明)
4. [推理和提交](#推理和提交)
5. [微调建议](#微调建议)
6. [常见问题](#常见问题)
7. [文件说明](#文件说明)

---

## 🚀 快速开始

### 方式1: 使用预训练模型（最快，无需训练）

```bash
cd /mnt/shared-storage-user/zhaoliangliang/code/Kronos/kaggle
python run_pretrained_inference.py
```

这会自动完成推理和提交文件生成，约需10-30分钟。

### 方式2: 微调后使用（推荐，效果更好）

```bash
# 步骤1: 训练模型
cd /mnt/shared-storage-user/zhaoliangliang/code/Kronos/kaggle
bash train_kaggle.sh --multi-gpu 8

# 步骤2: 推理和提交
python kaggle_inference.py
python generate_submission.py
```

---

## 📝 完整流程

### 阶段0: 数据预处理（只需运行一次）

```bash
cd /mnt/shared-storage-user/zhaoliangliang/code/Kronos/kaggle
python kaggle_data_preprocess.py
```

**输出**:
- `kaggle_data/train/` - 按ticker分组的训练数据
- `kaggle_data/test/` - 测试数据
- `kaggle_data/train_combined.csv` - 合并的训练数据（用于微调）
- `kaggle_data/test_ticker_info.csv` - 测试集元数据

**验证数据格式**:
```bash
python validate_data_format.py
```

### 阶段1: 模型训练（可选，但推荐）

#### 1.1 快速验证（1-2小时）

用于验证流程是否正确：

```bash
bash train_kaggle.sh --config kaggle_config_stage1.yaml
```

**配置特点**:
- 减少训练轮数（tokenizer: 10, basemodel: 8）
- 快速验证数据流程

#### 1.2 标准训练（4-8小时，单GPU）

```bash
bash train_kaggle.sh
```

或使用标准配置文件：

```bash
bash train_kaggle.sh --config kaggle_config.yaml
```

**配置特点**:
- 标准训练轮数（tokenizer: 20, basemodel: 15）
- 适合大多数场景

#### 1.3 多GPU训练（推荐，30-60分钟）

```bash
bash train_kaggle.sh --multi-gpu 8
```

**配置特点**:
- 使用8个GPU加速训练
- 实际批次大小 = batch_size × GPU数量

#### 1.4 精细微调（如果标准训练效果不理想）

```bash
bash train_kaggle.sh --config kaggle_config_stage3.yaml --multi-gpu 8
```

**配置特点**:
- 增加训练轮数（tokenizer: 30, basemodel: 20）
- 降低学习率（更精细的优化）
- 增加批次大小（如果GPU内存允许）

#### 1.5 分阶段训练

```bash
# 只训练tokenizer
bash train_kaggle.sh --skip-basemodel

# 只训练basemodel（需要先有训练好的tokenizer）
bash train_kaggle.sh --skip-tokenizer
```

#### 1.6 继续训练（跳过已存在的模型）

```bash
bash train_kaggle.sh --skip-existing
```

### 阶段2: 推理

#### 2.1 使用预训练模型推理

```bash
cd /mnt/shared-storage-user/zhaoliangliang/code/Kronos/kaggle
python kaggle_inference_pretrained.py
```

**输出**: `kaggle_predictions_pretrained.csv`

#### 2.2 使用微调模型推理

```bash
python kaggle_inference.py
```

**输出**: `kaggle_predictions.csv`

**注意**: 需要先完成模型训练

### 阶段3: 生成提交文件

#### 3.1 使用预训练模型结果

```bash
cd /mnt/shared-storage-user/zhaoliangliang/code/Kronos/kaggle
python generate_submission.py \
    --predictions ../kaggle_predictions_pretrained.csv \
    --test_csv /mnt/shared-storage-user/zhaoliangliang/dataset/predicting-stock-trends-rise-or-fall/test.csv \
    --output ../kaggle_submission_pretrained.csv
```

#### 3.2 使用微调模型结果

```bash
python generate_submission.py \
    --predictions ../kaggle_predictions.csv \
    --test_csv /mnt/shared-storage-user/zhaoliangliang/dataset/predicting-stock-trends-rise-or-fall/test.csv \
    --output ../kaggle_submission.csv
```

### 阶段4: 提交到Kaggle

将生成的提交文件（`kaggle_submission.csv` 或 `kaggle_submission_pretrained.csv`）上传到Kaggle比赛页面。

---

## ⚙️ 训练配置说明

### 配置文件对比

| 配置 | 用途 | Tokenizer Epochs | Basemodel Epochs | 学习率 | 批次大小 | 预计时间 |
|------|------|------------------|------------------|--------|----------|----------|
| `kaggle_config_stage1.yaml` | 快速验证 | 10 | 8 | 标准 | 32 | 1-2小时 |
| `kaggle_config.yaml` | 标准训练 | 20 | 15 | 标准 | 32 | 4-8小时 |
| `kaggle_config_stage3.yaml` | 精细微调 | 30 | 20 | 降低50% | 64 | 6-12小时 |

### 关键参数说明

#### 数据参数
```yaml
data:
  lookback_window: 256    # 历史窗口：使用256个时间点
  predict_window: 1        # 预测窗口：预测下一个时间点
  max_context: 512         # 最大上下文长度
  train_ratio: 0.9         # 训练集比例
  val_ratio: 0.1           # 验证集比例
```

#### 训练参数
```yaml
training:
  tokenizer_epochs: 20              # Tokenizer训练轮数
  basemodel_epochs: 15              # Basemodel训练轮数
  batch_size: 32                    # 批次大小
  tokenizer_learning_rate: 0.0002  # Tokenizer学习率
  predictor_learning_rate: 0.00004 # Predictor学习率
  accumulation_steps: 1             # 梯度累积步数
```

---

## 🔍 推理和提交

### 推理脚本对比

| 脚本 | 模型来源 | 输出文件 | 使用场景 |
|------|----------|----------|----------|
| `kaggle_inference_pretrained.py` | 预训练模型 | `kaggle_predictions_pretrained.csv` | 快速baseline，无需训练 |
| `kaggle_inference.py` | 微调模型 | `kaggle_predictions.csv` | 使用微调后的模型 |

### 一键运行脚本

#### 预训练模型（推荐首次使用）

```bash
cd /mnt/shared-storage-user/zhaoliangliang/code/Kronos/kaggle
python run_pretrained_inference.py
```

这会自动完成：
1. 使用预训练模型推理
2. 生成提交文件

#### 微调模型

```bash
cd /mnt/shared-storage-user/zhaoliangliang/code/Kronos/kaggle
python run_pretrained_inference.py  # 需要修改脚本中的模型路径
```

### 提交文件格式

Kaggle要求的格式：
```csv
ID,Pred
ticker_1,1
ticker_2,0
...
```

- `ID`: 测试样本ID（ticker名称）
- `Pred`: 预测结果（1=涨，0=跌）

---

## 💡 微调建议

### 1. 训练策略选择

#### 策略A: 渐进式训练（推荐）

```bash
# 步骤1: 快速验证（1-2小时）
bash train_kaggle.sh --config kaggle_config_stage1.yaml

# 步骤2: 如果验证通过，进行标准训练（4-8小时）
bash train_kaggle.sh --config kaggle_config.yaml --multi-gpu 8

# 步骤3: 如果效果不理想，进行精细微调（6-12小时）
bash train_kaggle.sh --config kaggle_config_stage3.yaml --skip-existing --multi-gpu 8
```

#### 策略B: 直接标准训练

```bash
# 如果时间充足，直接进行标准训练
bash train_kaggle.sh --multi-gpu 8
```

#### 策略C: 快速迭代

```bash
# 只训练basemodel（更快，假设tokenizer已经训练好）
bash train_kaggle.sh --skip-tokenizer --multi-gpu 8
```

### 2. 超参数调整建议

#### 学习率调整

**如果验证损失不下降**:
```yaml
# 降低学习率
tokenizer_learning_rate: 0.0001   # 从0.0002降低
predictor_learning_rate: 0.00002  # 从0.00004降低
```

**如果训练不稳定（损失震荡）**:
```yaml
# 降低学习率 + 增加梯度累积
tokenizer_learning_rate: 0.0001
predictor_learning_rate: 0.00002
accumulation_steps: 2
```

**如果收敛太慢**:
```yaml
# 适当提高学习率（不超过2倍）
tokenizer_learning_rate: 0.0003
predictor_learning_rate: 0.00006
```

#### 批次大小调整

**GPU内存充足**:
```yaml
batch_size: 64  # 从32增加到64
```

**GPU内存不足**:
```yaml
batch_size: 16
accumulation_steps: 2  # 等效批次大小 = 16 × 2 = 32
```

**多GPU训练**:
- 实际批次大小 = `batch_size × num_gpus`
- 例如：`batch_size: 32` + `8 GPUs` = 实际批次256

#### 训练轮数调整

**数据量大（20M+行）**:
```yaml
tokenizer_epochs: 15  # 可以减少
basemodel_epochs: 12
```

**数据量小（<1M行）**:
```yaml
tokenizer_epochs: 30  # 需要增加
basemodel_epochs: 25
```

**早停策略**:
- 监控验证损失
- 如果连续5个epoch不下降，考虑提前停止

#### 窗口大小调整

**短期趋势预测**:
```yaml
lookback_window: 128
max_context: 256
```

**长期趋势预测**:
```yaml
lookback_window: 512
max_context: 1024
```

**平衡策略**（推荐）:
```yaml
lookback_window: 256  # 保持当前配置
max_context: 512
```

### 3. 针对Kaggle比赛的优化

#### 多ticker数据处理

**当前方式**: 全局归一化（所有ticker使用相同的归一化参数）

**可尝试**: 按ticker归一化
- 优点: 保留每个ticker的相对波动模式
- 缺点: 需要修改数据预处理脚本

#### 二分类任务优化

**当前逻辑**: 
```python
prediction = 1 if predicted_close > current_close else 0
```

**可优化**:
```python
# 添加阈值，避免微小波动
threshold = 0.001  # 0.1%的价格变化
prediction = 1 if (predicted_close - current_close) / current_close > threshold else 0
```

#### 类别不平衡处理

如果涨跌分布不平衡：
1. 检查类别分布
2. 使用加权损失函数
3. 调整评估指标（使用AUC而不是准确率）

### 4. 实验对比建议

创建多个配置文件进行对比：

```bash
# 实验1: 标准配置
bash train_kaggle.sh --config kaggle_config.yaml --multi-gpu 8

# 实验2: 更大的批次大小
# 编辑配置文件: batch_size: 64
bash train_kaggle.sh --config kaggle_config_exp2.yaml --multi-gpu 8

# 实验3: 更小的学习率
# 编辑配置文件: predictor_learning_rate: 0.00002
bash train_kaggle.sh --config kaggle_config_exp3.yaml --multi-gpu 8

# 实验4: 更大的窗口
# 编辑配置文件: lookback_window: 512
bash train_kaggle.sh --config kaggle_config_exp4.yaml --multi-gpu 8
```

### 5. 性能优化建议

#### 训练速度优化

```yaml
# 增加数据加载并行度
num_workers: 8  # 根据CPU核心数调整

# 使用多GPU
# 8个GPU可以显著加速训练

# 增加批次大小（如果内存允许）
batch_size: 64
```

#### 内存优化

```yaml
# 如果GPU内存不足
batch_size: 16
accumulation_steps: 2
lookback_window: 128
max_context: 256
```

---

## 🚨 常见问题

### Q1: 训练时GPU内存不足

**解决方案**:
1. 减小 `batch_size`（32 → 16）
2. 增加 `accumulation_steps`（1 → 2）
3. 减小 `lookback_window`（256 → 128）
4. 减小 `max_context`（512 → 256）

### Q2: 训练速度太慢

**解决方案**:
1. 使用多GPU训练：`--multi-gpu 8`
2. 增加 `batch_size`（如果内存允许）
3. 增加 `num_workers`（4 → 8）

### Q3: 验证损失不下降

**解决方案**:
1. 降低学习率（降低2-5倍）
2. 增加训练轮数
3. 检查数据质量
4. 尝试不同的随机种子

### Q4: 过拟合（训练损失下降，验证损失上升）

**解决方案**:
1. 增加 `adam_weight_decay`（0.1 → 0.2）
2. 减少训练轮数
3. 使用更多训练数据
4. 使用dropout（如果模型支持）

### Q5: 推理时找不到模型文件

**解决方案**:
1. 检查模型路径是否正确
2. 确保训练已完成
3. 检查 `kaggle_finetuned/kaggle_stock_trends/` 目录

### Q6: 提交文件格式错误

**解决方案**:
1. 检查 `test_id` 列是否存在
2. 确保 `test.csv` 路径正确
3. 检查预测结果文件格式

### Q7: 预测结果全部为0或1

**可能原因**:
- 模型未正确训练
- 数据预处理有问题
- 阈值设置不当

**解决方案**:
1. 检查训练日志
2. 验证数据格式
3. 调整预测阈值

---

## 📁 文件说明

### 核心脚本

| 文件 | 用途 | 输入 | 输出 |
|------|------|------|------|
| `kaggle_data_preprocess.py` | 数据预处理 | Kaggle原始数据 | Kronos格式数据 |
| `validate_data_format.py` | 数据格式验证 | 训练数据 | 验证报告 |
| `train_kaggle.sh` | 训练启动脚本 | 配置文件 | 微调模型 |
| `kaggle_inference_pretrained.py` | 预训练模型推理 | 测试数据 | 预测结果 |
| `kaggle_inference.py` | 微调模型推理 | 测试数据 | 预测结果 |
| `generate_submission.py` | 生成提交文件 | 预测结果 | Kaggle提交文件 |
| `run_pretrained_inference.py` | 一键推理+提交 | - | 提交文件 |

### 配置文件

| 文件 | 用途 | 特点 |
|------|------|------|
| `kaggle_config.yaml` | 标准配置 | 推荐用于正式训练 |
| `kaggle_config_stage1.yaml` | 快速验证 | 减少epoch，快速测试 |
| `kaggle_config_stage3.yaml` | 精细微调 | 降低学习率，增加epoch |

### 文档

| 文件 | 内容 |
|------|------|
| `KAGGLE_README.md` | 快速开始指南 |
| `KAGGLE_FINETUNING_GUIDE.md` | 详细微调指南（推荐阅读） |
| `FINETUNING_GUIDE.md` | 通用微调指南 |
| `PRETRAINED_INFERENCE_README.md` | 预训练模型推理指南 |
| `COMPLETE_GUIDE.md` | 本文件，完整指南 |

### 输出文件

| 文件 | 说明 |
|------|------|
| `kaggle_predictions_pretrained.csv` | 预训练模型预测结果 |
| `kaggle_predictions.csv` | 微调模型预测结果 |
| `kaggle_submission_pretrained.csv` | 预训练模型提交文件 |
| `kaggle_submission.csv` | 微调模型提交文件 |

---

## 📊 训练检查清单

### 训练前
- [ ] 数据预处理完成
- [ ] 数据格式验证通过
- [ ] 配置文件路径正确
- [ ] 预训练模型路径正确
- [ ] GPU可用且数量正确
- [ ] 输出目录有写权限

### 训练中
- [ ] 训练损失正常下降
- [ ] 验证损失正常下降
- [ ] GPU利用率正常（>80%）
- [ ] 没有内存溢出错误
- [ ] 日志正常记录
- [ ] 模型检查点正常保存

### 训练后
- [ ] 模型文件已保存
- [ ] 验证损失达到预期
- [ ] 可以正常加载模型
- [ ] 推理脚本可以正常运行
- [ ] 预测结果格式正确
- [ ] 提交文件格式正确

---

## 🎯 推荐工作流程

### 首次使用（快速验证）

```bash
# 1. 数据预处理
python kaggle_data_preprocess.py
python validate_data_format.py

# 2. 使用预训练模型快速提交
python run_pretrained_inference.py

# 3. 提交到Kaggle，获得baseline分数
```

### 正式训练（追求更好效果）

```bash
# 1. 快速验证训练流程
bash train_kaggle.sh --config kaggle_config_stage1.yaml

# 2. 标准训练
bash train_kaggle.sh --config kaggle_config.yaml --multi-gpu 8

# 3. 推理和提交
python kaggle_inference.py
python generate_submission.py --predictions ../kaggle_predictions.csv \
    --test_csv /mnt/shared-storage-user/zhaoliangliang/dataset/predicting-stock-trends-rise-or-fall/test.csv \
    --output ../kaggle_submission.csv

# 4. 如果效果不理想，进行精细微调
bash train_kaggle.sh --config kaggle_config_stage3.yaml --skip-existing --multi-gpu 8
```

### 实验对比（优化超参数）

```bash
# 创建多个配置文件，对比不同超参数
# 实验1: 标准配置
bash train_kaggle.sh --config kaggle_config.yaml --multi-gpu 8

# 实验2: 更大的批次
# 编辑配置文件后
bash train_kaggle.sh --config kaggle_config_exp2.yaml --multi-gpu 8

# 实验3: 更小的学习率
# 编辑配置文件后
bash train_kaggle.sh --config kaggle_config_exp3.yaml --multi-gpu 8

# 对比结果，选择最佳配置
```

---

## 📈 性能参考

### 训练时间估算

| 配置 | 单GPU | 8 GPU |
|------|-------|-------|
| 快速验证（stage1） | 1-2小时 | 10-20分钟 |
| 标准训练 | 4-8小时 | 30-60分钟 |
| 精细微调（stage3） | 6-12小时 | 1-2小时 |

*注：实际时间取决于数据量、批次大小、GPU型号等因素*

### 推理时间

- **预训练模型**: 约5000个样本，单GPU预计10-30分钟
- **微调模型**: 约5000个样本，单GPU预计10-30分钟

### 内存占用

- **训练**: 约4-8GB GPU内存（取决于批次大小）
- **推理**: 约2-4GB GPU内存

---

## 🎓 最佳实践总结

1. **从简单开始**: 先用预训练模型获得baseline
2. **渐进式训练**: 快速验证 → 标准训练 → 精细微调
3. **监控验证损失**: 这是判断模型性能的关键指标
4. **多实验对比**: 尝试不同的超参数配置
5. **数据质量**: 确保数据预处理正确，数据质量比模型更重要
6. **时间序列特性**: 注意时间序列的特殊性，避免数据泄露
7. **保存检查点**: 定期保存模型，防止训练中断
8. **记录实验**: 记录每次实验的超参数和结果

---

## 📚 相关文档

- **快速开始**: `KAGGLE_README.md`
- **详细微调指南**: `KAGGLE_FINETUNING_GUIDE.md`（强烈推荐）
- **通用微调指南**: `FINETUNING_GUIDE.md`
- **预训练模型推理**: `PRETRAINED_INFERENCE_README.md`

---

**祝你在Kaggle比赛中取得好成绩！** 🏆

如有问题，请参考相关文档或检查常见问题部分。

