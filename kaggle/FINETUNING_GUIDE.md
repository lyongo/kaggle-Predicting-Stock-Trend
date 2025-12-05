# Kronos Kaggle 股票趋势预测微调指南

## 📋 目录
1. [快速开始](#快速开始)
2. [训练配置说明](#训练配置说明)
3. [微调建议](#微调建议)
4. [训练策略](#训练策略)
5. [常见问题](#常见问题)
6. [性能优化](#性能优化)

---

## 🚀 快速开始

### 1. 单GPU训练（推荐用于测试）
```bash
cd /mnt/shared-storage-user/zhaoliangliang/code/Kronos/kaggle
bash train_kaggle.sh
```

### 2. 多GPU训练（推荐用于生产）
```bash
# 使用8个GPU
bash train_kaggle.sh --multi-gpu 8

# 或使用4个GPU
bash train_kaggle.sh --num-gpus 4
```

### 3. 分阶段训练
```bash
# 只训练tokenizer
bash train_kaggle.sh --skip-basemodel

# 只训练basemodel（需要先有训练好的tokenizer）
bash train_kaggle.sh --skip-tokenizer
```

---

## ⚙️ 训练配置说明

### 关键配置参数（`kaggle_config.yaml`）

#### 数据配置
```yaml
data:
  data_path: "/path/to/train_combined.csv"  # 合并的训练数据
  lookback_window: 256      # 历史窗口：使用256个时间点预测未来
  predict_window: 1          # 预测窗口：预测下一个时间点
  max_context: 512          # 最大上下文长度
  clip: 5.0                 # 数据裁剪范围（防止异常值）
  train_ratio: 0.9          # 训练集比例
  val_ratio: 0.1            # 验证集比例
```

#### 训练超参数
```yaml
training:
  tokenizer_epochs: 20       # Tokenizer训练轮数
  basemodel_epochs: 15       # Basemodel训练轮数
  batch_size: 32             # 批次大小
  tokenizer_learning_rate: 0.0002    # Tokenizer学习率
  predictor_learning_rate: 0.00004   # Predictor学习率
  accumulation_steps: 1      # 梯度累积步数
```

---

## 💡 微调建议

### 1. 学习率调整策略

#### 初始学习率建议
- **Tokenizer**: `0.0002` (当前配置)
- **Predictor**: `0.00004` (当前配置)

#### 学习率调整原则
1. **如果验证损失不下降**：
   - 降低学习率 2-5倍
   - 例如：`predictor_learning_rate: 0.00002` 或 `0.00001`

2. **如果训练不稳定（损失震荡）**：
   - 降低学习率
   - 增加 `accumulation_steps` 到 2-4

3. **如果收敛太慢**：
   - 适当提高学习率（不超过2倍）
   - 但要注意过拟合风险

#### 学习率调度建议
当前配置使用固定学习率。如果需要，可以修改训练脚本添加：
- **Cosine Annealing**: 学习率从初始值逐渐降低到0
- **ReduceLROnPlateau**: 验证损失不下降时降低学习率
- **Warmup**: 前几个epoch使用较小的学习率

### 2. 批次大小调整

#### 当前配置
- `batch_size: 32`

#### 调整建议
- **GPU内存充足**：可以增加到 64 或 128，加快训练速度
- **GPU内存不足**：降低到 16 或 8，或使用梯度累积
- **多GPU训练**：实际批次大小 = `batch_size × num_gpus`

#### 梯度累积
如果批次大小受限，使用梯度累积：
```yaml
training:
  batch_size: 16
  accumulation_steps: 2  # 等效批次大小 = 16 × 2 = 32
```

### 3. 训练轮数调整

#### 当前配置
- `tokenizer_epochs: 20`
- `basemodel_epochs: 15`

#### 调整建议
- **数据量大（>1000万行）**：可以减少到 10-15 epochs
- **数据量小（<100万行）**：可以增加到 30-50 epochs
- **观察验证损失**：如果验证损失不再下降，可以提前停止

#### 早停策略
建议监控验证损失，如果连续3-5个epoch不下降，考虑：
1. 降低学习率
2. 提前停止训练
3. 检查数据质量

### 4. 窗口大小调整

#### 当前配置
- `lookback_window: 256`
- `predict_window: 1`

#### 调整建议
- **增加历史窗口**（256 → 512）：
  - 优点：更多历史信息，可能提高预测准确性
  - 缺点：训练时间增加，内存占用增加
  - 适用：数据充足，GPU内存充足

- **减少历史窗口**（256 → 128）：
  - 优点：训练更快，内存占用更少
  - 缺点：可能丢失长期依赖信息
  - 适用：数据较少，GPU内存受限

- **预测窗口**：保持为1（比赛要求预测下一个时间点）

### 5. 数据增强策略

虽然Kronos主要使用时间序列数据，但可以考虑：

1. **时间窗口滑动**：使用不同的起始点创建更多训练样本
2. **数据归一化**：确保不同ticker的数据在相似范围内
3. **数据清洗**：移除异常值和缺失值

---

## 🎯 训练策略

### 策略1：标准训练流程（推荐）

```bash
# 步骤1: 完整训练（tokenizer + basemodel）
bash train_kaggle.sh --multi-gpu 8

# 步骤2: 如果效果不好，调整学习率后重新训练
# 编辑 kaggle_config.yaml，降低学习率
# 然后使用 --skip-existing 跳过已训练的模型
bash train_kaggle.sh --skip-existing --multi-gpu 8
```

### 策略2：分阶段训练

```bash
# 阶段1: 只训练tokenizer（更快，用于测试）
bash train_kaggle.sh --skip-basemodel --multi-gpu 8

# 阶段2: 基于训练好的tokenizer训练basemodel
bash train_kaggle.sh --skip-tokenizer --multi-gpu 8
```

### 策略3：渐进式微调

1. **第一步**：使用较小的学习率微调tokenizer
   ```yaml
   tokenizer_epochs: 10
   tokenizer_learning_rate: 0.0001  # 降低学习率
   ```

2. **第二步**：使用更小的学习率微调predictor
   ```yaml
   basemodel_epochs: 10
   predictor_learning_rate: 0.00002  # 降低学习率
   ```

3. **第三步**：如果效果不理想，进一步降低学习率继续训练

### 策略4：多实验对比

创建多个配置文件，对比不同超参数：

```bash
# 实验1: 标准配置
cp kaggle_config.yaml kaggle_config_exp1.yaml
bash train_kaggle.sh --config kaggle_config_exp1.yaml

# 实验2: 更大的批次大小
cp kaggle_config.yaml kaggle_config_exp2.yaml
# 编辑 kaggle_config_exp2.yaml: batch_size: 64
bash train_kaggle.sh --config kaggle_config_exp2.yaml

# 实验3: 更小的学习率
cp kaggle_config.yaml kaggle_config_exp3.yaml
# 编辑 kaggle_config_exp3.yaml: predictor_learning_rate: 0.00002
bash train_kaggle.sh --config kaggle_config_exp3.yaml
```

---

## 🔧 常见问题

### Q1: 训练时GPU内存不足
**解决方案**：
1. 减小 `batch_size`（例如：32 → 16）
2. 增加 `accumulation_steps`（例如：1 → 2）
3. 减小 `lookback_window`（例如：256 → 128）
4. 减小 `max_context`（例如：512 → 256）

### Q2: 训练速度太慢
**解决方案**：
1. 使用多GPU训练：`--multi-gpu 8`
2. 增加 `batch_size`（如果内存允许）
3. 增加 `num_workers`（例如：4 → 8）
4. 使用混合精度训练（如果支持）

### Q3: 验证损失不下降
**解决方案**：
1. 降低学习率（降低2-5倍）
2. 增加训练轮数
3. 检查数据质量
4. 尝试不同的随机种子

### Q4: 过拟合（训练损失下降，验证损失上升）
**解决方案**：
1. 增加 `adam_weight_decay`（例如：0.1 → 0.2）
2. 使用更多的训练数据
3. 减少训练轮数
4. 使用dropout（如果模型支持）

### Q5: 训练中断后如何继续
**解决方案**：
```bash
# 使用 --skip-existing 跳过已完成的阶段
bash train_kaggle.sh --skip-existing --multi-gpu 8
```

---

## ⚡ 性能优化

### 1. 数据加载优化
```yaml
training:
  num_workers: 8  # 根据CPU核心数调整，通常为CPU核心数的一半
  batch_size: 64   # 根据GPU内存调整
```

### 2. 分布式训练优化
- 使用 `nccl` 后端（NVIDIA GPU）
- 确保GPU之间高速互联（NVLink/InfiniBand）
- 使用 `--nproc_per_node` 匹配实际GPU数量

### 3. 内存优化
- 使用梯度检查点（如果支持）
- 使用混合精度训练（FP16）
- 定期清理缓存：`torch.cuda.empty_cache()`

### 4. 训练监控
- 实时查看日志：`tail -f kaggle_finetuned/logs/train_*.log`
- 监控GPU使用：`watch -n 1 nvidia-smi`
- 监控训练进度：查看验证损失趋势

---

## 📊 训练监控指标

### 关键指标
1. **训练损失**：应该持续下降
2. **验证损失**：应该下降，但可能比训练损失高
3. **学习率**：如果使用调度器，会逐渐降低
4. **GPU利用率**：应该保持在80%以上

### 预期训练时间（估算）
- **单GPU**：Tokenizer ~2-4小时，Basemodel ~4-8小时
- **8 GPU**：Tokenizer ~20-30分钟，Basemodel ~30-60分钟

*注：实际时间取决于数据量、批次大小、GPU型号等因素*

---

## 🎓 最佳实践总结

1. **从小开始**：先用单GPU测试配置是否正确
2. **监控验证损失**：这是判断模型性能的关键指标
3. **保存检查点**：定期保存模型，防止训练中断
4. **实验记录**：记录每次实验的超参数和结果
5. **逐步调整**：一次只调整一个超参数，便于对比效果
6. **数据质量**：确保数据预处理正确，数据质量比模型更重要

---

## 📝 训练检查清单

训练前：
- [ ] 数据预处理完成
- [ ] 数据格式验证通过
- [ ] 配置文件路径正确
- [ ] 预训练模型路径正确
- [ ] GPU可用且数量正确

训练中：
- [ ] 训练损失正常下降
- [ ] 验证损失正常下降
- [ ] GPU利用率正常（>80%）
- [ ] 没有内存溢出错误
- [ ] 日志正常记录

训练后：
- [ ] 模型文件已保存
- [ ] 验证损失达到预期
- [ ] 可以正常加载模型进行推理

---

**祝训练顺利！** 🚀

