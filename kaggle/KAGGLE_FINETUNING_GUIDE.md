# Kaggle股票趋势预测比赛 - 专门微调指南

## 📊 比赛特点分析

### 任务特性
- **任务类型**: 二分类（涨/跌预测）
- **数据规模**: 5000个ticker，20M+行数据
- **时间跨度**: 1962-2024年（62年历史数据）
- **预测目标**: 预测下一个时间点的价格，转换为涨(1)/跌(0)分类
- **评估指标**: 分类准确率（通常使用AUC或准确率）

### 关键挑战
1. **多ticker数据**: 不同股票的价格范围、波动性差异很大
2. **长期依赖**: 需要捕捉长期趋势和短期波动
3. **类别不平衡**: 涨跌可能不平衡（需要检查）
4. **回归转分类**: 从连续价格预测转换为二分类

---

## 🎯 针对Kaggle比赛的微调策略

### 策略1: 渐进式微调（推荐）

#### 阶段1: 基础微调（快速验证）
```yaml
# kaggle_config_stage1.yaml
training:
  tokenizer_epochs: 10      # 减少轮数，快速验证
  basemodel_epochs: 8
  batch_size: 32
  tokenizer_learning_rate: 0.0002
  predictor_learning_rate: 0.00004
```

**目标**: 验证数据流程和模型加载是否正确

#### 阶段2: 标准微调（当前配置）
```yaml
# kaggle_config.yaml (当前配置)
training:
  tokenizer_epochs: 20
  basemodel_epochs: 15
  batch_size: 32
  tokenizer_learning_rate: 0.0002
  predictor_learning_rate: 0.00004
```

**目标**: 获得基础性能

#### 阶段3: 精细微调（优化性能）
```yaml
# kaggle_config_stage3.yaml
training:
  tokenizer_epochs: 30      # 增加轮数
  basemodel_epochs: 20
  batch_size: 64            # 增加批次大小（如果内存允许）
  tokenizer_learning_rate: 0.0001   # 降低学习率，更精细
  predictor_learning_rate: 0.00002
  accumulation_steps: 2    # 梯度累积
```

**目标**: 优化模型性能，提高准确率

---

## 🔧 参数调整建议

### 1. 学习率调整（针对二分类任务）

#### 当前配置分析
```yaml
tokenizer_learning_rate: 0.0002
predictor_learning_rate: 0.00004
```

#### 推荐调整

**保守策略**（推荐用于首次训练）:
```yaml
tokenizer_learning_rate: 0.0001      # 降低50%
predictor_learning_rate: 0.00002     # 降低50%
```
**原因**: 
- 二分类任务需要更精确的特征学习
- 较低的学习率有助于稳定训练
- 避免破坏预训练模型的权重

**激进策略**（如果收敛太慢）:
```yaml
tokenizer_learning_rate: 0.0003      # 提高50%
predictor_learning_rate: 0.00006      # 提高50%
```
**风险**: 可能导致训练不稳定，需要密切监控

**自适应策略**（最佳实践）:
- 前5个epoch使用标准学习率
- 如果验证损失不下降，降低学习率
- 使用学习率调度器（Cosine Annealing）

### 2. 批次大小调整

#### 多ticker数据的批次策略

**当前配置**: `batch_size: 32`

**推荐调整**:
```yaml
# 方案1: 增加批次大小（如果GPU内存充足）
batch_size: 64
# 优点: 更稳定的梯度，更快的训练
# 缺点: 需要更多GPU内存

# 方案2: 使用梯度累积（如果GPU内存不足）
batch_size: 16
accumulation_steps: 2
# 优点: 等效批次大小=32，但内存占用减半
# 缺点: 训练时间略增

# 方案3: 多GPU训练（推荐）
batch_size: 32
# 使用8个GPU，实际批次大小 = 32 × 8 = 256
# 优点: 训练速度快，批次大
```

**针对多ticker的特殊考虑**:
- 确保每个批次包含不同ticker的数据
- 避免批次内数据过于相似（可能导致过拟合）

### 3. 训练轮数调整

#### 针对20M+数据的轮数建议

**当前配置**:
```yaml
tokenizer_epochs: 20
basemodel_epochs: 15
```

**推荐调整**:

**数据量大（20M+行）**:
```yaml
tokenizer_epochs: 15      # 可以减少，因为数据量大
basemodel_epochs: 12      # 可以减少
```
**原因**: 数据量大，每个epoch已经看到足够多的样本

**数据量小（<1M行）**:
```yaml
tokenizer_epochs: 30      # 需要增加
basemodel_epochs: 25      # 需要增加
```
**原因**: 需要更多轮数才能充分学习

**早停策略**（推荐）:
- 监控验证损失
- 如果连续5个epoch验证损失不下降，提前停止
- 保存最佳模型（基于验证损失）

### 4. 窗口大小调整

#### 针对股票趋势预测的窗口策略

**当前配置**:
```yaml
lookback_window: 256
predict_window: 1
max_context: 512
```

**推荐调整**:

**短期趋势预测**（捕捉短期波动）:
```yaml
lookback_window: 128      # 减少窗口
max_context: 256
```
**适用场景**: 
- 关注短期价格波动
- 快速响应市场变化
- 计算资源有限

**长期趋势预测**（捕捉长期趋势）:
```yaml
lookback_window: 512      # 增加窗口
max_context: 1024         # 增加上下文
```
**适用场景**:
- 关注长期趋势
- 捕捉周期性模式
- GPU内存充足

**平衡策略**（推荐）:
```yaml
lookback_window: 256      # 保持当前配置
max_context: 512          # 保持当前配置
```
**原因**: 
- 平衡短期和长期信息
- 适合大多数股票预测场景
- 计算效率高

### 5. 数据裁剪参数

**当前配置**: `clip: 5.0`

**推荐调整**:
```yaml
# 方案1: 更严格的裁剪（处理异常值）
clip: 3.0
# 优点: 减少异常值影响
# 缺点: 可能丢失重要信息

# 方案2: 更宽松的裁剪（保留更多信息）
clip: 7.0
# 优点: 保留更多价格波动信息
# 缺点: 异常值可能影响训练

# 方案3: 自适应裁剪（推荐）
# 根据每个ticker的统计特性动态裁剪
# 需要修改数据预处理脚本
```

---

## 📈 数据处理建议

### 1. 多ticker数据归一化

#### 问题
不同ticker的价格范围差异很大（例如：$0.10 vs $100）

#### 解决方案

**方案1: 全局归一化**（当前使用）
- 所有ticker使用相同的归一化参数
- 优点: 简单，保持相对关系
- 缺点: 不同ticker的绝对价格差异被忽略

**方案2: 按ticker归一化**（推荐尝试）
```python
# 在数据预处理时，对每个ticker单独归一化
for ticker in tickers:
    ticker_df = normalize_by_ticker(ticker_df)
```
- 优点: 保留每个ticker的相对波动模式
- 缺点: 不同ticker之间的绝对关系丢失

**方案3: 混合归一化**（最佳）
- 价格特征（open, high, low, close）按ticker归一化
- 交易量特征（volume, amount）全局归一化
- 优点: 平衡相对和绝对信息

### 2. 类别不平衡处理

#### 检查类别分布
```python
# 在训练数据中检查涨跌分布
train_df = pd.read_csv('train_combined.csv')
# 计算每个样本的涨跌标签（需要根据实际数据计算）
```

#### 处理策略

**如果类别不平衡**:
1. **数据采样**:
   - 过采样少数类
   - 欠采样多数类
   - SMOTE等高级采样技术

2. **损失函数调整**:
   - 使用加权损失函数
   - 给少数类更高权重

3. **评估指标**:
   - 使用AUC而不是准确率
   - 使用F1-score（平衡精确率和召回率）

### 3. 时间序列数据增强

#### 数据增强策略

**方案1: 时间窗口滑动**
```python
# 使用不同的起始点创建更多训练样本
# 例如：从第0、10、20个时间点开始，创建多个样本
```

**方案2: 噪声注入**
```python
# 对价格数据添加小幅随机噪声
# 提高模型鲁棒性
noise = np.random.normal(0, 0.01, size=data.shape)
augmented_data = data + noise
```

**方案3: 时间扭曲**
```python
# 轻微调整时间序列的节奏
# 模拟不同的市场速度
```

**注意**: 数据增强需要谨慎，避免破坏时间序列的时序关系

### 4. 特征工程建议

#### 当前特征
- `open`, `high`, `low`, `close`, `volume`, `amount`

#### 可添加的特征

**技术指标**（如果预处理支持）:
- 移动平均线（MA）
- 相对强弱指数（RSI）
- MACD
- 布林带

**时间特征**（Kronos自动提取）:
- `minute`, `hour`, `weekday`, `day`, `month`
- 这些特征已经由Kronos自动生成

**价格变化特征**:
- 涨跌幅: `(close - open) / open`
- 波动率: `(high - low) / close`
- 成交量变化率

**注意**: 添加特征需要修改数据预处理脚本

---

## 🎯 训练策略建议

### 策略1: 完整训练流程（推荐）

```bash
# 步骤1: 快速验证（1-2小时）
bash train_kaggle.sh --config kaggle_config_stage1.yaml

# 步骤2: 标准训练（4-8小时）
bash train_kaggle.sh --config kaggle_config.yaml --multi-gpu 8

# 步骤3: 精细微调（如果效果不理想）
# 修改配置文件，降低学习率
bash train_kaggle.sh --config kaggle_config_stage3.yaml --skip-existing --multi-gpu 8
```

### 策略2: 分阶段训练

```bash
# 阶段1: 只训练tokenizer（更快）
bash train_kaggle.sh --skip-basemodel --multi-gpu 8

# 阶段2: 基于训练好的tokenizer训练basemodel
bash train_kaggle.sh --skip-tokenizer --multi-gpu 8
```

### 策略3: 多实验对比

```bash
# 实验1: 标准配置
bash train_kaggle.sh --config kaggle_config_exp1.yaml --multi-gpu 8

# 实验2: 更大的窗口
# 编辑配置文件: lookback_window: 512
bash train_kaggle.sh --config kaggle_config_exp2.yaml --multi-gpu 8

# 实验3: 更小的学习率
# 编辑配置文件: predictor_learning_rate: 0.00002
bash train_kaggle.sh --config kaggle_config_exp3.yaml --multi-gpu 8
```

---

## 🔍 针对二分类任务的特殊优化

### 1. 损失函数考虑

**当前**: Kronos使用MSE损失（回归任务）

**二分类任务的特殊性**:
- 模型预测连续价格值
- 然后转换为二分类（涨/跌）
- 需要确保价格预测的准确性，特别是方向预测

**优化建议**:
1. **方向损失**: 在训练时，可以添加方向预测损失
   ```python
   # 伪代码
   direction_loss = binary_cross_entropy(
       (predicted_price > current_price).float(),
       (true_price > current_price).float()
   )
   total_loss = mse_loss + alpha * direction_loss
   ```
   **注意**: 这需要修改训练脚本

2. **阈值优化**: 在推理时，可以调整涨跌判断阈值
   ```python
   # 当前: prediction = 1 if predicted_close > current_close else 0
   # 优化: 添加阈值
   threshold = 0.001  # 0.1%的价格变化才认为是涨
   prediction = 1 if (predicted_close - current_close) / current_close > threshold else 0
   ```

### 2. 验证策略

#### 时间序列交叉验证

**问题**: 标准K-fold不适合时间序列（会泄露未来信息）

**推荐**: 时间序列交叉验证
```python
# 按时间顺序分割
# 训练集: 前80%的时间
# 验证集: 后20%的时间
train_ratio: 0.8
val_ratio: 0.2
```

**Walk-forward验证**（更严格）:
```python
# 使用滚动窗口
# 训练: 时间t之前的数据
# 验证: 时间t的数据
# 逐步向前滚动
```

### 3. 模型集成

#### 多模型集成策略

**方案1: 多配置集成**
- 训练多个不同配置的模型
- 对预测结果投票或平均

**方案2: 时间窗口集成**
- 使用不同的lookback_window训练多个模型
- 集成预测结果

**方案3: 多ticker分组**
- 按ticker特征分组（例如：按波动率、市值）
- 每组训练专门模型
- 推理时使用对应组的模型

---

## 📊 性能优化建议

### 1. 训练速度优化

```yaml
# 增加数据加载并行度
num_workers: 8  # 根据CPU核心数调整

# 使用多GPU
# 8个GPU可以显著加速训练

# 增加批次大小（如果内存允许）
batch_size: 64
```

### 2. 内存优化

```yaml
# 如果GPU内存不足
batch_size: 16
accumulation_steps: 2
lookback_window: 128  # 减少窗口大小
max_context: 256
```

### 3. 训练监控

**关键指标**:
1. **训练损失**: 应该持续下降
2. **验证损失**: 应该下降，但可能比训练损失高
3. **验证准确率**: 涨跌预测的准确率（需要计算）
4. **GPU利用率**: 应该保持在80%以上

**监控命令**:
```bash
# 实时查看训练日志
tail -f kaggle_finetuned/logs/train_*.log

# 监控GPU使用
watch -n 1 nvidia-smi
```

---

## 🎓 最佳实践总结

### 针对Kaggle比赛的推荐流程

1. **数据验证** ✅
   - [x] 数据格式正确
   - [x] 数据量充足（20M+行）

2. **快速验证**（1-2小时）
   ```bash
   # 使用少量epoch快速验证流程
   bash train_kaggle.sh --config kaggle_config_stage1.yaml
   ```

3. **标准训练**（4-8小时）
   ```bash
   # 使用标准配置训练
   bash train_kaggle.sh --multi-gpu 8
   ```

4. **模型评估**
   - 检查验证损失
   - 在验证集上计算涨跌预测准确率
   - 分析错误案例

5. **精细调优**（如果需要）
   - 调整学习率
   - 调整窗口大小
   - 尝试不同的数据预处理方式

6. **推理和提交**
   ```bash
   # 使用训练好的模型进行预测
   python kaggle_inference.py
   
   # 生成提交文件
   python generate_submission.py
   ```

### 关键建议

1. **从简单开始**: 先用标准配置训练，验证流程
2. **监控验证损失**: 这是判断模型性能的关键
3. **关注方向预测**: 二分类任务中，价格方向比绝对值更重要
4. **多实验对比**: 尝试不同的超参数配置
5. **数据质量**: 确保数据预处理正确，数据质量比模型更重要
6. **时间序列特性**: 注意时间序列的特殊性，避免数据泄露

---

## 🚨 常见问题及解决方案

### Q1: 验证损失不下降
**可能原因**:
- 学习率太高或太低
- 数据质量问题
- 模型容量不足

**解决方案**:
1. 降低学习率（降低2-5倍）
2. 检查数据预处理
3. 增加训练轮数
4. 尝试更大的模型（如果可用）

### Q2: 训练损失下降但验证损失上升（过拟合）
**解决方案**:
1. 增加 `adam_weight_decay`（0.1 → 0.2）
2. 减少训练轮数
3. 使用更多训练数据
4. 添加dropout（如果模型支持）

### Q3: 涨跌预测准确率低
**可能原因**:
- 价格预测误差大
- 阈值设置不当
- 类别不平衡

**解决方案**:
1. 优化价格预测（降低学习率，增加训练轮数）
2. 调整涨跌判断阈值
3. 检查类别分布，使用加权损失

### Q4: 不同ticker预测效果差异大
**解决方案**:
1. 按ticker归一化数据
2. 按ticker特征分组训练
3. 使用ticker-specific的模型

---

## 📝 训练检查清单

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

---

