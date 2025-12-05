# 使用预训练模型进行Kaggle预测

本指南说明如何使用Kronos预训练模型（无需微调）直接进行预测并生成Kaggle提交文件。

## 快速开始

### 方式1: 使用Python脚本（推荐）

```bash
cd /mnt/shared-storage-user/zhaoliangliang/code/Kronos/kaggle
python run_pretrained_inference.py
```

这个脚本会自动：
1. 检查所有必要的路径
2. 使用预训练模型进行推理
3. 生成Kaggle提交文件

### 方式2: 使用Shell脚本

```bash
cd /mnt/shared-storage-user/zhaoliangliang/code/Kronos/kaggle
bash run_pretrained_inference.sh
```

### 方式3: 分步执行

#### 步骤1: 运行推理

```bash
cd /mnt/shared-storage-user/zhaoliangliang/code/Kronos/kaggle
python kaggle_inference_pretrained.py
```

这会生成预测结果文件：`kaggle_predictions_pretrained.csv`

#### 步骤2: 生成提交文件

```bash
python generate_submission.py \
    --predictions kaggle_predictions_pretrained.csv \
    --test_csv /mnt/shared-storage-user/zhaoliangliang/dataset/predicting-stock-trends-rise-or-fall/test.csv \
    --output kaggle_submission_pretrained.csv
```

## 输出文件

- **预测结果**: `kaggle_predictions_pretrained.csv`
  - 包含每个测试样本的详细预测信息
  - 列：test_id, ticker, date, prediction, predicted_close, current_close, price_change, price_change_pct

- **提交文件**: `kaggle_submission_pretrained.csv`
  - Kaggle要求的格式（ID, Pred列）
  - 可以直接上传到Kaggle

## 配置

预训练模型路径（在 `kaggle_inference_pretrained.py` 中配置）：
- Tokenizer: `/mnt/shared-storage-user/zhaoliangliang/model/Kronos-Tokenizer-base`
- Model: `/mnt/shared-storage-user/zhaoliangliang/model/Kronos-base`

如果需要修改路径，编辑 `kaggle_inference_pretrained.py` 文件。

## 预测参数

当前配置：
- **历史窗口**: 256个时间点
- **预测长度**: 1个时间点（下一个交易日）
- **设备**: cuda:0

可以在 `kaggle_inference_pretrained.py` 中修改这些参数。

## 注意事项

1. **数据预处理**: 确保已经运行 `kaggle_data_preprocess.py` 生成测试数据
2. **GPU可用**: 需要GPU支持（CUDA），如果没有GPU，可以修改 `device="cpu"`
3. **模型路径**: 确保预训练模型路径正确
4. **测试数据**: 确保测试数据文件存在

## 故障排查

### 问题1: 模型路径不存在
```
FileNotFoundError: 模型路径不存在
```
**解决**: 检查预训练模型是否已下载到指定路径

### 问题2: 测试数据不存在
```
FileNotFoundError: 测试集信息文件不存在
```
**解决**: 运行 `python kaggle_data_preprocess.py` 生成测试数据

### 问题3: GPU内存不足
```
CUDA out of memory
```
**解决**: 
- 减小 `lookback` 参数（例如：256 → 128）
- 或使用CPU：修改 `device="cpu"`

### 问题4: 预测结果格式错误
**解决**: 检查 `test_ticker_info.csv` 中是否有 `test_id` 列

## 性能

- **预测时间**: 约5000个样本，单GPU预计需要10-30分钟
- **内存占用**: 约4-8GB GPU内存
- **准确率**: 预训练模型的准确率可能不如微调后的模型，但可以作为baseline

## 下一步

如果预训练模型的效果不理想，可以：
1. 使用微调后的模型（运行 `train_kaggle.sh`）
2. 调整预测参数（历史窗口、预测策略等）
3. 尝试不同的阈值（当前使用简单的价格比较）

---

**祝你在Kaggle比赛中取得好成绩！** 🏆

