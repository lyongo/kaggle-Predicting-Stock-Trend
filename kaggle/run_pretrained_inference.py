#!/usr/bin/env python3
"""
使用预训练Kronos模型进行推理并生成Kaggle提交文件
一键运行脚本
"""
import os
import sys

# 添加路径
sys.path.append('/mnt/shared-storage-user/zhaoliangliang/code/Kronos/kaggle')

# 导入推理和提交函数
from kaggle_inference_pretrained import predict_trend_pretrained
from generate_submission import generate_submission

def main():
    print("=" * 70)
    print("使用预训练Kronos模型进行Kaggle预测和提交")
    print("=" * 70)
    
    # 路径配置
    pretrained_tokenizer_path = "/mnt/shared-storage-user/zhaoliangliang/model/Kronos-Tokenizer-base"
    pretrained_model_path = "/mnt/shared-storage-user/zhaoliangliang/model/Kronos-base"
    test_info_path = "/mnt/shared-storage-user/zhaoliangliang/code/Kronos/kaggle_data/test_ticker_info.csv"
    predictions_path = "/mnt/shared-storage-user/zhaoliangliang/code/Kronos/kaggle_predictions_pretrained.csv"
    test_csv_path = "/mnt/shared-storage-user/zhaoliangliang/dataset/predicting-stock-trends-rise-or-fall/test.csv"
    submission_path = "/mnt/shared-storage-user/zhaoliangliang/code/Kronos/kaggle_submission_pretrained.csv"
    
    # 检查路径
    print("\n检查路径...")
    if not os.path.exists(pretrained_tokenizer_path):
        print(f"❌ 错误: Tokenizer路径不存在: {pretrained_tokenizer_path}")
        return 1
    
    if not os.path.exists(pretrained_model_path):
        print(f"❌ 错误: 模型路径不存在: {pretrained_model_path}")
        return 1
    
    if not os.path.exists(test_info_path):
        print(f"❌ 错误: 测试集信息文件不存在: {test_info_path}")
        print("   请先运行 kaggle_data_preprocess.py")
        return 1
    
    print("✓ 所有路径检查通过")
    
    # 步骤1: 运行推理
    print("\n" + "=" * 70)
    print("步骤1: 运行推理")
    print("=" * 70)
    
    try:
        predictions = predict_trend_pretrained(
            model_path=pretrained_model_path,
            tokenizer_path=pretrained_tokenizer_path,
            test_data_path="",
            test_info_path=test_info_path,
            output_path=predictions_path,
            device="cuda:0",
            lookback=256,
            pred_len=1
        )
        print("✓ 推理完成")
    except Exception as e:
        print(f"❌ 推理失败: {str(e)}")
        import traceback
        traceback.print_exc()
        return 1
    
    # 步骤2: 生成提交文件
    print("\n" + "=" * 70)
    print("步骤2: 生成Kaggle提交文件")
    print("=" * 70)
    
    try:
        submission = generate_submission(
            predictions_path=predictions_path,
            test_csv_path=test_csv_path,
            output_path=submission_path
        )
        print("✓ 提交文件生成完成")
    except Exception as e:
        print(f"❌ 提交文件生成失败: {str(e)}")
        import traceback
        traceback.print_exc()
        return 1
    
    # 完成
    print("\n" + "=" * 70)
    print("✅ 完成！")
    print("=" * 70)
    print(f"预测结果: {predictions_path}")
    print(f"提交文件: {submission_path}")
    print("\n可以将提交文件上传到Kaggle进行提交")
    print("=" * 70)
    
    return 0

if __name__ == "__main__":
    sys.exit(main())

