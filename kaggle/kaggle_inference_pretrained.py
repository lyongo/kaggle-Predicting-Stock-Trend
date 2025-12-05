"""
Kaggle股票趋势预测推理脚本 - 使用预训练模型
使用Kronos预训练模型对测试集进行预测，并转换为涨跌分类
"""
import pandas as pd
import numpy as np
import os
import sys
import torch
from pathlib import Path
from tqdm import tqdm

# 添加Kronos模型路径
sys.path.append('/mnt/shared-storage-user/zhaoliangliang/code/Kronos')
from model import Kronos, KronosTokenizer, KronosPredictor


def predict_trend_pretrained(
    model_path: str,
    tokenizer_path: str,
    test_data_path: str,
    test_info_path: str,
    output_path: str,
    device: str = "cuda:0",
    lookback: int = 256,
    pred_len: int = 1
):
    """
    使用预训练模型对测试集进行预测
    
    Args:
        model_path: 预训练模型路径
        tokenizer_path: 预训练tokenizer路径
        test_data_path: 测试数据目录
        test_info_path: 测试集信息CSV路径
        output_path: 预测结果输出路径
        device: 设备
        lookback: 历史窗口长度
        pred_len: 预测长度
    """
    print("=" * 60)
    print("使用预训练Kronos模型进行预测")
    print("=" * 60)
    print(f"模型路径: {model_path}")
    print(f"Tokenizer路径: {tokenizer_path}")
    print(f"设备: {device}")
    print(f"历史窗口: {lookback}")
    print("=" * 60)
    
    # 检查模型路径
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"模型路径不存在: {model_path}")
    if not os.path.exists(tokenizer_path):
        raise FileNotFoundError(f"Tokenizer路径不存在: {tokenizer_path}")
    
    print("\n加载模型和tokenizer...")
    try:
        tokenizer = KronosTokenizer.from_pretrained(tokenizer_path, local_files_only=True)
        model = Kronos.from_pretrained(model_path, local_files_only=True)
        predictor = KronosPredictor(model, tokenizer, device=device, max_context=512)
        print("✓ 模型加载成功")
    except Exception as e:
        print(f"❌ 模型加载失败: {str(e)}")
        raise
    
    print("\n加载测试集信息...")
    test_info_df = pd.read_csv(test_info_path)
    print(f"✓ 测试集样本数: {len(test_info_df)}")
    
    predictions = []
    errors = 0
    
    print("\n开始预测...")
    for idx, row in tqdm(test_info_df.iterrows(), total=len(test_info_df), desc="预测进度"):
        ticker = row['ticker']
        test_date = pd.to_datetime(row['date'])
        test_file = row['file']
        test_id = row['test_id']
        
        if not os.path.exists(test_file):
            print(f"\n警告: 测试文件不存在 {test_file}，跳过")
            predictions.append({
                'test_id': test_id,
                'ticker': ticker,
                'date': test_date,
                'prediction': 0,  # 默认预测为跌
                'predicted_close': None,
                'current_close': None
            })
            errors += 1
            continue
        
        try:
            # 读取历史数据
            df = pd.read_csv(test_file)
            df['timestamps'] = pd.to_datetime(df['timestamps'])
            
            if len(df) < lookback:
                # 如果数据不足，使用所有可用数据
                if len(df) < 10:
                    predictions.append({
                        'test_id': test_id,
                        'ticker': ticker,
                        'date': test_date,
                        'prediction': 0,
                        'predicted_close': None,
                        'current_close': df['close'].iloc[-1] if len(df) > 0 else None
                    })
                    errors += 1
                    continue
                # 使用所有可用数据
                historical_df = df.copy()
            else:
                # 获取最后lookback个数据点
                historical_df = df.tail(lookback).copy()
            
            # 准备输入
            x_df = historical_df[['open', 'high', 'low', 'close', 'volume', 'amount']]
            x_timestamp = historical_df['timestamps']
            
            # 生成未来时间戳（预测下一个交易日）
            y_timestamp = pd.Series([test_date])
            
            # 获取当前收盘价
            current_close = historical_df['close'].iloc[-1]
            
            # 进行预测
            pred_df = predictor.predict(
                df=x_df,
                x_timestamp=x_timestamp,
                y_timestamp=y_timestamp,
                pred_len=pred_len,
                T=1.0,
                top_p=0.9,
                sample_count=1,
                verbose=False
            )
            
            # 获取预测的收盘价
            predicted_close = pred_df['close'].iloc[0]
            
            # 转换为分类：如果预测价格 > 当前价格，则为涨(1)，否则为跌(0)
            prediction = 1 if predicted_close > current_close else 0
            
            predictions.append({
                'test_id': test_id,
                'ticker': ticker,
                'date': test_date,
                'prediction': prediction,
                'predicted_close': predicted_close,
                'current_close': current_close,
                'price_change': predicted_close - current_close,
                'price_change_pct': (predicted_close - current_close) / current_close * 100 if current_close > 0 else 0
            })
                
        except Exception as e:
            print(f"\n处理 {ticker} (ID: {test_id}) 时出错: {str(e)}")
            import traceback
            traceback.print_exc()
            predictions.append({
                'test_id': test_id,
                'ticker': ticker,
                'date': test_date,
                'prediction': 0,
                'predicted_close': None,
                'current_close': None
            })
            errors += 1
    
    # 保存预测结果
    pred_df = pd.DataFrame(predictions)
    pred_df.to_csv(output_path, index=False)
    
    print("\n" + "=" * 60)
    print("预测完成！")
    print("=" * 60)
    print(f"结果保存在: {output_path}")
    print(f"\n预测统计:")
    print(f"  总样本数: {len(pred_df)}")
    print(f"  成功预测: {len(pred_df) - errors}")
    print(f"  失败/跳过: {errors}")
    print(f"  涨(1): {sum(pred_df['prediction'] == 1)} ({sum(pred_df['prediction'] == 1) / len(pred_df) * 100:.2f}%)")
    print(f"  跌(0): {sum(pred_df['prediction'] == 0)} ({sum(pred_df['prediction'] == 0) / len(pred_df) * 100:.2f}%)")
    
    # 显示价格变化统计
    valid_predictions = pred_df[pred_df['predicted_close'].notna()]
    if len(valid_predictions) > 0:
        print(f"\n价格预测统计:")
        print(f"  平均价格变化: {valid_predictions['price_change'].mean():.4f}")
        print(f"  平均价格变化率: {valid_predictions['price_change_pct'].mean():.4f}%")
        print(f"  最大涨幅: {valid_predictions['price_change_pct'].max():.2f}%")
        print(f"  最大跌幅: {valid_predictions['price_change_pct'].min():.2f}%")
    
    print("=" * 60)
    
    return pred_df


if __name__ == "__main__":
    # 使用预训练模型路径
    pretrained_tokenizer_path = "/mnt/shared-storage-user/zhaoliangliang/model/Kronos-Tokenizer-base"
    pretrained_model_path = "/mnt/shared-storage-user/zhaoliangliang/model/Kronos-base"
    
    # 测试数据路径
    test_info_path = "/mnt/shared-storage-user/zhaoliangliang/code/Kronos/kaggle_data/test_ticker_info.csv"
    output_path = "/mnt/shared-storage-user/zhaoliangliang/code/Kronos/kaggle_predictions_pretrained.csv"
    
    # 运行预测
    predictions = predict_trend_pretrained(
        model_path=pretrained_model_path,
        tokenizer_path=pretrained_tokenizer_path,
        test_data_path="",  # 不需要，因为test_info中有文件路径
        test_info_path=test_info_path,
        output_path=output_path,
        device="cuda:0",
        lookback=256,
        pred_len=1
    )

