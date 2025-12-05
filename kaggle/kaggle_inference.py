"""
Kaggle股票趋势预测推理脚本
使用微调后的Kronos模型对测试集进行预测，并转换为涨跌分类
"""
import pandas as pd
import numpy as np
import os
import sys
import torch
from pathlib import Path

# 添加Kronos模型路径
sys.path.append('/mnt/shared-storage-user/zhaoliangliang/code/Kronos')
from model import Kronos, KronosTokenizer, KronosPredictor


def predict_trend(
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
    对测试集进行预测
    
    Args:
        model_path: 微调后的模型路径
        tokenizer_path: 微调后的tokenizer路径
        test_data_path: 测试数据目录
        test_info_path: 测试集信息CSV路径
        output_path: 预测结果输出路径
        device: 设备
        lookback: 历史窗口长度
        pred_len: 预测长度
    """
    print("加载模型和tokenizer...")
    tokenizer = KronosTokenizer.from_pretrained(tokenizer_path, local_files_only=True)
    model = Kronos.from_pretrained(model_path, local_files_only=True)
    predictor = KronosPredictor(model, tokenizer, device=device, max_context=512)
    
    print("加载测试集信息...")
    test_info_df = pd.read_csv(test_info_path)
    print(f"测试集样本数: {len(test_info_df)}")
    
    predictions = []
    
    for idx, row in test_info_df.iterrows():
        ticker = row['ticker']
        test_date = pd.to_datetime(row['date'])
        test_file = row['file']
        test_id = row['test_id']
        
        if not os.path.exists(test_file):
            print(f"警告: 测试文件不存在 {test_file}，跳过")
            predictions.append({
                'test_id': test_id,
                'ticker': ticker,
                'date': test_date,
                'prediction': 0,  # 默认预测为跌
                'predicted_close': None,
                'current_close': None
            })
            continue
        
        try:
            # 读取历史数据
            df = pd.read_csv(test_file)
            df['timestamps'] = pd.to_datetime(df['timestamps'])
            
            if len(df) < lookback:
                print(f"警告: {ticker} 历史数据不足 ({len(df)} < {lookback})，跳过")
                predictions.append({
                    'test_id': test_id,
                    'ticker': ticker,
                    'date': test_date,
                    'prediction': 0,
                    'predicted_close': None,
                    'current_close': df['close'].iloc[-1] if len(df) > 0 else None
                })
                continue
            
            # 获取最后lookback个数据点
            historical_df = df.tail(lookback).copy()
            
            # 准备输入
            x_df = historical_df[['open', 'high', 'low', 'close', 'volume', 'amount']]
            x_timestamp = historical_df['timestamps']
            
            # 生成未来时间戳（预测下一个交易日）
            # 假设下一个交易日是测试日期
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
                'current_close': current_close
            })
            
            if (idx + 1) % 100 == 0:
                print(f"已处理 {idx + 1}/{len(test_info_df)} 个样本")
                
        except Exception as e:
            print(f"处理 {ticker} 时出错: {str(e)}")
            predictions.append({
                'test_id': test_id,
                'ticker': ticker,
                'date': test_date,
                'prediction': 0,
                'predicted_close': None,
                'current_close': None
            })
    
    # 保存预测结果
    pred_df = pd.DataFrame(predictions)
    pred_df.to_csv(output_path, index=False)
    print(f"\n预测完成！结果保存在: {output_path}")
    print(f"预测统计:")
    print(f"  涨(1): {sum(pred_df['prediction'] == 1)}")
    print(f"  跌(0): {sum(pred_df['prediction'] == 0)}")
    
    return pred_df


if __name__ == "__main__":
    # 设置路径
    model_path = "/mnt/shared-storage-user/zhaoliangliang/code/Kronos/kaggle_finetuned/kaggle_stock_trends/basemodel/best_model"
    tokenizer_path = "/mnt/shared-storage-user/zhaoliangliang/code/Kronos/kaggle_finetuned/kaggle_stock_trends/tokenizer/best_model"
    
    test_data_dir = "/mnt/shared-storage-user/zhaoliangliang/code/Kronos/kaggle_data/test"
    test_info_path = "/mnt/shared-storage-user/zhaoliangliang/code/Kronos/kaggle_data/test_ticker_info.csv"
    output_path = "/mnt/shared-storage-user/zhaoliangliang/code/Kronos/kaggle_predictions.csv"
    
    # 运行预测
    predictions = predict_trend(
        model_path=model_path,
        tokenizer_path=tokenizer_path,
        test_data_path=test_data_dir,
        test_info_path=test_info_path,
        output_path=output_path,
        device="cuda:0",
        lookback=256,
        pred_len=1
    )

