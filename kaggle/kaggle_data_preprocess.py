"""
Kaggle股票趋势预测比赛数据预处理脚本（优化版）
将Kaggle数据集转换为Kronos模型所需的格式
"""
import pandas as pd
import os
from pathlib import Path
import numpy as np
try:
    from tqdm import tqdm
except ImportError:
    # 如果没有tqdm，使用简单的进度显示
    def tqdm(iterable, total=None, desc=None):
        if desc:
            print(desc)
        return iterable

def preprocess_kaggle_data(
    train_csv_path: str,
    test_csv_path: str,
    output_dir: str,
    train_output_dir: str = None,
    test_output_dir: str = None
):
    """
    预处理Kaggle数据集，转换为Kronos格式
    
    Args:
        train_csv_path: 训练集CSV路径
        test_csv_path: 测试集CSV路径
        output_dir: 输出目录
        train_output_dir: 训练数据输出目录（默认：output_dir/train）
        test_output_dir: 测试数据输出目录（默认：output_dir/test）
    """
    print("开始预处理Kaggle数据集...")
    
    # 设置输出目录
    if train_output_dir is None:
        train_output_dir = os.path.join(output_dir, "train")
    if test_output_dir is None:
        test_output_dir = os.path.join(output_dir, "test")
    
    os.makedirs(train_output_dir, exist_ok=True)
    os.makedirs(test_output_dir, exist_ok=True)
    
    # 处理训练集
    print("\n处理训练集...")
    print("读取训练集CSV文件...")
    train_df = pd.read_csv(train_csv_path)
    print(f"训练集总行数: {len(train_df)}")
    print(f"训练集列: {train_df.columns.tolist()}")
    
    # 重命名列以匹配Kronos格式
    train_df = train_df.rename(columns={
        'Date': 'timestamps',
        'Open': 'open',
        'High': 'high',
        'Low': 'low',
        'Close': 'close',
        'Volume': 'volume'
    })
    
    # 添加amount列（如果没有，使用volume*close作为近似值）
    if 'amount' not in train_df.columns:
        train_df['amount'] = train_df['volume'] * train_df['close']
    
    # 确保timestamps是datetime类型
    train_df['timestamps'] = pd.to_datetime(train_df['timestamps'])
    
    # 优化：使用groupby一次性分组，而不是循环过滤
    print("\n按ticker分组处理...")
    train_ticker_info = []
    
    # 使用groupby批量处理所有ticker，避免重复过滤整个DataFrame
    grouped = train_df.groupby('Ticker')
    # 获取ticker数量（groupby对象需要先获取groups或使用size）
    total_tickers = grouped.ngroups
    print(f"训练集包含 {total_tickers} 个ticker")
    
    for ticker, ticker_df in tqdm(grouped, total=total_tickers, desc="处理ticker"):
        # 按时间排序
        ticker_df = ticker_df.sort_values('timestamps').reset_index(drop=True)
        
        # 选择Kronos需要的列
        kronos_df = ticker_df[['timestamps', 'open', 'high', 'low', 'close', 'volume', 'amount']].copy()
        
        # 处理缺失值
        kronos_df = kronos_df.ffill().bfill()
        
        # 移除无效数据（价格为0或负数的行）
        valid_mask = (kronos_df['open'] > 0) & (kronos_df['high'] > 0) & \
                     (kronos_df['low'] > 0) & (kronos_df['close'] > 0)
        kronos_df = kronos_df[valid_mask].reset_index(drop=True)
        
        if len(kronos_df) < 10:  # 数据太少，跳过
            continue
        
        # 保存为CSV
        output_file = os.path.join(train_output_dir, f"{ticker}.csv")
        kronos_df.to_csv(output_file, index=False)
        
        train_ticker_info.append({
            'ticker': ticker,
            'file': output_file,
            'rows': len(kronos_df),
            'start_date': kronos_df['timestamps'].min(),
            'end_date': kronos_df['timestamps'].max()
        })
    
    print(f"\n训练集处理完成，共生成 {len(train_ticker_info)} 个文件")
    
    # 处理测试集
    print("\n处理测试集...")
    test_df = pd.read_csv(test_csv_path)
    print(f"测试集总行数: {len(test_df)}")
    print(f"测试集列: {test_df.columns.tolist()}")
    
    # 优化：使用itertuples替代iterrows，速度更快
    test_ticker_info = []
    test_df['Date'] = pd.to_datetime(test_df['Date'])
    
    print("为测试样本准备历史数据...")
    for row in tqdm(test_df.itertuples(), total=len(test_df), desc="处理测试样本"):
        ticker = row.ID
        test_date = row.Date
        
        # 查找该ticker的训练数据
        ticker_train_file = os.path.join(train_output_dir, f"{ticker}.csv")
        
        if os.path.exists(ticker_train_file):
            ticker_train_df = pd.read_csv(ticker_train_file)
            ticker_train_df['timestamps'] = pd.to_datetime(ticker_train_df['timestamps'])
            
            # 获取测试日期之前的所有历史数据
            historical_df = ticker_train_df[ticker_train_df['timestamps'] <= test_date].copy()
            
            if len(historical_df) < 10:
                continue
            
            # 保存测试用的历史数据
            test_file = os.path.join(test_output_dir, f"{ticker}_{test_date.strftime('%Y%m%d')}.csv")
            historical_df.to_csv(test_file, index=False)
            
            test_ticker_info.append({
                'ticker': ticker,
                'date': test_date,
                'file': test_file,
                'rows': len(historical_df),
                'test_id': row.Index
            })
    
    print(f"测试集处理完成，共生成 {len(test_ticker_info)} 个文件")
    
    # 保存元数据
    train_info_df = pd.DataFrame(train_ticker_info)
    train_info_df.to_csv(os.path.join(output_dir, "train_ticker_info.csv"), index=False)
    
    test_info_df = pd.DataFrame(test_ticker_info)
    test_info_df.to_csv(os.path.join(output_dir, "test_ticker_info.csv"), index=False)
    
    # 合并所有ticker的训练数据用于Kronos训练
    print("\n合并训练数据...")
    combined_data = []
    for ticker_info in tqdm(train_ticker_info, desc="合并数据"):
        ticker_df = pd.read_csv(ticker_info['file'])
        combined_data.append(ticker_df)
    
    if combined_data:
        combined_df = pd.concat(combined_data, ignore_index=True)
        combined_df = combined_df.sort_values('timestamps').reset_index(drop=True)
        combined_output = os.path.join(output_dir, "train_combined.csv")
        print(f"保存合并的训练数据到: {combined_output}")
        combined_df.to_csv(combined_output, index=False)
        print(f"合并数据总行数: {len(combined_df)}")
    
    print(f"\n预处理完成！")
    print(f"训练数据保存在: {train_output_dir}")
    print(f"测试数据保存在: {test_output_dir}")
    print(f"元数据保存在: {output_dir}")
    
    return train_ticker_info, test_ticker_info


if __name__ == "__main__":
    # 设置路径
    dataset_dir = "/mnt/shared-storage-user/zhaoliangliang/dataset/predicting-stock-trends-rise-or-fall"
    train_csv = os.path.join(dataset_dir, "train.csv")
    test_csv = os.path.join(dataset_dir, "test.csv")
    
    output_dir = "/mnt/shared-storage-user/zhaoliangliang/code/Kronos/kaggle_data"
    
    # 运行预处理
    train_info, test_info = preprocess_kaggle_data(
        train_csv_path=train_csv,
        test_csv_path=test_csv,
        output_dir=output_dir
    )
    
    print(f"\n统计信息:")
    print(f"训练集ticker数量: {len(train_info)}")
    print(f"测试集样本数量: {len(test_info)}")

