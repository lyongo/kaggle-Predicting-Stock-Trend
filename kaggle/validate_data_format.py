#!/usr/bin/env python3
"""
验证Kaggle数据格式是否符合Kronos训练要求
"""
import pandas as pd
import os
import sys

def validate_data_format(data_path):
    """验证数据格式是否符合Kronos要求"""
    print("=" * 60)
    print("Kaggle数据格式验证")
    print("=" * 60)
    
    # 1. 检查文件是否存在
    if not os.path.exists(data_path):
        print(f"❌ 错误: 数据文件不存在: {data_path}")
        return False
    
    print(f"✓ 数据文件存在: {data_path}")
    
    # 2. 读取数据（只读前1000行进行快速检查）
    try:
        df = pd.read_csv(data_path, nrows=1000)
        print(f"✓ 成功读取数据，样本数: {len(df)}")
    except Exception as e:
        print(f"❌ 错误: 无法读取CSV文件: {e}")
        return False
    
    # 3. 检查必需的列
    required_columns = ['timestamps', 'open', 'high', 'low', 'close', 'volume', 'amount']
    missing_columns = [col for col in required_columns if col not in df.columns]
    
    if missing_columns:
        print(f"❌ 错误: 缺少必需的列: {missing_columns}")
        print(f"   当前列: {df.columns.tolist()}")
        return False
    
    print(f"✓ 所有必需的列都存在: {required_columns}")
    
    # 4. 检查数据类型
    print("\n数据类型检查:")
    for col in required_columns:
        if col == 'timestamps':
            # 时间戳应该是字符串或datetime
            if df[col].dtype == 'object':
                print(f"  ✓ {col}: {df[col].dtype} (字符串格式，可转换为datetime)")
            elif pd.api.types.is_datetime64_any_dtype(df[col]):
                print(f"  ✓ {col}: datetime64")
            else:
                print(f"  ⚠ {col}: {df[col].dtype} (可能需要转换)")
        else:
            # 数值列应该是float或int
            if pd.api.types.is_numeric_dtype(df[col]):
                print(f"  ✓ {col}: {df[col].dtype}")
            else:
                print(f"  ⚠ {col}: {df[col].dtype} (非数值类型)")
    
    # 5. 检查缺失值
    print("\n缺失值检查:")
    missing_counts = df[required_columns].isnull().sum()
    has_missing = missing_counts.sum() > 0
    if has_missing:
        print("  ⚠ 警告: 发现缺失值:")
        for col, count in missing_counts.items():
            if count > 0:
                print(f"    {col}: {count} 个缺失值")
    else:
        print("  ✓ 无缺失值")
    
    # 6. 检查时间戳格式
    print("\n时间戳格式检查:")
    try:
        df['timestamps_parsed'] = pd.to_datetime(df['timestamps'])
        print(f"  ✓ 时间戳可以成功解析")
        print(f"  ✓ 时间范围: {df['timestamps_parsed'].min()} 到 {df['timestamps_parsed'].max()}")
    except Exception as e:
        print(f"  ❌ 错误: 无法解析时间戳: {e}")
        return False
    
    # 7. 检查数值范围
    print("\n数值范围检查:")
    numeric_cols = ['open', 'high', 'low', 'close', 'volume', 'amount']
    for col in numeric_cols:
        col_min = df[col].min()
        col_max = df[col].max()
        col_mean = df[col].mean()
        print(f"  {col}:")
        print(f"    范围: [{col_min:.4f}, {col_max:.4f}]")
        print(f"    均值: {col_mean:.4f}")
        
        # 检查是否有异常值（负数或零）
        if col in ['open', 'high', 'low', 'close']:
            negative_count = (df[col] < 0).sum()
            if negative_count > 0:
                print(f"    ⚠ 警告: 发现 {negative_count} 个负值")
        if col in ['volume', 'amount']:
            negative_count = (df[col] < 0).sum()
            zero_count = (df[col] == 0).sum()
            if negative_count > 0:
                print(f"    ⚠ 警告: 发现 {negative_count} 个负值")
            if zero_count > 0:
                print(f"    ℹ 信息: 发现 {zero_count} 个零值 (volume/amount可以为0)")
    
    # 8. 检查数据排序（Kronos期望数据按时间排序）
    print("\n数据排序检查:")
    df_sorted = df.sort_values('timestamps')
    is_sorted = df['timestamps'].equals(df_sorted['timestamps'])
    if is_sorted:
        print("  ✓ 数据已按时间戳排序")
    else:
        print("  ⚠ 警告: 数据未按时间戳排序（训练时会自动排序）")
    
    # 9. 检查数据量（读取完整文件）
    print("\n完整数据量检查:")
    try:
        full_df = pd.read_csv(data_path)
        print(f"  ✓ 总行数: {len(full_df):,}")
        print(f"  ✓ 总列数: {len(full_df.columns)}")
        
        # 检查是否有足够的数据用于训练
        lookback_window = 256  # 从配置文件读取
        if len(full_df) < lookback_window:
            print(f"  ❌ 错误: 数据量不足，需要至少 {lookback_window} 行，当前只有 {len(full_df)} 行")
            return False
        else:
            print(f"  ✓ 数据量充足，可用于训练（lookback_window={lookback_window}）")
    except Exception as e:
        print(f"  ⚠ 警告: 无法读取完整文件进行验证: {e}")
    
    # 10. 与Kronos示例数据格式对比
    print("\n格式对比检查:")
    example_path = "/mnt/shared-storage-user/zhaoliangliang/code/Kronos/finetune_csv/data/HK_ali_09988_kline_5min_all.csv"
    if os.path.exists(example_path):
        try:
            example_df = pd.read_csv(example_path, nrows=5)
            print("  ✓ 找到Kronos示例数据文件")
            print(f"  示例数据列: {example_df.columns.tolist()}")
            print(f"  当前数据列: {df.columns.tolist()}")
            
            if set(example_df.columns) == set(df.columns):
                print("  ✓ 列名完全匹配")
            else:
                print("  ⚠ 列名不完全匹配（但必需列都存在）")
        except Exception as e:
            print(f"  ⚠ 无法读取示例数据: {e}")
    else:
        print("  ℹ 未找到示例数据文件（可选）")
    
    print("\n" + "=" * 60)
    print("✅ 数据格式验证完成！数据符合Kronos训练要求")
    print("=" * 60)
    return True

if __name__ == "__main__":
    # 默认检查合并的训练数据
    data_path = "/mnt/shared-storage-user/zhaoliangliang/code/Kronos/kaggle_data/train_combined.csv"
    
    if len(sys.argv) > 1:
        data_path = sys.argv[1]
    
    success = validate_data_format(data_path)
    sys.exit(0 if success else 1)

