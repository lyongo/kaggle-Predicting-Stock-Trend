"""
生成Kaggle提交文件
将预测结果格式化为Kaggle要求的提交格式
"""
import pandas as pd
import os
import argparse


def generate_submission(
    predictions_path: str,
    test_csv_path: str,
    output_path: str
):
    """
    生成Kaggle提交文件
    
    Args:
        predictions_path: 预测结果CSV路径（包含test_id和prediction列）
        test_csv_path: 原始测试集CSV路径（包含ID列）
        output_path: 输出提交文件路径
    """
    print("加载预测结果...")
    pred_df = pd.read_csv(predictions_path)
    print(f"预测结果数量: {len(pred_df)}")
    
    print("加载测试集...")
    test_df = pd.read_csv(test_csv_path)
    print(f"测试集数量: {len(test_df)}")
    
    # 创建ID到预测结果的映射
    # 如果predictions中有test_id，使用它；否则使用ticker和date匹配
    if 'test_id' in pred_df.columns:
        # test_id是测试集的行索引（从0开始）
        pred_dict = dict(zip(pred_df['test_id'], pred_df['prediction']))
        submission_df = test_df.copy()
        # 使用索引映射（test_id对应test_df的行索引）
        submission_df['Pred'] = submission_df.index.map(lambda x: pred_dict.get(x, 0))
    else:
        # 使用ticker和date匹配
        test_df['Date'] = pd.to_datetime(test_df['Date'])
        pred_df['date'] = pd.to_datetime(pred_df['date'])
        
        # 创建匹配键
        test_df['key'] = test_df['ID'] + '_' + test_df['Date'].astype(str)
        pred_df['key'] = pred_df['ticker'] + '_' + pred_df['date'].astype(str)
        
        pred_dict = dict(zip(pred_df['key'], pred_df['prediction']))
        submission_df = test_df.copy()
        submission_df['key'] = submission_df['ID'] + '_' + submission_df['Date'].astype(str)
        submission_df['Pred'] = submission_df['key'].map(lambda x: pred_dict.get(x, 0))
        submission_df = submission_df.drop('key', axis=1)
    
    # 确保Pred列是整数类型
    submission_df['Pred'] = submission_df['Pred'].astype(int)
    
    # 选择需要的列（ID和Pred）
    submission_df = submission_df[['ID', 'Pred']]
    
    # 保存提交文件
    submission_df.to_csv(output_path, index=False)
    print(f"\n提交文件已生成: {output_path}")
    print(f"提交文件统计:")
    print(f"  总样本数: {len(submission_df)}")
    print(f"  涨(1): {sum(submission_df['Pred'] == 1)}")
    print(f"  跌(0): {sum(submission_df['Pred'] == 0)}")
    print(f"\n前10行预览:")
    print(submission_df.head(10))
    
    return submission_df


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='生成Kaggle提交文件')
    parser.add_argument('--predictions', type=str, 
                       default="/mnt/shared-storage-user/zhaoliangliang/code/Kronos/kaggle_predictions.csv",
                       help='预测结果CSV路径')
    parser.add_argument('--test_csv', type=str,
                       default="/mnt/shared-storage-user/zhaoliangliang/dataset/predicting-stock-trends-rise-or-fall/test.csv",
                       help='原始测试集CSV路径')
    parser.add_argument('--output', type=str,
                       default="/mnt/shared-storage-user/zhaoliangliang/code/Kronos/kaggle_submission.csv",
                       help='输出提交文件路径')
    
    args = parser.parse_args()
    
    # 生成提交文件
    submission = generate_submission(
        predictions_path=args.predictions,
        test_csv_path=args.test_csv,
        output_path=args.output
    )

