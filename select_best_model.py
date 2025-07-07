#!/usr/bin/env python3
"""
分析训练过程中的MRR评估结果，帮助选择最佳模型
"""

import os
import json
import pandas as pd
import argparse
from pathlib import Path

def analyze_mrr_results(csv_file):
    """分析MRR评估结果"""
    if not os.path.exists(csv_file):
        print(f"❌ MRR评估结果文件不存在: {csv_file}")
        return None
    
    # 读取CSV文件
    df = pd.read_csv(csv_file)
    print(f"📊 MRR评估结果分析:")
    print(f"  总评估次数: {len(df)}")
    print(f"  最佳MRR: {df['MRR'].max():.4f}")
    print(f"  最佳MRR位置: Epoch {df.loc[df['MRR'].idxmax(), 'epoch']:.2f}, Steps {df.loc[df['MRR'].idxmax(), 'steps']}")
    
    # 显示所有评估结果
    print(f"\n📈 详细评估结果:")
    for idx, row in df.iterrows():
        print(f"  Epoch {row['epoch']:.2f}, Steps {row['steps']}: MRR = {row['MRR']:.4f}")
    
    return df

def find_checkpoints(model_dir):
    """查找所有checkpoint"""
    model_path = Path(model_dir)
    checkpoints = []
    
    # 查找checkpoint目录
    for item in model_path.iterdir():
        if item.is_dir() and item.name.startswith('checkpoint-'):
            steps = int(item.name.split('-')[1])
            checkpoints.append((steps, item))
    
    # 按步数排序
    checkpoints.sort(key=lambda x: x[0])
    
    print(f"🔍 找到的checkpoint:")
    for steps, path in checkpoints:
        print(f"  Steps {steps}: {path}")
    
    return checkpoints

def select_best_model(model_dir, mrr_csv=None):
    """选择最佳模型"""
    print(f"🎯 模型选择分析")
    print(f"模型目录: {model_dir}")
    
    # 分析MRR结果
    if mrr_csv:
        df = analyze_mrr_results(mrr_csv)
        if df is not None:
            best_steps = df.loc[df['MRR'].idxmax(), 'steps']
            print(f"\n🏆 基于MRR的最佳模型: Steps {best_steps}")
    
    # 查找checkpoint
    checkpoints = find_checkpoints(model_dir)
    
    if not checkpoints:
        print(f"⚠️  没有找到checkpoint，使用最终模型")
        return model_dir
    
    # 如果有MRR数据，推荐最佳checkpoint
    if mrr_csv and df is not None:
        best_steps = df.loc[df['MRR'].idxmax(), 'steps']
        for steps, path in checkpoints:
            if steps >= best_steps:
                print(f"\n✅ 推荐使用checkpoint: {path}")
                print(f"   对应MRR: {df.loc[df['MRR'].idxmax(), 'MRR']:.4f}")
                return str(path)
    
    # 否则使用最新的checkpoint
    latest_checkpoint = checkpoints[-1][1]
    print(f"\n✅ 使用最新checkpoint: {latest_checkpoint}")
    return str(latest_checkpoint)

def main():
    parser = argparse.ArgumentParser(description="选择最佳微调模型")
    parser.add_argument("--model_dir", type=str, required=True,
                       help="模型目录路径")
    parser.add_argument("--mrr_csv", type=str, default=None,
                       help="MRR评估结果CSV文件")
    
    args = parser.parse_args()
    
    best_model_path = select_best_model(args.model_dir, args.mrr_csv)
    
    print(f"\n🎉 最佳模型路径: {best_model_path}")
    print(f"💡 使用建议:")
    print(f"   - 在RAG系统中使用: {best_model_path}")
    print(f"   - 在评估脚本中使用: --model_name {best_model_path}")

if __name__ == "__main__":
    main() 