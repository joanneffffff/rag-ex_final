#!/usr/bin/env python3
"""
比较两个评估结果文件
"""

import json
import sys
from pathlib import Path

def load_results(file_path: str) -> dict:
    """加载评估结果文件"""
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            return json.load(f)
    except Exception as e:
        print(f"❌ 无法加载文件 {file_path}: {e}")
        return {}

def compare_results(file1: str, file2: str):
    """比较两个评估结果文件"""
    print("🔍 比较评估结果")
    print("=" * 60)
    
    # 加载结果
    results1 = load_results(file1)
    results2 = load_results(file2)
    
    if not results1 or not results2:
        print("❌ 无法加载结果文件")
        return
    
    print(f"📁 文件1: {file1}")
    print(f"📁 文件2: {file2}")
    print()
    
    # 获取所有模式
    modes1 = set(results1.keys())
    modes2 = set(results2.keys())
    all_modes = modes1.union(modes2)
    
    print("📊 模式对比:")
    print(f"   文件1模式: {sorted(modes1)}")
    print(f"   文件2模式: {sorted(modes2)}")
    print(f"   共同模式: {sorted(modes1.intersection(modes2))}")
    print()
    
    # 比较共同模式
    common_modes = modes1.intersection(modes2)
    if common_modes:
        print("📈 共同模式性能对比:")
        print("-" * 60)
        
        for mode in sorted(common_modes):
            print(f"\n🎯 模式: {mode}")
            print("-" * 40)
            
            mode1 = results1[mode]
            mode2 = results2[mode]
            
            # 获取所有top_k
            top_ks1 = set(mode1.keys())
            top_ks2 = set(mode2.keys())
            common_top_ks = top_ks1.intersection(top_ks2)
            
            for top_k in sorted(common_top_ks, key=lambda x: int(x.split('_')[1])):
                print(f"\n  Top-{top_k.split('_')[1]}:")
                
                result1 = mode1[top_k]
                result2 = mode2[top_k]
                
                # 比较指标
                metrics = ['MRR', 'Hit@1', 'Hit@3', 'Hit@5', 'Hit@10']
                for metric in metrics:
                    if metric in result1 and metric in result2:
                        val1 = result1[metric]
                        val2 = result2[metric]
                        diff = val2 - val1
                        change = "📈" if diff > 0 else "📉" if diff < 0 else "➡️"
                        print(f"    {metric}: {val1:.4f} → {val2:.4f} ({diff:+.4f}) {change}")
                
                # 比较时间
                time1 = result1.get('retrieval_time_seconds', 0)
                time2 = result2.get('retrieval_time_seconds', 0)
                time_diff = time2 - time1
                time_change = "🐌" if time_diff > 0 else "⚡" if time_diff < 0 else "➡️"
                print(f"    时间: {time1:.2f}s → {time2:.2f}s ({time_diff:+.2f}s) {time_change}")
    
    # 分析独有模式
    only_in_1 = modes1 - modes2
    only_in_2 = modes2 - modes1
    
    if only_in_1:
        print(f"\n📋 仅在文件1中存在的模式: {sorted(only_in_1)}")
        for mode in sorted(only_in_1):
            print(f"  {mode}:")
            for top_k, result in results1[mode].items():
                if 'MRR' in result:
                    print(f"    {top_k}: MRR={result['MRR']:.4f}, Hit@1={result.get('Hit@1', 'N/A')}")
    
    if only_in_2:
        print(f"\n📋 仅在文件2中存在的模式: {sorted(only_in_2)}")
        for mode in sorted(only_in_2):
            print(f"  {mode}:")
            for top_k, result in results2[mode].items():
                if 'MRR' in result:
                    print(f"    {top_k}: MRR={result['MRR']:.4f}, Hit@1={result.get('Hit@1', 'N/A')}")
    
    # 总结分析
    print("\n" + "=" * 60)
    print("📊 总结分析:")
    
    # 找出最佳性能
    best_mrr = 0
    best_config = ""
    
    for mode in all_modes:
        if mode in results1:
            for top_k, result in results1[mode].items():
                if 'MRR' in result and result['MRR'] > best_mrr:
                    best_mrr = result['MRR']
                    best_config = f"文件1-{mode}-{top_k}"
        
        if mode in results2:
            for top_k, result in results2[mode].items():
                if 'MRR' in result and result['MRR'] > best_mrr:
                    best_mrr = result['MRR']
                    best_config = f"文件2-{mode}-{top_k}"
    
    print(f"🏆 最佳MRR: {best_mrr:.4f} ({best_config})")
    
    # 分析趋势
    if common_modes:
        print("\n📈 性能趋势分析:")
        for mode in sorted(common_modes):
            mode1_best = max([results1[mode][top_k]['MRR'] for top_k in results1[mode] if 'MRR' in results1[mode][top_k]])
            mode2_best = max([results2[mode][top_k]['MRR'] for top_k in results2[mode] if 'MRR' in results2[mode][top_k]])
            trend = "改进" if mode2_best > mode1_best else "下降" if mode2_best < mode1_best else "持平"
            print(f"  {mode}: {mode1_best:.4f} → {mode2_best:.4f} ({trend})")

def main():
    if len(sys.argv) != 3:
        print("用法: python compare_evaluation_results.py <file1> <file2>")
        sys.exit(1)
    
    file1 = sys.argv[1]
    file2 = sys.argv[2]
    
    if not Path(file1).exists():
        print(f"❌ 文件不存在: {file1}")
        sys.exit(1)
    
    if not Path(file2).exists():
        print(f"❌ 文件不存在: {file2}")
        sys.exit(1)
    
    compare_results(file1, file2)

if __name__ == "__main__":
    main() 