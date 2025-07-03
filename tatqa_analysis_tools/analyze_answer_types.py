#!/usr/bin/env python3
"""
分析TAT-QA数据集中answer_from字段的分布
"""

import json
from collections import Counter
from pathlib import Path

def analyze_answer_types(file_path):
    """分析数据文件中的answer_from类型分布"""
    answer_types = []
    
    with open(file_path, 'r', encoding='utf-8') as f:
        for line in f:
            data = json.loads(line.strip())
            answer_from = data.get('answer_from', 'unknown')
            answer_types.append(answer_from)
    
    # 统计各类型数量
    type_counts = Counter(answer_types)
    total = len(answer_types)
    
    print(f"\n=== {Path(file_path).name} 分析结果 ===")
    print(f"总样本数: {total}")
    print("\n各类型分布:")
    for answer_type, count in type_counts.most_common():
        percentage = (count / total) * 100
        print(f"  {answer_type}: {count} ({percentage:.1f}%)")
    
    return type_counts

def main():
    # 分析训练集和评估集
    data_files = [
        "evaluate_mrr/tatqa_train_qc_enhanced.jsonl",
        "evaluate_mrr/tatqa_eval_enhanced.jsonl"
    ]
    
    all_results = {}
    
    for file_path in data_files:
        if Path(file_path).exists():
            results = analyze_answer_types(file_path)
            all_results[file_path] = results
        else:
            print(f"\n文件不存在: {file_path}")
    
    # 汇总统计
    if len(all_results) > 1:
        print("\n=== 汇总统计 ===")
        total_counts = Counter()
        for results in all_results.values():
            total_counts.update(results)
        
        total_samples = sum(total_counts.values())
        print(f"总样本数: {total_samples}")
        print("\n各类型分布:")
        for answer_type, count in total_counts.most_common():
            percentage = (count / total_samples) * 100
            print(f"  {answer_type}: {count} ({percentage:.1f}%)")

if __name__ == "__main__":
    main() 