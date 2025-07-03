#!/usr/bin/env python3
"""
分析TAT-QA数据集中context的结构类型分布
用于prompt模板优化
"""

import json
import re
from collections import Counter
from pathlib import Path

def determine_context_type(context):
    """根据context内容判断结构类型"""
    
    # 检查是否包含表格标识
    has_table = "Table ID:" in context
    
    # 检查是否包含明显的文本段落（非表格内容）
    lines = context.split('\n')
    table_lines = 0
    text_lines = 0
    
    for line in lines:
        line = line.strip()
        if line.startswith(('Table ID:', 'Headers:', 'Row', 'Category:')):
            table_lines += 1
        elif line and not line.startswith(('Table ID:', 'Headers:', 'Row', 'Category:')):
            # 检查是否是有效的文本内容（不是空行或纯数字）
            if len(line) > 10 and not re.match(r'^[\d\s\-\.,$%()]+$', line):
                text_lines += 1
    
    # 判断类型
    if has_table and text_lines > 2:
        return "table-text"
    elif has_table:
        return "table"
    else:
        return "text"

def analyze_context_types(file_path):
    """分析数据文件中的context类型分布"""
    context_types = []
    samples_by_type = {"table": [], "text": [], "table-text": []}
    
    with open(file_path, 'r', encoding='utf-8') as f:
        for i, line in enumerate(f):
            data = json.loads(line.strip())
            context = data.get('context', '')
            context_type = determine_context_type(context)
            context_types.append(context_type)
            
            # 保存样本信息
            sample_info = {
                "line": i + 1,
                "query": data.get('query', ''),
                "answer_from": data.get('answer_from', 'unknown'),
                "context_preview": context[:200] + "..." if len(context) > 200 else context
            }
            samples_by_type[context_type].append(sample_info)
    
    # 统计各类型数量
    type_counts = Counter(context_types)
    total = len(context_types)
    
    print(f"\n=== {Path(file_path).name} Context类型分析 ===")
    print(f"总样本数: {total}")
    print("\nContext类型分布:")
    for context_type, count in type_counts.most_common():
        percentage = (count / total) * 100
        print(f"  {context_type}: {count} ({percentage:.1f}%)")
    
    # 分析answer_from与context_type的关系
    print("\nAnswer_from vs Context_type 关系:")
    answer_context_mapping = {}
    for context_type in ["table", "text", "table-text"]:
        answer_context_mapping[context_type] = Counter()
        for sample in samples_by_type[context_type]:
            answer_from = sample["answer_from"]
            answer_context_mapping[context_type][answer_from] += 1
    
    for context_type in ["table", "text", "table-text"]:
        print(f"\n  Context类型 '{context_type}':")
        for answer_from, count in answer_context_mapping[context_type].most_common():
            print(f"    answer_from='{answer_from}': {count}")
    
    return context_types, samples_by_type

def main():
    """主函数"""
    files_to_analyze = [
        "evaluate_mrr/tatqa_train_qc_enhanced.jsonl",
        "evaluate_mrr/tatqa_eval_enhanced.jsonl",
        "evaluate_mrr/tatqa_test_15_samples.jsonl"
    ]
    
    all_context_types = []
    
    for file_path in files_to_analyze:
        if Path(file_path).exists():
            context_types, samples = analyze_context_types(file_path)
            all_context_types.extend(context_types)
        else:
            print(f"\n文件不存在: {file_path}")
    
    # 总体统计
    if all_context_types:
        total_counts = Counter(all_context_types)
        total = len(all_context_types)
        print(f"\n=== 总体Context类型分布 ===")
        print(f"总样本数: {total}")
        for context_type, count in total_counts.most_common():
            percentage = (count / total) * 100
            print(f"  {context_type}: {count} ({percentage:.1f}%)")

if __name__ == "__main__":
    main() 