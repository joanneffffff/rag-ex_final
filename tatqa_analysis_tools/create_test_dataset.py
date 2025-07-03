#!/usr/bin/env python3
"""
创建TAT-QA测试数据集脚本
分析问题类型分布并创建平衡的测试数据集
"""

import json
import re
from collections import Counter
from typing import List, Dict, Any
import random

def classify_question_type(context: str) -> str:
    """
    根据上下文内容分类问题类型
    """
    if "Table ID:" in context:
        return "table"
    elif "Table ID:" not in context and len(context.strip()) > 0:
        return "text"
    else:
        return "unknown"

def analyze_dataset_statistics(data_file: str) -> Dict[str, Any]:
    """
    分析数据集的统计信息
    """
    print(f"📊 分析数据集: {data_file}")
    
    question_types = []
    total_samples = 0
    
    with open(data_file, 'r', encoding='utf-8') as f:
        for line in f:
            try:
                sample = json.loads(line.strip())
                question_type = classify_question_type(sample["context"])
                question_types.append(question_type)
                total_samples += 1
            except json.JSONDecodeError:
                continue
    
    # 统计各类型数量
    type_counts = Counter(question_types)
    
    statistics = {
        "total_samples": total_samples,
        "type_distribution": dict(type_counts),
        "type_percentages": {
            qtype: (count / total_samples * 100) 
            for qtype, count in type_counts.items()
        }
    }
    
    return statistics

def create_balanced_test_dataset(data_file: str, output_file: str, 
                                table_samples: int = 5, 
                                text_samples: int = 5, 
                                mixed_samples: int = 5) -> Dict[str, Any]:
    """
    创建平衡的测试数据集
    """
    print(f"🔧 创建测试数据集: {output_file}")
    
    # 按类型分组样本
    table_samples_list = []
    text_samples_list = []
    
    with open(data_file, 'r', encoding='utf-8') as f:
        for line in f:
            try:
                sample = json.loads(line.strip())
                question_type = classify_question_type(sample["context"])
                
                if question_type == "table":
                    table_samples_list.append(sample)
                elif question_type == "text":
                    text_samples_list.append(sample)
            except json.JSONDecodeError:
                continue
    
    print(f"📋 找到 {len(table_samples_list)} 个表格问题")
    print(f"📝 找到 {len(text_samples_list)} 个文本问题")
    
    # 随机选择样本
    random.seed(42)  # 确保可重现性
    
    selected_samples = []
    
    # 选择表格样本
    if len(table_samples_list) >= table_samples:
        selected_table = random.sample(table_samples_list, table_samples)
        selected_samples.extend(selected_table)
        print(f"✅ 选择了 {len(selected_table)} 个表格样本")
    else:
        print(f"⚠️ 表格样本不足，只有 {len(table_samples_list)} 个")
        selected_samples.extend(table_samples_list)
    
    # 选择文本样本
    if len(text_samples_list) >= text_samples:
        selected_text = random.sample(text_samples_list, text_samples)
        selected_samples.extend(selected_text)
        print(f"✅ 选择了 {len(selected_text)} 个文本样本")
    else:
        print(f"⚠️ 文本样本不足，只有 {len(text_samples_list)} 个")
        selected_samples.extend(text_samples_list)
    
    # 对于混合样本，我们从两种类型中各选一些
    if mixed_samples > 0:
        remaining_table = [s for s in table_samples_list if s not in selected_samples]
        remaining_text = [s for s in text_samples_list if s not in selected_samples]
        
        mixed_table_count = min(mixed_samples // 2, len(remaining_table))
        mixed_text_count = mixed_samples - mixed_table_count
        
        if mixed_table_count > 0:
            mixed_table = random.sample(remaining_table, mixed_table_count)
            selected_samples.extend(mixed_table)
        
        if mixed_text_count > 0 and len(remaining_text) >= mixed_text_count:
            mixed_text = random.sample(remaining_text, mixed_text_count)
            selected_samples.extend(mixed_text)
        
        print(f"✅ 选择了 {mixed_samples} 个混合样本")
    
    # 打乱顺序
    random.shuffle(selected_samples)
    
    # 保存测试数据集
    with open(output_file, 'w', encoding='utf-8') as f:
        for sample in selected_samples:
            f.write(json.dumps(sample, ensure_ascii=False) + '\n')
    
    # 统计测试数据集
    test_stats = {
        "total_test_samples": len(selected_samples),
        "table_samples": len([s for s in selected_samples if classify_question_type(s["context"]) == "table"]),
        "text_samples": len([s for s in selected_samples if classify_question_type(s["context"]) == "text"]),
        "sample_ids": [s.get("doc_id", "unknown") for s in selected_samples]
    }
    
    return test_stats

def main():
    """主函数"""
    data_file = 'evaluate_mrr/tatqa_eval_enhanced.jsonl'
    test_output_file = 'evaluate_mrr/tatqa_test_15_samples.jsonl'
    
    print("🚀 TAT-QA数据集分析工具")
    print("="*50)
    
    # 1. 分析完整数据集统计
    print("\n📊 步骤1: 分析完整数据集统计")
    statistics = analyze_dataset_statistics(data_file)
    
    print(f"总样本数: {statistics['total_samples']}")
    print("类型分布:")
    for qtype, count in statistics['type_distribution'].items():
        percentage = statistics['type_percentages'][qtype]
        print(f"  {qtype}: {count} ({percentage:.1f}%)")
    
    # 2. 创建测试数据集
    print("\n🔧 步骤2: 创建平衡测试数据集")
    test_stats = create_balanced_test_dataset(
        data_file=data_file,
        output_file=test_output_file,
        table_samples=5,
        text_samples=5,
        mixed_samples=5
    )
    
    print(f"\n✅ 测试数据集创建完成!")
    print(f"测试样本总数: {test_stats['total_test_samples']}")
    print(f"表格样本: {test_stats['table_samples']}")
    print(f"文本样本: {test_stats['text_samples']}")
    print(f"输出文件: {test_output_file}")
    
    # 3. 显示一些样本示例
    print("\n📋 样本示例:")
    with open(test_output_file, 'r', encoding='utf-8') as f:
        for i, line in enumerate(f):
            if i >= 3:  # 只显示前3个样本
                break
            sample = json.loads(line.strip())
            qtype = classify_question_type(sample["context"])
            print(f"样本 {i+1} ({qtype}):")
            print(f"  问题: {sample['query'][:100]}...")
            print(f"  答案: {sample['answer']}")
            print(f"  文档ID: {sample.get('doc_id', 'unknown')}")
            print()

if __name__ == "__main__":
    main() 