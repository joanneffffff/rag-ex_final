#!/usr/bin/env python3
"""
从TAT-QA enhanced数据中创建平衡的15个测试样本
基于answer_from字段进行准确分类
"""

import json
import random
from pathlib import Path

def create_balanced_test_dataset(input_file, output_file, num_samples=15):
    """创建平衡的测试数据集"""
    
    # 按类型分组数据
    text_samples = []
    table_samples = []
    table_text_samples = []
    
    with open(input_file, 'r', encoding='utf-8') as f:
        for line in f:
            data = json.loads(line.strip())
            answer_from = data.get('answer_from', 'unknown')
            
            if answer_from == 'text':
                text_samples.append(data)
            elif answer_from == 'table':
                table_samples.append(data)
            elif answer_from == 'table-text':
                table_text_samples.append(data)
    
    print(f"找到的样本分布:")
    print(f"  文本样本: {len(text_samples)}")
    print(f"  表格样本: {len(table_samples)}")
    print(f"  表格+文本样本: {len(table_text_samples)}")
    
    # 计算每种类型的样本数量
    # 目标：5个表格，5个文本，5个表格+文本
    samples_per_type = num_samples // 3
    
    # 随机选择样本
    selected_samples = []
    
    # 选择表格样本
    if len(table_samples) >= samples_per_type:
        selected_table = random.sample(table_samples, samples_per_type)
        selected_samples.extend(selected_table)
        print(f"选择了 {len(selected_table)} 个表格样本")
    else:
        selected_samples.extend(table_samples)
        print(f"选择了所有 {len(table_samples)} 个表格样本")
    
    # 选择文本样本
    if len(text_samples) >= samples_per_type:
        selected_text = random.sample(text_samples, samples_per_type)
        selected_samples.extend(selected_text)
        print(f"选择了 {len(selected_text)} 个文本样本")
    else:
        selected_samples.extend(text_samples)
        print(f"选择了所有 {len(text_samples)} 个文本样本")
    
    # 选择表格+文本样本
    if len(table_text_samples) >= samples_per_type:
        selected_table_text = random.sample(table_text_samples, samples_per_type)
        selected_samples.extend(selected_table_text)
        print(f"选择了 {len(selected_table_text)} 个表格+文本样本")
    else:
        selected_samples.extend(table_text_samples)
        print(f"选择了所有 {len(table_text_samples)} 个表格+文本样本")
    
    # 打乱顺序
    random.shuffle(selected_samples)
    
    # 保存到文件
    with open(output_file, 'w', encoding='utf-8') as f:
        for sample in selected_samples:
            f.write(json.dumps(sample, ensure_ascii=False) + '\n')
    
    print(f"\n成功创建测试数据集: {output_file}")
    print(f"总样本数: {len(selected_samples)}")
    
    # 统计最终分布
    final_counts = {}
    for sample in selected_samples:
        answer_from = sample.get('answer_from', 'unknown')
        final_counts[answer_from] = final_counts.get(answer_from, 0) + 1
    
    print("\n最终分布:")
    for answer_type, count in final_counts.items():
        print(f"  {answer_type}: {count}")
    
    return selected_samples

def main():
    input_file = "evaluate_mrr/tatqa_eval_enhanced.jsonl"
    output_file = "evaluate_mrr/tatqa_test_15_samples.json"
    
    # 设置随机种子以确保可重复性
    random.seed(42)
    
    if not Path(input_file).exists():
        print(f"输入文件不存在: {input_file}")
        return
    
    create_balanced_test_dataset(input_file, output_file, num_samples=15)

if __name__ == "__main__":
    main() 