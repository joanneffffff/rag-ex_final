#!/usr/bin/env python3
"""
从alphafin_eval.jsonl中随机选择100个样本用于评估（JSONL格式）
"""

import json
import random
from pathlib import Path

def create_eval_samples_jsonl():
    """从alphafin_eval.jsonl中随机选择100个样本，保存为JSONL格式"""
    
    input_file = "evaluate_mrr/alphafin_eval.jsonl"
    output_file = "data/alphafin/alphafin_eval_samples.json"
    
    # 确保输出目录存在
    Path("data/alphafin").mkdir(parents=True, exist_ok=True)
    
    print(f"读取评估数据: {input_file}")
    
    # 读取所有数据
    all_samples = []
    with open(input_file, 'r', encoding='utf-8') as f:
        for line in f:
            try:
                sample = json.loads(line.strip())
                all_samples.append(sample)
            except json.JSONDecodeError as e:
                print(f"跳过无效行: {e}")
                continue
    
    print(f"总样本数: {len(all_samples)}")
    
    # 随机选择100个样本
    if len(all_samples) >= 100:
        selected_samples = random.sample(all_samples, 100)
    else:
        selected_samples = all_samples
        print(f"警告: 总样本数少于100，使用全部 {len(all_samples)} 个样本")
    
    print(f"选择的样本数: {len(selected_samples)}")
    
    # 保存为JSONL格式（每行一个JSON对象）
    with open(output_file, 'w', encoding='utf-8') as f:
        for sample in selected_samples:
            f.write(json.dumps(sample, ensure_ascii=False) + '\n')
    
    print(f"✅ 评估样本已保存到: {output_file} (JSONL格式)")
    
    # 显示样本统计信息
    print("\n样本统计信息:")
    print(f"- 总样本数: {len(selected_samples)}")
    
    # 检查必要字段
    valid_samples = 0
    for sample in selected_samples:
        if 'generated_question' in sample and 'summary' in sample:
            valid_samples += 1
    
    print(f"- 有效样本数: {valid_samples}")
    print(f"- 包含generated_question和summary字段的样本: {valid_samples}")
    
    # 显示前3个样本的字段
    if selected_samples:
        print(f"\n第一个样本的字段:")
        for key in selected_samples[0].keys():
            print(f"  - {key}")
        
        print(f"\n第一个样本的generated_question:")
        print(f"  {selected_samples[0].get('generated_question', 'N/A')[:100]}...")

if __name__ == "__main__":
    # 设置随机种子以确保可重复性
    random.seed(42)
    create_eval_samples_jsonl() 