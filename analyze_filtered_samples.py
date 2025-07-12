#!/usr/bin/env python3
"""
分析筛选后的评估样本
提供详细的统计信息
"""

import json
import re
from collections import Counter
from pathlib import Path

def analyze_filtered_samples(file_path: str):
    """分析筛选后的样本"""
    
    samples = []
    with open(file_path, 'r', encoding='utf-8') as f:
        for line in f:
            samples.append(json.loads(line.strip()))
    
    print(f"📊 样本分析报告")
    print(f"📁 文件: {file_path}")
    print(f"📈 总样本数: {len(samples)}")
    
    # 统计instruction情况
    no_instruction_count = 0
    has_instruction_count = 0
    
    # 统计answer模式
    pattern_answers = []
    normal_answers = []
    
    # 统计公司分布
    companies = []
    
    for sample in samples:
        instruction = sample.get("instruction", "").strip()
        answer = sample.get("answer", "")
        company = sample.get("company_name", "Unknown")
        
        # 统计instruction
        if not instruction:
            no_instruction_count += 1
        else:
            has_instruction_count += 1
        
        # 统计answer模式
        if re.search(r"这个股票的下月最终收益结果是:.*?(上涨|下跌)概率:", answer):
            pattern_answers.append(sample)
        else:
            normal_answers.append(sample)
        
        companies.append(company)
    
    print(f"\n📋 详细统计:")
    print(f"   - 无instruction样本: {no_instruction_count} ({no_instruction_count/len(samples)*100:.1f}%)")
    print(f"   - 有instruction样本: {has_instruction_count} ({has_instruction_count/len(samples)*100:.1f}%)")
    print(f"   - 包含特定模式answer: {len(pattern_answers)} ({len(pattern_answers)/len(samples)*100:.1f}%)")
    print(f"   - 普通answer: {len(normal_answers)} ({len(normal_answers)/len(samples)*100:.1f}%)")
    
    # 公司分布
    company_counter = Counter(companies)
    print(f"\n🏢 公司分布 (Top 10):")
    for company, count in company_counter.most_common(10):
        print(f"   - {company}: {count} 次")
    
    # 显示示例
    print(f"\n📝 示例样本:")
    
    # 无instruction示例
    no_instruction_examples = [s for s in samples if not s.get("instruction", "").strip()]
    if no_instruction_examples:
        print(f"\n1. 无instruction示例:")
        sample = no_instruction_examples[0]
        print(f"   问题: {sample.get('question', 'N/A')}")
        print(f"   答案: {sample.get('answer', 'N/A')}")
        print(f"   公司: {sample.get('company_name', 'N/A')}")
    
    # 特定模式示例
    if pattern_answers:
        print(f"\n2. 特定模式answer示例:")
        sample = pattern_answers[0]
        print(f"   问题: {sample.get('question', 'N/A')}")
        print(f"   答案: {sample.get('answer', 'N/A')}")
        print(f"   Instruction: {sample.get('instruction', 'N/A')}")
        print(f"   公司: {sample.get('company_name', 'N/A')}")
    
    # 保存统计结果
    stats = {
        "total_samples": len(samples),
        "no_instruction_count": no_instruction_count,
        "has_instruction_count": has_instruction_count,
        "pattern_answer_count": len(pattern_answers),
        "normal_answer_count": len(normal_answers),
        "company_distribution": dict(company_counter.most_common(20))
    }
    
    stats_file = file_path.replace('.jsonl', '_stats.json')
    with open(stats_file, 'w', encoding='utf-8') as f:
        json.dump(stats, f, ensure_ascii=False, indent=2)
    
    print(f"\n💾 统计结果已保存到: {stats_file}")

if __name__ == "__main__":
    file_path = "data/alphafin/alphafin_eval_filtered.jsonl"
    analyze_filtered_samples(file_path) 