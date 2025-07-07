#!/usr/bin/env python3
"""
从 TAT-QA 评估数据中按 answer_from 类型选择测试样本
选择 5 个 table、5 个 text、5 个 table-text 样本
"""

import json
from pathlib import Path
from typing import List, Dict, Any

def select_test_samples():
    """按 answer_from 类型选择测试样本"""
    
    input_file = "evaluate_mrr/tatqa_eval_enhanced_optimized.jsonl"
    output_file = "evaluate_mrr/tatqa_test_15_samples.json"
    
    print("🔄 开始选择测试样本...")
    
    # 按类型收集样本
    table_samples = []
    text_samples = []
    table_text_samples = []
    
    # 读取输入文件
    print(f"📖 读取文件: {input_file}")
    with open(input_file, 'r', encoding='utf-8') as f:
        for line_num, line in enumerate(f, 1):
            if line.strip():
                try:
                    item = json.loads(line)
                    answer_from = item.get('answer_from', '').lower()
                    
                    # 按类型分类
                    if answer_from == 'table':
                        if len(table_samples) < 5:
                            table_samples.append(item)
                    elif answer_from == 'text':
                        if len(text_samples) < 5:
                            text_samples.append(item)
                    elif answer_from == 'table-text':
                        if len(table_text_samples) < 5:
                            table_text_samples.append(item)
                    
                    # 如果所有类型都收集够了，就停止
                    if len(table_samples) >= 5 and len(text_samples) >= 5 and len(table_text_samples) >= 5:
                        break
                        
                except json.JSONDecodeError as e:
                    print(f"⚠️ 第{line_num}行JSON解析错误: {e}")
                    continue
    
    # 检查是否收集到足够的样本
    print(f"📊 收集统计:")
    print(f"   - Table 样本: {len(table_samples)}/5")
    print(f"   - Text 样本: {len(text_samples)}/5")
    print(f"   - Table-Text 样本: {len(table_text_samples)}/5")
    
    # 合并所有样本
    all_samples = table_samples + text_samples + table_text_samples
    
    if len(all_samples) < 15:
        print(f"⚠️ 警告：只收集到 {len(all_samples)} 个样本，少于预期的 15 个")
    
    # 写入输出文件
    print(f"💾 写入文件: {output_file}")
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(all_samples, f, ensure_ascii=False, indent=2)
    
    print(f"🎉 选择完成！")
    print(f"📊 最终统计:")
    print(f"   - 总样本数: {len(all_samples)}")
    print(f"   - Table 样本: {len(table_samples)}")
    print(f"   - Text 样本: {len(text_samples)}")
    print(f"   - Table-Text 样本: {len(table_text_samples)}")
    print(f"   - 输出文件: {output_file}")
    
    # 显示样本预览
    print(f"\n📋 样本预览:")
    
    print(f"\n🔢 Table 样本 ({len(table_samples)} 个):")
    for i, sample in enumerate(table_samples, 1):
        print(f"  {i}. {sample['query'][:60]}...")
        print(f"     答案: {sample['answer']}")
        print(f"     来源: {sample['answer_from']}")
    
    print(f"\n📝 Text 样本 ({len(text_samples)} 个):")
    for i, sample in enumerate(text_samples, 1):
        print(f"  {i}. {sample['query'][:60]}...")
        print(f"     答案: {sample['answer']}")
        print(f"     来源: {sample['answer_from']}")
    
    print(f"\n🔗 Table-Text 样本 ({len(table_text_samples)} 个):")
    for i, sample in enumerate(table_text_samples, 1):
        print(f"  {i}. {sample['query'][:60]}...")
        print(f"     答案: {sample['answer']}")
        print(f"     来源: {sample['answer_from']}")
    
    return output_file

if __name__ == "__main__":
    select_test_samples() 