#!/usr/bin/env python3
"""
展示从原始TatQA数据集到处理后数据集的转换示例
"""

import json
from pathlib import Path

def show_conversion_example():
    """展示数据转换示例"""
    
    print("=" * 80)
    print("TatQA数据集转换示例：从原始数据到处理后数据")
    print("=" * 80)
    
    # 加载原始数据
    with open('data/tatqa_dataset_raw/tatqa_dataset_train.json', 'r') as f:
        raw_data = json.load(f)
    
    # 加载处理后的评估数据
    with open('evaluate_mrr/tatqa_eval_enhanced.jsonl', 'r') as f:
        processed_data = []
        for line in f:
            processed_data.append(json.loads(line))
    
    # 选择第一个样本作为示例
    raw_sample = raw_data[0]
    
    print("\n📋 原始TatQA数据集样本:")
    print("-" * 40)
    print(f"样本键: {list(raw_sample.keys())}")
    print(f"表格存在: {'table' in raw_sample}")
    print(f"段落数量: {len(raw_sample.get('paragraphs', []))}")
    print(f"问题数量: {len(raw_sample.get('questions', []))}")
    
    # 显示表格
    if 'table' in raw_sample:
        table = raw_sample['table']
        print(f"\n📊 表格信息:")
        print(f"  表格UID: {table.get('uid', 'N/A')}")
        print(f"  表格行数: {len(table.get('table', []))}")
        if table.get('table'):
            print(f"  表格列数: {len(table['table'][0])}")
            print(f"  表格内容预览:")
            for i, row in enumerate(table['table'][:3]):  # 只显示前3行
                print(f"    行{i+1}: {row}")
            if len(table['table']) > 3:
                print(f"    ... (还有{len(table['table'])-3}行)")
    
    # 显示段落
    print(f"\n📝 段落信息:")
    for i, para in enumerate(raw_sample.get('paragraphs', [])):
        print(f"  段落{i+1} (UID: {para.get('uid', 'N/A')}):")
        print(f"    {para.get('text', '')[:100]}{'...' if len(para.get('text', '')) > 100 else ''}")
    
    # 显示问题
    print(f"\n❓ 问题信息:")
    for i, q in enumerate(raw_sample.get('questions', [])):
        print(f"  问题{i+1}: {q.get('question', '')[:80]}{'...' if len(q.get('question', '')) > 80 else ''}")
        answer = q.get('answer', '')
        if isinstance(answer, (list, tuple)):
            answer_str = str(answer)
        else:
            answer_str = str(answer)
        print(f"    答案: {answer_str[:50]}{'...' if len(answer_str) > 50 else ''}")
    
    # 查找对应的处理后数据
    print(f"\n" + "=" * 80)
    print("🔄 转换后的评估数据样本:")
    print("=" * 80)
    
    # 查找与第一个原始样本相关的问题
    # 通过检查context内容来匹配
    first_para_text = raw_sample['paragraphs'][0]['text'] if raw_sample.get('paragraphs') else ""
    first_question = raw_sample['questions'][0]['question'] if raw_sample.get('questions') else ""
    
    matching_samples = []
    for item in processed_data:
        if (first_para_text[:50] in item.get('context', '') or 
            first_question[:50] in item.get('query', '')):
            matching_samples.append(item)
    
    if matching_samples:
        print(f"找到 {len(matching_samples)} 个相关的处理后样本:")
        for i, sample in enumerate(matching_samples[:3]):  # 只显示前3个
            print(f"\n📄 处理后样本 {i+1}:")
            print(f"  问题: {sample.get('query', '')[:100]}{'...' if len(sample.get('query', '')) > 100 else ''}")
            print(f"  Context类型: {'表格' if 'Details for item' in sample.get('context', '') else '段落'}")
            print(f"  Context长度: {len(sample.get('context', ''))}")
            print(f"  doc_id: {sample.get('doc_id', 'N/A')}")
            print(f"  relevant_doc_ids: {sample.get('relevant_doc_ids', [])}")
            answer = sample.get('answer', '')
            answer_str = str(answer)
            print(f"  答案: {answer_str[:50]}{'...' if len(answer_str) > 50 else ''}")
    else:
        print("未找到完全匹配的样本，显示前3个处理后样本作为示例:")
        for i, sample in enumerate(processed_data[:3]):
            print(f"\n📄 处理后样本 {i+1}:")
            print(f"  问题: {sample.get('query', '')[:100]}{'...' if len(sample.get('query', '')) > 100 else ''}")
            print(f"  Context类型: {'表格' if 'Details for item' in sample.get('context', '') else '段落'}")
            print(f"  Context长度: {len(sample.get('context', ''))}")
            print(f"  doc_id: {sample.get('doc_id', 'N/A')}")
            print(f"  relevant_doc_ids: {sample.get('relevant_doc_ids', [])}")
            answer = sample.get('answer', '')
            answer_str = str(answer)
            print(f"  答案: {answer_str[:50]}{'...' if len(answer_str) > 50 else ''}")
    
    # 显示转换过程
    print(f"\n" + "=" * 80)
    print("🔄 转换过程说明:")
    print("=" * 80)
    print("1. 📊 原始数据: 1个样本包含1个表格 + 多个段落 + 多个问题")
    print("2. 🔪 数据切分: 每个段落和表格被转换为独立的chunk")
    print("3. 📝 问题分离: 每个问题成为独立的评估样本")
    print("4. 🔗 关联保持: 通过relevant_doc_ids保持问题与原始文档的关联")
    print("5. 📋 格式统一: 转换为标准的query-context-answer格式")
    
    print(f"\n📈 转换统计:")
    print(f"  原始样本数: {len(raw_data)}")
    print(f"  处理后样本数: {len(processed_data)}")
    print(f"  平均每个原始样本产生的问题数: {len(processed_data) / len(raw_data):.2f}")

if __name__ == "__main__":
    show_conversion_example() 