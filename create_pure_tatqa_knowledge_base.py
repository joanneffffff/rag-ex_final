#!/usr/bin/env python3
"""
创建纯TAT-QA知识库
只包含原始TAT-QA数据，不包含AlphaFin数据
"""

import json
import os
from pathlib import Path

def create_pure_tatqa_knowledge_base():
    """创建纯TAT-QA知识库"""
    
    # 输入文件路径
    tatqa_train_path = "data/tatqa/tatqa_train_qc.jsonl"
    tatqa_dev_path = "data/tatqa/tatqa_dev_qc.jsonl"
    
    # 输出文件路径
    output_path = "evaluate_mrr/pure_tatqa_knowledge_base.jsonl"
    
    print("🔄 创建纯TAT-QA知识库...")
    
    # 收集所有TAT-QA数据
    all_tatqa_data = []
    
    # 处理训练数据
    if os.path.exists(tatqa_train_path):
        print(f"📖 加载训练数据: {tatqa_train_path}")
        with open(tatqa_train_path, 'r', encoding='utf-8') as f:
            for line in f:
                if line.strip():
                    data = json.loads(line)
                    all_tatqa_data.append(data)
        print(f"✅ 加载了 {len(all_tatqa_data)} 条训练数据")
    
    # 处理开发数据
    if os.path.exists(tatqa_dev_path):
        print(f"📖 加载开发数据: {tatqa_dev_path}")
        dev_count = 0
        with open(tatqa_dev_path, 'r', encoding='utf-8') as f:
            for line in f:
                if line.strip():
                    data = json.loads(line)
                    all_tatqa_data.append(data)
                    dev_count += 1
        print(f"✅ 加载了 {dev_count} 条开发数据")
    
    print(f"📊 总计: {len(all_tatqa_data)} 条TAT-QA数据")
    
    # 转换为知识库格式
    knowledge_base = []
    
    for i, data in enumerate(all_tatqa_data):
        # 提取上下文信息
        context = data.get('context', '')
        question = data.get('question', '')
        
        # 创建知识库条目
        kb_entry = {
            "text": context,
            "doc_id": f"tatqa_{i:06d}",
            "source_type": "tatqa",
            "question": question,
            "answer": data.get('answer', ''),
            "answer_from": data.get('answer_from', '')
        }
        
        knowledge_base.append(kb_entry)
    
    # 保存知识库
    print(f"💾 保存知识库到: {output_path}")
    with open(output_path, 'w', encoding='utf-8') as f:
        for entry in knowledge_base:
            f.write(json.dumps(entry, ensure_ascii=False) + '\n')
    
    print(f"✅ 纯TAT-QA知识库创建完成!")
    print(f"📊 包含 {len(knowledge_base)} 条记录")
    
    # 显示样本数据
    print("\n📋 样本数据:")
    for i in range(min(3, len(knowledge_base))):
        entry = knowledge_base[i]
        print(f"\n--- 样本 {i+1} ---")
        print(f"问题: {entry['question']}")
        print(f"答案来源: {entry['answer_from']}")
        print(f"上下文长度: {len(entry['text'])} 字符")
        print(f"上下文预览: {entry['text'][:100]}...")
    
    return output_path

if __name__ == "__main__":
    create_pure_tatqa_knowledge_base() 