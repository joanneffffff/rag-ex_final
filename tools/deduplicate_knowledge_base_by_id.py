#!/usr/bin/env python3
"""
基于Table ID和Paragraph ID的知识库去重脚本
解决知识库中相同ID但不同单位说明的重复问题
"""

import json
import re
from pathlib import Path
from typing import List, Dict, Any

def extract_table_id(context: str) -> str:
    """从context中提取Table ID"""
    if "Table ID:" in context:
        match = re.search(r'Table ID:\s*([a-f0-9-]+)', context)
        if match:
            return match.group(1)
    return ""

def extract_paragraph_id(context: str) -> str:
    """从context中提取Paragraph ID"""
    if "Paragraph ID:" in context:
        match = re.search(r'Paragraph ID:\s*([a-f0-9-]+)', context)
        if match:
            return match.group(1)
    return ""

def deduplicate_knowledge_base_by_id():
    """基于Table ID和Paragraph ID去重知识库"""
    
    # 输入和输出文件
    input_file = "data/unified/tatqa_knowledge_base_combined.jsonl"
    output_file = "data/unified/tatqa_knowledge_base_deduplicated.jsonl"
    
    print("🔄 开始基于Table ID和Paragraph ID去重知识库...")
    
    if not Path(input_file).exists():
        print(f"❌ 输入文件不存在: {input_file}")
        return None
    
    # 读取所有文档
    documents = []
    with open(input_file, 'r', encoding='utf-8') as f:
        for line_num, line in enumerate(f, 1):
            if line.strip():
                try:
                    doc = json.loads(line)
                    documents.append(doc)
                except json.JSONDecodeError as e:
                    print(f"⚠️ 第{line_num}行JSON解析错误: {e}")
                    continue
    
    print(f"📖 读取了 {len(documents)} 个文档")
    
    # 分类文档
    table_docs = []  # 包含Table ID的文档
    paragraph_docs = []  # 只包含Paragraph ID的文档
    other_docs = []  # 其他文档
    
    for doc in documents:
        context = doc.get('context', '')
        table_id = extract_table_id(context)
        paragraph_id = extract_paragraph_id(context)
        
        if table_id:
            # 包含Table ID的文档（包括表格+文本）
            table_docs.append({
                'doc': doc,
                'table_id': table_id,
                'paragraph_id': paragraph_id
            })
        elif paragraph_id:
            # 只包含Paragraph ID的文档
            paragraph_docs.append({
                'doc': doc,
                'paragraph_id': paragraph_id
            })
        else:
            # 其他文档
            other_docs.append(doc)
    
    print(f"📊 文档分类:")
    print(f"   - 表格文档: {len(table_docs)}")
    print(f"   - 段落文档: {len(paragraph_docs)}")
    print(f"   - 其他文档: {len(other_docs)}")
    
    # 基于Table ID去重
    print("\n🔄 基于Table ID去重...")
    unique_table_docs = []
    seen_table_ids = set()
    table_duplicates = 0
    
    for item in table_docs:
        table_id = item['table_id']
        if table_id in seen_table_ids:
            table_duplicates += 1
            print(f"  ❌ 跳过重复Table ID: {table_id}")
            continue
        
        seen_table_ids.add(table_id)
        unique_table_docs.append(item['doc'])
        print(f"  ✅ 保留Table ID: {table_id}")
    
    print(f"  📊 Table ID去重结果: {len(unique_table_docs)} 个文档，移除 {table_duplicates} 个重复")
    
    # 基于Paragraph ID去重
    print("\n🔄 基于Paragraph ID去重...")
    unique_paragraph_docs = []
    seen_paragraph_ids = set()
    paragraph_duplicates = 0
    
    for item in paragraph_docs:
        paragraph_id = item['paragraph_id']
        if paragraph_id in seen_paragraph_ids:
            paragraph_duplicates += 1
            print(f"  ❌ 跳过重复Paragraph ID: {paragraph_id}")
            continue
        
        seen_paragraph_ids.add(paragraph_id)
        unique_paragraph_docs.append(item['doc'])
        print(f"  ✅ 保留Paragraph ID: {paragraph_id}")
    
    print(f"  📊 Paragraph ID去重结果: {len(unique_paragraph_docs)} 个文档，移除 {paragraph_duplicates} 个重复")
    
    # 合并所有唯一文档
    all_unique_docs = unique_table_docs + unique_paragraph_docs + other_docs
    
    print(f"\n📊 去重总结:")
    print(f"   - 原始文档数: {len(documents)}")
    print(f"   - 去重后文档数: {len(all_unique_docs)}")
    print(f"   - 移除重复数: {len(documents) - len(all_unique_docs)}")
    print(f"   - 去重率: {(len(documents) - len(all_unique_docs)) / len(documents) * 100:.2f}%")
    
    # 写入去重后的文件
    print(f"\n💾 写入去重文件: {output_file}")
    Path(output_file).parent.mkdir(parents=True, exist_ok=True)
    
    with open(output_file, 'w', encoding='utf-8') as f:
        for doc in all_unique_docs:
            f.write(json.dumps(doc, ensure_ascii=False) + '\n')
    
    # 显示一些样本
    print(f"\n📋 去重后样本预览:")
    for i, doc in enumerate(all_unique_docs[:3]):
        print(f"\n样本 {i+1} (ID: {doc['doc_id']}):")
        context = doc['context']
        table_id = extract_table_id(context)
        paragraph_id = extract_paragraph_id(context)
        
        if table_id:
            print(f"  Table ID: {table_id}")
        if paragraph_id:
            print(f"  Paragraph ID: {paragraph_id}")
        
        if len(context) > 150:
            print(f"  内容: {context[:150]}...")
        else:
            print(f"  内容: {context}")
    
    print(f"\n🎉 去重完成！")
    print(f"📁 输出文件: {output_file}")
    
    return output_file

if __name__ == "__main__":
    deduplicate_knowledge_base_by_id() 