#!/usr/bin/env python3
"""
检查知识库中相同Table ID但不同context的情况
"""

import json
import re
from collections import defaultdict
from pathlib import Path

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

def check_table_id_duplicates():
    """检查知识库中相同Table ID但不同context的情况"""
    
    knowledge_base_file = "data/unified/tatqa_knowledge_base_combined.jsonl"
    
    if not Path(knowledge_base_file).exists():
        print(f"❌ 知识库文件不存在: {knowledge_base_file}")
        return
    
    print("🔍 检查知识库中相同Table ID但不同context的情况...")
    
    # 读取知识库
    documents = []
    with open(knowledge_base_file, 'r', encoding='utf-8') as f:
        for line_num, line in enumerate(f, 1):
            if line.strip():
                try:
                    doc = json.loads(line)
                    documents.append(doc)
                except json.JSONDecodeError as e:
                    print(f"⚠️ 第{line_num}行JSON解析错误: {e}")
                    continue
    
    print(f"📊 知识库总文档数: {len(documents)}")
    
    # 按Table ID分组
    table_id_groups = defaultdict(list)
    paragraph_id_groups = defaultdict(list)
    no_id_docs = []
    
    for doc in documents:
        context = doc.get('context', '')
        if context:
            table_id = extract_table_id(context)
            paragraph_id = extract_paragraph_id(context)
            
            if table_id:
                table_id_groups[table_id].append(doc)
            elif paragraph_id:
                paragraph_id_groups[paragraph_id].append(doc)
            else:
                no_id_docs.append(doc)
    
    print(f"\n📋 文档分类统计:")
    print(f"   - 包含Table ID的文档: {sum(len(docs) for docs in table_id_groups.values())}")
    print(f"   - 包含Paragraph ID的文档: {sum(len(docs) for docs in paragraph_id_groups.values())}")
    print(f"   - 无ID的文档: {len(no_id_docs)}")
    
    # 检查Table ID重复
    table_id_duplicates = {table_id: docs for table_id, docs in table_id_groups.items() if len(docs) > 1}
    
    print(f"\n🔍 Table ID重复统计:")
    print(f"   - 唯一Table ID数量: {len(table_id_groups)}")
    print(f"   - 有重复的Table ID数量: {len(table_id_duplicates)}")
    print(f"   - Table ID重复文档总数: {sum(len(docs) for docs in table_id_duplicates.values())}")
    
    if table_id_duplicates:
        print(f"\n🔍 Table ID重复详情:")
        for i, (table_id, docs) in enumerate(list(table_id_duplicates.items())[:10]):  # 只显示前10个
            print(f"\n重复Table ID {i+1}: {table_id} (共{len(docs)}个文档)")
            
            for j, doc in enumerate(docs):
                doc_id = doc.get('doc_id', '')
                source = doc.get('source', '')
                context = doc.get('context', '')
                
                print(f"  文档 {j+1}: {doc_id} ({source})")
                print(f"    Context预览: {context[:100]}...")
                
                # 检查context是否真的不同
                if j > 0:
                    prev_context = docs[j-1].get('context', '')
                    if context == prev_context:
                        print(f"    ⚠️ 与前一文档context相同")
                    else:
                        print(f"    ✅ 与前一文档context不同")
    
    # 检查Paragraph ID重复
    paragraph_id_duplicates = {paragraph_id: docs for paragraph_id, docs in paragraph_id_groups.items() if len(docs) > 1}
    
    print(f"\n🔍 Paragraph ID重复统计:")
    print(f"   - 唯一Paragraph ID数量: {len(paragraph_id_groups)}")
    print(f"   - 有重复的Paragraph ID数量: {len(paragraph_id_duplicates)}")
    print(f"   - Paragraph ID重复文档总数: {sum(len(docs) for docs in paragraph_id_duplicates.values())}")
    
    if paragraph_id_duplicates:
        print(f"\n🔍 Paragraph ID重复详情:")
        for i, (paragraph_id, docs) in enumerate(list(paragraph_id_duplicates.items())[:5]):  # 只显示前5个
            print(f"\n重复Paragraph ID {i+1}: {paragraph_id} (共{len(docs)}个文档)")
            
            for j, doc in enumerate(docs):
                doc_id = doc.get('doc_id', '')
                source = doc.get('source', '')
                context = doc.get('context', '')
                
                print(f"  文档 {j+1}: {doc_id} ({source})")
                print(f"    Context预览: {context[:100]}...")
    
    # 总结
    print(f"\n📊 总结:")
    total_duplicates = len(table_id_duplicates) + len(paragraph_id_duplicates)
    if total_duplicates > 0:
        print(f"   ❌ 发现 {total_duplicates} 个重复ID组")
        print(f"   - Table ID重复: {len(table_id_duplicates)} 组")
        print(f"   - Paragraph ID重复: {len(paragraph_id_duplicates)} 组")
        print(f"   这可能是UI中显示重复内容的原因")
    else:
        print(f"   ✅ 没有发现重复ID")
    
    return table_id_duplicates, paragraph_id_duplicates

if __name__ == "__main__":
    check_table_id_duplicates() 