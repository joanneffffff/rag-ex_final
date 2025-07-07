#!/usr/bin/env python3
"""
分析知识库中的重复内容
"""

import json
from collections import defaultdict
import hashlib

def analyze_duplicates():
    """分析知识库中的重复内容"""
    
    knowledge_base_file = "data/unified/tatqa_knowledge_base_combined.jsonl"
    
    print("🔍 分析知识库重复内容...")
    
    # 收集所有文档
    documents = []
    with open(knowledge_base_file, 'r', encoding='utf-8') as f:
        for line in f:
            if line.strip():
                doc = json.loads(line)
                documents.append(doc)
    
    print(f"📊 总文档数: {len(documents)}")
    
    # 按context分组
    context_groups = defaultdict(list)
    for doc in documents:
        context = doc.get('context', '')
        if context:
            # 标准化context用于比较
            normalized_context = ' '.join(context.split())
            context_groups[normalized_context].append(doc)
    
    # 找出重复的context
    duplicates = {context: docs for context, docs in context_groups.items() if len(docs) > 1}
    
    print(f"\n📋 重复内容统计:")
    print(f"   - 唯一context数量: {len(context_groups)}")
    print(f"   - 有重复的context数量: {len(duplicates)}")
    print(f"   - 重复文档总数: {sum(len(docs) for docs in duplicates.values())}")
    
    if duplicates:
        print(f"\n🔍 重复内容详情:")
        for i, (context, docs) in enumerate(list(duplicates.items())[:10]):  # 只显示前10个
            print(f"\n重复组 {i+1} (共{len(docs)}个文档):")
            print(f"Context预览: {context[:100]}...")
            
            for j, doc in enumerate(docs):
                doc_id = doc.get('doc_id', '')
                source = doc.get('source', '')
                print(f"  {j+1}. {doc_id} ({source})")
                
                # 显示ID相关字段
                if 'all_doc_ids' in doc:
                    print(f"     所有doc_ids: {doc['all_doc_ids']}")
                if 'all_table_ids' in doc:
                    print(f"     所有table_ids: {doc['all_table_ids']}")
                if 'all_paragraph_ids' in doc:
                    print(f"     所有paragraph_ids: {doc['all_paragraph_ids']}")
        
        if len(duplicates) > 10:
            print(f"\n... 还有 {len(duplicates) - 10} 个重复组未显示")
    
    # 分析重复的原因
    print(f"\n🔍 重复原因分析:")
    
    # 按source统计
    source_stats = defaultdict(int)
    for docs in duplicates.values():
        for doc in docs:
            source = doc.get('source', '')
            source_stats[source] += 1
    
    print(f"重复文档的来源分布:")
    for source, count in source_stats.items():
        print(f"  {source}: {count} 个文档")
    
    # 检查是否有跨source的重复
    cross_source_duplicates = 0
    for context, docs in duplicates.items():
        sources = set(doc.get('source', '') for doc in docs)
        if len(sources) > 1:
            cross_source_duplicates += 1
    
    print(f"\n跨来源重复的context数量: {cross_source_duplicates}")
    
    return duplicates

if __name__ == "__main__":
    analyze_duplicates() 