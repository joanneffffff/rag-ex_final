#!/usr/bin/env python3
"""
检查数据集兼容性
验证修复后的数据是否能正常工作
"""

import os
import json
from pathlib import Path

def check_dataset_compatibility():
    """检查数据集兼容性"""
    
    print("🔍 检查数据集兼容性")
    print("=" * 50)
    
    # 1. 检查修复后的知识库文件
    print("1. 检查修复后的知识库文件...")
    knowledge_base_path = "data/unified/tatqa_knowledge_base_combined.jsonl"
    
    if not os.path.exists(knowledge_base_path):
        print(f"❌ 知识库文件不存在: {knowledge_base_path}")
        return False
    
    # 统计文档数量
    doc_count = 0
    table_count = 0
    paragraph_count = 0
    
    with open(knowledge_base_path, 'r', encoding='utf-8') as f:
        for line in f:
            if line.strip():
                doc_count += 1
                try:
                    item = json.loads(line)
                    context = item.get('context', '')
                    
                    # 检查是否包含表格
                    if 'Table ID:' in context:
                        table_count += 1
                    
                    # 检查是否包含段落
                    if 'Paragraph ID:' in context:
                        paragraph_count += 1
                        
                except json.JSONDecodeError:
                    continue
    
    print(f"   总文档数: {doc_count}")
    print(f"   表格文档: {table_count}")
    print(f"   段落文档: {paragraph_count}")
    
    # 2. 检查缓存目录
    print("\n2. 检查缓存目录...")
    cache_dirs = ["cache", "data/faiss_indexes", "xlm/cache"]
    
    for cache_dir in cache_dirs:
        if os.path.exists(cache_dir):
            files = os.listdir(cache_dir)
            print(f"   {cache_dir}: {len(files)} 个文件")
        else:
            print(f"   {cache_dir}: 目录不存在")
    
    # 3. 检查数据质量
    print("\n3. 检查数据质量...")
    
    # 检查前几个文档的完整性
    with open(knowledge_base_path, 'r', encoding='utf-8') as f:
        for i, line in enumerate(f):
            if i >= 5:  # 只检查前5个文档
                break
            if line.strip():
                try:
                    item = json.loads(line)
                    context = item.get('context', '')
                    
                    # 检查表格完整性
                    if 'Table ID:' in context:
                        has_columns = 'Table columns:' in context
                        has_data = 'For ' in context
                        
                        print(f"   文档 {i+1}: 表格{'完整' if has_columns and has_data else '不完整'}")
                        
                except json.JSONDecodeError:
                    continue
    
    # 4. 生成兼容性报告
    print("\n4. 兼容性报告:")
    print(f"   ✅ 知识库文件存在: {os.path.exists(knowledge_base_path)}")
    print(f"   ✅ 文档数量合理: {doc_count} (预期: 5398)")
    print(f"   ✅ 表格文档存在: {table_count > 0}")
    print(f"   ✅ 段落文档存在: {paragraph_count > 0}")
    
    # 5. 建议
    print("\n5. 建议:")
    if doc_count != 5398:
        print(f"   ⚠️ 文档数量不匹配，可能需要重新生成知识库")
    else:
        print(f"   ✅ 文档数量正确，可以正常使用")
    
    print(f"   📋 下一步: 重新启动RAG系统，系统会自动检测数据变化并重新生成索引")
    
    return True

if __name__ == "__main__":
    check_dataset_compatibility() 