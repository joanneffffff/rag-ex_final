#!/usr/bin/env python3
"""
为中文知识库添加doc_id
"""

import json
import hashlib
from pathlib import Path

def add_doc_ids_to_chinese_kb():
    """为中文知识库添加doc_id"""
    
    kb_path = "data/alphafin/alphafin_final_clean.json"
    
    if not Path(kb_path).exists():
        print(f"❌ 中文知识库文件不存在：{kb_path}")
        return
    
    print(f"🔧 正在为中文知识库添加doc_id：{kb_path}")
    
    # 读取数据
    with open(kb_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    print(f"📊 原始记录数：{len(data)}")
    
    # 检查现有doc_id情况
    existing_doc_ids = sum(1 for record in data if record.get('doc_id'))
    print(f"📊 现有doc_id数量：{existing_doc_ids}")
    
    # 为没有doc_id的记录添加doc_id
    added_count = 0
    for i, record in enumerate(data):
        if not record.get('doc_id'):
            # 使用内容哈希作为doc_id
            content = record.get('context', '') or record.get('content', '')
            if content:
                doc_id = hashlib.md5(content.encode('utf-8')).hexdigest()[:16]
                record['doc_id'] = doc_id
                added_count += 1
            else:
                # 如果内容为空，使用索引
                record['doc_id'] = f"chinese_doc_{i}"
                added_count += 1
    
    # 保存修复后的数据
    with open(kb_path, 'w', encoding='utf-8') as f:
        json.dump(data, f, ensure_ascii=False, indent=2)
    
    print(f"✅ 修复完成！")
    print(f"   📊 添加了 {added_count} 个doc_id")
    print(f"   📊 总doc_id覆盖率：{len(data)}/{len(data)} (100%)")
    
    # 显示一些示例
    print(f"\n📋 示例doc_id：")
    for i, record in enumerate(data[:5]):
        print(f"   记录 {i+1}: {record.get('doc_id', 'N/A')}")

if __name__ == "__main__":
    add_doc_ids_to_chinese_kb() 