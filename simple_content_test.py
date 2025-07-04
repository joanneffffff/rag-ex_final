#!/usr/bin/env python3
"""
简单的数据内容测试
"""

import json
from pathlib import Path

def test_data_content():
    """测试数据内容"""
    print("=" * 60)
    print("数据内容测试")
    print("=" * 60)
    
    # 检查中文数据文件
    chinese_file = Path("data/unified/alphafin_unified.json")
    if chinese_file.exists():
        print(f"✅ 中文数据文件存在: {chinese_file}")
        print(f"   文件大小: {chinese_file.stat().st_size / 1024 / 1024:.2f} MB")
        
        # 读取前几条记录
        with open(chinese_file, 'r', encoding='utf-8') as f:
            data = json.load(f)
            print(f"   总记录数: {len(data)}")
            
            if len(data) > 0:
                first_record = data[0]
                print(f"   第一条记录字段: {list(first_record.keys())}")
                
                original_context = first_record.get('original_context', '')
                print(f"   第一条记录original_context长度: {len(original_context)} 字符")
                print(f"   前200字符: {original_context[:200]}...")
                print()
    else:
        print(f"❌ 中文数据文件不存在: {chinese_file}")
    
    # 检查英文数据文件
    english_file = Path("data/unified/tatqa_knowledge_base_unified.jsonl")
    if english_file.exists():
        print(f"✅ 英文数据文件存在: {english_file}")
        print(f"   文件大小: {english_file.stat().st_size / 1024 / 1024:.2f} MB")
        
        # 读取前几条记录
        count = 0
        with open(english_file, 'r', encoding='utf-8') as f:
            for line in f:
                if count >= 3:
                    break
                data = json.loads(line.strip())
                count += 1
                print(f"   第{count}条记录字段: {list(data.keys())}")
                
                context = data.get('context', '')
                print(f"   第{count}条记录context长度: {len(context)} 字符")
                print(f"   前200字符: {context[:200]}...")
                print()
    else:
        print(f"❌ 英文数据文件不存在: {english_file}")

if __name__ == "__main__":
    test_data_content() 