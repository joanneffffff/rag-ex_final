#!/usr/bin/env python3
"""
检查中文数据的详细内容
"""

import json
from pathlib import Path

def check_chinese_data():
    """检查中文数据"""
    print("=" * 60)
    print("中文数据详细检查")
    print("=" * 60)
    
    chinese_file = Path("data/unified/alphafin_unified.json")
    if not chinese_file.exists():
        print(f"❌ 中文数据文件不存在: {chinese_file}")
        return
    
    with open(chinese_file, 'r', encoding='utf-8') as f:
        data = json.load(f)
        print(f"总记录数: {len(data)}")
        
        # 检查前5条记录的所有字段
        for i in range(min(5, len(data))):
            record = data[i]
            print(f"\n记录 {i+1}:")
            print(f"  字段: {list(record.keys())}")
            
            # 检查各个字段的长度
            for field in ['original_context', 'context', 'summary', 'original_content']:
                if field in record:
                    content = record[field]
                    if isinstance(content, str):
                        print(f"  {field}: {len(content)} 字符")
                        if len(content) > 0:
                            print(f"    前100字符: {content[:100]}...")
                    else:
                        print(f"  {field}: 非字符串类型 ({type(content)})")
                else:
                    print(f"  {field}: 字段不存在")
            
            print()

if __name__ == "__main__":
    check_chinese_data() 