#!/usr/bin/env python3
"""
为原始数据文件添加doc_id字段
目标：为alphafin_unified.json的每条记录添加唯一doc_id，便于RAG链路溯源
"""

import json
import hashlib
import os
from typing import Dict, Any

def generate_doc_id(record: Dict[str, Any], index: int) -> str:
    """
    为记录生成唯一doc_id
    策略：使用context字段的hash + 索引，确保唯一性
    """
    # 使用context字段作为主要标识
    context = record.get('context', '')
    if not context:
        # 如果没有context，使用original_content
        context = record.get('original_content', '')
    
    # 生成hash
    content_hash = hashlib.md5(context.encode('utf-8')).hexdigest()[:8]
    
    # 结合索引确保唯一性
    doc_id = f"doc_{content_hash}_{index:06d}"
    
    return doc_id

def add_doc_id_to_data(input_file: str, output_file: str | None = None) -> None:
    """
    为数据文件添加doc_id字段
    
    Args:
        input_file: 输入文件路径
        output_file: 输出文件路径，如果为None则覆盖原文件
    """
    if output_file is None:
        output_file = input_file
    
    print(f"正在处理文件: {input_file}")
    
    # 读取原始数据
    with open(input_file, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    print(f"原始数据包含 {len(data)} 条记录")
    
    # 为每条记录添加doc_id
    for i, record in enumerate(data):
        doc_id = generate_doc_id(record, i)
        record['doc_id'] = doc_id
        
        # 每处理1000条记录打印一次进度
        if (i + 1) % 1000 == 0:
            print(f"已处理 {i + 1}/{len(data)} 条记录")
    
    # 保存更新后的数据
    print(f"正在保存到: {output_file}")
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(data, f, ensure_ascii=False, indent=2)
    
    print(f"完成！已为 {len(data)} 条记录添加doc_id")
    
    # 显示前几条记录的doc_id示例
    print("\n前3条记录的doc_id示例:")
    for i in range(min(3, len(data))):
        print(f"记录 {i+1}: {data[i]['doc_id']}")

def main():
    """主函数"""
    input_file = "data/unified/alphafin_unified.json"
    
    # 检查文件是否存在
    if not os.path.exists(input_file):
        print(f"错误：文件 {input_file} 不存在")
        return
    
    # 创建备份文件
    backup_file = input_file + ".backup"
    print(f"创建备份文件: {backup_file}")
    
    # 复制原文件作为备份
    import shutil
    shutil.copy2(input_file, backup_file)
    
    # 添加doc_id
    add_doc_id_to_data(input_file)
    
    print(f"\n处理完成！")
    print(f"原文件已备份为: {backup_file}")
    print(f"更新后的文件: {input_file}")

if __name__ == "__main__":
    main() 