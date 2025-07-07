#!/usr/bin/env python3
"""
测试Table ID去重功能
"""

import re

def test_table_id_deduplication():
    """测试Table ID去重功能"""
    print("=" * 80)
    print("测试Table ID去重功能")
    print("=" * 80)
    
    # 模拟您提供的重复Table ID内容
    table_content_1 = """Table ID: 268b1d1e-62dc-4e74-9efe-78f5b9c492a8
Table columns: , December 31,, .
All monetary amounts are in million of USD.
Row 1 data: December 31, is 2019, Value is 2018.
For Internally developed software: December 31, is 808.2, Value is 746.0.
For Purchased software: December 31, is 78.9, Value is 60.7.
For Computer software: December 31, is 887.1, Value is 806.7.
For Accumulated amortization: December 31, is a negative 481.1, Value is a negative 401.1.
For Computer software, net: December 31, is 406.0, Value is 405.6."""

    table_content_2 = """Table ID: 268b1d1e-62dc-4e74-9efe-78f5b9c492a8
Table columns: , December 31,, .
All monetary amounts are in percentage.
Row 1 data: December 31, is 2019, Value is 2018.
For Internally developed software: December 31, is 808.2, Value is 746.0.
For Purchased software: December 31, is 78.9, Value is 60.7.
For Computer software: December 31, is 887.1, Value is 806.7.
For Accumulated amortization: December 31, is a negative 481.1, Value is a negative 401.1.
For Computer software, net: December 31, is 406.0, Value is 405.6."""

    different_content = """Table ID: 12345678-1234-1234-1234-123456789abc
Table columns: Revenue, Year.
All amounts in USD.
Row 1: Revenue is 1000, Year is 2023."""

    # 模拟文档列表
    mock_docs = [
        ("doc1", 0.95, table_content_1),
        ("doc2", 0.92, table_content_2),  # 相同Table ID，不同单位
        ("doc3", 0.88, different_content),  # 不同Table ID
        ("doc4", 0.85, "No table content here"),  # 无Table ID
    ]
    
    print("原始文档列表:")
    for i, (doc_id, score, content) in enumerate(mock_docs):
        print(f"  文档 {i+1}: {doc_id} (分数: {score:.2f})")
        print(f"    内容前50字符: {content[:50]}...")
        if "Table ID:" in content:
            table_id_match = re.search(r'Table ID:\s*([a-f0-9-]+)', content)
            if table_id_match:
                print(f"    Table ID: {table_id_match.group(1)}")
    
    # 应用Table ID去重逻辑
    print("\n应用Table ID去重逻辑...")
    unique_docs = []
    seen_table_ids = set()
    seen_hashes = set()
    
    for doc_id, score, content in mock_docs:
        # 检查是否包含Table ID
        if "Table ID:" in content:
            table_id_match = re.search(r'Table ID:\s*([a-f0-9-]+)', content)
            if table_id_match:
                table_id = table_id_match.group(1)
                if table_id in seen_table_ids:
                    print(f"  ❌ 跳过重复Table ID: {table_id} (文档: {doc_id})")
                    continue
                seen_table_ids.add(table_id)
                print(f"  ✅ 保留Table ID: {table_id} (文档: {doc_id})")
        
        # 常规内容去重
        content_hash = hash(content)
        if content_hash in seen_hashes:
            print(f"  ❌ 跳过重复内容: {doc_id}")
            continue
        
        seen_hashes.add(content_hash)
        unique_docs.append((doc_id, score, content))
        print(f"  ✅ 保留文档: {doc_id}")
    
    print(f"\n去重结果:")
    print(f"  原始文档数: {len(mock_docs)}")
    print(f"  去重后文档数: {len(unique_docs)}")
    print(f"  移除重复数: {len(mock_docs) - len(unique_docs)}")
    
    print("\n去重后的文档:")
    for i, (doc_id, score, content) in enumerate(unique_docs):
        print(f"  文档 {i+1}: {doc_id} (分数: {score:.2f})")
        print(f"    内容前50字符: {content[:50]}...")
        if "Table ID:" in content:
            table_id_match = re.search(r'Table ID:\s*([a-f0-9-]+)', content)
            if table_id_match:
                print(f"    Table ID: {table_id_match.group(1)}")
    
    # 验证修复效果
    print("\n验证修复效果:")
    if len(unique_docs) < len(mock_docs):
        print("✅ Table ID去重逻辑正常工作，重复Table ID已被移除")
    else:
        print("❌ Table ID去重逻辑可能有问题")
    
    # 检查是否还有重复的Table ID
    final_table_ids = set()
    for doc_id, score, content in unique_docs:
        if "Table ID:" in content:
            table_id_match = re.search(r'Table ID:\s*([a-f0-9-]+)', content)
            if table_id_match:
                table_id = table_id_match.group(1)
                if table_id in final_table_ids:
                    print(f"❌ 仍然存在重复Table ID: {table_id}")
                else:
                    final_table_ids.add(table_id)
    
    if len(final_table_ids) == len([d for d in unique_docs if "Table ID:" in d[2]]):
        print("✅ 最终结果中无重复Table ID")
    else:
        print("❌ 最终结果中仍有重复Table ID")

if __name__ == "__main__":
    test_table_id_deduplication() 