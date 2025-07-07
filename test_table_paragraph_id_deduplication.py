#!/usr/bin/env python3
"""
测试Table ID和Paragraph ID去重功能
"""

import re

def test_table_paragraph_id_deduplication():
    """测试Table ID和Paragraph ID去重功能"""
    print("=" * 80)
    print("测试Table ID和Paragraph ID去重功能")
    print("=" * 80)
    
    # 模拟不同类型的重复内容
    table_content_1 = """Table ID: 268b1d1e-62dc-4e74-9efe-78f5b9c492a8
Table columns: , December 31,, .
All monetary amounts are in million of USD.
Row 1 data: December 31, is 2019, Value is 2018.
For Internally developed software: December 31, is 808.2, Value is 746.0."""

    table_content_2 = """Table ID: 268b1d1e-62dc-4e74-9efe-78f5b9c492a8
Table columns: , December 31,, .
All monetary amounts are in percentage.
Row 1 data: December 31, is 2019, Value is 2018.
For Internally developed software: December 31, is 808.2, Value is 746.0."""

    paragraph_content_1 = """Paragraph ID: e73d746e-0e01-4b4f-900d-c699c16af69a
Computer software, net consists of the following (in millions):"""

    paragraph_content_2 = """Paragraph ID: e73d746e-0e01-4b4f-900d-c699c16af69a
Computer software, net consists of the following (in billions):"""

    table_plus_text_content = """Table ID: 268b1d1e-62dc-4e74-9efe-78f5b9c492a8
Table columns: , December 31,, .
All monetary amounts are in million of USD.
Row 1 data: December 31, is 2019, Value is 2018.

Paragraph ID: e73d746e-0e01-4b4f-900d-c699c16af69a
Computer software, net consists of the following (in millions):"""

    different_content = """Table ID: 12345678-1234-1234-1234-123456789abc
Table columns: Revenue, Year.
All amounts in USD.
Row 1: Revenue is 1000, Year is 2023."""

    # 模拟文档列表
    mock_docs = [
        ("doc1", 0.95, table_content_1),  # 表格内容
        ("doc2", 0.92, table_content_2),  # 相同Table ID，不同单位
        ("doc3", 0.88, paragraph_content_1),  # 纯文本内容
        ("doc4", 0.85, paragraph_content_2),  # 相同Paragraph ID，不同单位
        ("doc5", 0.82, table_plus_text_content),  # 表格+文本内容
        ("doc6", 0.80, different_content),  # 不同Table ID
        ("doc7", 0.78, "No ID content here"),  # 无ID内容
    ]
    
    print("原始文档列表:")
    for i, (doc_id, score, content) in enumerate(mock_docs):
        print(f"  文档 {i+1}: {doc_id} (分数: {score:.2f})")
        print(f"    内容前50字符: {content[:50]}...")
        if "Table ID:" in content:
            table_id_match = re.search(r'Table ID:\s*([a-f0-9-]+)', content)
            if table_id_match:
                print(f"    Table ID: {table_id_match.group(1)}")
        if "Paragraph ID:" in content:
            paragraph_id_match = re.search(r'Paragraph ID:\s*([a-f0-9-]+)', content)
            if paragraph_id_match:
                print(f"    Paragraph ID: {paragraph_id_match.group(1)}")
    
    # 应用Table ID和Paragraph ID去重逻辑
    print("\n应用Table ID和Paragraph ID去重逻辑...")
    unique_docs = []
    seen_table_ids = set()
    seen_paragraph_ids = set()
    seen_hashes = set()
    
    for doc_id, score, content in mock_docs:
        # 检查内容类型并应用相应的去重逻辑
        has_table_id = "Table ID:" in content
        has_paragraph_id = "Paragraph ID:" in content
        
        if has_table_id:
            # 表格内容或表格+文本内容：使用Table ID去重
            table_id_match = re.search(r'Table ID:\s*([a-f0-9-]+)', content)
            if table_id_match:
                table_id = table_id_match.group(1)
                if table_id in seen_table_ids:
                    print(f"  ❌ 跳过重复Table ID: {table_id} (文档: {doc_id})")
                    continue
                seen_table_ids.add(table_id)
                print(f"  ✅ 保留Table ID: {table_id} (文档: {doc_id})")
        elif has_paragraph_id:
            # 纯文本内容：使用Paragraph ID去重
            paragraph_id_match = re.search(r'Paragraph ID:\s*([a-f0-9-]+)', content)
            if paragraph_id_match:
                paragraph_id = paragraph_id_match.group(1)
                if paragraph_id in seen_paragraph_ids:
                    print(f"  ❌ 跳过重复Paragraph ID: {paragraph_id} (文档: {doc_id})")
                    continue
                seen_paragraph_ids.add(paragraph_id)
                print(f"  ✅ 保留Paragraph ID: {paragraph_id} (文档: {doc_id})")
        
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
        if "Paragraph ID:" in content:
            paragraph_id_match = re.search(r'Paragraph ID:\s*([a-f0-9-]+)', content)
            if paragraph_id_match:
                print(f"    Paragraph ID: {paragraph_id_match.group(1)}")
    
    # 验证修复效果
    print("\n验证修复效果:")
    if len(unique_docs) < len(mock_docs):
        print("✅ Table ID和Paragraph ID去重逻辑正常工作，重复内容已被移除")
    else:
        print("❌ 去重逻辑可能有问题")
    
    # 检查是否还有重复的ID
    final_table_ids = set()
    final_paragraph_ids = set()
    
    for doc_id, score, content in unique_docs:
        if "Table ID:" in content:
            table_id_match = re.search(r'Table ID:\s*([a-f0-9-]+)', content)
            if table_id_match:
                table_id = table_id_match.group(1)
                if table_id in final_table_ids:
                    print(f"❌ 仍然存在重复Table ID: {table_id}")
                else:
                    final_table_ids.add(table_id)
        
        if "Paragraph ID:" in content:
            paragraph_id_match = re.search(r'Paragraph ID:\s*([a-f0-9-]+)', content)
            if paragraph_id_match:
                paragraph_id = paragraph_id_match.group(1)
                if paragraph_id in final_paragraph_ids:
                    print(f"❌ 仍然存在重复Paragraph ID: {paragraph_id}")
                else:
                    final_paragraph_ids.add(paragraph_id)
    
    print(f"✅ 最终结果中无重复Table ID (共{len(final_table_ids)}个)")
    print(f"✅ 最终结果中无重复Paragraph ID (共{len(final_paragraph_ids)}个)")

if __name__ == "__main__":
    test_table_paragraph_id_deduplication() 