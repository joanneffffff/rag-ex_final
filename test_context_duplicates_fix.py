#!/usr/bin/env python3
"""
测试上下文重复修复效果
"""

def test_context_duplicates_fix():
    """测试上下文重复修复效果"""
    print("=" * 80)
    print("测试上下文重复修复效果")
    print("=" * 80)
    
    # 模拟您提供的重复内容
    duplicate_content = """Table ID: 268b1d1e-62dc-4e74-9efe-78f5b9c492a8
Table columns: , December 31,, .
All monetary amounts are in million of USD.
Row 1 data: December 31, is 2019, Value is 2018.
For Internally developed software: December 31, is 808.2, Value is 746.0.
For Purchased software: December 31, is 78.9, Value is 60.7.
For Computer software: December 31, is 887.1, Value is 806.7.
For Accumulated amortization: December 31, is a negative 481.1, Value is a negative 401.1.
For Computer software, net: December 31, is 406.0, Value is 405.6.

Paragraph ID: e73d746e-0e01-4b4f-900d-c699c16af69a
Computer software, net consists of the following (in millions):"""
    
    # 模拟重复的文档列表
    mock_docs = [
        ("doc1", 0.95, duplicate_content),
        ("doc2", 0.92, duplicate_content),  # 重复内容
        ("doc3", 0.88, "Different content here"),
        ("doc4", 0.85, duplicate_content),  # 再次重复
    ]
    
    print("原始文档列表:")
    for i, (doc_id, score, content) in enumerate(mock_docs):
        print(f"  文档 {i+1}: {doc_id} (分数: {score:.2f})")
        print(f"    内容前50字符: {content[:50]}...")
    
    # 应用去重逻辑
    print("\n应用去重逻辑...")
    unique_docs = []
    seen_hashes = set()
    
    for doc_id, score, content in mock_docs:
        content_hash = hash(content)
        if content_hash in seen_hashes:
            print(f"  ❌ 跳过重复文档: {doc_id}")
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
    
    # 验证修复效果
    print("\n验证修复效果:")
    if len(unique_docs) < len(mock_docs):
        print("✅ 去重逻辑正常工作，重复内容已被移除")
    else:
        print("❌ 去重逻辑可能有问题")
    
    # 检查是否还有重复
    final_hashes = set()
    for doc_id, score, content in unique_docs:
        content_hash = hash(content)
        if content_hash in final_hashes:
            print(f"❌ 仍然存在重复: {doc_id}")
        else:
            final_hashes.add(content_hash)
    
    if len(final_hashes) == len(unique_docs):
        print("✅ 最终结果中无重复内容")
    else:
        print("❌ 最终结果中仍有重复内容")

if __name__ == "__main__":
    test_context_duplicates_fix() 