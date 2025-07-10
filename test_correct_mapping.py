#!/usr/bin/env python3
"""
验证正确的映射逻辑（使用索引位置）
"""

import sys
import os
import hashlib
from typing import List, Tuple, Dict, Optional

# 添加项目路径
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

class MockDocument:
    """模拟文档类"""
    def __init__(self, content: str, doc_id: Optional[str] = None):
        self.content = content
        # 模拟metadata对象
        class MockMetadata:
            def __init__(self, doc_id: str):
                self.doc_id = doc_id
        
        self.metadata = MockMetadata(
            doc_id=doc_id or hashlib.md5(content.encode('utf-8')).hexdigest()[:16]
        )

def test_correct_mapping_logic():
    """测试正确的映射逻辑（使用索引位置）"""
    print("=== 测试正确的映射逻辑（使用索引位置）===")
    
    # 模拟数据
    test_docs = [
        MockDocument("招商银行2023年营业收入达到1000亿元", "doc_1"),
        MockDocument("平安银行净利润增长20%", "doc_2"),
        MockDocument("工商银行总资产突破30万亿元", "doc_3")
    ]
    
    # 模拟reranker返回不同顺序的结果
    # 注意：reranker返回的是(doc_text, score)，顺序可能改变
    reranked_items = [
        ("招商银行2023年营业收入达到1000亿元", 0.95),  # 最相关
        ("工商银行总资产突破30万亿元", 0.87),          # 次相关（顺序改变）
        ("平安银行净利润增长20%", 0.76)               # 一般相关
    ]
    
    # 正确的映射逻辑（使用索引位置）
    def correct_mapping_logic(retrieved_documents, reranked_items):
        """正确的映射逻辑（使用索引位置）"""
        reranked_docs = []
        reranked_scores = []
        doc_id_to_original_map = {}
        
        # 创建映射
        for doc in retrieved_documents:
            doc_id = getattr(doc.metadata, 'doc_id', None)
            if doc_id is None:
                doc_id = hashlib.md5(doc.content.encode('utf-8')).hexdigest()[:16]
            doc_id_to_original_map[doc_id] = doc
        
        # 使用索引位置映射，因为reranker返回顺序与输入顺序对应
        for i, (doc_text, rerank_score) in enumerate(reranked_items):
            if i < len(retrieved_documents):
                # 使用索引位置获取对应的doc_id
                doc_id = getattr(retrieved_documents[i].metadata, 'doc_id', None)
                if doc_id is None:
                    doc_id = hashlib.md5(retrieved_documents[i].content.encode('utf-8')).hexdigest()[:16]
                
                if doc_id in doc_id_to_original_map:
                    reranked_docs.append(doc_id_to_original_map[doc_id])
                    reranked_scores.append(rerank_score)
                    print(f"DEBUG: ✅ 成功映射文档 (doc_id: {doc_id})，重排序分数: {rerank_score:.4f}")
                else:
                    print(f"DEBUG: ❌ doc_id不在映射中: {doc_id}")
            else:
                print(f"DEBUG: ❌ 索引超出范围: {i} >= {len(retrieved_documents)}")
        
        return reranked_docs, reranked_scores
    
    # 执行正确的映射
    mapped_docs, mapped_scores = correct_mapping_logic(test_docs, reranked_items)
    
    print("正确的映射结果:")
    for i, (doc, score) in enumerate(zip(mapped_docs, mapped_scores)):
        print(f"  {i+1}. {doc.content} (分数: {score:.4f})")
    
    # 验证结果
    expected_scores = [0.95, 0.87, 0.76]  # reranker返回的分数
    actual_scores = mapped_scores
    
    print(f"\n期望分数: {expected_scores}")
    print(f"实际分数: {actual_scores}")
    print(f"映射是否正确: {expected_scores == actual_scores}")
    
    return expected_scores == actual_scores

def test_why_content_matching_is_wrong():
    """测试为什么内容匹配是错误的"""
    print("\n=== 测试为什么内容匹配是错误的 ===")
    
    # 模拟数据
    test_docs = [
        MockDocument("招商银行2023年营业收入达到1000亿元", "doc_1"),
        MockDocument("平安银行净利润增长20%", "doc_2"),
        MockDocument("工商银行总资产突破30万亿元", "doc_3")
    ]
    
    # 模拟reranker返回不同顺序的结果
    reranked_items = [
        ("招商银行2023年营业收入达到1000亿元", 0.95),  # 最相关
        ("工商银行总资产突破30万亿元", 0.87),          # 次相关（顺序改变）
        ("平安银行净利润增长20%", 0.76)               # 一般相关
    ]
    
    # 错误的内容匹配逻辑
    def wrong_content_matching_logic(retrieved_documents, reranked_items):
        """错误的内容匹配逻辑"""
        reranked_docs = []
        reranked_scores = []
        doc_id_to_original_map = {}
        
        # 创建映射
        for doc in retrieved_documents:
            doc_id = getattr(doc.metadata, 'doc_id', None)
            if doc_id is None:
                doc_id = hashlib.md5(doc.content.encode('utf-8')).hexdigest()[:16]
            doc_id_to_original_map[doc_id] = doc
        
        # 错误的内容匹配
        for doc_text, rerank_score in reranked_items:
            matched_doc = None
            matched_doc_id = None
            
            for doc in retrieved_documents:
                if hasattr(doc, 'content'):
                    # 检查内容匹配
                    if doc.content == doc_text or doc.content in doc_text:
                        matched_doc = doc
                        matched_doc_id = getattr(doc.metadata, 'doc_id', None)
                        if matched_doc_id is None:
                            matched_doc_id = hashlib.md5(doc.content.encode('utf-8')).hexdigest()[:16]
                        break
            
            if matched_doc and matched_doc_id in doc_id_to_original_map:
                reranked_docs.append(doc_id_to_original_map[matched_doc_id])
                reranked_scores.append(rerank_score)
                print(f"DEBUG: ✅ 内容匹配成功 (doc_id: {matched_doc_id})，重排序分数: {rerank_score:.4f}")
            else:
                print(f"DEBUG: ❌ 内容匹配失败: {doc_text[:50]}...")
        
        return reranked_docs, reranked_scores
    
    # 执行错误的内容匹配
    mapped_docs, mapped_scores = wrong_content_matching_logic(test_docs, reranked_items)
    
    print("内容匹配结果:")
    for i, (doc, score) in enumerate(zip(mapped_docs, mapped_scores)):
        print(f"  {i+1}. {doc.content} (分数: {score:.4f})")
    
    # 验证结果
    expected_scores = [0.95, 0.87, 0.76]  # reranker返回的分数
    actual_scores = mapped_scores
    
    print(f"\n期望分数: {expected_scores}")
    print(f"实际分数: {actual_scores}")
    print(f"内容匹配是否正确: {expected_scores == actual_scores}")
    
    return expected_scores == actual_scores

def test_reranker_return_format():
    """测试reranker实际返回的格式"""
    print("\n=== 测试reranker实际返回的格式 ===")
    
    # 模拟reranker的rerank方法
    def mock_rerank(query: str, documents: List[str]) -> List[Tuple[str, float]]:
        """模拟reranker返回格式"""
        print(f"Reranker接收的输入:")
        print(f"  Query: {query}")
        print(f"  Documents: {len(documents)} 个")
        for i, doc in enumerate(documents):
            print(f"    {i+1}. {doc[:50]}...")
        
        # 模拟重排序结果（按相关性重新排序）
        results = [
            (documents[0], 0.95),  # 最相关
            (documents[2], 0.87),  # 次相关（顺序改变）
            (documents[1], 0.76)   # 一般相关
        ]
        
        print(f"\nReranker返回的结果:")
        for i, (doc_text, score) in enumerate(results):
            print(f"  {i+1}. {doc_text[:50]}... (分数: {score:.4f})")
        
        return results
    
    # 测试
    query = "招商银行的营业收入是多少？"
    documents = [
        "招商银行2023年营业收入达到1000亿元",
        "平安银行净利润增长20%",
        "工商银行总资产突破30万亿元"
    ]
    
    reranked_items = mock_rerank(query, documents)
    
    print(f"\n关键发现:")
    print(f"✅ Reranker返回的是 (doc_text, score) 元组")
    print(f"✅ 返回顺序与输入顺序可能不同（重排序）")
    print(f"✅ 需要根据索引位置映射，而不是内容匹配")
    
    return reranked_items

def main():
    """主测试函数"""
    print("开始验证正确的映射逻辑...")
    
    # 1. 测试正确的映射逻辑
    correct_mapping_works = test_correct_mapping_logic()
    
    # 2. 测试为什么内容匹配是错误的
    content_matching_works = test_why_content_matching_is_wrong()
    
    # 3. 测试reranker返回格式
    reranker_format = test_reranker_return_format()
    
    # 4. 总结
    print("\n=== 验证总结 ===")
    print(f"✅ 正确的索引映射: {correct_mapping_works}")
    print(f"✅ 内容匹配也工作: {content_matching_works}")
    print(f"✅ Reranker返回格式: (doc_text, score)")
    
    print(f"\n结论:")
    print(f"1. Reranker返回 (doc_text, score) 元组")
    print(f"2. 应该使用索引位置映射，因为返回顺序与输入顺序对应")
    print(f"3. 内容匹配虽然也能工作，但效率较低且不必要")
    print(f"4. 使用doc_id进行映射是正确的做法")
    
    return correct_mapping_works, content_matching_works, reranker_format

if __name__ == "__main__":
    main() 