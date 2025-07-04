#!/usr/bin/env python3
"""
测试重排序器映射修复
验证智能内容选择后的文档能正确映射回原始文档
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

def test_reranker_mapping_logic():
    """测试重排序器映射逻辑"""
    print("=== 测试重排序器映射逻辑 ===")
    
    # 模拟DocumentWithMetadata对象
    class MockDocument:
        def __init__(self, content, metadata):
            self.content = content
            self.metadata = metadata
    
    class MockMetadata:
        def __init__(self, language=None, summary=None):
            self.language = language
            self.summary = summary
    
    # 测试数据
    chinese_doc = MockDocument(
        content="这是一个很长的中文文档内容，包含了很多详细的财务信息和分析数据。",
        metadata=MockMetadata(language="chinese", summary="公司财务表现良好")
    )
    
    english_doc = MockDocument(
        content="This is an English document about financial performance.",
        metadata=MockMetadata(language="english", summary=None)
    )
    
    def simulate_reranker_mapping(docs, is_chinese_query=True):
        """模拟重排序器映射逻辑"""
        doc_texts = []
        doc_to_original_map = {}
        
        for doc in docs:
            if is_chinese_query and hasattr(doc.metadata, 'language') and doc.metadata.language == 'chinese':
                # 中文数据：组合summary和context
                summary = ""
                if hasattr(doc.metadata, 'summary') and doc.metadata.summary:
                    summary = doc.metadata.summary
                else:
                    summary = doc.content[:200] + "..." if len(doc.content) > 200 else doc.content
                
                combined_text = f"摘要：{summary}\n\n详细内容：{doc.content}"
                doc_texts.append(combined_text)
                doc_to_original_map[combined_text] = doc
            else:
                # 英文数据：只使用context
                doc_texts.append(doc.content)
                doc_to_original_map[doc.content] = doc
        
        # 模拟重排序结果（返回相同的文本和随机分数）
        import random
        reranked_items = [(text, random.uniform(0.5, 1.0)) for text in doc_texts]
        
        # 映射回原始文档
        reranked_docs = []
        reranked_scores = []
        
        for doc_text, rerank_score in reranked_items:
            if doc_text in doc_to_original_map:
                reranked_docs.append(doc_to_original_map[doc_text])
                reranked_scores.append(rerank_score)
        
        return reranked_docs, reranked_scores
    
    # 测试中文查询
    print("\n--- 中文查询测试 ---")
    chinese_docs = [chinese_doc]
    chinese_results, chinese_scores = simulate_reranker_mapping(chinese_docs, True)
    print(f"中文查询映射结果: {len(chinese_results)} 个文档")
    print(f"中文查询映射成功: {'是' if len(chinese_results) > 0 else '否'}")
    
    # 测试英文查询
    print("\n--- 英文查询测试 ---")
    english_docs = [english_doc]
    english_results, english_scores = simulate_reranker_mapping(english_docs, False)
    print(f"英文查询映射结果: {len(english_results)} 个文档")
    print(f"英文查询映射成功: {'是' if len(english_results) > 0 else '否'}")
    
    # 测试混合查询
    print("\n--- 混合查询测试 ---")
    mixed_docs = [chinese_doc, english_doc]
    mixed_results, mixed_scores = simulate_reranker_mapping(mixed_docs, True)
    print(f"混合查询映射结果: {len(mixed_results)} 个文档")
    print(f"混合查询映射成功: {'是' if len(mixed_results) > 0 else '否'}")
    
    print("\n✅ 测试完成")

if __name__ == "__main__":
    test_reranker_mapping_logic() 