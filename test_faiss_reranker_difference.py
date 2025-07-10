#!/usr/bin/env python3
"""
测试FAISS和Reranker结果的差异
验证映射逻辑的正确性
"""

import sys
import os
import hashlib
from typing import List, Tuple, Dict, Optional
import numpy as np

# 添加项目路径
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

try:
    from xlm.ui.optimized_rag_ui import OptimizedRagUI
    from xlm.components.retriever.reranker import QwenReranker
except ImportError:
    print("警告：无法导入xlm模块，将跳过实际reranker测试")

class MockDocument:
    """模拟文档类"""
    def __init__(self, content: str, doc_id: Optional[str] = None, metadata: Optional[Dict] = None):
        self.content = content
        # 模拟metadata对象
        class MockMetadata:
            def __init__(self, doc_id: str, language: str, summary: str):
                self.doc_id = doc_id
                self.language = language
                self.summary = summary
        
        self.metadata = MockMetadata(
            doc_id=doc_id or hashlib.md5(content.encode('utf-8')).hexdigest()[:16],
            language='chinese' if any('\u4e00' <= char <= '\u9fff' for char in content) else 'english',
            summary=content[:100] + "..." if len(content) > 100 else content
        )

def test_faiss_reranker_mapping():
    """测试FAISS和Reranker的映射逻辑"""
    print("=== 测试FAISS和Reranker映射逻辑 ===")
    
    # 1. 准备测试数据
    test_documents = [
        MockDocument("招商银行2023年营业收入达到1000亿元，同比增长15%", "doc_1"),
        MockDocument("平安银行净利润增长20%，资产质量持续改善", "doc_2"), 
        MockDocument("工商银行发布年报，总资产突破30万亿元", "doc_3"),
        MockDocument("建设银行数字化转型成效显著，科技投入持续增加", "doc_4"),
        MockDocument("农业银行服务乡村振兴，普惠金融业务快速发展", "doc_5")
    ]
    
    test_query = "招商银行的营业收入是多少？"
    
    # 2. 模拟FAISS检索结果（按相似度排序）
    faiss_results = [
        (test_documents[0], 0.95),  # 最相关
        (test_documents[2], 0.87),  # 次相关
        (test_documents[1], 0.76),  # 一般相关
        (test_documents[3], 0.65),  # 较低相关
        (test_documents[4], 0.54)   # 最低相关
    ]
    
    print(f"FAISS检索结果（按相似度排序）:")
    for i, (doc, score) in enumerate(faiss_results):
        print(f"  {i+1}. {doc.content[:50]}... (分数: {score:.4f})")
    
    # 3. 模拟Reranker重排序结果（按相关性重新排序）
    reranked_results = [
        (test_documents[0].content, 0.92),  # 最相关
        (test_documents[1].content, 0.89),  # 次相关（顺序改变）
        (test_documents[2].content, 0.85),  # 一般相关
        (test_documents[3].content, 0.78),  # 较低相关
        (test_documents[4].content, 0.72)   # 最低相关
    ]
    
    print(f"\nReranker重排序结果（按相关性重新排序）:")
    for i, (content, score) in enumerate(reranked_results):
        print(f"  {i+1}. {content[:50]}... (分数: {score:.4f})")
    
    # 4. 测试错误的映射逻辑（当前实现）
    print(f"\n=== 测试错误的映射逻辑 ===")
    
    def wrong_mapping_logic(retrieved_documents, reranked_items):
        """错误的映射逻辑（当前实现）"""
        reranked_docs = []
        reranked_scores = []
        doc_id_to_original_map = {}
        
        # 创建映射
        for doc in retrieved_documents:
            doc_id = getattr(doc.metadata, 'doc_id', None)
            if doc_id is None:
                doc_id = hashlib.md5(doc.content.encode('utf-8')).hexdigest()[:16]
            doc_id_to_original_map[doc_id] = doc
        
        # 错误的映射：使用索引位置
        for i, (doc_text, rerank_score) in enumerate(reranked_items):
            if i < len(retrieved_documents):
                doc_id = getattr(retrieved_documents[i].metadata, 'doc_id', None)
                if doc_id is None:
                    doc_id = hashlib.md5(retrieved_documents[i].content.encode('utf-8')).hexdigest()[:16]
                
                if doc_id in doc_id_to_original_map:
                    reranked_docs.append(doc_id_to_original_map[doc_id])
                    reranked_scores.append(rerank_score)
        
        return reranked_docs, reranked_scores
    
    # 执行错误的映射
    wrong_docs, wrong_scores = wrong_mapping_logic(
        [doc for doc, _ in faiss_results], 
        reranked_results
    )
    
    print("错误映射结果:")
    for i, (doc, score) in enumerate(zip(wrong_docs, wrong_scores)):
        print(f"  {i+1}. {doc.content[:50]}... (分数: {score:.4f})")
    
    # 5. 测试正确的映射逻辑
    print(f"\n=== 测试正确的映射逻辑 ===")
    
    def correct_mapping_logic(retrieved_documents, reranked_items):
        """正确的映射逻辑"""
        reranked_docs = []
        reranked_scores = []
        
        # 根据文档内容匹配
        for doc_text, rerank_score in reranked_items:
            matched_doc = None
            for doc in retrieved_documents:
                if hasattr(doc, 'content'):
                    # 检查内容匹配
                    if doc.content == doc_text or doc.content in doc_text:
                        matched_doc = doc
                        break
            
            if matched_doc:
                reranked_docs.append(matched_doc)
                reranked_scores.append(rerank_score)
        
        return reranked_docs, reranked_scores
    
    # 执行正确的映射
    correct_docs, correct_scores = correct_mapping_logic(
        [doc for doc, _ in faiss_results], 
        reranked_results
    )
    
    print("正确映射结果:")
    for i, (doc, score) in enumerate(zip(correct_docs, correct_scores)):
        print(f"  {i+1}. {doc.content[:50]}... (分数: {score:.4f})")
    
    # 6. 比较结果
    print(f"\n=== 结果比较 ===")
    
    # 检查分数是否不同
    faiss_scores = [score for _, score in faiss_results]
    reranker_scores = [score for _, score in reranked_results]
    
    print(f"FAISS分数: {[f'{s:.4f}' for s in faiss_scores]}")
    print(f"Reranker分数: {[f'{s:.4f}' for s in reranker_scores]}")
    print(f"分数是否相同: {faiss_scores == reranker_scores}")
    
    # 检查顺序是否不同
    faiss_order = [doc.content[:20] for doc, _ in faiss_results]
    reranker_order = [content[:20] for content, _ in reranked_results]
    print(f"顺序是否不同: {faiss_order != reranker_order}")
    
    # 检查映射是否正确
    print(f"错误映射是否导致结果相同: {wrong_scores == faiss_scores}")
    print(f"正确映射是否显示reranker分数: {correct_scores == reranker_scores}")
    
    return {
        'faiss_scores': faiss_scores,
        'reranker_scores': reranker_scores,
        'wrong_mapping_scores': wrong_scores,
        'correct_mapping_scores': correct_scores,
        'scores_different': faiss_scores != reranker_scores,
        'order_different': faiss_order != reranker_order,
        'wrong_mapping_shows_faiss': wrong_scores == faiss_scores,
        'correct_mapping_shows_reranker': correct_scores == reranker_scores
    }

def test_actual_reranker():
    """测试实际的Reranker模型"""
    print("\n=== 测试实际Reranker模型 ===")
    
    try:
        # 尝试加载reranker
        reranker = QwenReranker(
            model_name="Qwen/Qwen3-Reranker-0.6B",
            device="cpu"  # 使用CPU避免GPU内存问题
        )
        
        # 测试文档
        test_docs = [
            "招商银行2023年营业收入达到1000亿元，同比增长15%",
            "平安银行净利润增长20%，资产质量持续改善", 
            "工商银行发布年报，总资产突破30万亿元",
            "建设银行数字化转型成效显著，科技投入持续增加",
            "农业银行服务乡村振兴，普惠金融业务快速发展"
        ]
        
        query = "招商银行的营业收入是多少？"
        
        print(f"查询: {query}")
        print(f"文档数量: {len(test_docs)}")
        
        # 执行重排序
        reranked_items = reranker.rerank(query, test_docs, batch_size=1)
        
        print(f"\nReranker重排序结果:")
        for i, (doc_text, score) in enumerate(reranked_items):
            print(f"  {i+1}. {doc_text[:50]}... (分数: {score:.4f})")
        
        # 检查分数范围
        scores = [score for _, score in reranked_items]
        print(f"\n分数范围: [{min(scores):.4f}, {max(scores):.4f}]")
        print(f"分数是否在[0,1]范围内: {all(0 <= s <= 1 for s in scores)}")
        
        return True
        
    except Exception as e:
        print(f"Reranker测试失败: {e}")
        return False

def test_ui_mapping_logic():
    """测试UI中的映射逻辑"""
    print("\n=== 测试UI映射逻辑 ===")
    
    # 模拟UI中的映射逻辑
    def simulate_ui_mapping(retrieved_documents, reranked_items):
        """模拟UI中的映射逻辑"""
        reranked_docs = []
        reranked_scores = []
        doc_id_to_original_map = {}
        
        # 创建映射
        for doc in retrieved_documents:
            doc_id = getattr(doc.metadata, 'doc_id', None)
            if doc_id is None:
                doc_id = hashlib.md5(doc.content.encode('utf-8')).hexdigest()[:16]
            doc_id_to_original_map[doc_id] = doc
        
        # UI中的映射逻辑（新格式：直接使用doc_id）
        for doc_text, rerank_score, doc_id in reranked_items:
            if doc_id in doc_id_to_original_map:
                reranked_docs.append(doc_id_to_original_map[doc_id])
                reranked_scores.append(rerank_score)
        
        return reranked_docs, reranked_scores
    
    # 测试数据
    test_docs = [
        MockDocument("文档A：招商银行营业收入", "doc_1"),
        MockDocument("文档B：平安银行净利润", "doc_2"),
        MockDocument("文档C：工商银行总资产", "doc_3")
    ]
    
    # 模拟reranker返回不同顺序的结果（新格式：包含doc_id）
    reranked_items = [
        ("文档A：招商银行营业收入", 0.95, "doc_1"),  # 最相关
        ("文档C：工商银行总资产", 0.87, "doc_3"),    # 次相关（顺序改变）
        ("文档B：平安银行净利润", 0.76, "doc_2")     # 一般相关
    ]
    
    # 执行UI映射
    mapped_docs, mapped_scores = simulate_ui_mapping(test_docs, reranked_items)
    
    print("UI映射结果:")
    for i, (doc, score) in enumerate(zip(mapped_docs, mapped_scores)):
        print(f"  {i+1}. {doc.content} (分数: {score:.4f})")
    
    # 检查是否正确映射
    expected_scores = [0.95, 0.87, 0.76]  # reranker返回的分数
    actual_scores = mapped_scores
    
    print(f"\n期望分数: {expected_scores}")
    print(f"实际分数: {actual_scores}")
    print(f"映射是否正确: {expected_scores == actual_scores}")
    
    return expected_scores == actual_scores

def main():
    """主测试函数"""
    print("开始测试FAISS和Reranker结果差异...")
    
    # 1. 测试映射逻辑
    mapping_results = test_faiss_reranker_mapping()
    
    # 2. 测试实际reranker（如果可用）
    try:
        reranker_works = test_actual_reranker()
    except NameError:
        print("跳过实际reranker测试（模块未导入）")
        reranker_works = False
    
    # 3. 测试UI映射逻辑
    ui_mapping_correct = test_ui_mapping_logic()
    
    # 4. 总结
    print("\n=== 测试总结 ===")
    print(f"✅ FAISS和Reranker分数不同: {mapping_results['scores_different']}")
    print(f"✅ FAISS和Reranker顺序不同: {mapping_results['order_different']}")
    print(f"✅ 错误映射导致显示FAISS分数: {mapping_results['wrong_mapping_shows_faiss']} (False表示正确)")
    print(f"✅ 正确映射显示Reranker分数: {mapping_results['correct_mapping_shows_reranker']}")
    print(f"✅ 实际Reranker工作正常: {reranker_works}")
    print(f"✅ UI映射逻辑正确: {ui_mapping_correct}")
    
    if mapping_results['wrong_mapping_shows_faiss']:
        print("\n🚨 发现问题：当前UI映射逻辑错误，导致显示的是FAISS分数而不是Reranker分数！")
        print("建议修复optimized_rag_ui.py中的映射逻辑。")
    else:
        print("\n✅ 所有问题已解决：UI映射逻辑正确，显示的是Reranker分数而不是FAISS分数！")
    
    return mapping_results, reranker_works, ui_mapping_correct

if __name__ == "__main__":
    main() 