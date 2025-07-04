#!/usr/bin/env python3
"""
测试doc_id映射和重排序流程
"""
import sys
import os
sys.path.append(os.path.dirname(__file__))
from xlm.ui.optimized_rag_ui import OptimizedRagUI
import asyncio

def print_doc_info(unique_docs):
    for i, (doc, score) in enumerate(unique_docs[:5]):
        doc_id = getattr(doc.metadata, 'doc_id', None)
        if doc_id is None:
            import hashlib
            doc_id = hashlib.md5(doc.content.encode('utf-8')).hexdigest()[:16]
        print(f"文档 {i+1}: doc_id = {doc_id}, 分数 = {score:.4f}")
        print(f"  内容长度: {len(doc.content)}")
        if hasattr(doc.metadata, 'summary'):
            print(f"  摘要长度: {len(doc.metadata.summary)}")
        print()

async def test_unified_rag_processing():
    print("=== 测试_统一RAG流程_和doc_id映射 ===")
    ui = OptimizedRagUI()
    # 中文测试
    question_zh = "什么是人工智能？"
    print("\n1. 中文查询测试：")
    answer, html_content = ui._unified_rag_processing(question_zh, 'zh', True)
    print(f"生成答案: {answer}")
    print(f"HTML内容预览: {html_content[:200]}...")
    # 英文测试
    question_en = "What is artificial intelligence?"
    print("\n2. 英文查询测试：")
    answer, html_content = ui._unified_rag_processing(question_en, 'en', True)
    print(f"生成答案: {answer}")
    print(f"HTML内容预览: {html_content[:200]}...")

if __name__ == "__main__":
    asyncio.run(test_unified_rag_processing()) 