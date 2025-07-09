#!/usr/bin/env python3
"""
测试重排序器是否真的在工作
"""

import sys
import os
from pathlib import Path
sys.path.append(str(Path(__file__).parent))

from config.parameters import config
from alphafin_data_process.rag_system_adapter import RagSystemAdapter

def test_reranker():
    """测试重排序器是否工作"""
    print("=" * 80)
    print("测试重排序器是否工作")
    print("=" * 80)
    
    # 初始化RAG系统
    adapter = RagSystemAdapter(config=config)
    
    # 测试查询
    test_query = "招商银行的营业收入是多少？"
    
    print(f"测试查询: {test_query}")
    
    # 测试baseline模式（无重排序）
    print("\n1. 测试baseline模式（无重排序）:")
    baseline_results = adapter.get_ranked_documents_for_evaluation(
        query=test_query,
        top_k=5,
        mode="baseline",
        use_prefilter=True
    )
    
    print(f"Baseline模式返回 {len(baseline_results)} 个文档")
    for i, doc in enumerate(baseline_results[:3]):
        print(f"  {i+1}. {doc.get('doc_id', 'N/A')} - {doc.get('content', '')[:100]}...")
    
    # 测试reranker模式（有重排序）
    print("\n2. 测试reranker模式（有重排序）:")
    reranker_results = adapter.get_ranked_documents_for_evaluation(
        query=test_query,
        top_k=5,
        mode="reranker",
        use_prefilter=True
    )
    
    print(f"Reranker模式返回 {len(reranker_results)} 个文档")
    for i, doc in enumerate(reranker_results[:3]):
        print(f"  {i+1}. {doc.get('doc_id', 'N/A')} - {doc.get('content', '')[:100]}...")
    
    # 比较结果
    print("\n3. 结果比较:")
    baseline_doc_ids = [doc.get('doc_id') for doc in baseline_results[:3]]
    reranker_doc_ids = [doc.get('doc_id') for doc in reranker_results[:3]]
    
    print(f"Baseline前3个doc_id: {baseline_doc_ids}")
    print(f"Reranker前3个doc_id: {reranker_doc_ids}")
    
    # 检查是否有差异
    if baseline_doc_ids == reranker_doc_ids:
        print("⚠️  警告: Baseline和Reranker模式返回相同的文档顺序")
        print("这可能意味着重排序器没有真正工作")
    else:
        print("✅ 重排序器正在工作，文档顺序发生了变化")
    
    # 检查重排序器是否被调用
    print("\n4. 检查重排序器调用:")
    if hasattr(adapter, 'ui') and adapter.ui and hasattr(adapter.ui, 'reranker'):
        print("✅ 重排序器已加载")
        print(f"重排序器类型: {type(adapter.ui.reranker)}")
    else:
        print("❌ 重排序器未加载")

if __name__ == "__main__":
    test_reranker() 