#!/usr/bin/env python3
"""
测试重排序器修复是否有效
"""

import sys
import os
from pathlib import Path
sys.path.append(str(Path(__file__).parent))

from config.parameters import config
from alphafin_data_process.rag_system_adapter import RagSystemAdapter

def test_reranker_fix():
    """测试重排序器修复"""
    print("=" * 80)
    print("测试重排序器修复是否有效")
    print("=" * 80)
    
    # 初始化RAG系统
    adapter = RagSystemAdapter(config=config)
    
    # 测试查询
    test_query = "招商银行的营业收入是多少？"
    
    print(f"测试查询: {test_query}")
    
    # 测试baseline模式
    print("\n1. 测试baseline模式（无重排序）:")
    baseline_results = adapter.get_ranked_documents_for_evaluation(
        query=test_query,
        top_k=3,
        mode="baseline",
        use_prefilter=True
    )
    
    print(f"Baseline模式返回 {len(baseline_results)} 个文档")
    baseline_doc_ids = []
    for i, doc in enumerate(baseline_results[:3]):
        doc_id = doc.get('doc_id', 'N/A')
        baseline_doc_ids.append(doc_id)
        print(f"  {i+1}. {doc_id}")
    
    # 测试reranker模式
    print("\n2. 测试reranker模式（有重排序）:")
    reranker_results = adapter.get_ranked_documents_for_evaluation(
        query=test_query,
        top_k=3,
        mode="reranker",
        use_prefilter=True
    )
    
    print(f"Reranker模式返回 {len(reranker_results)} 个文档")
    reranker_doc_ids = []
    for i, doc in enumerate(reranker_results[:3]):
        doc_id = doc.get('doc_id', 'N/A')
        reranker_doc_ids.append(doc_id)
        print(f"  {i+1}. {doc_id}")
    
    # 比较结果
    print("\n3. 结果比较:")
    print(f"Baseline前3个doc_id: {baseline_doc_ids}")
    print(f"Reranker前3个doc_id: {reranker_doc_ids}")
    
    if baseline_doc_ids == reranker_doc_ids:
        print("❌ 修复失败：Baseline和Reranker模式返回相同的文档顺序")
        print("重排序器仍然没有工作")
    else:
        print("✅ 修复成功：Baseline和Reranker模式返回不同的文档顺序")
        print("重排序器现在正在工作！")
    
    # 检查调试日志
    print("\n4. 检查调试日志:")
    print("如果看到以下调试信息，说明重排序器正在工作：")
    print("- 'DEBUG: 重排序器返回 X 个结果'")
    print("- 'DEBUG: 找到匹配文档，重排序分数: X.XXXX'")
    print("- 'DEBUG: 重排序后前3个文档:'")

if __name__ == "__main__":
    test_reranker_fix() 