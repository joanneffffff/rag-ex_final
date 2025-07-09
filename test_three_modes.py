#!/usr/bin/env python3
"""
测试三个模式：baseline、reranker、reranker_no_prefilter
"""

import sys
import os
from pathlib import Path
sys.path.append(str(Path(__file__).parent))

from config.parameters import config
from alphafin_data_process.rag_system_adapter import RagSystemAdapter

def test_three_modes():
    """测试三个模式"""
    print("=" * 80)
    print("测试三个模式：baseline、reranker、reranker_no_prefilter")
    print("=" * 80)
    
    # 初始化RAG系统
    adapter = RagSystemAdapter(config=config)
    
    # 测试查询
    test_query = "招商银行的营业收入是多少？"
    
    print(f"测试查询: {test_query}")
    
    # 测试三个模式
    modes = ["baseline", "reranker", "reranker_no_prefilter"]
    results = {}
    
    for mode in modes:
        print(f"\n{'='*50}")
        print(f"测试模式: {mode}")
        print(f"{'='*50}")
        
        try:
            mode_results = adapter.get_ranked_documents_for_evaluation(
                query=test_query,
                top_k=5,
                mode=mode,
                use_prefilter=None  # 让系统自动决定
            )
            
            results[mode] = mode_results
            print(f"✅ {mode}模式返回 {len(mode_results)} 个文档")
            
            # 显示前3个文档
            for i, doc in enumerate(mode_results[:3]):
                doc_id = doc.get('doc_id', 'N/A')
                content = doc.get('content', '')[:100]
                print(f"  {i+1}. {doc_id} - {content}...")
                
        except Exception as e:
            print(f"❌ {mode}模式失败: {e}")
            results[mode] = []
    
    # 比较结果
    print(f"\n{'='*80}")
    print("结果比较")
    print(f"{'='*80}")
    
    for mode in modes:
        if mode in results and results[mode]:
            doc_ids = [doc.get('doc_id') for doc in results[mode][:3]]
            print(f"{mode}前3个doc_id: {doc_ids}")
        else:
            print(f"{mode}: 无结果")
    
    # 检查差异
    print(f"\n{'='*80}")
    print("差异分析")
    print(f"{'='*80}")
    
    baseline_ids = [doc.get('doc_id') for doc in results.get('baseline', [])[:3]]
    reranker_ids = [doc.get('doc_id') for doc in results.get('reranker', [])[:3]]
    reranker_no_prefilter_ids = [doc.get('doc_id') for doc in results.get('reranker_no_prefilter', [])[:3]]
    
    print(f"Baseline vs Reranker: {'相同' if baseline_ids == reranker_ids else '不同'}")
    print(f"Baseline vs Reranker_no_prefilter: {'相同' if baseline_ids == reranker_no_prefilter_ids else '不同'}")
    print(f"Reranker vs Reranker_no_prefilter: {'相同' if reranker_ids == reranker_no_prefilter_ids else '不同'}")
    
    # 分析重排序器效果
    if baseline_ids != reranker_ids:
        print("✅ 重排序器正在工作（reranker模式改变了文档顺序）")
    else:
        print("⚠️  重排序器可能没有工作（reranker模式与baseline相同）")
    
    # 分析预过滤效果
    if reranker_ids != reranker_no_prefilter_ids:
        print("✅ 预过滤正在工作（reranker_no_prefilter模式改变了文档顺序）")
    else:
        print("⚠️  预过滤可能没有工作（reranker_no_prefilter模式与reranker相同）")

if __name__ == "__main__":
    test_three_modes() 