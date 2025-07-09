#!/usr/bin/env python3
"""
测试新的reranker_no_prefilter模式
"""

import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent))

from alphafin_data_process.rag_system_adapter import RagSystemAdapter
from config.parameters import config

def test_reranker_no_prefilter_mode():
    """测试新的reranker_no_prefilter模式"""
    print("=" * 60)
    print("测试新的reranker_no_prefilter模式")
    print("=" * 60)
    
    try:
        # 初始化RAG系统适配器
        print("初始化RAG系统适配器...")
        adapter = RagSystemAdapter(config=config)
        print("✅ RAG系统适配器初始化成功")
        
        # 测试查询
        test_query = "平安银行2023年净利润是多少？"
        print(f"\n测试查询: {test_query}")
        
        # 测试不同模式
        modes = ['baseline', 'prefilter', 'reranker', 'reranker_no_prefilter']
        
        for mode in modes:
            print(f"\n--- 测试模式: {mode} ---")
            try:
                results = adapter.get_ranked_documents_for_evaluation(
                    query=test_query,
                    top_k=5,
                    mode=mode
                )
                
                print(f"✅ {mode}模式测试成功")
                print(f"   返回文档数量: {len(results)}")
                
                if results:
                    print("   前3个文档:")
                    for i, doc in enumerate(results[:3]):
                        doc_id = doc.get('doc_id', 'unknown')
                        content = doc.get('content', '')
                        content_preview = content[:100] + "..." if len(content) > 100 else content
                        print(f"     {i+1}. doc_id: {doc_id}")
                        print(f"        content: {content_preview}")
                
            except Exception as e:
                print(f"❌ {mode}模式测试失败: {e}")
        
        print(f"\n✅ 所有模式测试完成")
        
    except Exception as e:
        print(f"❌ 测试失败: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    test_reranker_no_prefilter_mode() 