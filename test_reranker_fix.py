#!/usr/bin/env python3
"""
测试Reranker修复的脚本
验证重排序是否真正生效
"""

import sys
import json
from pathlib import Path

# 添加项目路径
sys.path.append(str(Path(__file__).parent))

from xlm.components.retriever.reranker import QwenReranker

def test_reranker_fix():
    """测试Reranker修复是否有效"""
    
    print("=== 测试Reranker修复 ===")
    
    # 模拟测试数据
    query = "What is the revenue growth rate?"
    
    # 创建测试文档，故意让相关性不明显的文档排在前面
    documents_with_metadata = [
        {
            'content': 'The company reported quarterly earnings.',
            'metadata': {'doc_id': 'doc1', 'source': 'earnings_report'}
        },
        {
            'content': 'Revenue growth rate increased by 15% year-over-year.',
            'metadata': {'doc_id': 'doc2', 'source': 'financial_report'}
        },
        {
            'content': 'The stock price rose by 5%.',
            'metadata': {'doc_id': 'doc3', 'source': 'market_data'}
        },
        {
            'content': 'Revenue growth was driven by strong sales performance.',
            'metadata': {'doc_id': 'doc4', 'source': 'sales_report'}
        }
    ]
    
    try:
        # 初始化Reranker
        print("1. 初始化Reranker...")
        reranker = QwenReranker(
            model_name="Qwen/Qwen3-Reranker-0.6B",
            device="cuda:0" if torch.cuda.is_available() else "cpu",
            use_quantization=True
        )
        print("✅ Reranker初始化成功")
        
        # 执行重排序
        print("2. 执行重排序...")
        reranked_results = reranker.rerank_with_metadata(
            query=query,
            documents_with_metadata=documents_with_metadata,
            batch_size=2
        )
        
        print("3. 分析重排序结果...")
        print(f"查询: {query}")
        print("\n重排序结果:")
        
        for i, result in enumerate(reranked_results, 1):
            content = result.get('content', '')
            score = result.get('reranker_score', 0.0)
            doc_id = result.get('metadata', {}).get('doc_id', 'unknown')
            
            print(f"  {i}. Doc {doc_id}: {content[:50]}... (Score: {score:.4f})")
        
        # 检查重排序是否生效
        first_result = reranked_results[0] if reranked_results else {}
        first_score = first_result.get('reranker_score', 0.0)
        first_content = first_result.get('content', '')
        
        # 检查是否包含关键词的文档排在前面
        if 'revenue' in first_content.lower() or 'growth' in first_content.lower():
            print("\n✅ 重排序生效：相关文档排在前面")
            return True
        else:
            print("\n❌ 重排序可能未生效：相关文档未排在前面")
            return False
            
    except Exception as e:
        print(f"❌ 测试失败: {e}")
        return False

if __name__ == "__main__":
    import torch
    success = test_reranker_fix()
    if success:
        print("\n🎉 Reranker修复验证成功！")
    else:
        print("\n⚠️ Reranker修复验证失败，需要进一步检查") 