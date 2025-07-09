#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import sys
import os
# 添加项目根目录到Python路径
project_root = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, project_root)

from alphafin_data_process.rag_system_adapter import RagSystemAdapter
from config.parameters import Config

def test_reranker_mapping():
    """测试重排序器映射逻辑"""
    print("开始测试重排序器映射逻辑...")
    
    # 初始化适配器
    config = Config()
    adapter = RagSystemAdapter(config)
    
    # 测试查询
    test_query = "腾讯公司的营业收入是多少？"
    
    print(f"测试查询: {test_query}")
    
    # 测试reranker模式
    print("\n=== 测试reranker模式 ===")
    try:
        results = adapter.get_ranked_documents_for_evaluation(
            query=test_query,
            top_k=5,
            mode="reranker"
        )
        
        print(f"reranker模式返回 {len(results)} 个结果")
        for i, result in enumerate(results[:3]):
            print(f"  {i+1}. doc_id: {result['doc_id']}, score: {result['faiss_score']:.4f}")
            
    except Exception as e:
        print(f"reranker模式测试失败: {e}")
    
    # 测试reranker_no_prefilter模式
    print("\n=== 测试reranker_no_prefilter模式 ===")
    try:
        results = adapter.get_ranked_documents_for_evaluation(
            query=test_query,
            top_k=5,
            mode="reranker_no_prefilter"
        )
        
        print(f"reranker_no_prefilter模式返回 {len(results)} 个结果")
        for i, result in enumerate(results[:3]):
            print(f"  {i+1}. doc_id: {result['doc_id']}, score: {result['faiss_score']:.4f}")
            
    except Exception as e:
        print(f"reranker_no_prefilter模式测试失败: {e}")

if __name__ == "__main__":
    test_reranker_mapping() 