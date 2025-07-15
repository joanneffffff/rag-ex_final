#!/usr/bin/env python3
"""
测试RAG系统适配器的使用
"""

import sys
import os
import json
from typing import Dict, List, Any

sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from alphafin_data_process.rag_system_adapter import RagSystemAdapter
from config.parameters import Config

def test_rag_adapter():
    """测试RAG系统适配器"""
    print("🧪 测试RAG系统适配器")
    print("=" * 50)
    
    # 初始化配置
    config = Config()
    print(f"📋 使用配置: {config.generator.model_name}")
    
    # 初始化RAG系统适配器
    print("🔧 初始化RAG系统适配器...")
    rag_adapter = RagSystemAdapter(config=config)
    
    # 测试查询
    test_query = "什么是股票投资？"
    print(f"\n🔍 测试查询: {test_query}")
    
    try:
        # 获取检索结果
        results = rag_adapter.get_ranked_documents_for_evaluation(
            query=test_query,
            top_k=5,
            mode="baseline"
        )
        
        print(f"✅ 检索成功，获得 {len(results)} 个文档")
        
        # 显示前3个结果
        for i, result in enumerate(results[:3]):
            print(f"\n📄 文档 {i+1}:")
            print(f"  ID: {result.get('doc_id', 'N/A')}")
            print(f"  分数: {result.get('faiss_score', 'N/A')}")
            print(f"  内容: {result.get('content', 'N/A')[:100]}...")
            
    except Exception as e:
        print(f"❌ 测试失败: {e}")
        import traceback
        traceback.print_exc()

def test_with_sample_data():
    """使用样本数据测试"""
    print("\n🧪 使用样本数据测试")
    print("=" * 50)
    
    # 加载样本数据
    sample_data_path = "data/alphafin/alphafin_eval_samples_updated.jsonl"
    
    if not os.path.exists(sample_data_path):
        print(f"❌ 样本数据文件不存在: {sample_data_path}")
        return
    
    # 加载第一个样本
    with open(sample_data_path, 'r', encoding='utf-8') as f:
        first_line = f.readline().strip()
        sample = json.loads(first_line)
    
    print(f"📊 样本数据: {sample.get('question', 'N/A')}")
    
    # 初始化RAG系统适配器
    config = Config()
    rag_adapter = RagSystemAdapter(config=config)
    
    try:
        # 测试检索
        results = rag_adapter.get_ranked_documents_for_evaluation(
            query=sample['question'],
            top_k=3,
            mode="baseline"
        )
        
        print(f"✅ 样本检索成功，获得 {len(results)} 个文档")
        
        # 显示结果
        for i, result in enumerate(results):
            print(f"\n📄 结果 {i+1}:")
            print(f"  ID: {result.get('doc_id', 'N/A')}")
            print(f"  分数: {result.get('faiss_score', 'N/A')}")
            print(f"  内容: {result.get('content', 'N/A')[:150]}...")
            
    except Exception as e:
        print(f"❌ 样本测试失败: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    test_rag_adapter()
    test_with_sample_data() 