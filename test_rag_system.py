#!/usr/bin/env python3
"""
简单的RAG系统测试脚本
用于验证RAG系统是否正常工作，包括预过滤和映射功能
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from alphafin_data_process.rag_system_adapter import RagSystemAdapter
from config.parameters import Config

def test_rag_system():
    """测试RAG系统的基本功能"""
    print("=" * 60)
    print("开始测试RAG系统...")
    print("=" * 60)
    
    try:
        # 1. 初始化RAG系统适配器
        print("1. 初始化RAG系统适配器...")
        adapter = RagSystemAdapter()
        print("✅ RAG系统适配器初始化成功")
        
        # 2. 测试配置文件读取
        print("\n2. 测试配置文件读取...")
        if adapter.ui and adapter.ui.config:
            use_prefilter = adapter.ui.config.retriever.use_prefilter
            print(f"✅ 配置文件读取成功，use_prefilter: {use_prefilter}")
        else:
            print("❌ 配置文件读取失败")
            return False
        
        # 3. 测试股票代码和公司名称映射加载
        print("\n3. 测试股票代码和公司名称映射加载...")
        if adapter.ui and adapter.ui.chinese_retrieval_system:
            mapping_count = len(adapter.ui.chinese_retrieval_system.stock_company_mapping)
            print(f"✅ 股票代码映射加载成功，数量: {mapping_count}")
        else:
            print("❌ 股票代码映射加载失败")
            return False
        
        # 4. 测试中文查询检索（baseline模式）
        print("\n4. 测试中文查询检索（baseline模式）...")
        test_query = "宝莱特2023年营业收入是多少？"
        results = adapter.get_ranked_documents_for_evaluation(
            query=test_query,
            top_k=3,
            mode="baseline"
        )
        print(f"✅ baseline模式检索成功，返回 {len(results)} 个文档")
        
        # 5. 测试中文查询检索（prefilter模式）
        print("\n5. 测试中文查询检索（prefilter模式）...")
        results = adapter.get_ranked_documents_for_evaluation(
            query=test_query,
            top_k=3,
            mode="prefilter"
        )
        print(f"✅ prefilter模式检索成功，返回 {len(results)} 个文档")
        
        # 6. 测试英文查询检索
        print("\n6. 测试英文查询检索...")
        test_query_en = "What is the revenue of Apple in 2023?"
        results = adapter.get_ranked_documents_for_evaluation(
            query=test_query_en,
            top_k=3,
            mode="baseline"
        )
        print(f"✅ 英文查询检索成功，返回 {len(results)} 个文档")
        
        print("\n" + "=" * 60)
        print("🎉 所有测试通过！RAG系统工作正常")
        print("=" * 60)
        return True
        
    except Exception as e:
        print(f"\n❌ 测试失败: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_config_parameters():
    """测试配置参数"""
    print("\n" + "=" * 60)
    print("测试配置参数...")
    print("=" * 60)
    
    try:
        config = Config()
        print(f"✅ 配置加载成功")
        print(f"   use_prefilter: {config.retriever.use_prefilter}")
        print(f"   retrieval_top_k: {config.retriever.retrieval_top_k}")
        print(f"   rerank_top_k: {config.retriever.rerank_top_k}")
        return True
    except Exception as e:
        print(f"❌ 配置测试失败: {e}")
        return False

if __name__ == "__main__":
    print("RAG系统测试脚本")
    print("测试内容：")
    print("1. 配置文件读取")
    print("2. RAG系统初始化")
    print("3. 股票代码和公司名称映射加载")
    print("4. 中文查询检索（baseline模式）")
    print("5. 中文查询检索（prefilter模式）")
    print("6. 英文查询检索")
    print()
    
    # 测试配置参数
    config_ok = test_config_parameters()
    
    # 测试RAG系统
    rag_ok = test_rag_system()
    
    if config_ok and rag_ok:
        print("\n🎉 所有测试通过！系统可以正常使用")
        sys.exit(0)
    else:
        print("\n❌ 测试失败，请检查系统配置")
        sys.exit(1) 