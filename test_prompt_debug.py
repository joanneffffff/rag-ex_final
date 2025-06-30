#!/usr/bin/env python3
"""
测试Prompt调试功能
"""

from alphafin_data_process.multi_stage_retrieval_final import MultiStageRetrievalSystem
from pathlib import Path

def test_prompt_debug():
    """测试Prompt调试功能"""
    print("=== 测试Prompt调试功能 ===\n")
    
    # 1. 初始化多阶段检索系统
    print("1. 初始化多阶段检索系统...")
    try:
        retrieval_system = MultiStageRetrievalSystem(
            data_path=Path("data/alphafin/alphafin_merged_generated_qa.json"),
            dataset_type="chinese"
        )
        print("✅ 多阶段检索系统初始化成功")
    except Exception as e:
        print(f"❌ 多阶段检索系统初始化失败: {e}")
        return
    
    # 2. 测试查询
    print("\n2. 执行测试查询...")
    test_query = "德赛电池（000049）2021年利润持续增长的主要原因是什么？"
    
    try:
        # 执行搜索
        results = retrieval_system.search(
            query=test_query,
            company_name="德赛电池",
            stock_code="000049",
            report_date="2021",
            top_k=5
        )
        
        print("✅ 查询执行成功")
        print(f"返回结果数: {len(results.get('retrieved_documents', []))}")
        print(f"LLM答案长度: {len(results.get('llm_answer', ''))}")
        
        # 显示LLM答案
        llm_answer = results.get('llm_answer', '')
        if llm_answer:
            print(f"\nLLM答案预览: {llm_answer[:200]}...")
        
    except Exception as e:
        print(f"❌ 查询执行失败: {e}")

if __name__ == "__main__":
    test_prompt_debug() 