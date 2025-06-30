#!/usr/bin/env python3
"""
测试优化后的上下文提取效果
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from alphafin_data_process.multi_stage_retrieval_final import MultiStageRetrievalSystem
from pathlib import Path

def test_context_extraction():
    """测试上下文提取优化效果"""
    
    print("🧪 测试优化后的上下文提取效果")
    print("=" * 60)
    
    # 初始化多阶段检索系统
    data_path = Path("data/alphafin/alphafin_merged_generated_qa.json")
    if not data_path.exists():
        print(f"❌ 数据文件不存在: {data_path}")
        return
    
    print(f"📁 加载数据: {data_path}")
    retrieval_system = MultiStageRetrievalSystem(
        data_path=data_path,
        dataset_type="chinese",
        use_existing_config=True
    )
    
    # 测试查询
    test_queries = [
        "德赛电池（000049）2021年利润持续增长的主要原因是什么？",
        "德赛电池(000049)的下一季度收益预测如何？"
    ]
    
    for query in test_queries:
        print(f"\n🔍 测试查询: {query}")
        print("-" * 40)
        
        try:
            # 提取股票代码和公司名称
            import re
            stock_code = None
            company_name = None
            
            # 提取股票代码
            stock_match = re.search(r'(\d{6})', query)
            if stock_match:
                stock_code = stock_match.group(1)
            
            # 提取公司名称
            company_match = re.search(r'([^（(]+)（', query)
            if company_match:
                company_name = company_match.group(1).strip()
            
            print(f"🔍 提取的元数据:")
            print(f"   公司名称: {company_name}")
            print(f"   股票代码: {stock_code}")
            
            # 执行检索，使用元数据过滤
            results = retrieval_system.search(
                query=query,
                company_name=company_name,
                stock_code=stock_code,
                top_k=10
            )
            
            if 'retrieved_documents' in results:
                print(f"✅ 检索成功，获得 {len(results['retrieved_documents'])} 个文档")
                
                # 显示前3个文档的摘要
                print("\n📄 前3个文档摘要:")
                for i, doc in enumerate(results['retrieved_documents'][:3]):
                    print(f"文档 {i+1} (分数: {doc.get('combined_score', 0):.4f}):")
                    context = doc.get('context', '')[:200] + '...' if len(doc.get('context', '')) > 200 else doc.get('context', '')
                    print(f"  {context}")
                
                # 显示LLM答案
                if 'llm_answer' in results:
                    print(f"\n🤖 LLM答案:")
                    print(f"  {results['llm_answer'][:300]}...")
                
            else:
                print("❌ 检索失败")
                
        except Exception as e:
            print(f"❌ 测试失败: {e}")
            import traceback
            traceback.print_exc()

if __name__ == "__main__":
    test_context_extraction() 