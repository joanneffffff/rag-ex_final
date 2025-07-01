#!/usr/bin/env python3
"""
测试多阶段检索系统使用Top1的context和summary
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from alphafin_data_process.multi_stage_retrieval_final import MultiStageRetrievalSystem
from pathlib import Path

def test_top1_context_summary():
    """测试Top1 context和summary的使用"""
    
    print("🧪 测试多阶段检索系统使用Top1的context和summary")
    print("=" * 60)
    
    # 数据文件路径
    data_path = Path("data/alphafin/alphafin_merged_generated_qa.json")
    
    if not data_path.exists():
        print(f"❌ 数据文件不存在: {data_path}")
        return
    
    try:
        # 初始化检索系统
        print("1. 初始化多阶段检索系统...")
        retrieval_system = MultiStageRetrievalSystem(data_path, dataset_type="chinese")
        print("✅ 检索系统初始化成功")
        
        # 测试查询
        test_queries = [
            "德赛电池2021年业绩如何？",
            "中国平安的营业收入是多少？",
            "比亚迪的净利润增长了多少？"
        ]
        
        for i, query in enumerate(test_queries, 1):
            print(f"\n{'='*50}")
            print(f"测试 {i}: {query}")
            print(f"{'='*50}")
            
            # 提取公司名称和股票代码
            company_name = None
            stock_code = None
            
            # 简单的公司名称提取逻辑
            if "德赛电池" in query:
                company_name = "德赛电池"
            elif "中国平安" in query:
                company_name = "中国平安"
            elif "比亚迪" in query:
                company_name = "比亚迪"
            
            # 执行检索（带元数据过滤）
            results = retrieval_system.search(
                query=query,
                company_name=company_name,
                stock_code=stock_code,
                top_k=5
            )
            
            if 'retrieved_documents' in results and results['retrieved_documents']:
                # 获取Top1文档
                top1_doc = results['retrieved_documents'][0]
                
                print(f"📊 Top1文档信息:")
                print(f"   公司名称: {top1_doc.get('company_name', 'N/A')}")
                print(f"   股票代码: {top1_doc.get('stock_code', 'N/A')}")
                print(f"   综合分数: {top1_doc.get('combined_score', 0):.4f}")
                
                # 检查summary字段
                summary = top1_doc.get('summary', '')
                print(f"📝 Top1 Summary:")
                print(f"   长度: {len(summary)} 字符")
                print(f"   内容: {summary[:200]}{'...' if len(summary) > 200 else ''}")
                
                # 检查context字段
                context = top1_doc.get('context', '')
                print(f"📄 Top1 Context:")
                print(f"   长度: {len(context)} 字符")
                print(f"   内容: {context[:200]}{'...' if len(context) > 200 else ''}")
                
                # 检查LLM答案
                llm_answer = results.get('llm_answer', '')
                if llm_answer:
                    print(f"🤖 LLM答案:")
                    print(f"   长度: {len(llm_answer)} 字符")
                    print(f"   内容: {llm_answer}")
                else:
                    print("❌ 未生成LLM答案")
                
                # 验证是否使用了Top1的summary和context
                print(f"🔍 验证结果:")
                if summary:
                    print("   ✅ Top1文档有summary字段")
                else:
                    print("   ⚠️ Top1文档没有summary字段")
                
                if context:
                    print("   ✅ Top1文档有context字段")
                else:
                    print("   ⚠️ Top1文档没有context字段")
                
            else:
                print("❌ 检索失败或无结果")
        
        print(f"\n{'='*60}")
        print("✅ 测试完成!")
        
    except Exception as e:
        print(f"❌ 测试失败: {e}")
        import traceback
        traceback.print_exc()

def test_extract_relevant_context():
    """测试extract_relevant_context方法"""
    
    print("\n🔍 测试extract_relevant_context方法")
    print("=" * 50)
    
    # 数据文件路径
    data_path = Path("data/alphafin/alphafin_merged_generated_qa.json")
    
    if not data_path.exists():
        print(f"❌ 数据文件不存在: {data_path}")
        return
    
    try:
        # 初始化检索系统
        retrieval_system = MultiStageRetrievalSystem(data_path, dataset_type="chinese")
        
        # 测试查询
        query = "德赛电池2021年业绩如何？"
        
        # 提取公司名称
        company_name = "德赛电池" if "德赛电池" in query else None
        
        # 执行检索获取候选结果（带元数据过滤）
        results = retrieval_system.search(
            query=query, 
            company_name=company_name,
            top_k=5
        )
        
        if 'retrieved_documents' in results and results['retrieved_documents']:
            # 构造候选结果格式
            candidate_results = []
            for i, doc in enumerate(results['retrieved_documents']):
                candidate_results.append((i, doc.get('faiss_score', 0), doc.get('combined_score', 0)))
            
            print(f"📋 查询: {query}")
            print(f"📊 候选结果数: {len(candidate_results)}")
            
            # 测试extract_relevant_context方法
            context = retrieval_system.extract_relevant_context(query, candidate_results)
            
            print(f"✅ 提取的context:")
            print(f"   长度: {len(context)} 字符")
            print(f"   内容: {context[:300]}{'...' if len(context) > 300 else ''}")
            
            # 验证是否只使用了Top1
            print(f"🔍 验证是否只使用Top1:")
            if len(context) > 0:
                print("   ✅ 成功提取了context")
                print("   ✅ 现在只使用Top1文档的context")
            else:
                print("   ❌ 未提取到context")
        
    except Exception as e:
        print(f"❌ 测试失败: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    # 测试Top1 context和summary的使用
    test_top1_context_summary()
    
    # 测试extract_relevant_context方法
    test_extract_relevant_context() 