#!/usr/bin/env python3
"""
测试LLM生成器修复是否有效
验证多阶段检索系统中的LLM生成器是否能正常工作
"""

import sys
import os
from pathlib import Path

# 添加项目根目录到路径
sys.path.append(str(Path(__file__).parent))

def test_llm_generator():
    """测试LLM生成器是否正常工作"""
    print("=== 测试LLM生成器修复 ===")
    
    try:
        # 导入多阶段检索系统
        from alphafin_data_process.multi_stage_retrieval_final import MultiStageRetrievalSystem
        
        # 初始化系统
        data_path = Path("data/alphafin/alphafin_merged_generated_qa.json")
        print(f"正在初始化多阶段检索系统...")
        print(f"数据路径: {data_path}")
        
        retrieval_system = MultiStageRetrievalSystem(
            data_path=data_path,
            dataset_type="chinese",
            use_existing_config=True
        )
        
        # 检查LLM生成器状态
        if retrieval_system.llm_generator:
            print("✅ LLM生成器已成功初始化")
            
            # 测试生成器功能
            test_prompt = "基于以下上下文回答问题：\n\n德赛电池是一家专注于电池制造的公司。\n\n问题：德赛电池的主要业务是什么？\n\n回答："
            
            print("正在测试LLM生成器...")
            try:
                response = retrieval_system.llm_generator.generate(texts=[test_prompt])
                if response and len(response) > 0:
                    print("✅ LLM生成器正常工作")
                    print(f"生成的回答: {response[0][:100]}...")
                else:
                    print("❌ LLM生成器返回空响应")
            except Exception as e:
                print(f"❌ LLM生成器测试失败: {e}")
        else:
            print("❌ LLM生成器未初始化")
            return False
        
        # 测试完整的检索和生成流程
        print("\n=== 测试完整检索和生成流程 ===")
        test_query = "德赛电池（000049）2021年利润持续增长的主要原因是什么？"
        print(f"测试查询: {test_query}")
        
        results = retrieval_system.search(
            query=test_query,
            company_name="德赛电池",
            stock_code="000049",
            report_date="2021",
            top_k=5
        )
        
        # 检查结果
        if isinstance(results, dict) and 'llm_answer' in results:
            llm_answer = results['llm_answer']
            if llm_answer and llm_answer != "未配置LLM生成器。":
                print("✅ 完整流程测试成功")
                print(f"LLM生成的答案: {llm_answer[:200]}...")
                return True
            else:
                print("❌ LLM生成器未生成答案")
                print(f"返回的答案: {llm_answer}")
                return False
        else:
            print("❌ 检索结果格式错误")
            return False
            
    except Exception as e:
        print(f"❌ 测试过程中出现错误: {e}")
        import traceback
        traceback.print_exc()
        return False

def main():
    """主函数"""
    print("开始测试LLM生成器修复...")
    
    success = test_llm_generator()
    
    if success:
        print("\n🎉 测试成功！LLM生成器修复有效")
        print("现在多阶段检索系统可以正常生成答案了")
    else:
        print("\n❌ 测试失败！LLM生成器仍有问题")
    
    return success

if __name__ == "__main__":
    main() 