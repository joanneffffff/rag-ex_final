#!/usr/bin/env python3
"""
测试当前运行的LLM生成器是否使用了修复后的代码
"""

import sys
import os
from pathlib import Path

# 添加项目根目录到路径
sys.path.append(str(Path(__file__).parent))

def test_current_llm_generator():
    """测试当前LLM生成器"""
    print("=== 测试当前LLM生成器 ===")
    
    try:
        # 导入多阶段检索系统
        from alphafin_data_process.multi_stage_retrieval_final import MultiStageRetrievalSystem
        
        # 初始化系统（不加载模型，只检查代码）
        data_path = Path("data/alphafin/alphafin_merged_generated_qa.json")
        
        print("正在初始化多阶段检索系统...")
        retrieval_system = MultiStageRetrievalSystem(
            data_path=data_path,
            dataset_type="chinese",
            use_existing_config=True
        )
        
        # 检查LLM生成器是否正确初始化
        if retrieval_system.llm_generator is None:
            print("❌ LLM生成器未初始化")
            return
        
        print("✅ LLM生成器已初始化")
        
        # 检查LLM生成器类型
        from xlm.components.generator.local_llm_generator import LocalLLMGenerator
        if isinstance(retrieval_system.llm_generator, LocalLLMGenerator):
            print("✅ LLM生成器类型正确: LocalLLMGenerator")
        else:
            print(f"❌ LLM生成器类型错误: {type(retrieval_system.llm_generator)}")
            return
        
        # 检查是否有修正方法
        if hasattr(retrieval_system.llm_generator, '_fix_company_name_translation'):
            print("✅ LLM生成器包含 _fix_company_name_translation 方法")
        else:
            print("❌ LLM生成器缺少 _fix_company_name_translation 方法")
            return
        
        # 测试修正方法
        test_text = "德赛 battery (00) 的业绩表现良好"
        fixed_text = retrieval_system.llm_generator._fix_company_name_translation(test_text)
        
        print(f"测试修正方法:")
        print(f"  原始: {test_text}")
        print(f"  修正后: {fixed_text}")
        
        if "德赛电池" in fixed_text:
            print("✅ 修正方法工作正常")
        else:
            print("❌ 修正方法没有正常工作")
            return
        
        # 测试完整的生成流程（使用简单的prompt）
        print("\n=== 测试完整生成流程 ===")
        
        # 创建一个简单的测试prompt
        test_prompt = """===SYSTEM===
你是一位专业的金融分析师。请用中文回答用户的问题。

**极度重要：请严格遵守以下输出规范。你的回答：**
* **公司名称处理：** **严格禁止**将中文公司名称翻译为英文或修改公司名称。必须保持原始的中文公司名称不变，包括股票代码格式。

===USER===
德赛电池（000049）2021年利润持续增长的主要原因是什么？

===ASSISTANT===
"""
        
        print(f"测试Prompt长度: {len(test_prompt)} 字符")
        
        try:
            # 生成答案
            print("正在生成答案...")
            answers = retrieval_system.llm_generator.generate(texts=[test_prompt])
            
            if answers and len(answers) > 0:
                answer = answers[0]
                print(f"\n生成的答案:")
                print(f"'{answer}'")
                
                # 检查是否包含翻译问题
                if "德赛 battery" in answer or "德赛 Battery" in answer:
                    print("❌ 答案仍包含翻译问题")
                    print("可能的原因:")
                    print("1. 模型行为：Fin-R1模型可能忽略prompt指令")
                    print("2. 需要重启：系统可能需要重启以加载新代码")
                    print("3. 缓存问题：可能有缓存的文件没有更新")
                else:
                    print("✅ 答案没有翻译问题")
                
                # 检查是否包含正确的中文名称
                if "德赛电池" in answer:
                    print("✅ 答案包含正确的中文公司名称")
                else:
                    print("❌ 答案缺少正确的中文公司名称")
            else:
                print("❌ 没有生成答案")
                
        except Exception as e:
            print(f"❌ 生成答案时出错: {e}")
            import traceback
            traceback.print_exc()
        
    except Exception as e:
        print(f"❌ 测试失败: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    test_current_llm_generator() 