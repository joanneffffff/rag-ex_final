#!/usr/bin/env python3
"""
测试包含明确答案的上下文
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

def test_with_answer():
    """测试包含明确答案的上下文"""
    print("开始测试包含明确答案的上下文...")
    
    try:
        from xlm.components.generator.local_llm_generator import LocalLLMGenerator
        
        # 初始化生成器
        print("初始化LLM生成器...")
        generator = LocalLLMGenerator()
        print(f"生成器初始化成功: {generator.model_name}")
        
        # 测试包含明确答案的Prompt
        print("\n=== 测试包含明确答案的Prompt ===")
        full_prompt = """你是一位专业的金融分析师。你的核心任务是根据以下提供的【公司财务报告片段】，针对用户的问题提供一个**纯粹、直接且准确的回答**。

**极度重要：请严格遵守以下输出规范。你的回答：**
* **必须是纯粹的答案**：不包含任何自我反思、思考过程、对Prompt的分析、与回答无关的额外注释、任何格式标记（如 boxed、数字列表、加粗、下划线、代码块）、或任何形式的元评论。
* **请勿引用或复述Prompt内容**。
* **必须直接、简洁地结束**：不带任何引导语、开场白、后续说明或总结性语句。
* **信息来源严格限定**：你的回答中的所有信息**必须且仅能**来源于提供的【公司财务报告片段】。
* **处理信息缺失**：如果提供的【公司财务报告片段】中**无法找到**用户问题所需的明确、完整的答案，你的回答应是"根据现有信息，无法提供此项信息。"。请直接给出这句话，不带任何其他修饰。

---
**【公司财务报告片段】**
德赛电池（000049）2021年利润持续增长的主要原因包括：1、iPhone 12 Pro Max需求佳及盈利能力提升；2、5G iPhone周期叠加非手机业务增量；3、Watch、AirPods需求量增长；4、iPad、Mac份额提升；5、新品盈利能力提升驱动盈利水平同比提升。
---

**【用户问题】**
德赛电池（000049）2021年利润持续增长的主要原因是什么？
---

**【回答】**
"""
        
        print(f"完整Prompt长度: {len(full_prompt)} 字符")
        print("🚀 开始生成答案...")
        
        responses = generator.generate([full_prompt])
        response = responses[0] if responses else "无响应"
        
        print("\n=== 生成结果 ===")
        print("📤 发送的Prompt长度:", len(full_prompt), "字符")
        print("📥 生成的答案:")
        print(response)
        print("=" * 50)
        
        # 分析结果
        print("\n=== 结果分析 ===")
        if "iPhone" in response and ("需求" in response or "盈利能力" in response):
            print("✅ 答案相关性良好 - 包含关键信息")
        elif "根据现有信息，无法提供此项信息" in response:
            print("❌ 答案格式正确但内容缺失 - 模型没有找到答案")
        elif "德赛电池" in response and ("利润" in response or "增长" in response):
            print("✅ 答案包含公司名称和利润相关信息 - 基本相关")
        else:
            print("❌ 答案可能有问题 - 未包含预期内容")
            
        return True
        
    except Exception as e:
        print(f"测试失败: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    test_with_answer() 