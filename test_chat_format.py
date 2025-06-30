#!/usr/bin/env python3
"""
测试Fin-R1聊天格式
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

def test_chat_format():
    """测试聊天格式"""
    print("开始测试Fin-R1聊天格式...")
    
    try:
        from xlm.components.generator.local_llm_generator import LocalLLMGenerator
        
        # 初始化生成器
        print("初始化LLM生成器...")
        generator = LocalLLMGenerator()
        print(f"生成器初始化成功: {generator.model_name}")
        
        # 测试1: 简单聊天格式
        print("\n=== 测试1: 简单聊天格式 ===")
        chat_prompt = "<|im_start|>system\n你是一位专业的金融分析师。<|im_end|>\n<|im_start|>user\n德赛电池2021年利润增长的原因是什么？<|im_end|>\n<|im_start|>assistant\n"
        
        print(f"聊天Prompt: {chat_prompt}")
        responses = generator.generate([chat_prompt])
        response1 = responses[0] if responses else "无响应"
        print(f"生成结果: {response1}")
        
        # 测试2: 完整Prompt格式
        print("\n=== 测试2: 完整Prompt格式 ===")
        full_prompt = """你是一位专业的金融分析师。你的核心任务是根据以下提供的【公司财务报告片段】，针对用户的问题提供一个**纯粹、直接且准确的回答**。

**极度重要：请严格遵守以下输出规范。你的回答：**
* **必须是纯粹的答案**：不包含任何自我反思、思考过程、对Prompt的分析、与回答无关的额外注释、任何格式标记（如 boxed、数字列表、加粗、下划线、代码块）、或任何形式的元评论。
* **请勿引用或复述Prompt内容**。
* **必须直接、简洁地结束**：不带任何引导语、开场白、后续说明或总结性语句。
* **信息来源严格限定**：你的回答中的所有信息**必须且仅能**来源于提供的【公司财务报告片段】。
* **处理信息缺失**：如果提供的【公司财务报告片段】中**无法找到**用户问题所需的明确、完整的答案，你的回答应是"根据现有信息，无法提供此项信息。"。请直接给出这句话，不带任何其他修饰。

---
**【公司财务报告片段】**
德赛电池（000049）2021年业绩预告显示，公司营收约193.9亿元，同比增长5%，净利润7.07亿元，同比增长45.13%，归母净利润6.37亿元，同比增长25.5%。业绩超出预期主要源于iPhone 12 Pro Max需求佳及盈利能力提升。
---

**【用户问题】**
德赛电池（000049）2021年利润持续增长的主要原因是什么？
---

**【回答】**
"""
        
        print(f"完整Prompt长度: {len(full_prompt)} 字符")
        responses = generator.generate([full_prompt])
        response2 = responses[0] if responses else "无响应"
        print(f"生成结果: {response2}")
        
        return True
        
    except Exception as e:
        print(f"测试失败: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    test_chat_format() 