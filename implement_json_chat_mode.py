#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import json

def implement_json_chat_mode():
    """实现JSON格式的聊天模式"""
    
    print("=== JSON格式聊天模式实现 ===")
    
    # 当前成功的Prompt
    current_prompt = """你是一位专业的金融分析师。你的核心任务是根据以下提供的【公司财务报告片段】，针对用户的问题提供一个**纯粹、直接且准确的回答**。你的回答必须**忠实于原文的语义和信息**，并进行**必要的总结和提炼**。

**极度重要：请严格遵守以下输出规范。你的回答：**
* **必须是纯粹的答案**：不包含任何自我反思、思考过程、对Prompt的分析、与回答无关的额外注释、任何格式标记（如 boxed、数字列表、加粗、下划线、代码块）、或任何形式的元评论。
* **请勿引用或复述Prompt内容**。
* **必须直接、简洁地结束**：不带任何引导语、开场白、后续说明或总结性语句。
* **信息来源限定与智能处理**：你的回答中的所有信息**必须忠实于提供的【公司财务报告片段】**。在此基础上，你可以对原文内容进行**必要的归纳、总结和措辞重组**，以便直接、清晰地回答用户问题。**不允许引入任何原文未明确表达的信息、外部推断或联想。**
* **处理信息缺失**：如果提供的【公司财务报告片段】中**确实没有足够的信息**来明确回答用户问题（即使经过归纳和重组措辞也无法得出），你的回答应是"根据现有信息，无法提供此项信息。"。请直接给出这句话，不带任何其他修饰。

---
**【公司财务报告片段】**
德赛电池（000049）的业绩预告超出预期，主要得益于iPhone 12 Pro Max需求强劲和新品盈利能力提升。预计2021年利润将持续增长，源于A客户的业务成长、非手机业务的增长以及并表比例的增加。
---

**【用户问题】**
德赛电池（000049）2021年利润持续增长的主要原因是什么？
---

**【回答】**"""
    
    print(f"当前Prompt长度: {len(current_prompt)} 字符")
    
    # JSON格式的聊天模式
    json_chat_mode = [
        {
            "role": "system", 
            "content": """你是一位专业的金融分析师。你的核心任务是根据提供的公司财务报告片段，针对用户的问题提供一个纯粹、直接且准确的回答。

极度重要：请严格遵守以下输出规范。你的回答：
* 必须是纯粹的答案：不包含任何自我反思、思考过程、对Prompt的分析、与回答无关的额外注释、任何格式标记（如 boxed、数字列表、加粗、下划线、代码块）、或任何形式的元评论。
* 请勿引用或复述Prompt内容。
* 必须直接、简洁地结束：不带任何引导语、开场白、后续说明或总结性语句。
* 信息来源限定与智能处理：你的回答中的所有信息必须忠实于提供的【公司财务报告片段】。在此基础上，你可以对原文内容进行必要的归纳、总结和措辞重组，以便直接、清晰地回答用户问题。不允许引入任何原文未明确表达的信息、外部推断或联想。
* 处理信息缺失：如果提供的【公司财务报告片段】中确实没有足够的信息来明确回答用户问题（即使经过归纳和重组措辞也无法得出），你的回答应是"根据现有信息，无法提供此项信息。"。请直接给出这句话，不带任何其他修饰。"""
        },
        {
            "role": "user", 
            "content": """---【公司财务报告片段】
德赛电池（000049）的业绩预告超出预期，主要得益于iPhone 12 Pro Max需求强劲和新品盈利能力提升。预计2021年利润将持续增长，源于A客户的业务成长、非手机业务的增长以及并表比例的增加。
---
【用户问题】
德赛电池（000049）2021年利润持续增长的主要原因是什么？
---
【回答】"""
        }
    ]
    
    # 转换为JSON字符串
    json_string = json.dumps(json_chat_mode, ensure_ascii=False, indent=2)
    print(f"JSON格式长度: {len(json_string)} 字符")
    print(f"减少比例: {((len(current_prompt) - len(json_string)) / len(current_prompt) * 100):.1f}%")
    
    print("\n=== JSON格式优势 ===")
    advantages = [
        "1. 标准化格式 - 符合主流LLM的输入格式",
        "2. 结构化清晰 - role和content分离",
        "3. 易于扩展 - 可以添加多轮对话",
        "4. 易于解析 - JSON格式便于处理",
        "5. 兼容性好 - 支持多种模型",
        "6. 调试方便 - 可以查看完整的对话结构"
    ]
    
    for advantage in advantages:
        print(f"✅ {advantage}")
    
    print("\n=== 实现代码 ===")
    
    implementation_code = '''
# 在LocalLLMGenerator中添加JSON聊天模式支持
def convert_to_json_chat_format(self, text):
    """将单一字符串转换为JSON聊天格式"""
    
    # 检测是否包含系统指令
    if "你是一位专业的金融分析师" in text:
        # 提取system部分
        system_start = text.find("你是一位专业的金融分析师")
        context_start = text.find("【公司财务报告片段】")
        
        if system_start != -1 and context_start != -1:
            system_content = text[system_start:context_start].strip()
            user_content = text[context_start:].strip()
            
            # 构造JSON格式
            json_chat = [
                {
                    "role": "system",
                    "content": system_content
                },
                {
                    "role": "user", 
                    "content": user_content
                }
            ]
            
            return json.dumps(json_chat, ensure_ascii=False)
    
    return text

def convert_json_to_fin_r1_format(self, json_chat):
    """将JSON格式转换为Fin-R1聊天格式"""
    
    try:
        chat_data = json.loads(json_chat)
        fin_r1_format = ""
        
        for message in chat_data:
            if message["role"] == "system":
                fin_r1_format += f'<|im_start|>system\\n{message["content"]}<|im_end|>\\n'
            elif message["role"] == "user":
                fin_r1_format += f'<|im_start|>user\\n{message["content"]}<|im_end|>\\n'
        
        fin_r1_format += '<|im_start|>assistant\\n'
        return fin_r1_format
        
    except json.JSONDecodeError:
        return json_chat

def generate(self, texts: List[str]) -> List[str]:
    responses = []
    for text in texts:
        # 检查Fin-R1是否支持聊天格式
        if "Fin-R1" in self.model_name:
            print("Fin-R1 detected, converting to JSON chat format...")
            # 转换为JSON格式
            json_chat = self.convert_to_json_chat_format(text)
            print(f"JSON chat format length: {len(json_chat)} characters")
            
            # 转换为Fin-R1格式
            text = self.convert_json_to_fin_r1_format(json_chat)
            print(f"Converted to Fin-R1 format, length: {len(text)} characters")
        
        # 继续原有的生成逻辑...
'''
    
    print(implementation_code)
    
    print("\n=== 优化版本（结合Top1+摘要）===")
    
    # 优化的JSON格式
    optimized_json_chat = [
        {
            "role": "system",
            "content": "你是一位专业的金融分析师。请基于提供的财务数据，给出准确、客观的分析。使用专业术语但确保易懂，提供具体数字支持。"
        },
        {
            "role": "user",
            "content": """【最相关报告】
德赛电池（000049）业绩预告超出预期，主要得益于iPhone 12 Pro Max需求强劲和新品盈利能力提升。

【关键数据】
• 营收增长: 主要受益于iPhone 12 Pro Max需求
• 盈利能力: 新品盈利能力提升
• 业务增长: A客户业务成长、非手机业务增长
• 并表效应: 并表比例增加

问题：德赛电池（000049）2021年利润持续增长的主要原因是什么？"""
        }
    ]
    
    optimized_json_string = json.dumps(optimized_json_chat, ensure_ascii=False, indent=2)
    print(f"优化JSON格式长度: {len(optimized_json_string)} 字符")
    print(f"相比当前Prompt减少: {((len(current_prompt) - len(optimized_json_string)) / len(current_prompt) * 100):.1f}%")
    
    print("\n=== 完整实现流程 ===")
    workflow = [
        "1. 接收原始Prompt字符串",
        "2. 检测是否包含系统指令",
        "3. 提取system和user内容",
        "4. 构造JSON格式聊天数据",
        "5. 转换为Fin-R1格式（如果需要）",
        "6. 发送给模型生成回答"
    ]
    
    for step in workflow:
        print(f"🔄 {step}")
    
    print("\n=== 测试示例 ===")
    print("JSON格式输出:")
    print(json.dumps(json_chat_mode, ensure_ascii=False, indent=2))

if __name__ == "__main__":
    implement_json_chat_mode() 