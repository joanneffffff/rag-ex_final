#!/usr/bin/env python3
"""
调试system内容提取
"""

def debug_system_extraction():
    """调试system内容提取"""
    text = """你是一位专业的金融分析师。你的核心任务是根据以下提供的【公司财务报告片段】，针对用户的问题提供一个**纯粹、直接且准确的回答**。

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
    
    print("=== 调试system内容提取 ===")
    print(f"原始文本长度: {len(text)} 字符")
    
    # 查找关键位置
    system_start = text.find("你是一位专业的金融分析师")
    context_start = text.find("【公司财务报告片段】")
    
    print(f"system_start: {system_start}")
    print(f"context_start: {context_start}")
    
    if system_start != -1 and context_start != -1:
        system_content = text[system_start:context_start].strip()
        user_content = text[context_start:].strip()
        
        print(f"\n=== 提取结果 ===")
        print(f"System content length: {len(system_content)} 字符")
        print(f"User content length: {len(user_content)} 字符")
        
        print(f"\n=== System content ===")
        print(system_content)
        
        print(f"\n=== User content (前200字符) ===")
        print(user_content[:200] + "..." if len(user_content) > 200 else user_content)
        
        # 构造聊天格式
        messages = [
            {"role": "system", "content": system_content},
            {"role": "user", "content": user_content}
        ]
        
        chat_text = ""
        for message in messages:
            role = message["role"]
            content = message["content"]
            if role == "system":
                chat_text += f"<|im_start|>system\n{content}<|im_end|>\n"
            elif role == "user":
                chat_text += f"<|im_start|>user\n{content}<|im_end|>\n"
            elif role == "assistant":
                chat_text += f"<|im_start|>assistant\n{content}<|im_end|>\n"
        
        chat_text += "<|im_start|>assistant\n"
        
        print(f"\n=== 聊天格式长度 ===")
        print(f"Chat text length: {len(chat_text)} 字符")
        
        print(f"\n=== 聊天格式预览 ===")
        print(chat_text[:500] + "..." if len(chat_text) > 500 else chat_text)
        
    else:
        print("❌ 无法找到关键标记")

if __name__ == "__main__":
    debug_system_extraction() 