#!/usr/bin/env python3
# -*- coding: utf-8 -*-

def analyze_chat_mode_vs_current():
    """分析聊天模式与当前模式的对比"""
    
    print("=== 聊天模式 vs 当前模式分析 ===")
    
    # 当前模式（单一字符串）
    current_mode = """你是一位专业的金融分析师，擅长分析公司财务报告并回答相关问题。

请基于以下公司财务报告片段，回答用户的问题。你的回答必须：
1. 准确、客观，基于提供的财务数据
2. 使用专业术语，但确保易于理解
3. 如果信息不足，明确指出并说明需要哪些额外信息
4. 提供具体的数字和百分比支持你的分析

【公司财务报告片段】
---
德赛电池（000049）2021年业绩预告显示，公司营收约193.9亿元，同比增长5%，净利润7.07亿元，同比增长45.13%，归母净利润6.37亿元，同比增长25.5%。业绩超出预期主要源于iPhone 12 Pro Max需求佳及盈利能力提升。
---

【用户问题】
德赛电池（000049）2021年利润持续增长的主要原因是什么？
---

请基于上述信息提供详细的分析。"""

    # 聊天模式（分离system和user）
    chat_mode = {
        "system": """你是一位专业的金融分析师，擅长分析公司财务报告并回答相关问题。

你的回答必须：
1. 准确、客观，基于提供的财务数据
2. 使用专业术语，但确保易于理解
3. 如果信息不足，明确指出并说明需要哪些额外信息
4. 提供具体的数字和百分比支持你的分析
5. 直接回答，不要包含思考过程或元评论""",
        
        "user": """请分析以下公司财务报告片段：

德赛电池（000049）2021年业绩预告显示，公司营收约193.9亿元，同比增长5%，净利润7.07亿元，同比增长45.13%，归母净利润6.37亿元，同比增长25.5%。业绩超出预期主要源于iPhone 12 Pro Max需求佳及盈利能力提升。

问题：德赛电池（000049）2021年利润持续增长的主要原因是什么？"""
    }
    
    print(f"当前模式长度: {len(current_mode)} 字符")
    print(f"聊天模式system长度: {len(chat_mode['system'])} 字符")
    print(f"聊天模式user长度: {len(chat_mode['user'])} 字符")
    print(f"聊天模式总长度: {len(chat_mode['system']) + len(chat_mode['user'])} 字符")
    
    print("\n=== 聊天模式优势 ===")
    chat_advantages = [
        "1. 角色分离清晰 - system定义角色，user提供具体任务",
        "2. 指令更简洁 - 避免重复的角色定义",
        "3. 模型理解更好 - Fin-R1等模型原生支持聊天格式",
        "4. 上下文更清晰 - system和user内容分离",
        "5. 易于扩展 - 可以添加多轮对话",
        "6. 格式标准化 - 符合主流LLM的输入格式"
    ]
    
    for advantage in chat_advantages:
        print(f"✅ {advantage}")
    
    print("\n=== 当前模式问题 ===")
    current_problems = [
        "1. 角色定义重复 - 在长prompt中可能被忽略",
        "2. 格式不标准 - 不是标准的聊天格式",
        "3. 模型理解困难 - 需要自己解析角色和任务",
        "4. 上下文混乱 - 指令、数据、问题混在一起",
        "5. 难以扩展 - 不支持多轮对话",
        "6. 长度冗余 - 包含不必要的格式标记"
    ]
    
    for problem in current_problems:
        print(f"❌ {problem}")
    
    print("\n=== 聊天模式实现示例 ===")
    
    # Fin-R1聊天格式
    fin_r1_chat_format = f"""<|im_start|>system
{chat_mode['system']}<|im_end|>
<|im_start|>user
{chat_mode['user']}<|im_end|>
<|im_start|>assistant
"""
    
    print("Fin-R1聊天格式:")
    print(f"长度: {len(fin_r1_chat_format)} 字符")
    print("格式: <|im_start|>system...<|im_end|> + <|im_start|>user...<|im_end|> + <|im_start|>assistant")
    
    # 通用聊天格式
    generic_chat_format = f"""<s>[INST] <<SYS>>
{chat_mode['system']}
<</SYS>>

{chat_mode['user']} [/INST]"""
    
    print(f"\n通用聊天格式长度: {len(generic_chat_format)} 字符")
    
    print("\n=== 优化建议 ===")
    
    # 结合Top1+摘要的聊天模式
    optimized_chat_mode = {
        "system": """你是一位专业的金融分析师。请基于提供的财务数据，给出准确、客观的分析。使用专业术语但确保易懂，提供具体数字支持。""",
        
        "user": """【最相关报告】
德赛电池（000049）业绩预告超出预期，主要得益于iPhone 12 Pro Max需求佳及盈利能力提升。

【关键数据】
• 营收: 193.9亿元 (+5%)
• 净利润: 7.07亿元 (+45.13%)
• 归母净利润: 6.37亿元 (+25.5%)

问题：德赛电池2021年利润持续增长的主要原因是什么？"""
    }
    
    optimized_chat = f"""<|im_start|>system
{optimized_chat_mode['system']}<|im_end|>
<|im_start|>user
{optimized_chat_mode['user']}<|im_end|>
<|im_start|>assistant
"""
    
    print(f"优化聊天模式长度: {len(optimized_chat)} 字符")
    print(f"相比当前模式减少: {((len(current_mode) - len(optimized_chat)) / len(current_mode) * 100):.1f}%")
    
    print("\n=== 结论 ===")
    conclusions = [
        "1. 聊天模式确实比当前模式更好",
        "2. 结合Top1+摘要策略效果最佳",
        "3. 符合Fin-R1等模型的预期格式",
        "4. 提高模型理解和回答质量",
        "5. 减少prompt长度和复杂度"
    ]
    
    for conclusion in conclusions:
        print(f"🎯 {conclusion}")

if __name__ == "__main__":
    analyze_chat_mode_vs_current() 