#!/usr/bin/env python3
# -*- coding: utf-8 -*-

def implement_chat_mode():
    """实现聊天模式的具体代码"""
    
    print("=== 聊天模式实现方案 ===")
    
    # 当前成功的Prompt（12,579字符）
    current_successful_prompt = """你是一位专业的金融分析师。你的核心任务是根据以下提供的【公司财务报告片段】，针对用户的问题提供一个**纯粹、直接且准确的回答**。你的回答必须**忠实于原文的语义和信息**，并进行**必要的总结和提炼**。

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
    
    print(f"当前成功Prompt长度: {len(current_successful_prompt)} 字符")
    
    # 转换为聊天模式
    chat_mode_prompt = {
        "system": """你是一位专业的金融分析师。你的核心任务是根据提供的财务报告片段，针对用户的问题提供一个纯粹、直接且准确的回答。

你的回答必须：
1. 忠实于原文的语义和信息，进行必要的总结和提炼
2. 不包含任何自我反思、思考过程或元评论
3. 直接、简洁地结束，不带引导语或总结性语句
4. 不允许引入任何原文未明确表达的信息或外部推断
5. 如果信息不足，直接回答"根据现有信息，无法提供此项信息。"
""",
        
        "user": """请分析以下财务报告片段：

德赛电池（000049）的业绩预告超出预期，主要得益于iPhone 12 Pro Max需求强劲和新品盈利能力提升。预计2021年利润将持续增长，源于A客户的业务成长、非手机业务的增长以及并表比例的增加。

问题：德赛电池（000049）2021年利润持续增长的主要原因是什么？"""
    }
    
    # Fin-R1聊天格式
    fin_r1_chat = f"""<|im_start|>system
{chat_mode_prompt['system']}<|im_end|>
<|im_start|>user
{chat_mode_prompt['user']}<|im_end|>
<|im_start|>assistant
"""
    
    print(f"聊天模式长度: {len(fin_r1_chat)} 字符")
    print(f"减少比例: {((len(current_successful_prompt) - len(fin_r1_chat)) / len(current_successful_prompt) * 100):.1f}%")
    
    print("\n=== 实现代码 ===")
    
    implementation_code = '''
# 在LocalLLMGenerator中添加聊天模式支持
def convert_to_chat_format(self, text):
    """将单一字符串转换为Fin-R1聊天格式"""
    
    # 检测是否包含系统指令
    if "你是一位专业的金融分析师" in text:
        # 提取system部分
        system_start = text.find("你是一位专业的金融分析师")
        context_start = text.find("【公司财务报告片段】")
        
        if system_start != -1 and context_start != -1:
            system_content = text[system_start:context_start].strip()
            
            # 提取user部分（上下文和问题）
            user_content = text[context_start:].strip()
            
            # 构造Fin-R1聊天格式
            chat_format = f"""<|im_start|>system
{system_content}<|im_end|>
<|im_start|>user
{user_content}<|im_end|>
<|im_start|>assistant
"""
            return chat_format
    
    return text

def generate(self, texts: List[str]) -> List[str]:
    responses = []
    for text in texts:
        # 检查Fin-R1是否支持聊天格式
        if "Fin-R1" in self.model_name:
            print("Fin-R1 detected, converting to chat format...")
            # 转换为聊天格式
            text = self.convert_to_chat_format(text)
            print(f"Converted to chat format, length: {len(text)} characters")
        
        # 继续原有的生成逻辑...
'''
    
    print(implementation_code)
    
    print("\n=== 优化建议 ===")
    
    # 结合Top1+摘要的聊天模式
    optimized_chat = {
        "system": """你是一位专业的金融分析师。请基于提供的财务数据，给出准确、客观的分析。使用专业术语但确保易懂，提供具体数字支持。""",
        
        "user": """【最相关报告】
德赛电池（000049）业绩预告超出预期，主要得益于iPhone 12 Pro Max需求强劲和新品盈利能力提升。

【关键数据】
• 营收增长: 主要受益于iPhone 12 Pro Max需求
• 盈利能力: 新品盈利能力提升
• 业务增长: A客户业务成长、非手机业务增长
• 并表效应: 并表比例增加

问题：德赛电池（000049）2021年利润持续增长的主要原因是什么？"""
    }
    
    optimized_fin_r1 = f"""<|im_start|>system
{optimized_chat['system']}<|im_end|>
<|im_start|>user
{optimized_chat['user']}<|im_end|>
<|im_start|>assistant
"""
    
    print(f"优化聊天模式长度: {len(optimized_fin_r1)} 字符")
    print(f"相比当前成功模式减少: {((len(current_successful_prompt) - len(optimized_fin_r1)) / len(current_successful_prompt) * 100):.1f}%")
    
    print("\n=== 实施步骤 ===")
    steps = [
        "1. 修改LocalLLMGenerator，添加convert_to_chat_format方法",
        "2. 在generate方法中检测Fin-R1模型并转换格式",
        "3. 测试聊天格式的效果",
        "4. 结合Top1+摘要策略进一步优化",
        "5. 监控回答质量改进"
    ]
    
    for step in steps:
        print(f"🔧 {step}")
    
    print("\n=== 预期效果 ===")
    benefits = [
        "✅ 减少Prompt长度，提高处理效率",
        "✅ 符合Fin-R1模型预期格式",
        "✅ 提高模型理解和回答质量",
        "✅ 保持当前回答的高质量",
        "✅ 为未来多轮对话做准备"
    ]
    
    for benefit in benefits:
        print(benefit)

if __name__ == "__main__":
    implement_chat_mode() 