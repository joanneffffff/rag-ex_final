#!/usr/bin/env python3
"""
测试严格的token控制和prompt注入清理
验证max_new_tokens=200和答案清理逻辑是否生效
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from xlm.components.generator.local_llm_generator import LocalLLMGenerator

def load_chat_template(path):
    with open(path, 'r', encoding='utf-8') as f:
        content = f.read()
    parts = content.split('===USER===')
    if len(parts) != 2:
        raise ValueError("模板文件必须包含===SYSTEM===和===USER===分隔")
    system = parts[0].replace('===SYSTEM===', '').strip()
    user = parts[1].strip()
    return system, user

def test_strict_token_control():
    """测试严格的token控制和prompt注入清理"""
    print("=" * 80)
    print("🔧 测试严格的token控制和prompt注入清理")
    print("=" * 80)
    
    try:
        # 初始化LLM生成器
        print("1. 初始化LLM生成器...")
        generator = LocalLLMGenerator()
        print(f"✅ LLM生成器初始化成功: {generator.model_name}")
        print(f"📏 配置的max_new_tokens: {generator.max_new_tokens}")
        
        # 加载chat分段Prompt模板
        print("\n2. 加载chat分段Prompt模板...")
        system_prompt, user_prompt = load_chat_template("data/prompt_templates/multi_stage_chinese_template.txt")
        print(f"✅ SYSTEM段长度: {len(system_prompt)} 字符")
        print(f"✅ USER段长度: {len(user_prompt)} 字符")
        
        # 准备测试数据
        print("\n3. 准备测试数据...")
        context = "德赛电池（000049）2021年业绩预告显示，公司营收约193.9亿元，同比增长5%，净利润7.07亿元，同比增长45.13%，归母净利润6.37亿元，同比增长25.5%。业绩超出预期主要源于iPhone 12 Pro Max需求佳及盈利能力提升。"
        query = "德赛电池（000049）2021年利润持续增长的主要原因是什么？"
        summary = context[:200] + "..." if len(context) > 200 else context
        
        # 格式化USER段Prompt
        print("\n4. 格式化USER段Prompt...")
        prompt = user_prompt.format(context=context, query=query, summary=summary)
        print(f"✅ Prompt格式化完成，长度: {len(prompt)} 字符")
        
        # 检查Prompt是否包含"【回答】"标记
        if "【回答】" in prompt:
            print("❌ 发现Prompt中包含'【回答】'标记，这可能导致prompt注入！")
        else:
            print("✅ Prompt中不包含'【回答】'标记")
        
        # 打印Prompt预览
        print("\n5. Prompt预览:")
        print("-" * 50)
        print(prompt[:500] + "..." if len(prompt) > 500 else prompt)
        print("-" * 50)
        
        # 调用LLM生成器
        print("\n6. 调用LLM生成器...")
        print("🚀 开始生成答案...")
        
        answer = generator.generate(texts=[prompt])[0]
        
        print("\n7. 生成结果分析:")
        print("-" * 50)
        print(f"📏 答案长度: {len(answer)} 字符")
        print(f"📝 答案内容: '{answer}'")
        print("-" * 50)
        
        # 检查答案质量
        print("\n8. 答案质量检查:")
        
        # 检查是否包含prompt注入
        injection_indicators = ["【回答】", "Answer:", "回答：", "---", "===", "boxed{", "\\boxed{"]
        found_injections = []
        for indicator in injection_indicators:
            if indicator in answer:
                found_injections.append(indicator)
        
        if found_injections:
            print(f"❌ 发现prompt注入标记: {found_injections}")
        else:
            print("✅ 未发现prompt注入标记")
        
        # 检查答案长度
        if len(answer) > 200:
            print(f"⚠️  答案长度({len(answer)})超过200字符，可能超出预期")
        else:
            print(f"✅ 答案长度({len(answer)})在合理范围内")
        
        # 检查答案内容
        if "德赛电池" in answer and ("iPhone" in answer or "需求" in answer or "盈利" in answer):
            print("✅ 答案内容相关且准确")
        else:
            print("❌ 答案内容可能不相关或不准确")
        
        # 检查是否包含重复内容
        words = answer.split()
        if len(words) > 10:
            unique_words = set(words)
            repetition_ratio = 1 - (len(unique_words) / len(words))
            if repetition_ratio > 0.3:
                print(f"⚠️  答案重复率较高: {repetition_ratio:.2%}")
            else:
                print(f"✅ 答案重复率正常: {repetition_ratio:.2%}")
        
        print("\n9. 测试完成！")
        
    except Exception as e:
        print(f"❌ 测试失败: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    test_strict_token_control() 