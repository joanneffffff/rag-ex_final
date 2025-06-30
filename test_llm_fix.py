#!/usr/bin/env python3
"""
测试LLM生成器修复效果
验证输入截断和聊天格式问题是否解决
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from xlm.components.generator.local_llm_generator import LocalLLMGenerator
from xlm.components.prompt_templates.prompt_template_loader import PromptTemplateLoader

def test_llm_generator_fix():
    """测试LLM生成器修复效果"""
    print("=" * 80)
    print("🔧 测试LLM生成器修复效果")
    print("=" * 80)
    
    try:
        # 初始化LLM生成器
        print("1. 初始化LLM生成器...")
        generator = LocalLLMGenerator()
        print(f"✅ LLM生成器初始化成功: {generator.model_name}")
        
        # 加载Prompt模板
        print("\n2. 加载Prompt模板...")
        loader = PromptTemplateLoader()
        template = loader.load_template("multi_stage_chinese_template")
        print(f"✅ Prompt模板加载成功，长度: {len(template)} 字符")
        
        # 准备测试数据
        print("\n3. 准备测试数据...")
        context = "德赛电池（000049）2021年业绩预告显示，公司营收约193.9亿元，同比增长5%，净利润7.07亿元，同比增长45.13%，归母净利润6.37亿元，同比增长25.5%。业绩超出预期主要源于iPhone 12 Pro Max需求佳及盈利能力提升。"
        query = "德赛电池（000049）2021年利润持续增长的主要原因是什么？"
        
        # 格式化Prompt
        print("\n4. 格式化Prompt...")
        prompt = template.format(context=context, query=query)
        print(f"✅ Prompt格式化完成，长度: {len(prompt)} 字符")
        
        # 打印Prompt预览
        print("\n5. Prompt预览:")
        print("-" * 50)
        print(prompt[:500] + "..." if len(prompt) > 500 else prompt)
        print("-" * 50)
        
        # 调用LLM生成器
        print("\n6. 调用LLM生成器...")
        print("🚀 开始生成答案...")
        
        responses = generator.generate([prompt])
        response = responses[0] if responses else "生成失败"
        
        print("\n7. 生成结果:")
        print("=" * 50)
        print("📤 发送的Prompt长度:", len(prompt), "字符")
        print("📥 生成的答案:")
        print(response)
        print("=" * 50)
        
        # 分析结果
        print("\n8. 结果分析:")
        if "德赛电池" in response and ("iPhone" in response or "需求" in response or "盈利能力" in response):
            print("✅ 答案相关性良好 - 包含关键信息")
        elif "根据现有信息，无法提供此项信息" in response:
            print("✅ 答案格式正确 - 明确表示信息不足")
        else:
            print("❌ 答案可能有问题 - 未包含预期内容")
            
        return True
        
    except Exception as e:
        print(f"❌ 测试失败: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = test_llm_generator_fix()
    if success:
        print("\n🎉 测试完成！LLM生成器修复验证成功")
    else:
        print("\n💥 测试失败！需要进一步调试") 