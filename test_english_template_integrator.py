#!/usr/bin/env python3
"""
测试英文模板集成器的修改
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from xlm.components.prompts.english_prompt_integrator import english_prompt_integrator

def test_english_template_integrator():
    """测试英文模板集成器"""
    print("=== 测试英文模板集成器 ===")
    
    # 1. 测试模板信息
    print("\n1. 模板信息:")
    template_info = english_prompt_integrator.get_template_info()
    for key, value in template_info.items():
        print(f"   {key}: {value}")
    
    # 2. 测试Prompt创建
    print("\n2. 测试Prompt创建:")
    
    # 测试上下文和问题
    test_context = """
    Apple Inc. reported Q3 2023 revenue of $81.8 billion, down 1.4% year-over-year. 
    iPhone sales increased 2.8% to $39.7 billion. The company's services revenue grew 8.2% to $21.2 billion.
    """
    
    test_question = "What was Apple's revenue in Q3 2023?"
    
    # 创建英文prompt
    english_prompt = english_prompt_integrator.create_english_prompt(
        context=test_context,
        question=test_question
    )
    
    print(f"生成的英文Prompt长度: {len(english_prompt)} 字符")
    print("Prompt预览:")
    print("-" * 50)
    print(english_prompt[:500] + "..." if len(english_prompt) > 500 else english_prompt)
    print("-" * 50)
    
    # 3. 验证Prompt内容
    print("\n3. 验证Prompt内容:")
    
    # 检查是否包含必要的元素
    required_elements = ["Context:", "Question:", "Answer:"]
    missing_elements = []
    
    for element in required_elements:
        if element not in english_prompt:
            missing_elements.append(element)
    
    if missing_elements:
        print(f"❌ 缺少必要元素: {missing_elements}")
    else:
        print("✅ 包含所有必要元素")
    
    # 检查是否包含测试内容
    if "Apple Inc." in english_prompt and "Q3 2023" in english_prompt:
        print("✅ 包含测试上下文内容")
    else:
        print("❌ 缺少测试上下文内容")
    
    if "What was Apple's revenue" in english_prompt:
        print("✅ 包含测试问题内容")
    else:
        print("❌ 缺少测试问题内容")
    
    # 4. 测试与中文模板的一致性
    print("\n4. 测试与中文模板的一致性:")
    
    from xlm.components.prompt_templates.template_loader import template_loader
    
    # 检查是否使用相同的模板加载器
    if hasattr(english_prompt_integrator, 'template_name'):
        template_name = english_prompt_integrator.template_name
        template_content = template_loader.get_template(template_name)
        
        if template_content:
            print(f"✅ 成功加载模板: {template_name}")
            print(f"   模板长度: {len(template_content)} 字符")
        else:
            print(f"❌ 无法加载模板: {template_name}")
    
    # 5. 测试错误处理
    print("\n5. 测试错误处理:")
    
    # 测试空上下文
    empty_prompt = english_prompt_integrator.create_english_prompt(
        context="",
        question="Test question"
    )
    
    if empty_prompt:
        print("✅ 空上下文处理正常")
    else:
        print("❌ 空上下文处理失败")
    
    print("\n=== 测试完成 ===")

if __name__ == "__main__":
    test_english_template_integrator() 