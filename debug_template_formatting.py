#!/usr/bin/env python3
"""
调试模板格式化失败问题
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from xlm.components.prompt_templates.template_loader import template_loader

def debug_template_formatting():
    """调试模板格式化问题"""
    
    print("🔍 调试模板格式化失败问题")
    print("=" * 60)
    
    # 1. 检查模板内容
    print("1. 检查模板内容:")
    template = template_loader.get_template("multi_stage_chinese_template")
    if template:
        print(f"✅ 模板加载成功，长度: {len(template)} 字符")
        print("模板内容预览:")
        print("-" * 40)
        print(template[:500] + "..." if len(template) > 500 else template)
        print("-" * 40)
    else:
        print("❌ 模板加载失败")
        return
    
    # 2. 检查模板中的参数
    print("\n2. 检查模板中的参数:")
    import re
    param_pattern = r'\{(\w+)\}'
    params = re.findall(param_pattern, template)
    print(f"模板中的参数: {params}")
    
    # 3. 测试错误的调用方式（缺少summary参数）
    print("\n3. 测试错误的调用方式（缺少summary参数）:")
    try:
        wrong_prompt = template_loader.format_template(
            "multi_stage_chinese_template",
            context="测试上下文",
            query="测试查询"
        )
        print(f"❌ 错误调用应该失败，但返回: {wrong_prompt}")
    except Exception as e:
        print(f"✅ 错误调用正确失败: {e}")
    
    # 4. 测试正确的调用方式
    print("\n4. 测试正确的调用方式:")
    try:
        correct_prompt = template_loader.format_template(
            "multi_stage_chinese_template",
            summary="测试摘要",
            context="测试上下文",
            query="测试查询"
        )
        print(f"✅ 正确调用成功")
        print("格式化后的prompt预览:")
        print("-" * 40)
        if correct_prompt:
            print(correct_prompt[:500] + "..." if len(correct_prompt) > 500 else correct_prompt)
        else:
            print("格式化失败，返回None")
        print("-" * 40)
    except Exception as e:
        print(f"❌ 正确调用失败: {e}")
    
    # 5. 检查其他可能的调用点
    print("\n5. 检查其他可能的调用点:")
    print("需要检查以下文件中是否有错误的调用:")
    print("- test_chinese_prompt.py (已确认缺少summary参数)")
    print("- 其他可能调用模板的文件")
    
    # 6. 提供修复建议
    print("\n6. 修复建议:")
    print("所有调用 multi_stage_chinese_template 的地方都需要传递三个参数:")
    print("```python")
    print("prompt = template_loader.format_template(")
    print("    'multi_stage_chinese_template',")
    print("    summary=summary,  # 摘要或context前200字")
    print("    context=context,  # 完整上下文")
    print("    query=query       # 用户查询")
    print(")")
    print("```")

if __name__ == "__main__":
    debug_template_formatting() 