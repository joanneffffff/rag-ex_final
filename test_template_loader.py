#!/usr/bin/env python3
"""
测试模板加载器
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from xlm.components.prompt_templates.template_loader import template_loader

def test_template_loader():
    """测试模板加载器功能"""
    print("=== 测试模板加载器 ===")
    
    # 列出所有可用的模板
    templates = template_loader.list_templates()
    print(f"可用的模板: {templates}")
    
    # 测试英文模板
    print("\n=== 测试英文模板 ===")
    english_prompt = template_loader.format_template(
        "rag_english_template",
        context="Apple Inc. reported Q3 2023 revenue of $81.8 billion.",
        question="What was Apple's revenue in Q3 2023?"
    )
    print(f"英文模板结果:\n{english_prompt}")
    
    # 测试中文模板
    print("\n=== 测试中文模板 ===")
    chinese_prompt = template_loader.format_template(
        "rag_chinese_clean_template",
        context="中国平安2023年第一季度实现营业收入2,345.67亿元。",
        question="中国平安的营业收入是多少？"
    )
    print(f"中文模板结果:\n{chinese_prompt}")
    
    # 测试多阶段模板
    print("\n=== 测试多阶段模板 ===")
    multi_stage_prompt = template_loader.format_template(
        "multi_stage_chinese_template",
        context="测试上下文内容",
        query="测试查询"
    )
    print(f"多阶段模板结果:\n{multi_stage_prompt}")

if __name__ == "__main__":
    test_template_loader() 