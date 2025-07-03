#!/usr/bin/env python3
"""
测试模板加载功能
"""

import sys
import os
from pathlib import Path

# 添加项目根目录到Python路径
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

def test_template_loading():
    """测试模板加载功能"""
    print("=" * 60)
    print("🔧 模板加载测试")
    print("=" * 60)
    
    # 导入模板加载函数
    from comprehensive_evaluation_enhanced import load_and_format_template, get_final_prompt
    
    # 测试数据
    test_context = "This is a test context with some financial data."
    test_query = "What is the test question?"
    
    print("1. 测试模板文件存在性:")
    template_files = [
        "data/prompt_templates/template_for_table_answer.txt",
        "data/prompt_templates/template_for_text_answer.txt", 
        "data/prompt_templates/template_for_hybrid_answer.txt"
    ]
    
    for template_file in template_files:
        if Path(template_file).exists():
            print(f"   ✅ {template_file} 存在")
        else:
            print(f"   ❌ {template_file} 不存在")
    
    print("\n2. 测试直接模板加载:")
    try:
        # 测试文本模板
        text_messages = load_and_format_template("template_for_text_answer.txt", test_context, test_query)
        print(f"   ✅ 文本模板加载成功，消息数量: {len(text_messages)}")
        
        # 测试表格模板
        table_messages = load_and_format_template("template_for_table_answer.txt", test_context, test_query)
        print(f"   ✅ 表格模板加载成功，消息数量: {len(table_messages)}")
        
        # 测试混合模板
        hybrid_messages = load_and_format_template("template_for_hybrid_answer.txt", test_context, test_query)
        print(f"   ✅ 混合模板加载成功，消息数量: {len(hybrid_messages)}")
        
    except Exception as e:
        print(f"   ❌ 模板加载失败: {e}")
        return False
    
    print("\n3. 测试动态路由:")
    try:
        # 测试不同类型的context
        text_context = "This is a paragraph about financial performance."
        table_context = "Table: Revenue | 2023 | 2024\nRow: Sales | $100M | $120M"
        hybrid_context = "Table: Revenue | 2023 | 2024\nRow: Sales | $100M | $120M\n\nNote: The increase was due to market expansion."
        
        # 测试文本路由
        text_prompt = get_final_prompt(text_context, "What is the revenue?")
        print(f"   ✅ 文本路由成功，使用模板: {len(text_prompt)} 条消息")
        
        # 测试表格路由
        table_prompt = get_final_prompt(table_context, "What is the revenue?")
        print(f"   ✅ 表格路由成功，使用模板: {len(table_prompt)} 条消息")
        
        # 测试混合路由
        hybrid_prompt = get_final_prompt(hybrid_context, "What is the revenue?")
        print(f"   ✅ 混合路由成功，使用模板: {len(hybrid_prompt)} 条消息")
        
    except Exception as e:
        print(f"   ❌ 动态路由失败: {e}")
        return False
    
    print("\n" + "=" * 60)
    print("🎉 模板加载测试完成")
    print("=" * 60)
    return True

if __name__ == "__main__":
    success = test_template_loading()
    sys.exit(0 if success else 1) 