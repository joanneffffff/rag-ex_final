#!/usr/bin/env python3
"""
测试中英文查询的prompt模板选择
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from xlm.components.prompt_templates.template_loader import template_loader
from langdetect import detect

def test_language_detection():
    """测试语言检测功能"""
    
    print("🔍 测试语言检测功能")
    print("=" * 50)
    
    test_queries = [
        "德赛电池2021年业绩如何？",
        "What was Apple's revenue in Q3 2023?",
        "How did Tesla perform in Q2 2023?",
        "中国平安的营业收入是多少？",
        "What is the stock price of Microsoft?",
        "比亚迪的净利润增长了多少？"
    ]
    
    for query in test_queries:
        try:
            lang = detect(query)
            is_chinese = lang.startswith('zh')
            print(f"查询: {query}")
            print(f"  检测语言: {lang}")
            print(f"  是否中文: {is_chinese}")
            print()
        except Exception as e:
            print(f"查询: {query}")
            print(f"  语言检测失败: {e}")
            # 使用字符检测作为备选
            is_chinese = any('\u4e00' <= char <= '\u9fff' for char in query)
            print(f"  字符检测是否中文: {is_chinese}")
            print()

def test_prompt_template_selection():
    """测试prompt模板选择"""
    
    print("🔍 测试prompt模板选择")
    print("=" * 50)
    
    # 模拟上下文
    context = """
    Apple Inc. reported Q3 2023 revenue of $81.8 billion, down 1.4% year-over-year. 
    iPhone sales increased 2.8% to $39.7 billion. The company's services revenue grew 8.2% to $21.2 billion.
    """
    
    test_queries = [
        ("德赛电池2021年业绩如何？", "中文查询"),
        ("What was Apple's revenue in Q3 2023?", "英文查询"),
        ("How did Tesla perform in Q2 2023?", "英文查询"),
        ("中国平安的营业收入是多少？", "中文查询")
    ]
    
    for query, expected_type in test_queries:
        print(f"\n测试: {expected_type}")
        print(f"查询: {query}")
        
        # 语言检测
        try:
            query_language = detect(query)
            is_chinese_query = query_language.startswith('zh')
        except:
            is_chinese_query = any('\u4e00' <= char <= '\u9fff' for char in query)
        
        print(f"  检测结果: {'中文' if is_chinese_query else '英文'}")
        
        # 选择模板
        if is_chinese_query:
            # 中文查询使用中文prompt模板
            summary = context[:200] + "..." if len(context) > 200 else context
            prompt = template_loader.format_template(
                "multi_stage_chinese_template",
                summary=summary,
                context=context,
                query=query
            )
            template_name = "multi_stage_chinese_template"
        else:
            # 英文查询使用英文prompt模板
            prompt = template_loader.format_template(
                "rag_english_template",
                context=context,
                question=query
            )
            template_name = "rag_english_template"
        
        if prompt:
            print(f"  使用模板: {template_name}")
            print(f"  Prompt长度: {len(prompt)} 字符")
            print(f"  Prompt预览: {prompt[:100]}...")
            
            # 检查prompt语言
            chinese_chars = sum(1 for char in prompt if '\u4e00' <= char <= '\u9fff')
            english_words = len([word for word in prompt.split() if word.isalpha()])
            
            print(f"  中文字符数: {chinese_chars}")
            print(f"  英文单词数: {english_words}")
            
            if is_chinese_query and chinese_chars > 10:
                print("  ✅ 中文查询正确使用中文模板")
            elif not is_chinese_query and english_words > 10:
                print("  ✅ 英文查询正确使用英文模板")
            else:
                print("  ❌ 模板语言不匹配")
        else:
            print(f"  ❌ 模板格式化失败")

def test_ui_logic():
    """测试UI逻辑中的prompt选择"""
    
    print("\n🔍 测试UI逻辑中的prompt选择")
    print("=" * 50)
    
    # 模拟UI中的逻辑
    def simulate_ui_prompt_selection(question: str, context_str: str):
        """模拟UI中的prompt选择逻辑"""
        
        # 根据查询语言动态选择prompt模板
        try:
            from langdetect import detect
            query_language = detect(question)
            is_chinese_query = query_language.startswith('zh')
        except:
            # 如果语言检测失败，根据查询内容判断
            is_chinese_query = any('\u4e00' <= char <= '\u9fff' for char in question)
        
        if is_chinese_query:
            # 中文查询使用中文prompt模板
            summary = context_str[:200] + "..." if len(context_str) > 200 else context_str
            prompt = template_loader.format_template(
                "multi_stage_chinese_template",
                summary=summary,
                context=context_str,
                query=question
            )
            if prompt is None:
                # 回退到简单中文prompt
                prompt = f"基于以下上下文回答问题：\n\n{context_str}\n\n问题：{question}\n\n回答："
            template_type = "中文模板"
        else:
            # 英文查询使用英文prompt模板
            prompt = template_loader.format_template(
                "rag_english_template",
                context=context_str,
                question=question
            )
            if prompt is None:
                # 回退到简单英文prompt
                prompt = f"Context: {context_str}\nQuestion: {question}\nAnswer:"
            template_type = "英文模板"
        
        return prompt, template_type, is_chinese_query
    
    # 测试用例
    test_cases = [
        ("德赛电池2021年业绩如何？", "德赛电池2021年实现营业收入45.67亿元，同比增长12.3%。"),
        ("What was Apple's revenue in Q3 2023?", "Apple Inc. reported Q3 2023 revenue of $81.8 billion."),
        ("How did Tesla perform?", "Tesla delivered 466,140 vehicles in Q2 2023."),
        ("中国平安的营业收入是多少？", "中国平安2023年第一季度实现营业收入2,345.67亿元。")
    ]
    
    for question, context in test_cases:
        print(f"\n查询: {question}")
        print(f"上下文: {context}")
        
        prompt, template_type, is_chinese = simulate_ui_prompt_selection(question, context)
        
        print(f"  模板类型: {template_type}")
        print(f"  是否中文查询: {is_chinese}")
        print(f"  Prompt长度: {len(prompt)} 字符")
        print(f"  Prompt预览: {prompt[:150]}...")
        
        # 验证语言匹配
        chinese_chars = sum(1 for char in prompt if '\u4e00' <= char <= '\u9fff')
        english_words = len([word for word in prompt.split() if word.isalpha()])
        
        if is_chinese and chinese_chars > 5:
            print("  ✅ 语言匹配正确")
        elif not is_chinese and english_words > 5:
            print("  ✅ 语言匹配正确")
        else:
            print("  ❌ 语言匹配错误")

if __name__ == "__main__":
    print("🧪 测试中英文查询的prompt模板选择")
    print("=" * 60)
    
    # 测试语言检测
    test_language_detection()
    
    # 测试模板选择
    test_prompt_template_selection()
    
    # 测试UI逻辑
    test_ui_logic()
    
    print("\n✅ 测试完成!") 