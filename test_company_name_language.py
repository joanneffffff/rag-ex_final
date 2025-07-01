#!/usr/bin/env python3
"""
公司名称语言一致性测试
验证公司名称保持原样，回答语言与查询语言一致
"""

import sys
import os
from pathlib import Path

# Add parent directory to path
sys.path.append(str(Path(__file__).parent))

def test_chinese_company_english_query():
    """测试中文公司名称 + 英文查询"""
    
    print("=== 中文公司名称 + 英文查询测试 ===")
    print("=" * 50)
    
    try:
        from xlm.components.generator.local_llm_generator import LocalLLMGenerator
        
        # 初始化生成器
        generator = LocalLLMGenerator(device="cuda:1")
        
        # 测试 Prompt：中文公司名称，英文查询
        test_prompt = """===SYSTEM===
你是一位专业的金融分析师。请基于以下信息回答问题：

**要求：**
1. 回答简洁，控制在2-3句话内
2. 用中文回答
3. 公司名称保持原样
4. 句子要完整

===USER===
德赛电池（000049）2021年业绩预告显示，公司预计实现归属于上市公司股东的净利润为6.5亿元至7.5亿元，
同比增长11.02%至28.23%。业绩增长的主要原因是：
1. iPhone 12 Pro Max等高端产品需求强劲，带动公司电池业务增长
2. 新产品盈利能力提升，毛利率改善
3. A客户业务持续成长，非手机业务稳步增长

Question: What are the main reasons for Desay Battery's profit growth in 2021?

Answer: ==="""
        
        print("🚀 开始生成...")
        responses = generator.generate([test_prompt])
        response = responses[0] if responses else "生成失败"
        
        print(f"问题: What are the main reasons for Desay Battery's profit growth in 2021?")
        print(f"答案: {response}")
        
        # 评估响应
        length = len(response.strip())
        has_chinese = any('\u4e00' <= char <= '\u9fff' for char in response)
        has_english = any(char.isalpha() and ord(char) < 128 for char in response)
        
        # 检查公司名称
        chinese_company_names = ["德赛电池", "000049"]
        english_company_names = ["Desay Battery"]
        product_names = ["iPhone", "12 Pro Max"]
        
        has_chinese_company = any(name in response for name in chinese_company_names)
        has_english_company = any(name in response for name in english_company_names)
        has_product_names = any(name in response for name in product_names)
        
        # 语言一致性评估
        is_chinese_answer = has_chinese  # 包含中文字符表示中文回答
        company_name_consistent = has_chinese_company  # 包含中文公司名称
        
        print(f"\n评估结果:")
        print(f"响应长度: {length} 字符")
        print(f"包含中文字符: {'是' if has_chinese else '否'}")
        print(f"包含英文字符: {'是' if has_english else '否'}")
        print(f"包含中文公司名称: {'是' if has_chinese_company else '否'}")
        print(f"包含英文公司名称: {'是' if has_english_company else '否'}")
        print(f"包含产品名称: {'是' if has_product_names else '否'}")
        print(f"中文回答: {'是' if is_chinese_answer else '否'}")
        print(f"公司名称一致: {'是' if company_name_consistent else '否'}")
        
        # 评分
        score = 0
        if 20 <= length <= 200: score += 20
        if is_chinese_answer: score += 20
        if company_name_consistent: score += 20
        if has_product_names: score += 20
        if response.strip().endswith(("。", "！", "？")): score += 20
        
        print(f"评分: {score}/100")
        
        return score >= 80
        
    except Exception as e:
        print(f"❌ 测试失败: {e}")
        return False

def test_english_company_chinese_query():
    """测试英文公司名称 + 中文查询"""
    
    print("\n=== 英文公司名称 + 中文查询测试 ===")
    print("=" * 50)
    
    try:
        from xlm.components.generator.local_llm_generator import LocalLLMGenerator
        
        # 初始化生成器
        generator = LocalLLMGenerator(device="cuda:1")
        
        # 测试 Prompt：英文公司名称，中文查询
        test_prompt = """===SYSTEM===
你是一位专业的金融分析师。请基于以下信息回答问题：

**要求：**
1. 回答简洁，控制在2-3句话内
2. 用中文回答
3. 公司名称保持原样
4. 句子要完整

===USER===
Apple Inc. reported Q3 2023 revenue of $81.8 billion, down 1.4% year-over-year. 
iPhone sales increased 2.8% to $39.7 billion. The company's services revenue 
grew 8.2% to $21.2 billion, while Mac and iPad sales declined.

问题：苹果公司2023年第三季度的表现如何？

回答：==="""
        
        print("🚀 开始生成...")
        responses = generator.generate([test_prompt])
        response = responses[0] if responses else "生成失败"
        
        print(f"问题: 苹果公司2023年第三季度的表现如何？")
        print(f"答案: {response}")
        
        # 评估响应
        length = len(response.strip())
        has_chinese = any('\u4e00' <= char <= '\u9fff' for char in response)
        has_english = any(char.isalpha() and ord(char) < 128 for char in response)
        
        # 检查公司名称
        english_company_names = ["Apple", "iPhone", "Mac", "iPad"]
        chinese_company_names = ["苹果公司", "苹果"]
        
        has_english_company = any(name in response for name in english_company_names)
        has_chinese_company = any(name in response for name in chinese_company_names)
        
        # 语言一致性评估
        is_chinese_answer = has_chinese  # 包含中文字符表示中文回答
        company_name_consistent = has_english_company  # 包含英文公司名称
        
        print(f"\n评估结果:")
        print(f"响应长度: {length} 字符")
        print(f"包含中文字符: {'是' if has_chinese else '否'}")
        print(f"包含英文字符: {'是' if has_english else '否'}")
        print(f"包含英文公司名称: {'是' if has_english_company else '否'}")
        print(f"包含中文公司名称: {'是' if has_chinese_company else '否'}")
        print(f"中文回答: {'是' if is_chinese_answer else '否'}")
        print(f"公司名称一致: {'是' if company_name_consistent else '否'}")
        
        # 评分
        score = 0
        if 20 <= length <= 200: score += 20
        if is_chinese_answer: score += 20
        if company_name_consistent: score += 20
        if "revenue" in response.lower() or "billion" in response.lower(): score += 20
        if response.strip().endswith(("。", "！", "？")): score += 20
        
        print(f"评分: {score}/100")
        
        return score >= 80
        
    except Exception as e:
        print(f"❌ 测试失败: {e}")
        return False

def test_mixed_company_names():
    """测试混合公司名称"""
    
    print("\n=== 混合公司名称测试 ===")
    print("=" * 50)
    
    try:
        from xlm.components.generator.local_llm_generator import LocalLLMGenerator
        
        # 初始化生成器
        generator = LocalLLMGenerator(device="cuda:1")
        
        # 测试 Prompt：混合公司名称
        test_prompt = """===SYSTEM===
你是一位专业的金融分析师。请基于以下信息回答问题：

**要求：**
1. 回答简洁，控制在2-3句话内
2. 用中文回答
3. 公司名称保持原样
4. 句子要完整

===USER===
德赛电池（000049）为Apple Inc.提供iPhone电池，2021年业绩增长显著。
同时，用友网络（600588）的云服务业务也表现良好。

问题：这些公司的业务关系如何？

回答：==="""
        
        print("🚀 开始生成...")
        responses = generator.generate([test_prompt])
        response = responses[0] if responses else "生成失败"
        
        print(f"问题: 这些公司的业务关系如何？")
        print(f"答案: {response}")
        
        # 评估响应
        length = len(response.strip())
        has_chinese = any('\u4e00' <= char <= '\u9fff' for char in response)
        has_english = any(char.isalpha() and ord(char) < 128 for char in response)
        
        # 检查公司名称
        chinese_company_names = ["德赛电池", "000049", "用友网络", "600588"]
        english_company_names = ["Apple", "iPhone"]
        
        has_chinese_company = any(name in response for name in chinese_company_names)
        has_english_company = any(name in response for name in english_company_names)
        
        # 语言一致性评估
        is_chinese_answer = has_chinese
        company_names_preserved = has_chinese_company and has_english_company
        
        print(f"\n评估结果:")
        print(f"响应长度: {length} 字符")
        print(f"包含中文字符: {'是' if has_chinese else '否'}")
        print(f"包含英文字符: {'是' if has_english else '否'}")
        print(f"包含中文公司名称: {'是' if has_chinese_company else '否'}")
        print(f"包含英文公司名称: {'是' if has_english_company else '否'}")
        print(f"中文回答: {'是' if is_chinese_answer else '否'}")
        print(f"公司名称保持原样: {'是' if company_names_preserved else '否'}")
        
        # 评分
        score = 0
        if 20 <= length <= 200: score += 20
        if is_chinese_answer: score += 20
        if company_names_preserved: score += 20
        if "电池" in response or "云服务" in response: score += 20
        if response.strip().endswith(("。", "！", "？")): score += 20
        
        print(f"评分: {score}/100")
        
        return score >= 80
        
    except Exception as e:
        print(f"❌ 测试失败: {e}")
        return False

def main():
    """主测试函数"""
    
    print("🚀 公司名称语言一致性测试")
    print("验证公司名称保持原样，回答语言与查询语言一致")
    print("=" * 60)
    
    # 测试中文公司名称 + 英文查询
    test1_result = test_chinese_company_english_query()
    
    # 测试英文公司名称 + 中文查询
    test2_result = test_english_company_chinese_query()
    
    # 测试混合公司名称
    test3_result = test_mixed_company_names()
    
    # 总结
    print(f"\n=== 测试结果总结 ===")
    print(f"中文公司名称 + 英文查询: {'✅ 通过' if test1_result else '❌ 失败'}")
    print(f"英文公司名称 + 中文查询: {'✅ 通过' if test2_result else '❌ 失败'}")
    print(f"混合公司名称: {'✅ 通过' if test3_result else '❌ 失败'}")
    
    if all([test1_result, test2_result, test3_result]):
        print("🎉 所有测试通过！公司名称语言一致性良好。")
    else:
        print("⚠️ 部分测试失败，需要优化 Prompt 或模型参数。")

if __name__ == "__main__":
    main() 