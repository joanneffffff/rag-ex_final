#!/usr/bin/env python3
"""
测试句子完整性检测和动态token调整功能
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from xlm.components.generator.local_llm_generator import LocalLLMGenerator
from config.parameters import Config

def test_sentence_completion():
    """测试句子完整性检测功能"""
    
    print("🔍 测试句子完整性检测功能...")
    
    # 初始化配置
    config = Config()
    
    # 初始化生成器
    generator = LocalLLMGenerator()
    
    # 测试用例
    test_cases = [
        # 完整句子
        "德赛电池业绩增长主要源于iPhone需求强劲和新品盈利能力提升。",
        "公司营收同比增长15%，净利润增长20%。",
        
        # 不完整句子
        "德赛电池业绩增长主要源于iPhone需求强劲和新品盈利能力提",
        "公司营收同比增长15%，净利润增长",
        "根据财务报告显示，公司",
        
        # 英文完整句子
        "The company's revenue increased by 15% year-over-year.",
        "Net profit grew by 20% compared to last year.",
        
        # 英文不完整句子
        "The company's revenue increased by 15% year-over-",
        "Net profit grew by 20% compared to",
        
        # 空字符串
        "",
        "   ",
    ]
    
    print("\n📝 句子完整性检测结果:")
    print("-" * 60)
    
    for i, text in enumerate(test_cases, 1):
        is_complete = generator._is_sentence_complete(text)
        status = "✅ 完整" if is_complete else "❌ 不完整"
        print(f"{i:2d}. {status} | {repr(text)}")
    
    print("\n" + "=" * 60)

def test_dynamic_token_generation():
    """测试动态token生成功能"""
    
    print("🚀 测试动态token生成功能...")
    
    # 初始化配置
    config = Config()
    
    # 初始化生成器
    generator = LocalLLMGenerator()
    
    # 测试prompt
    test_prompt = """
请分析德赛电池的业绩表现，重点关注营收增长和盈利能力。
"""
    
    print(f"\n📋 测试Prompt: {test_prompt.strip()}")
    print("-" * 60)
    
    try:
        # 生成回答
        response = generator.generate([test_prompt])[0]
        
        print(f"✅ 生成成功!")
        print(f"📊 回答长度: {len(response)} 字符")
        print(f"📝 回答内容: {response}")
        
        # 检查句子完整性
        is_complete = generator._is_sentence_complete(response)
        print(f"🔍 句子完整性: {'✅ 完整' if is_complete else '❌ 不完整'}")
        
    except Exception as e:
        print(f"❌ 生成失败: {e}")

def test_token_optimization():
    """测试token优化效果"""
    
    print("⚡ 测试token优化效果...")
    
    # 初始化配置
    config = Config()
    
    # 初始化生成器
    generator = LocalLLMGenerator()
    
    # 测试不同max_new_tokens的效果
    test_cases = [
        ("德赛电池业绩如何？", 100),
        ("分析公司营收增长原因", 150),
        ("详细说明盈利能力变化", 200),
        ("综合评估公司发展前景", 250),
    ]
    
    print("\n📊 不同token数量的生成效果对比:")
    print("-" * 80)
    
    for query, max_tokens in test_cases:
        print(f"\n🔍 查询: {query}")
        print(f"🎯 目标token数: {max_tokens}")
        
        # 临时修改max_new_tokens
        original_max_tokens = generator.max_new_tokens
        generator.max_new_tokens = max_tokens
        
        try:
            response = generator.generate([query])[0]
            is_complete = generator._is_sentence_complete(response)
            
            print(f"📝 回答: {response}")
            print(f"📊 长度: {len(response)} 字符")
            print(f"✅ 完整性: {'完整' if is_complete else '不完整'}")
            
        except Exception as e:
            print(f"❌ 失败: {e}")
        
        # 恢复原始设置
        generator.max_new_tokens = original_max_tokens
        
        print("-" * 40)

if __name__ == "__main__":
    print("🧪 句子完整性检测和动态token调整测试")
    print("=" * 60)
    
    # 测试句子完整性检测
    test_sentence_completion()
    
    # 测试动态token生成
    test_dynamic_token_generation()
    
    # 测试token优化效果
    test_token_optimization()
    
    print("\n✅ 测试完成!") 