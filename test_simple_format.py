#!/usr/bin/env python3
"""
测试简单格式，不使用聊天格式
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

def test_simple_format():
    """测试简单格式"""
    print("开始测试简单格式...")
    
    try:
        from xlm.components.generator.local_llm_generator import LocalLLMGenerator
        
        # 初始化生成器
        print("初始化LLM生成器...")
        generator = LocalLLMGenerator()
        print(f"生成器初始化成功: {generator.model_name}")
        
        # 测试简单格式
        print("\n=== 测试简单格式 ===")
        simple_prompt = """德赛电池（000049）2021年利润持续增长的主要原因包括：1、iPhone 12 Pro Max需求佳及盈利能力提升；2、5G iPhone周期叠加非手机业务增量；3、Watch、AirPods需求量增长；4、iPad、Mac份额提升；5、新品盈利能力提升驱动盈利水平同比提升。

问题：德赛电池（000049）2021年利润持续增长的主要原因是什么？

回答："""
        
        print(f"简单Prompt长度: {len(simple_prompt)} 字符")
        print("🚀 开始生成答案...")
        
        responses = generator.generate([simple_prompt])
        response = responses[0] if responses else "无响应"
        
        print("\n=== 生成结果 ===")
        print("📤 发送的Prompt长度:", len(simple_prompt), "字符")
        print("📥 生成的答案:")
        print(response)
        print("=" * 50)
        
        # 分析结果
        print("\n=== 结果分析 ===")
        if "iPhone" in response and ("需求" in response or "盈利能力" in response):
            print("✅ 答案相关性良好 - 包含关键信息")
        elif "根据现有信息，无法提供此项信息" in response:
            print("❌ 答案格式正确但内容缺失 - 模型没有找到答案")
        elif "德赛电池" in response and ("利润" in response or "增长" in response):
            print("✅ 答案包含公司名称和利润相关信息 - 基本相关")
        else:
            print("❌ 答案可能有问题 - 未包含预期内容")
            
        return True
        
    except Exception as e:
        print(f"测试失败: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    test_simple_format() 