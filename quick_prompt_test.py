#!/usr/bin/env python3
"""
快速 Prompt 测试脚本
用于快速测试单个 Prompt 变体的效果
"""

import sys
import os
from pathlib import Path

# Add parent directory to path
sys.path.append(str(Path(__file__).parent))

def test_single_prompt():
    """测试单个 Prompt 变体"""
    
    print("=== 快速 Prompt 测试 ===")
    print("测试问题：德赛电池（000049）2021年利润持续增长的主要原因是什么？")
    print("=" * 60)
    
    try:
        from xlm.components.generator.local_llm_generator import LocalLLMGenerator
        
        # 初始化生成器
        print("1. 初始化 LLM 生成器...")
        generator = LocalLLMGenerator()
        print(f"✅ 生成器初始化成功: {generator.model_name}")
        
        # 测试数据
        test_context = """
        德赛电池（000049）2021年业绩预告显示，公司预计实现归属于上市公司股东的净利润为6.5亿元至7.5亿元，
        同比增长11.02%至28.23%。业绩增长的主要原因是：
        1. iPhone 12 Pro Max等高端产品需求强劲，带动公司电池业务增长
        2. 新产品盈利能力提升，毛利率改善
        3. A客户业务持续成长，非手机业务稳步增长
        4. 并表比例增加，贡献业绩增量
        """
        
        test_summary = "德赛电池2021年业绩增长主要受益于iPhone 12 Pro Max需求强劲和新品盈利能力提升。"
        test_query = "德赛电池（000049）2021年利润持续增长的主要原因是什么？"
        
        # 当前测试的 Prompt 变体（可以在这里修改测试不同的版本）
        current_prompt = f"""你是一位金融分析师。请基于以下信息回答问题：

摘要：{test_summary}

详细内容：{test_context}

问题：{test_query}

回答："""
        
        print(f"\n2. 当前测试的 Prompt:")
        print("-" * 40)
        print(current_prompt)
        print("-" * 40)
        print(f"Prompt 长度: {len(current_prompt)} 字符")
        
        # 测试参数（可以在这里修改）
        test_params = {
            "temperature": 0.2,
            "top_p": 0.8,
            "max_new_tokens": 200
        }
        
        print(f"\n3. 测试参数:")
        print(f"   Temperature: {test_params['temperature']}")
        print(f"   Top-p: {test_params['top_p']}")
        print(f"   Max tokens: {test_params['max_new_tokens']}")
        
        # 临时修改参数
        original_temp = generator.temperature
        original_top_p = generator.top_p
        original_max_tokens = generator.max_new_tokens
        
        try:
            generator.temperature = test_params["temperature"]
            generator.top_p = test_params["top_p"]
            generator.max_new_tokens = test_params["max_new_tokens"]
            
            # 生成响应
            print(f"\n4. 生成响应...")
            print("🚀 开始生成，请稍候...")
            
            responses = generator.generate([current_prompt])
            response = responses[0] if responses else "生成失败"
            
            print(f"\n5. 生成结果:")
            print("=" * 60)
            print(f"问题: {test_query}")
            print(f"答案: {response}")
            print("=" * 60)
            
            # 简单评估
            print(f"\n6. 简单评估:")
            length = len(response.strip())
            print(f"   响应长度: {length} 字符")
            print(f"   简洁性: {'✅' if 50 <= length <= 200 else '❌'} (理想: 50-200字符)")
            
            key_terms = ["德赛电池", "iPhone", "需求", "增长", "利润", "业绩"]
            found_terms = [term for term in key_terms if term in response]
            print(f"   关键信息: {found_terms}")
            print(f"   准确性: {'✅' if len(found_terms) >= 3 else '❌'} (找到{len(found_terms)}个关键词)")
            
            unwanted_patterns = ["【", "】", "回答：", "Answer:", "---", "===", "___"]
            has_unwanted = any(pattern in response for pattern in unwanted_patterns)
            print(f"   纯粹性: {'✅' if not has_unwanted else '❌'} (无格式标记)")
            
            is_complete = response.strip().endswith(("。", "！", "？", ".", "!", "?"))
            print(f"   完整性: {'✅' if is_complete else '❌'} (句子完整)")
            
            # 总体评分
            score = 0
            if 50 <= length <= 200: score += 25
            if len(found_terms) >= 3: score += 25
            if not has_unwanted: score += 25
            if is_complete: score += 25
            
            print(f"\n🎯 总体评分: {score}/100 ({score}%)")
            
            if score >= 75:
                print("🎉 效果很好！")
            elif score >= 50:
                print("⚠️ 效果一般，可以继续优化")
            else:
                print("❌ 效果不佳，需要重新设计")
            
        finally:
            # 恢复原始参数
            generator.temperature = original_temp
            generator.top_p = original_top_p
            generator.max_new_tokens = original_max_tokens
        
        return True
        
    except Exception as e:
        print(f"❌ 测试失败: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    test_single_prompt() 