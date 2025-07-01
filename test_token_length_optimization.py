#!/usr/bin/env python3
"""
Token 长度优化测试
测试不同的 max_new_tokens 设置对响应质量的影响
"""

import sys
import os
from pathlib import Path

# Add parent directory to path
sys.path.append(str(Path(__file__).parent))

def test_token_length_settings():
    """测试不同的 token 长度设置"""
    
    print("🚀 Token 长度优化测试")
    print("测试不同的 max_new_tokens 设置对响应质量的影响")
    print("=" * 60)
    
    try:
        from xlm.components.generator.local_llm_generator import LocalLLMGenerator
        
        # 硬编码测试 Prompt
        test_prompt = """===SYSTEM===
你是一位专业的金融分析师。请基于以下信息回答问题：

**要求：**
1. 回答简洁，控制在2-3句话内
2. 只包含核心信息
3. 用中文回答
4. 句子要完整

===USER===
Apple Inc. reported Q3 2023 revenue of $81.8 billion, down 1.4% year-over-year. 
iPhone sales increased 2.8% to $39.7 billion. The company's services revenue 
grew 8.2% to $21.2 billion, while Mac and iPad sales declined.

问题：How did Apple perform in Q3 2023?

回答：==="""
        
        # 不同的 token 设置
        token_settings = [
            {"name": "当前设置", "max_new_tokens": 700, "max_total_tokens": 1000},
            {"name": "增加设置", "max_new_tokens": 1000, "max_total_tokens": 1500},
            {"name": "保守设置", "max_new_tokens": 500, "max_total_tokens": 800},
            {"name": "激进设置", "max_new_tokens": 1500, "max_total_tokens": 2000},
        ]
        
        results = []
        
        for setting in token_settings:
            print(f"\n=== 测试 {setting['name']} ===")
            print(f"max_new_tokens: {setting['max_new_tokens']}")
            print(f"max_total_tokens: {setting['max_total_tokens']}")
            
            try:
                # 创建生成器实例，临时修改配置
                generator = LocalLLMGenerator(device="cuda:1")
                
                # 临时修改配置
                original_max_new_tokens = generator.max_new_tokens
                original_max_total_tokens = getattr(generator.config.generator, 'max_total_tokens', 1000)
                
                generator.max_new_tokens = setting['max_new_tokens']
                generator.config.generator.max_total_tokens = setting['max_total_tokens']
                
                print(f"✅ 生成器配置更新成功")
                print(f"   原始 max_new_tokens: {original_max_new_tokens}")
                print(f"   当前 max_new_tokens: {generator.max_new_tokens}")
                
                # 生成响应
                print("🚀 开始生成...")
                responses = generator.generate([test_prompt])
                response = responses[0] if responses else "生成失败"
                
                print(f"响应: {response}")
                
                # 评估响应
                length = len(response.strip())
                has_chinese = any('\u4e00' <= char <= '\u9fff' for char in response)
                has_english = any(char.isalpha() and ord(char) < 128 for char in response)
                is_complete = response.strip().endswith(("。", "！", "？", ".", "!", "?"))
                
                # 检查关键信息
                key_terms = ["Apple", "revenue", "billion", "iPhone", "sales", "Q3"]
                found_terms = [term for term in key_terms if term.lower() in response.lower()]
                
                # 检查语言一致性（区分公司名称和回答语言）
                # 公司名称应该保持原样，回答语言应该与查询语言一致
                company_names = ["Apple", "iPhone", "Mac", "iPad"]  # 英文公司/产品名称
                chinese_company_names = ["德赛电池", "用友网络", "首钢股份"]  # 中文公司名称
                
                # 检查是否包含英文公司名称（这是正常的）
                has_english_company = any(name in response for name in company_names)
                # 检查是否包含中文字符（表示回答用中文）
                has_chinese_answer = has_chinese
                
                # 语言一致性评分：英文查询应该得到中文回答，但可以包含英文公司名称
                language_consistent = has_chinese_answer  # 只要包含中文字符就算语言一致
                
                # 评分
                score = 0
                if 20 <= length <= 200: score += 25
                if language_consistent: score += 25  # 修改：只要包含中文回答就算一致
                if len(found_terms) >= 3: score += 25
                if is_complete: score += 25
                
                results.append({
                    "name": setting['name'],
                    "max_new_tokens": setting['max_new_tokens'],
                    "max_total_tokens": setting['max_total_tokens'],
                    "response": response,
                    "length": length,
                    "language_consistent": language_consistent,
                    "is_complete": is_complete,
                    "key_terms_found": len(found_terms),
                    "score": score
                })
                
                print(f"评分: {score}/100")
                print(f"长度: {length} 字符")
                print(f"包含中文字符: {'是' if has_chinese else '否'}")
                print(f"包含英文字符: {'是' if has_english else '否'}")
                print(f"包含英文公司名称: {'是' if has_english_company else '否'}")
                print(f"语言一致: {'是' if language_consistent else '否'} (中文回答)")
                print(f"句子完整: {'是' if is_complete else '否'}")
                print(f"关键信息: {found_terms}")
                
                # 恢复原始配置
                generator.max_new_tokens = original_max_new_tokens
                generator.config.generator.max_total_tokens = original_max_total_tokens
                
            except Exception as e:
                print(f"❌ {setting['name']} 测试失败: {e}")
                results.append({
                    "name": setting['name'],
                    "max_new_tokens": setting['max_new_tokens'],
                    "max_total_tokens": setting['max_total_tokens'],
                    "response": "测试失败",
                    "length": 0,
                    "language_consistent": False,
                    "is_complete": False,
                    "key_terms_found": 0,
                    "score": 0
                })
        
        # 总结结果
        print(f"\n=== 测试结果总结 ===")
        print("-" * 80)
        for result in results:
            status = "✅" if result["score"] >= 75 else "⚠️" if result["score"] >= 50 else "❌"
            print(f"{status} {result['name']}: {result['score']}/100")
            print(f"   max_new_tokens: {result['max_new_tokens']}, max_total_tokens: {result['max_total_tokens']}")
            print(f"   响应长度: {result['length']} 字符")
            print(f"   语言一致: {'是' if result['language_consistent'] else '否'}")
            print(f"   句子完整: {'是' if result['is_complete'] else '否'}")
            print(f"   关键信息: {result['key_terms_found']} 个")
            print()
        
        # 推荐最佳设置
        best_result = max(results, key=lambda x: x['score'])
        print(f"🎯 推荐设置: {best_result['name']}")
        print(f"   max_new_tokens: {best_result['max_new_tokens']}")
        print(f"   max_total_tokens: {best_result['max_total_tokens']}")
        print(f"   评分: {best_result['score']}/100")
        
        return True
        
    except Exception as e:
        print(f"❌ Token 长度测试失败: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_sentence_completion():
    """测试句子完整性检测机制"""
    
    print("\n=== 句子完整性检测测试 ===")
    print("=" * 50)
    
    try:
        from xlm.components.generator.local_llm_generator import LocalLLMGenerator
        
        # 初始化生成器
        generator = LocalLLMGenerator(device="cuda:1")
        
        # 测试 Prompt
        test_prompt = """===SYSTEM===
你是一位金融分析师。请回答以下问题：

**要求：**
1. 回答要完整，句子要结束
2. 控制在2句话内
3. 用中文回答

===USER===
Apple Inc. reported Q3 2023 revenue of $81.8 billion.

问题：How did Apple perform in Q3 2023?

回答：==="""
        
        print("1. 启用句子完整性检测...")
        generator.config.generator.enable_sentence_completion = True
        generator.config.generator.max_completion_attempts = 3
        generator.config.generator.token_increment = 100
        
        print("🚀 开始生成...")
        responses = generator.generate([test_prompt])
        response_with_completion = responses[0] if responses else "生成失败"
        
        print(f"启用完整性检测的响应: {response_with_completion}")
        
        print("\n2. 禁用句子完整性检测...")
        generator.config.generator.enable_sentence_completion = False
        
        print("🚀 开始生成...")
        responses = generator.generate([test_prompt])
        response_without_completion = responses[0] if responses else "生成失败"
        
        print(f"禁用完整性检测的响应: {response_without_completion}")
        
        # 比较结果
        print(f"\n3. 结果比较:")
        print(f"启用完整性检测: {len(response_with_completion)} 字符")
        print(f"禁用完整性检测: {len(response_without_completion)} 字符")
        
        is_complete_with = response_with_completion.strip().endswith(("。", "！", "？", ".", "!", "?"))
        is_complete_without = response_without_completion.strip().endswith(("。", "！", "？", ".", "!", "?"))
        
        print(f"启用完整性检测 - 句子完整: {'是' if is_complete_with else '否'}")
        print(f"禁用完整性检测 - 句子完整: {'是' if is_complete_without else '否'}")
        
        return True
        
    except Exception as e:
        print(f"❌ 句子完整性检测测试失败: {e}")
        import traceback
        traceback.print_exc()
        return False

def main():
    """主测试函数"""
    
    # 测试不同的 token 长度设置
    test_token_length_settings()
    
    # 测试句子完整性检测
    test_sentence_completion()
    
    print("\n🎉 Token 长度优化测试完成！")

if __name__ == "__main__":
    main() 