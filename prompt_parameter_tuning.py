#!/usr/bin/env python3
"""
Generator LLM Prompt 和参数调优测试
固定测试问题：德赛电池（000049）2021年利润持续增长的主要原因是什么？
"""

import sys
import os
from pathlib import Path
import json
import time

# Add parent directory to path
sys.path.append(str(Path(__file__).parent))

def test_prompt_variations():
    """测试不同的 Prompt 变体"""
    
    print("=== Generator LLM Prompt 调优测试 ===")
    print("测试问题：德赛电池（000049）2021年利润持续增长的主要原因是什么？")
    print("=" * 70)
    
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
    
    # 不同的 Prompt 变体
    prompt_variations = {
        "简洁版": f"""你是一位金融分析师。请基于以下信息回答问题：

摘要：{test_summary}

详细内容：{test_context}

问题：{test_query}

回答：""",
        
        "详细版": f"""你是一位专业的金融分析师，擅长分析公司财务报告。

请基于以下公司财务报告信息，准确回答用户问题：

【财务报告摘要】
{test_summary}

【详细财务数据】
{test_context}

【用户问题】
{test_query}

请提供准确、简洁的分析回答：""",
        
        "指令版": f"""你是一位金融分析师。请严格按照以下要求回答：

要求：
1. 基于提供的财务信息回答
2. 回答简洁，控制在2-3句话内
3. 如果信息不足，回答"根据现有信息，无法提供此项信息。"
4. 不要包含任何格式标记或额外说明

信息：
{test_summary}

{test_context}

问题：{test_query}

回答：""",
        
        "问答版": f"""基于以下财务信息回答问题：

{test_summary}

{test_context}

问题：{test_query}

答案：""",
        
        "分析版": f"""作为金融分析师，请分析以下财务数据并回答问题：

财务摘要：{test_summary}

详细数据：{test_context}

分析问题：{test_query}

分析结果："""
    }
    
    return prompt_variations, test_query

def test_parameter_variations():
    """测试不同的参数组合"""
    
    parameter_sets = {
        "保守型": {
            "temperature": 0.1,
            "top_p": 0.7,
            "max_new_tokens": 200,
            "repetition_penalty": 1.2
        },
        "平衡型": {
            "temperature": 0.2,
            "top_p": 0.8,
            "max_new_tokens": 300,
            "repetition_penalty": 1.3
        },
        "创造性": {
            "temperature": 0.3,
            "top_p": 0.9,
            "max_new_tokens": 400,
            "repetition_penalty": 1.1
        },
        "精确型": {
            "temperature": 0.05,
            "top_p": 0.6,
            "max_new_tokens": 150,
            "repetition_penalty": 1.4
        }
    }
    
    return parameter_sets

def evaluate_response(response, query):
    """评估响应质量"""
    
    # 质量指标
    indicators = {
        "简洁性": {
            "score": 0,
            "max": 25,
            "description": "回答长度适中（50-200字符）"
        },
        "准确性": {
            "score": 0,
            "max": 25,
            "description": "包含关键信息（德赛电池、iPhone、需求等）"
        },
        "纯粹性": {
            "score": 0,
            "max": 25,
            "description": "无格式标记、引导语等"
        },
        "完整性": {
            "score": 0,
            "max": 25,
            "description": "句子完整，有明确结论"
        }
    }
    
    # 简洁性评分
    length = len(response.strip())
    if 50 <= length <= 200:
        indicators["简洁性"]["score"] = 25
    elif 30 <= length <= 300:
        indicators["简洁性"]["score"] = 15
    else:
        indicators["简洁性"]["score"] = 5
    
    # 准确性评分
    key_terms = ["德赛电池", "iPhone", "需求", "增长", "利润", "业绩"]
    found_terms = sum(1 for term in key_terms if term in response)
    indicators["准确性"]["score"] = min(25, found_terms * 4)
    
    # 纯粹性评分
    unwanted_patterns = ["【", "】", "回答：", "Answer:", "---", "===", "___"]
    has_unwanted = any(pattern in response for pattern in unwanted_patterns)
    indicators["纯粹性"]["score"] = 0 if has_unwanted else 25
    
    # 完整性评分
    if response.strip().endswith(("。", "！", "？", ".", "!", "?")):
        indicators["完整性"]["score"] = 25
    elif len(response.strip()) > 20:
        indicators["完整性"]["score"] = 15
    else:
        indicators["完整性"]["score"] = 5
    
    # 计算总分
    total_score = sum(ind["score"] for ind in indicators.values())
    max_score = sum(ind["max"] for ind in indicators.values())
    
    return indicators, total_score, max_score

def run_single_test(generator, prompt, params, test_name):
    """运行单次测试"""
    
    print(f"\n🔍 测试: {test_name}")
    print("-" * 50)
    
    # 临时修改参数
    original_params = {
        "temperature": generator.temperature,
        "top_p": generator.top_p,
        "max_new_tokens": generator.max_new_tokens
    }
    
    try:
        # 应用新参数
        generator.temperature = params.get("temperature", 0.2)
        generator.top_p = params.get("top_p", 0.8)
        generator.max_new_tokens = params.get("max_new_tokens", 300)
        
        print(f"参数: temp={generator.temperature}, top_p={generator.top_p}, max_tokens={generator.max_new_tokens}")
        print(f"Prompt长度: {len(prompt)} 字符")
        
        # 生成响应
        start_time = time.time()
        responses = generator.generate([prompt])
        end_time = time.time()
        
        response = responses[0] if responses else "生成失败"
        generation_time = end_time - start_time
        
        print(f"生成时间: {generation_time:.2f}秒")
        print(f"响应长度: {len(response)} 字符")
        print(f"响应内容: {response}")
        
        return response, generation_time
        
    finally:
        # 恢复原始参数
        generator.temperature = original_params["temperature"]
        generator.top_p = original_params["top_p"]
        generator.max_new_tokens = original_params["max_new_tokens"]

def main():
    """主测试函数"""
    
    try:
        # 导入模块
        from xlm.components.generator.local_llm_generator import LocalLLMGenerator
        
        print("1. 初始化 LLM 生成器...")
        generator = LocalLLMGenerator()
        print(f"✅ 生成器初始化成功: {generator.model_name}")
        
        # 获取测试数据
        prompt_variations, test_query = test_prompt_variations()
        parameter_sets = test_parameter_variations()
        
        # 存储测试结果
        results = []
        
        # 测试所有组合
        for prompt_name, prompt in prompt_variations.items():
            for param_name, params in parameter_sets.items():
                test_name = f"{prompt_name} + {param_name}"
                
                try:
                    response, generation_time = run_single_test(generator, prompt, params, test_name)
                    
                    # 评估质量
                    indicators, total_score, max_score = evaluate_response(response, test_query)
                    
                    # 存储结果
                    result = {
                        "test_name": test_name,
                        "prompt_name": prompt_name,
                        "param_name": param_name,
                        "response": response,
                        "generation_time": generation_time,
                        "total_score": total_score,
                        "max_score": max_score,
                        "score_percentage": (total_score / max_score) * 100,
                        "indicators": indicators
                    }
                    results.append(result)
                    
                    print(f"质量评分: {total_score}/{max_score} ({result['score_percentage']:.1f}%)")
                    
                except Exception as e:
                    print(f"❌ 测试失败: {e}")
                    continue
        
        # 分析结果
        print("\n" + "=" * 70)
        print("📊 测试结果汇总")
        print("=" * 70)
        
        # 按评分排序
        results.sort(key=lambda x: x["score_percentage"], reverse=True)
        
        print("\n🏆 最佳组合:")
        for i, result in enumerate(results[:5]):
            print(f"{i+1}. {result['test_name']}: {result['score_percentage']:.1f}%")
            print(f"   响应: {result['response'][:100]}...")
        
        print("\n📈 详细分析:")
        for result in results:
            print(f"\n{result['test_name']}: {result['score_percentage']:.1f}%")
            print(f"  简洁性: {result['indicators']['简洁性']['score']}/25")
            print(f"  准确性: {result['indicators']['准确性']['score']}/25")
            print(f"  纯粹性: {result['indicators']['纯粹性']['score']}/25")
            print(f"  完整性: {result['indicators']['完整性']['score']}/25")
            print(f"  生成时间: {result['generation_time']:.2f}秒")
        
        # 保存结果
        with open("prompt_tuning_results.json", "w", encoding="utf-8") as f:
            json.dump(results, f, ensure_ascii=False, indent=2)
        
        print(f"\n💾 详细结果已保存到: prompt_tuning_results.json")
        
        return True
        
    except Exception as e:
        print(f"❌ 测试失败: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    main() 