#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
调试LLM Judge输出
找出为什么评分总是0
"""

import sys
import os
import json
import re
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

def debug_llm_judge():
    """调试LLM Judge输出"""
    print("🔍 调试LLM Judge输出...")
    
    try:
        from llm_comparison.chinese_llm_judge import llm_judge_singleton
        
        # 初始化LLM Judge
        llm_judge_singleton.initialize()
        
        # 测试用例
        query = "2023年公司营收增长情况如何？"
        expected_answer = "根据2023年财报，公司营收增长20%，净利润达到5000万元。"
        model_final_answer = "根据2018年财报，公司营收增长20%，净利润达到5000万元。"
        
        print(f"问题: {query}")
        print(f"期望答案: {expected_answer}")
        print(f"模型答案: {model_final_answer}")
        
        # 执行评估
        print("\n🤖 执行LLM Judge评估...")
        result = llm_judge_singleton.evaluate(query, expected_answer, model_final_answer)
        
        print(f"\n📊 评估结果:")
        print(f"准确性: {result.get('accuracy', 'N/A')}")
        print(f"简洁性: {result.get('conciseness', 'N/A')}")
        print(f"专业性: {result.get('professionalism', 'N/A')}")
        print(f"总体评分: {result.get('overall_score', 'N/A')}")
        print(f"推理: {result.get('reasoning', 'N/A')}")
        
        # 检查原始输出
        raw_output = result.get('raw_output', '')
        print(f"\n🔍 原始输出 (长度: {len(raw_output)}):")
        print(f"'{raw_output}'")
        
        # 检查是否包含JSON
        if '{' in raw_output and '}' in raw_output:
            print("✅ 输出包含JSON结构")
            try:
                json_start = raw_output.find('{')
                json_end = raw_output.rfind('}') + 1
                json_str = raw_output[json_start:json_end]
                parsed_json = json.loads(json_str)
                print(f"✅ JSON解析成功: {parsed_json}")
            except json.JSONDecodeError as e:
                print(f"❌ JSON解析失败: {e}")
        else:
            print("❌ 输出不包含JSON结构")
            
            # 尝试从文本中提取评分
            accuracy_match = re.search(r'准确性[：:]\s*(\d+)', raw_output)
            conciseness_match = re.search(r'简洁性[：:]\s*(\d+)', raw_output)
            professionalism_match = re.search(r'专业性[：:]\s*(\d+)', raw_output)
            
            if accuracy_match:
                print(f"✅ 从文本提取到准确性: {accuracy_match.group(1)}")
            if conciseness_match:
                print(f"✅ 从文本提取到简洁性: {conciseness_match.group(1)}")
            if professionalism_match:
                print(f"✅ 从文本提取到专业性: {professionalism_match.group(1)}")
        
        print("\n🎯 调试完成！")
        
    except Exception as e:
        print(f"❌ 调试失败: {str(e)}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    debug_llm_judge() 