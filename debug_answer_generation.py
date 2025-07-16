#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
调试答案生成和评估过程
找出为什么LLM Judge评分为0和F1/EM为1的问题
"""

import sys
import os
import json
from typing import List, Dict, Any
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

def debug_answer_generation():
    """调试答案生成过程"""
    print("🔍 调试答案生成过程...")
    
    try:
        from rag_perturbation_experiment import RAGPerturbationExperiment
        
        # 初始化实验系统
        experiment = RAGPerturbationExperiment()
        
        # 测试样本
        test_context = "2023年公司营收增长20%，净利润达到5000万元。"
        test_question = "2023年公司营收增长情况如何？"
        
        print(f"测试上下文: {test_context}")
        print(f"测试问题: {test_question}")
        
        # 获取原始答案
        print("\n🔍 获取原始答案...")
        original_answer = experiment.get_original_answer(test_context, test_question)
        print(f"原始答案: {original_answer}")
        print(f"原始答案长度: {len(original_answer)}")
        
        if not original_answer or original_answer.strip() == "":
            print("❌ 原始答案为空！")
            return False
        
        # 应用扰动
        print("\n🔍 应用扰动...")
        perturbations = experiment.apply_perturbation(test_context, "year")
        
        if not perturbations:
            print("❌ 没有生成扰动！")
            return False
        
        for i, perturbation in enumerate(perturbations):
            print(f"\n--- 扰动 {i+1} ---")
            print(f"扰动后上下文: {perturbation.perturbed_text}")
            
            # 获取扰动后答案
            print(f"🔍 获取扰动后答案...")
            perturbed_answer = experiment.get_perturbed_answer(perturbation.perturbed_text, test_question, "year")
            print(f"扰动后答案: {perturbed_answer}")
            print(f"扰动后答案长度: {len(perturbed_answer)}")
            
            if not perturbed_answer or perturbed_answer.strip() == "":
                print("❌ 扰动后答案为空！")
                continue
            
            # 计算评估指标
            print(f"\n🔍 计算评估指标...")
            similarity_score, importance_score, f1_score, em_score = experiment.calculate_importance_score(original_answer, perturbed_answer)
            
            print(f"相似度: {similarity_score}")
            print(f"重要性: {importance_score}")
            print(f"F1分数: {f1_score}")
            print(f"EM分数: {em_score}")
            
            # LLM Judge评估
            print(f"\n🔍 LLM Judge评估...")
            judge_result = experiment.run_llm_judge_evaluation(original_answer, perturbed_answer, test_question)
            
            print(f"LLM Judge结果: {judge_result}")
            
            # 检查答案是否真的不同
            if original_answer == perturbed_answer:
                print("⚠️ 原始答案和扰动后答案相同！")
            else:
                print("✅ 答案确实发生了变化")
            
            break  # 只测试第一个扰动
        
        return True
        
    except Exception as e:
        print(f"❌ 调试失败: {str(e)}")
        import traceback
        traceback.print_exc()
        return False

def debug_llm_judge():
    """调试LLM Judge评估"""
    print("\n🔍 调试LLM Judge评估...")
    
    try:
        from llm_comparison.chinese_llm_judge import SingletonLLMJudge
        
        # 初始化LLM Judge
        judge = SingletonLLMJudge()
        judge.initialize(model_name="Qwen3-8B", device="cuda:1")
        
        # 测试评估
        query = "2023年公司营收增长情况如何？"
        expected_answer = "2023年公司营收增长20%。"
        model_answer = "根据报告显示，2023年公司营收增长20%，净利润达到5000万元。"
        
        print(f"查询: {query}")
        print(f"期望答案: {expected_answer}")
        print(f"模型答案: {model_answer}")
        
        result = judge.evaluate(query, expected_answer, model_answer)
        
        print(f"LLM Judge评估结果:")
        print(f"  准确性: {result.get('accuracy', 'N/A')}")
        print(f"  简洁性: {result.get('conciseness', 'N/A')}")
        print(f"  专业性: {result.get('professionalism', 'N/A')}")
        print(f"  总体评分: {result.get('overall_score', 'N/A')}")
        print(f"  推理: {result.get('reasoning', 'N/A')}")
        print(f"  原始输出: {result.get('raw_output', 'N/A')[:200]}...")
        
        # 清理资源
        judge.cleanup()
        
        return True
        
    except Exception as e:
        print(f"❌ LLM Judge调试失败: {str(e)}")
        import traceback
        traceback.print_exc()
        return False

def debug_f1_em_calculation():
    """调试F1和EM计算"""
    print("\n🔍 调试F1和EM计算...")
    
    try:
        from rag_perturbation_experiment import RAGPerturbationExperiment
        
        # 初始化实验系统
        experiment = RAGPerturbationExperiment()
        
        # 测试案例
        test_cases = [
            {
                "original": "2023年公司营收增长20%。",
                "perturbed": "2023年公司营收增长20%。",
                "description": "相同答案"
            },
            {
                "original": "2023年公司营收增长20%。",
                "perturbed": "2022年公司营收增长20%。",
                "description": "年份不同"
            },
            {
                "original": "2023年公司营收增长20%。",
                "perturbed": "",
                "description": "空答案"
            },
            {
                "original": "",
                "perturbed": "2023年公司营收增长20%。",
                "description": "原始答案为空"
            }
        ]
        
        for i, case in enumerate(test_cases):
            print(f"\n--- 测试案例 {i+1}: {case['description']} ---")
            print(f"原始答案: '{case['original']}'")
            print(f"扰动答案: '{case['perturbed']}'")
            
            f1_score = experiment.calculate_f1_score(case['original'], case['perturbed'])
            em_score = experiment.calculate_exact_match(case['original'], case['perturbed'])
            
            print(f"F1分数: {f1_score}")
            print(f"EM分数: {em_score}")
            
            # 检查归一化结果
            normalized_original = experiment.normalize_answer_chinese(case['original'])
            normalized_perturbed = experiment.normalize_answer_chinese(case['perturbed'])
            
            print(f"归一化原始答案: '{normalized_original}'")
            print(f"归一化扰动答案: '{normalized_perturbed}'")
        
        return True
        
    except Exception as e:
        print(f"❌ F1/EM计算调试失败: {str(e)}")
        import traceback
        traceback.print_exc()
        return False

def main():
    """主调试函数"""
    print("🚀 开始调试答案生成和评估过程...")
    
    # 调试答案生成
    answer_success = debug_answer_generation()
    
    # 调试LLM Judge
    judge_success = debug_llm_judge()
    
    # 调试F1/EM计算
    f1_em_success = debug_f1_em_calculation()
    
    # 总结
    print(f"\n{'='*60}")
    print(f"📊 调试结果总结")
    print(f"{'='*60}")
    
    results = [
        ("答案生成", answer_success),
        ("LLM Judge", judge_success),
        ("F1/EM计算", f1_em_success)
    ]
    
    for test_name, success in results:
        status = "✅ 成功" if success else "❌ 失败"
        print(f"{test_name}: {status}")
    
    if all(success for _, success in results):
        print("\n🎉 所有调试通过！")
    else:
        print("\n⚠️ 部分调试失败，请检查相关组件。")

if __name__ == "__main__":
    main() 