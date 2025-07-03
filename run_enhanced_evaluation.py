#!/usr/bin/env python3
"""
极致强化后的评估运行脚本
测试clean_llm_response和evaluate_answer_quality的优化效果
"""

import sys
import os
from pathlib import Path

# 添加项目根目录到路径
sys.path.append(str(Path(__file__).parent))

def run_enhanced_evaluation():
    """运行极致强化后的评估"""
    print("🚀 开始运行极致强化后的评估...")
    
    # 导入评估模块
    try:
        from comprehensive_evaluation import ComprehensiveEvaluator
        print("✅ 成功导入ComprehensiveEvaluator")
    except ImportError as e:
        print(f"❌ 导入失败: {e}")
        return
    
    # 创建评估器
    evaluator = ComprehensiveEvaluator()
    
    # 运行小规模测试
    print("\n📊 运行小规模测试 (50个样本)...")
    test_results = evaluator.run_comprehensive_evaluation(50)
    
    # 打印结果摘要
    evaluator.print_analysis_summary(test_results["analysis"])
    
    # 保存结果
    import json
    with open("enhanced_evaluation_results.json", "w", encoding="utf-8") as f:
        json.dump(test_results, f, ensure_ascii=False, indent=2)
    
    print(f"\n✅ 评估结果已保存到 enhanced_evaluation_results.json")
    
    # 分析格式违规情况
    format_violations_count = 0
    total_samples = len(test_results["results"])
    
    for result in test_results["results"]:
        violations = result.get("evaluation", {}).get("format_violations", [])
        if violations:
            format_violations_count += 1
            print(f"\n⚠️ 样本 {result.get('sample_id')} 存在格式违规:")
            for violation in violations:
                print(f"   - {violation}")
    
    print(f"\n📊 格式违规统计:")
    print(f"   总样本数: {total_samples}")
    print(f"   存在格式违规的样本数: {format_violations_count}")
    print(f"   格式违规率: {format_violations_count/total_samples*100:.1f}%")

def run_failure_analysis():
    """运行失败模式分析"""
    print("\n🔍 运行失败模式分析...")
    
    try:
        from analyze_failure_patterns import main as analyze_failures
        analyze_failures()
    except ImportError as e:
        print(f"❌ 失败模式分析导入失败: {e}")
        print("请先运行 enhanced_evaluation_results.json 生成评估结果")

def main():
    """主函数"""
    print("="*80)
    print("🎯 极致强化后的评估测试")
    print("="*80)
    
    # 运行评估
    run_enhanced_evaluation()
    
    # 运行失败分析
    run_failure_analysis()
    
    print("\n" + "="*80)
    print("✅ 极致强化评估完成")
    print("="*80)

if __name__ == "__main__":
    main() 