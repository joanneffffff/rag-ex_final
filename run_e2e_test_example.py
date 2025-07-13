#!/usr/bin/env python3
"""
端到端测试使用示例
展示如何运行RAG系统的端到端测试
"""

import sys
from pathlib import Path

# 添加项目根目录到Python路径
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

from test_rag_system_e2e import run_e2e_test


def run_basic_test():
    """运行基础端到端测试"""
    print("🚀 运行基础端到端测试...")
    
    # 使用示例数据文件（请根据实际情况修改路径）
    data_path = "data/alphafin/alphafin_eval_samples_updated.jsonl"
    output_path = "e2e_test_results_basic.json"
    
    try:
        test_summary = run_e2e_test(
            data_path=data_path,
            output_path=output_path,
            sample_size=10,  # 只测试10个样本
            enable_reranker=True,
            enable_stock_prediction=False
        )
        
        print("✅ 基础测试完成！")
        return test_summary
        
    except Exception as e:
        print(f"❌ 基础测试失败: {e}")
        return None


def run_stock_prediction_test():
    """运行股票预测端到端测试"""
    print("🚀 运行股票预测端到端测试...")
    
    # 使用示例数据文件（请根据实际情况修改路径）
    data_path = "data/alphafin/alphafin_eval_samples_updated.jsonl"
    output_path = "e2e_test_results_stock_prediction.json"
    
    try:
        test_summary = run_e2e_test(
            data_path=data_path,
            output_path=output_path,
            sample_size=10,  # 只测试10个样本
            enable_reranker=True,
            enable_stock_prediction=True
        )
        
        print("✅ 股票预测测试完成！")
        return test_summary
        
    except Exception as e:
        print(f"❌ 股票预测测试失败: {e}")
        return None


def run_comparison_test():
    """运行对比测试：有重排序器 vs 无重排序器"""
    print("🚀 运行重排序器对比测试...")
    
    data_path = "data/alphafin/alphafin_eval_samples_updated.jsonl"
    
    # 测试1：启用重排序器
    print("📊 测试1：启用重排序器")
    test_with_reranker = run_e2e_test(
        data_path=data_path,
        output_path="e2e_test_results_with_reranker.json",
        sample_size=10,
        enable_reranker=True,
        enable_stock_prediction=False
    )
    
    # 测试2：禁用重排序器
    print("📊 测试2：禁用重排序器")
    test_without_reranker = run_e2e_test(
        data_path=data_path,
        output_path="e2e_test_results_without_reranker.json",
        sample_size=10,
        enable_reranker=False,
        enable_stock_prediction=False
    )
    
    # 输出对比结果
    print("\n" + "="*60)
    print("📊 重排序器对比结果")
    print("="*60)
    print(f"{'指标':<20} {'启用重排序器':<15} {'禁用重排序器':<15}")
    print("-" * 60)
    print(f"{'平均F1-score':<20} {test_with_reranker['overall_metrics']['average_f1_score']:<15.4f} {test_without_reranker['overall_metrics']['average_f1_score']:<15.4f}")
    print(f"{'平均Exact Match':<20} {test_with_reranker['overall_metrics']['average_exact_match']:<15.4f} {test_without_reranker['overall_metrics']['average_exact_match']:<15.4f}")
    print(f"{'成功率':<20} {test_with_reranker['overall_metrics']['success_rate']:<15.2%} {test_without_reranker['overall_metrics']['success_rate']:<15.2%}")
    print(f"{'平均处理时间':<20} {test_with_reranker['overall_metrics']['average_processing_time']:<15.2f} {test_without_reranker['overall_metrics']['average_processing_time']:<15.2f}")
    print("="*60)
    
    return test_with_reranker, test_without_reranker


def main():
    """主函数"""
    print("🎯 RAG系统端到端测试示例")
    print("="*50)
    
    # 检查数据文件是否存在
    data_path = "data/alphafin/alphafin_eval_samples_updated.jsonl"
    if not Path(data_path).exists():
        print(f"❌ 数据文件不存在: {data_path}")
        print("请确保数据文件存在，或修改脚本中的data_path变量")
        return
    
    print(f"📁 使用数据文件: {data_path}")
    print()
    
    # 运行基础测试
    print("1️⃣ 运行基础端到端测试...")
    basic_result = run_basic_test()
    
    if basic_result:
        print("✅ 基础测试成功")
        print(f"   平均F1-score: {basic_result['overall_metrics']['average_f1_score']:.4f}")
        print(f"   平均Exact Match: {basic_result['overall_metrics']['average_exact_match']:.4f}")
        print(f"   成功率: {basic_result['overall_metrics']['success_rate']:.2%}")
    else:
        print("❌ 基础测试失败")
    
    print()
    
    # 运行股票预测测试
    print("2️⃣ 运行股票预测端到端测试...")
    stock_result = run_stock_prediction_test()
    
    if stock_result:
        print("✅ 股票预测测试成功")
        print(f"   平均F1-score: {stock_result['overall_metrics']['average_f1_score']:.4f}")
        print(f"   平均Exact Match: {stock_result['overall_metrics']['average_exact_match']:.4f}")
        print(f"   成功率: {stock_result['overall_metrics']['success_rate']:.2%}")
    else:
        print("❌ 股票预测测试失败")
    
    print()
    
    # 运行对比测试
    print("3️⃣ 运行重排序器对比测试...")
    try:
        reranker_result, no_reranker_result = run_comparison_test()
        print("✅ 对比测试成功")
    except Exception as e:
        print(f"❌ 对比测试失败: {e}")
    
    print("\n🎉 所有测试完成！")
    print("📁 详细结果已保存到相应的JSON文件中")


if __name__ == "__main__":
    main() 