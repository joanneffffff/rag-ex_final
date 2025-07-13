#!/usr/bin/env python3
"""
多语言端到端测试使用示例
展示如何运行RAG系统的多语言端到端测试
"""

import sys
from pathlib import Path

# 添加项目根目录到Python路径
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

from test_rag_system_e2e_multilingual import run_multilingual_e2e_test


def run_basic_multilingual_test():
    """运行基础多语言端到端测试"""
    print("🚀 运行基础多语言端到端测试...")
    
    # 使用您提供的数据集路径
    chinese_data_path = "data/alphafin/alphafin_eval_samples_updated.jsonl"
    english_data_path = "evaluate_mrr/tatqa_eval_balanced_100.jsonl"
    output_dir = "e2e_test_results_multilingual"
    
    try:
        combined_summary = run_multilingual_e2e_test(
            chinese_data_path=chinese_data_path,
            english_data_path=english_data_path,
            output_dir=output_dir,
            sample_size=20,  # 每种语言测试20个样本
            enable_reranker=True,
            enable_stock_prediction=False
        )
        
        print("✅ 基础多语言测试完成！")
        return combined_summary
        
    except Exception as e:
        print(f"❌ 基础多语言测试失败: {e}")
        return None


def run_stock_prediction_multilingual_test():
    """运行股票预测多语言端到端测试"""
    print("🚀 运行股票预测多语言端到端测试...")
    
    chinese_data_path = "data/alphafin/alphafin_eval_samples_updated.jsonl"
    english_data_path = "evaluate_mrr/tatqa_eval_balanced_100.jsonl"
    output_dir = "e2e_test_results_stock_prediction"
    
    try:
        combined_summary = run_multilingual_e2e_test(
            chinese_data_path=chinese_data_path,
            english_data_path=english_data_path,
            output_dir=output_dir,
            sample_size=20,  # 每种语言测试20个样本
            enable_reranker=True,
            enable_stock_prediction=True
        )
        
        print("✅ 股票预测多语言测试完成！")
        return combined_summary
        
    except Exception as e:
        print(f"❌ 股票预测多语言测试失败: {e}")
        return None


def run_full_dataset_test():
    """运行完整数据集测试"""
    print("🚀 运行完整数据集测试...")
    
    chinese_data_path = "data/alphafin/alphafin_eval_samples_updated.jsonl"
    english_data_path = "evaluate_mrr/tatqa_eval_balanced_100.jsonl"
    output_dir = "e2e_test_results_full"
    
    try:
        combined_summary = run_multilingual_e2e_test(
            chinese_data_path=chinese_data_path,
            english_data_path=english_data_path,
            output_dir=output_dir,
            sample_size=None,  # 使用全部数据
            enable_reranker=True,
            enable_stock_prediction=False
        )
        
        print("✅ 完整数据集测试完成！")
        return combined_summary
        
    except Exception as e:
        print(f"❌ 完整数据集测试失败: {e}")
        return None


def main():
    """主函数"""
    print("🎯 RAG系统多语言端到端测试示例")
    print("="*60)
    
    # 检查数据文件是否存在
    chinese_data_path = "data/alphafin/alphafin_eval_samples_updated.jsonl"
    english_data_path = "evaluate_mrr/tatqa_eval_balanced_100.jsonl"
    
    if not Path(chinese_data_path).exists():
        print(f"❌ 中文数据文件不存在: {chinese_data_path}")
        print("请确保数据文件存在，或修改脚本中的chinese_data_path变量")
        return
    
    if not Path(english_data_path).exists():
        print(f"❌ 英文数据文件不存在: {english_data_path}")
        print("请确保数据文件存在，或修改脚本中的english_data_path变量")
        return
    
    print(f"📁 使用中文数据文件: {chinese_data_path}")
    print(f"📁 使用英文数据文件: {english_data_path}")
    print()
    
    # 运行基础多语言测试
    print("1️⃣ 运行基础多语言端到端测试...")
    basic_result = run_basic_multilingual_test()
    
    if basic_result:
        print("✅ 基础多语言测试成功")
        print(f"   加权平均F1-score: {basic_result['weighted_f1_score']:.4f}")
        print(f"   加权平均Exact Match: {basic_result['weighted_exact_match']:.4f}")
        print(f"   整体成功率: {basic_result['overall_success_rate']:.2%}")
        
        # 显示分语言结果
        for language, result in basic_result['language_specific_results'].items():
            print(f"   {result['language_name']}: F1={result['average_f1_score']:.4f}, EM={result['average_exact_match']:.4f}")
    else:
        print("❌ 基础多语言测试失败")
    
    print()
    
    # 运行股票预测多语言测试
    print("2️⃣ 运行股票预测多语言端到端测试...")
    stock_result = run_stock_prediction_multilingual_test()
    
    if stock_result:
        print("✅ 股票预测多语言测试成功")
        print(f"   加权平均F1-score: {stock_result['weighted_f1_score']:.4f}")
        print(f"   加权平均Exact Match: {stock_result['weighted_exact_match']:.4f}")
        print(f"   整体成功率: {stock_result['overall_success_rate']:.2%}")
    else:
        print("❌ 股票预测多语言测试失败")
    
    print()
    
    # 运行完整数据集测试（可选）
    print("3️⃣ 运行完整数据集测试...")
    full_result = run_full_dataset_test()
    
    if full_result:
        print("✅ 完整数据集测试成功")
        print(f"   加权平均F1-score: {full_result['weighted_f1_score']:.4f}")
        print(f"   加权平均Exact Match: {full_result['weighted_exact_match']:.4f}")
        print(f"   整体成功率: {full_result['overall_success_rate']:.2%}")
    else:
        print("❌ 完整数据集测试失败")
    
    print("\n🎉 所有多语言测试完成！")
    print("📁 详细结果已保存到相应的目录中")


if __name__ == "__main__":
    main() 