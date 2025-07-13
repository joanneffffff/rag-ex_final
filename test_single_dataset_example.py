#!/usr/bin/env python3
"""
单数据集测试使用示例
展示如何分别测试中文和英文数据集
"""

import sys
from pathlib import Path

# 添加项目根目录到Python路径
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

from test_rag_system_e2e_multilingual import test_single_dataset


def test_chinese_dataset():
    """测试中文数据集"""
    print("🚀 测试中文数据集...")
    
    chinese_data_path = "data/alphafin/alphafin_eval_samples_updated.jsonl"
    
    if not Path(chinese_data_path).exists():
        print(f"❌ 中文数据文件不存在: {chinese_data_path}")
        return None
    
    try:
        summary = test_single_dataset(
            data_path=chinese_data_path,
            sample_size=20,  # 测试20个样本
            enable_reranker=True,
            enable_stock_prediction=False
        )
        
        print("✅ 中文数据集测试完成！")
        return summary
        
    except Exception as e:
        print(f"❌ 中文数据集测试失败: {e}")
        return None


def test_english_dataset():
    """测试英文数据集"""
    print("🚀 测试英文数据集...")
    
    english_data_path = "evaluate_mrr/tatqa_eval_balanced_100.jsonl"
    
    if not Path(english_data_path).exists():
        print(f"❌ 英文数据文件不存在: {english_data_path}")
        return None
    
    try:
        summary = test_single_dataset(
            data_path=english_data_path,
            sample_size=20,  # 测试20个样本
            enable_reranker=True,
            enable_stock_prediction=False
        )
        
        print("✅ 英文数据集测试完成！")
        return summary
        
    except Exception as e:
        print(f"❌ 英文数据集测试失败: {e}")
        return None


def test_chinese_with_stock_prediction():
    """测试中文数据集（启用股票预测）"""
    print("🚀 测试中文数据集（启用股票预测）...")
    
    chinese_data_path = "data/alphafin/alphafin_eval_samples_updated.jsonl"
    
    if not Path(chinese_data_path).exists():
        print(f"❌ 中文数据文件不存在: {chinese_data_path}")
        return None
    
    try:
        summary = test_single_dataset(
            data_path=chinese_data_path,
            sample_size=10,  # 测试10个样本
            enable_reranker=True,
            enable_stock_prediction=True
        )
        
        print("✅ 中文数据集股票预测测试完成！")
        return summary
        
    except Exception as e:
        print(f"❌ 中文数据集股票预测测试失败: {e}")
        return None


def main():
    """主函数"""
    print("🎯 RAG系统单数据集测试示例")
    print("="*60)
    
    # 检查数据文件
    chinese_data_path = "data/alphafin/alphafin_eval_samples_updated.jsonl"
    english_data_path = "evaluate_mrr/tatqa_eval_balanced_100.jsonl"
    
    if not Path(chinese_data_path).exists():
        print(f"❌ 中文数据文件不存在: {chinese_data_path}")
        return
    
    if not Path(english_data_path).exists():
        print(f"❌ 英文数据文件不存在: {english_data_path}")
        return
    
    print(f"📁 中文数据文件: {chinese_data_path}")
    print(f"📁 英文数据文件: {english_data_path}")
    print()
    
    # 测试中文数据集
    print("1️⃣ 测试中文数据集...")
    chinese_result = test_chinese_dataset()
    
    if chinese_result:
        print(f"   平均F1-score: {chinese_result['average_f1_score']:.4f}")
        print(f"   平均Exact Match: {chinese_result['average_exact_match']:.4f}")
        print(f"   成功率: {chinese_result['success_rate']:.2%}")
    else:
        print("❌ 中文数据集测试失败")
    
    print()
    
    # 测试英文数据集
    print("2️⃣ 测试英文数据集...")
    english_result = test_english_dataset()
    
    if english_result:
        print(f"   平均F1-score: {english_result['average_f1_score']:.4f}")
        print(f"   平均Exact Match: {english_result['average_exact_match']:.4f}")
        print(f"   成功率: {english_result['success_rate']:.2%}")
    else:
        print("❌ 英文数据集测试失败")
    
    print()
    
    # 测试中文数据集（启用股票预测）
    print("3️⃣ 测试中文数据集（启用股票预测）...")
    stock_result = test_chinese_with_stock_prediction()
    
    if stock_result:
        print(f"   平均F1-score: {stock_result['average_f1_score']:.4f}")
        print(f"   平均Exact Match: {stock_result['average_exact_match']:.4f}")
        print(f"   成功率: {stock_result['success_rate']:.2%}")
    else:
        print("❌ 中文数据集股票预测测试失败")
    
    print("\n🎉 所有数据集测试完成！")


if __name__ == "__main__":
    main() 