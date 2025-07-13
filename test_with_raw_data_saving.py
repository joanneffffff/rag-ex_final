#!/usr/bin/env python3
"""
测试原始数据保存功能
展示如何每10个数据保存一次原始数据
"""

import sys
from pathlib import Path

# 添加项目根目录到Python路径
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

from test_rag_system_e2e_multilingual import test_single_dataset


def test_chinese_with_raw_data_saving():
    """测试中文数据集并保存原始数据"""
    print("🚀 测试中文数据集并保存原始数据...")
    
    chinese_data_path = "data/alphafin/alphafin_eval_samples_updated.jsonl"
    
    if not Path(chinese_data_path).exists():
        print(f"❌ 中文数据文件不存在: {chinese_data_path}")
        return None
    
    try:
        summary = test_single_dataset(
            data_path=chinese_data_path,
            sample_size=25,  # 测试25个样本，会产生3个批次（10+10+5）
            enable_reranker=True,
            enable_stock_prediction=False
        )
        
        print("✅ 中文数据集测试完成！")
        print(f"   原始数据已保存到: raw_data_alphafin_eval_samples_updated/")
        return summary
        
    except Exception as e:
        print(f"❌ 中文数据集测试失败: {e}")
        return None


def test_english_with_raw_data_saving():
    """测试英文数据集并保存原始数据"""
    print("🚀 测试英文数据集并保存原始数据...")
    
    english_data_path = "evaluate_mrr/tatqa_eval_balanced_100.jsonl"
    
    if not Path(english_data_path).exists():
        print(f"❌ 英文数据文件不存在: {english_data_path}")
        return None
    
    try:
        summary = test_single_dataset(
            data_path=english_data_path,
            sample_size=25,  # 测试25个样本，会产生3个批次（10+10+5）
            enable_reranker=True,
            enable_stock_prediction=False
        )
        
        print("✅ 英文数据集测试完成！")
        print(f"   原始数据已保存到: raw_data_tatqa_eval_balanced_100/")
        return summary
        
    except Exception as e:
        print(f"❌ 英文数据集测试失败: {e}")
        return None


def show_raw_data_format():
    """显示原始数据格式"""
    print("📋 原始数据格式说明:")
    print("="*60)
    
    raw_data_example = {
        "sample_id": 0,
        "query": "这个股票的下月最终收益结果是？",
        "context": "<div>检索到的上下文信息...</div>",
        "answer": "这个股票的下月最终收益结果是:'涨',上涨/下跌概率:极大",
        "expected_answer": "这个股票的下月最终收益结果是:'涨',上涨/下跌概率:极大",
        "em": 1.0,
        "f1": 1.0,
        "processing_time": 6.23,
        "success": True,
        "language": "chinese"
    }
    
    print("每个原始数据记录包含以下字段:")
    for key, value in raw_data_example.items():
        print(f"   {key}: {type(value).__name__} - {value}")
    
    print("\n📁 保存位置:")
    print("   - 中文数据: raw_data_alphafin_eval_samples_updated/")
    print("   - 英文数据: raw_data_tatqa_eval_balanced_100/")
    print("   - 文件格式: batch_001.json, batch_002.json, ...")
    print("   - 每批次包含10个数据记录（最后一批可能少于10个）")
    
    print("="*60)


def main():
    """主函数"""
    print("🎯 RAG系统原始数据保存测试")
    print("="*60)
    
    # 显示原始数据格式
    show_raw_data_format()
    print()
    
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
    print("1️⃣ 测试中文数据集并保存原始数据...")
    chinese_result = test_chinese_with_raw_data_saving()
    
    if chinese_result:
        print(f"   平均F1-score: {chinese_result['average_f1_score']:.4f}")
        print(f"   平均Exact Match: {chinese_result['average_exact_match']:.4f}")
        print(f"   成功率: {chinese_result['success_rate']:.2%}")
    else:
        print("❌ 中文数据集测试失败")
    
    print()
    
    # 测试英文数据集
    print("2️⃣ 测试英文数据集并保存原始数据...")
    english_result = test_english_with_raw_data_saving()
    
    if english_result:
        print(f"   平均F1-score: {english_result['average_f1_score']:.4f}")
        print(f"   平均Exact Match: {english_result['average_exact_match']:.4f}")
        print(f"   成功率: {english_result['success_rate']:.2%}")
    else:
        print("❌ 英文数据集测试失败")
    
    print("\n🎉 所有数据集测试完成！")
    print("📁 原始数据已保存到相应的目录中")


if __name__ == "__main__":
    main() 