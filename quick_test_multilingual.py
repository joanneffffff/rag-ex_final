#!/usr/bin/env python3
"""
快速测试多语言端到端测试功能
"""

import sys
from pathlib import Path

# 添加项目根目录到Python路径
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

from test_rag_system_e2e_multilingual import run_multilingual_e2e_test


def quick_test():
    """快速测试多语言端到端测试功能"""
    print("🚀 快速测试多语言端到端测试功能...")
    
    # 检查数据文件
    chinese_data_path = "data/alphafin/alphafin_eval_samples_updated.jsonl"
    english_data_path = "evaluate_mrr/tatqa_eval_balanced_100.jsonl"
    
    if not Path(chinese_data_path).exists():
        print(f"❌ 中文数据文件不存在: {chinese_data_path}")
        return False
    
    if not Path(english_data_path).exists():
        print(f"❌ 英文数据文件不存在: {english_data_path}")
        return False
    
    print(f"✅ 数据文件检查通过")
    print(f"   中文数据: {chinese_data_path}")
    print(f"   英文数据: {english_data_path}")
    
    try:
        # 运行快速测试（每种语言5个样本）
        combined_summary = run_multilingual_e2e_test(
            chinese_data_path=chinese_data_path,
            english_data_path=english_data_path,
            output_dir="quick_test_results",
            sample_size=5,  # 每种语言测试5个样本
            enable_reranker=True,
            enable_stock_prediction=False
        )
        
        print("✅ 快速测试完成！")
        print(f"   加权平均F1-score: {combined_summary['weighted_f1_score']:.4f}")
        print(f"   加权平均Exact Match: {combined_summary['weighted_exact_match']:.4f}")
        print(f"   整体成功率: {combined_summary['overall_success_rate']:.2%}")
        
        return True
        
    except Exception as e:
        print(f"❌ 快速测试失败: {e}")
        return False


if __name__ == "__main__":
    success = quick_test()
    if success:
        print("\n🎉 多语言端到端测试功能验证成功！")
    else:
        print("\n❌ 多语言端到端测试功能验证失败！") 