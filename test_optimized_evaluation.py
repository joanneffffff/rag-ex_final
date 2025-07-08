#!/usr/bin/env python3
"""
测试优化后的RAG系统检索评估流程
验证一次检索，多指标计算的功能
"""

import sys
import os
import json
from pathlib import Path

# 添加项目根目录到路径
sys.path.append(str(Path(__file__).parent))

from alphafin_data_process.rag_system_adapter import RagSystemAdapter
from alphafin_data_process.run_retrieval_evaluation_background import calculate_metrics_from_raw_results

def test_optimized_evaluation():
    """测试优化后的评估流程"""
    print("=" * 60)
    print("测试优化后的RAG系统检索评估流程")
    print("=" * 60)
    
    # 创建测试数据
    test_data = [
        {
            "generated_question": "什么是股票投资？",
            "doc_id": "doc_001"
        },
        {
            "generated_question": "债券的基本概念是什么？",
            "doc_id": "doc_002"
        },
        {
            "generated_question": "基金投资与股票投资有什么区别？",
            "doc_id": "doc_003"
        }
    ]
    
    print(f"创建测试数据: {len(test_data)} 个样本")
    
    try:
        # 初始化适配器
        print("\n1. 初始化RAG系统适配器...")
        adapter = RagSystemAdapter()
        print("✅ 适配器初始化成功")
        
        # 测试一次性检索
        print("\n2. 执行一次性检索（top_k=10）...")
        raw_results = adapter.evaluate_retrieval_performance(
            eval_dataset=test_data,
            top_k=10,  # 最大检索深度
            mode="baseline",
            use_prefilter=False
        )
        print(f"✅ 检索完成，返回 {len(raw_results)} 个原始结果")
        
        # 显示原始结果结构
        if raw_results:
            print("\n原始结果结构示例:")
            sample_result = raw_results[0]
            print(f"  query_text: {sample_result['query_text'][:50]}...")
            print(f"  ground_truth_doc_ids: {sample_result['ground_truth_doc_ids']}")
            print(f"  retrieved_doc_ids_ranked: {sample_result['retrieved_doc_ids_ranked'][:5]}...")
        
        # 测试多指标计算
        print("\n3. 测试多指标计算...")
        top_k_list = [1, 3, 5, 10]
        
        for top_k in top_k_list:
            print(f"\n计算 Top-{top_k} 指标...")
            metrics = calculate_metrics_from_raw_results(raw_results, top_k)
            print(f"  MRR: {metrics['MRR']:.4f}")
            print(f"  Hit@{top_k}: {metrics[f'Hit@{top_k}']:.4f}")
            print(f"  样本数: {metrics['total_samples']}")
        
        print("\n✅ 优化后的评估流程测试成功！")
        
        # 保存测试结果
        test_output = {
            "raw_results": raw_results,
            "metrics": {}
        }
        
        for top_k in top_k_list:
            test_output["metrics"][f"top_{top_k}"] = calculate_metrics_from_raw_results(raw_results, top_k)
        
        output_file = "test_optimized_evaluation_results.json"
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(test_output, f, ensure_ascii=False, indent=2)
        print(f"测试结果已保存到: {output_file}")
        
        return True
        
    except Exception as e:
        print(f"❌ 测试失败: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_metrics_calculation():
    """测试指标计算函数"""
    print("\n" + "=" * 60)
    print("测试指标计算函数")
    print("=" * 60)
    
    # 创建模拟的原始检索结果
    mock_raw_results = [
        {
            "query_text": "什么是股票投资？",
            "ground_truth_doc_ids": ["doc_001"],
            "retrieved_doc_ids_ranked": ["doc_001", "doc_002", "doc_003", "doc_004", "doc_005"]
        },
        {
            "query_text": "债券的基本概念是什么？",
            "ground_truth_doc_ids": ["doc_002"],
            "retrieved_doc_ids_ranked": ["doc_003", "doc_002", "doc_001", "doc_004", "doc_005"]
        },
        {
            "query_text": "基金投资与股票投资有什么区别？",
            "ground_truth_doc_ids": ["doc_003"],
            "retrieved_doc_ids_ranked": ["doc_001", "doc_002", "doc_004", "doc_003", "doc_005"]
        }
    ]
    
    print("模拟原始检索结果:")
    for i, result in enumerate(mock_raw_results):
        print(f"  样本 {i+1}: 正确答案排名 {result['retrieved_doc_ids_ranked'].index(result['ground_truth_doc_ids'][0]) + 1}")
    
    # 测试不同top_k值的指标计算
    top_k_list = [1, 3, 5]
    
    for top_k in top_k_list:
        print(f"\nTop-{top_k} 指标:")
        metrics = calculate_metrics_from_raw_results(mock_raw_results, top_k)
        print(f"  MRR: {metrics['MRR']:.4f}")
        print(f"  Hit@{top_k}: {metrics[f'Hit@{top_k}']:.4f}")
        print(f"  样本数: {metrics['total_samples']}")
    
    print("\n✅ 指标计算函数测试成功！")

if __name__ == "__main__":
    # 先测试指标计算函数
    test_metrics_calculation()
    
    # 再测试完整的优化流程
    test_optimized_evaluation() 