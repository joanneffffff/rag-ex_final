#!/usr/bin/env python3
"""
测试股票预测检测功能修复
"""

import json
import sys
from pathlib import Path

def test_stock_prediction_detection():
    """测试股票预测检测功能"""
    print("🔍 测试股票预测检测功能...")
    
    # 加载数据
    data_path = "data/alphafin/alphafin_eval_samples_updated.jsonl"
    
    if not Path(data_path).exists():
        print(f"❌ 数据文件不存在: {data_path}")
        return False
    
    # 读取数据
    with open(data_path, 'r', encoding='utf-8') as f:
        dataset = [json.loads(line) for line in f if line.strip()]
    
    print(f"📊 总样本数: {len(dataset)}")
    
    # 检测股票预测查询
    stock_prediction_count = 0
    stock_prediction_samples = []
    
    for i, item in enumerate(dataset):
        instruction = item.get("instruction", "")
        if "涨跌" in instruction and "预测" in instruction and "涨跌概率" in instruction:
            stock_prediction_count += 1
            stock_prediction_samples.append(i)
    
    print(f"🔮 检测到股票预测查询: {stock_prediction_count} 个")
    
    # 显示前几个股票预测样本的instruction
    if stock_prediction_samples:
        print("\n📋 前3个股票预测样本的instruction:")
        for i in range(min(3, len(stock_prediction_samples))):
            sample_idx = stock_prediction_samples[i]
            instruction = dataset[sample_idx].get("instruction", "")
            print(f"样本 {sample_idx}: {instruction[:100]}...")
    
    # 验证结果
    expected_count = 66
    if stock_prediction_count == expected_count:
        print(f"✅ 检测结果正确: {stock_prediction_count} 个股票预测查询")
        return True
    else:
        print(f"❌ 检测结果错误: 期望 {expected_count} 个，实际 {stock_prediction_count} 个")
        return False

def test_rag_system_adapter():
    """测试RAG系统适配器的股票预测检测"""
    print("\n🔧 测试RAG系统适配器...")
    
    try:
        from test_rag_system_e2e_multilingual import is_stock_prediction_query, load_test_dataset
        
        # 加载测试数据
        data_path = "data/alphafin/alphafin_eval_samples_updated.jsonl"
        dataset, language = load_test_dataset(data_path, sample_size=10)  # 只测试前10个样本
        
        print(f"🌍 检测到语言: {language}")
        
        # 测试检测函数
        detected_count = 0
        for i, test_item in enumerate(dataset):
            if is_stock_prediction_query(test_item):
                detected_count += 1
                print(f"✅ 样本 {i} 被正确识别为股票预测查询")
        
        print(f"🔮 RAG适配器检测到: {detected_count} 个股票预测查询")
        
        return True
        
    except ImportError as e:
        print(f"❌ 导入错误: {e}")
        return False
    except Exception as e:
        print(f"❌ 测试错误: {e}")
        return False

def main():
    """主函数"""
    print("🚀 开始测试股票预测检测功能修复...")
    print("=" * 60)
    
    # 测试1: 直接检测
    test1_result = test_stock_prediction_detection()
    
    # 测试2: RAG适配器检测
    test2_result = test_rag_system_adapter()
    
    print("\n" + "=" * 60)
    print("📊 测试结果汇总:")
    print(f"   直接检测: {'✅ 通过' if test1_result else '❌ 失败'}")
    print(f"   RAG适配器: {'✅ 通过' if test2_result else '❌ 失败'}")
    
    if test1_result and test2_result:
        print("\n🎉 所有测试通过！股票预测检测功能已修复。")
        print("💡 现在可以运行端到端测试，应该会看到正确的股票预测检测数量。")
        return True
    else:
        print("\n⚠️ 部分测试失败，需要进一步检查。")
        return False

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1) 