#!/usr/bin/env python3
"""
调试股票预测检测逻辑
"""

import json
from pathlib import Path

def is_stock_prediction_query(test_item):
    """检测数据项是否为股票预测指令"""
    instruction = test_item.get("instruction", "")
    if instruction and instruction.strip():
        return True
    return False

def analyze_dataset(data_path):
    """分析数据集中的股票预测检测情况"""
    print(f"🔍 分析数据集: {data_path}")
    print("="*60)
    
    dataset = []
    with open(data_path, 'r', encoding='utf-8') as f:
        for line in f:
            if line.strip():
                dataset.append(json.loads(line))
    
    print(f"📊 总样本数: {len(dataset)}")
    
    # 分析instruction字段
    instruction_stats = {}
    stock_prediction_detected = []
    
    for i, item in enumerate(dataset):
        instruction = item.get("instruction", "")
        
        # 统计instruction类型
        if instruction is None:
            instruction_type = "None"
        elif instruction == "":
            instruction_type = "空字符串"
        elif instruction.strip() == "":
            instruction_type = "空白字符串"
        else:
            instruction_type = "非空内容"
        
        if instruction_type not in instruction_stats:
            instruction_stats[instruction_type] = 0
        instruction_stats[instruction_type] += 1
        
        # 检测是否为股票预测查询
        is_stock = is_stock_prediction_query(item)
        if is_stock:
            stock_prediction_detected.append(i)
        
        # 显示前5个样本的详细信息
        if i < 5:
            print(f"\n样本 {i}:")
            print(f"  instruction: '{instruction}' (类型: {instruction_type})")
            print(f"  is_stock_prediction: {is_stock}")
            print(f"  question: {item.get('question', '')[:50]}...")
    
    print(f"\n📈 instruction字段统计:")
    for instruction_type, count in instruction_stats.items():
        print(f"  {instruction_type}: {count} 个")
    
    print(f"\n🔮 股票预测检测结果:")
    print(f"  检测到的股票预测查询: {len(stock_prediction_detected)} 个")
    if stock_prediction_detected:
        print(f"  样本索引: {stock_prediction_detected[:10]}{'...' if len(stock_prediction_detected) > 10 else ''}")
    
    return len(stock_prediction_detected) > 0

if __name__ == "__main__":
    data_path = "data/alphafin/alphafin_eval_samples_updated.jsonl"
    
    if Path(data_path).exists():
        has_stock_queries = analyze_dataset(data_path)
        print(f"\n🎯 结论: 数据集{'包含' if has_stock_queries else '不包含'}股票预测查询")
    else:
        print(f"❌ 数据文件不存在: {data_path}") 