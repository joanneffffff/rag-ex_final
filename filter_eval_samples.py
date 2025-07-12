#!/usr/bin/env python3
"""
筛选评估样本脚本
选择无instruction或者answer包含特定模式的样本
"""

import json
import re
from pathlib import Path
from typing import List, Dict, Any

def filter_eval_samples(input_file: str, output_file: str, max_samples: int = 100):
    """
    筛选评估样本
    
    Args:
        input_file: 输入文件路径
        output_file: 输出文件路径
        max_samples: 最大样本数量
    """
    filtered_samples = []
    
    # 定义筛选条件 - 只保留无instruction的样本，排除股票涨跌预测
    def should_include_sample(sample: Dict[str, Any]) -> bool:
        # 只保留无instruction的样本
        if not sample.get("instruction", "").strip():
            return True
        
        # 排除所有其他样本（包括股票涨跌预测）
        return False
    
    print(f"🔍 开始筛选样本...")
    print(f"📁 输入文件: {input_file}")
    print(f"🎯 筛选条件: 只保留无instruction的样本，排除股票涨跌预测")
    
    with open(input_file, 'r', encoding='utf-8') as f:
        for line_num, line in enumerate(f, 1):
            try:
                sample = json.loads(line.strip())
                
                if should_include_sample(sample):
                    filtered_samples.append(sample)
                    
                    if len(filtered_samples) >= max_samples:
                        break
                        
            except json.JSONDecodeError as e:
                print(f"⚠️ 第{line_num}行JSON解析错误: {e}")
                continue
    
    # 保存筛选结果
    with open(output_file, 'w', encoding='utf-8') as f:
        for sample in filtered_samples:
            f.write(json.dumps(sample, ensure_ascii=False) + '\n')
    
    print(f"✅ 筛选完成!")
    print(f"📊 总筛选样本数: {len(filtered_samples)}")
    print(f"📁 输出文件: {output_file}")
    
    # 统计信息
    print(f"📈 统计信息:")
    print(f"   - 无instruction样本: {len(filtered_samples)}")
    print(f"   - 排除的股票涨跌预测样本: 已排除")
    
    # 显示前几个样本的示例
    print(f"\n📋 样本示例:")
    for i, sample in enumerate(filtered_samples[:3]):
        print(f"\n样本 {i+1}:")
        print(f"  问题: {sample.get('question', 'N/A')}")
        print(f"  答案: {sample.get('answer', 'N/A')[:100]}...")
        print(f"  Instruction: {sample.get('instruction', 'N/A')[:50]}...")
        print(f"  公司: {sample.get('company_name', 'N/A')}")

if __name__ == "__main__":
    input_file = "evaluate_mrr/alphafin_eval.jsonl"
    output_file = "data/alphafin/alphafin_eval_filtered.jsonl"
    
    # 确保输出目录存在
    Path(output_file).parent.mkdir(parents=True, exist_ok=True)
    
    # 筛选样本
    filter_eval_samples(input_file, output_file, max_samples=100) 