#!/usr/bin/env python3
"""
筛选评估样本脚本 V2
删除所有以"这个股票的下月最终收益结果是"开头的答案
"""

import json
import re
from pathlib import Path
from typing import List, Dict, Any

def filter_eval_samples_v2(input_file: str, output_file: str, target_samples: int = 100):
    """
    筛选评估样本 V2
    
    Args:
        input_file: 输入文件路径
        output_file: 输出文件路径
        target_samples: 目标样本数量
    """
    filtered_samples = []
    excluded_count = 0
    
    # 定义要排除的模式
    exclude_pattern = r"^这个股票的下月最终收益结果是"
    
    print(f"🔍 开始筛选样本...")
    print(f"📁 输入文件: {input_file}")
    print(f"🎯 筛选条件: 排除以'这个股票的下月最终收益结果是'开头的答案")
    print(f"📊 目标样本数: {target_samples}")
    
    with open(input_file, 'r', encoding='utf-8') as f:
        for line_num, line in enumerate(f, 1):
            try:
                sample = json.loads(line.strip())
                answer = sample.get("answer", "").strip()
                
                # 检查答案是否包含指定模式
                if "这个股票的下月最终收益结果是" in answer:
                    excluded_count += 1
                    if excluded_count <= 10:  # 只显示前10个被排除的样本
                        print(f"❌ 排除样本 {line_num}: {answer[:50]}...")
                    elif excluded_count == 11:
                        print(f"❌ ... (还有更多被排除的样本)")
                    continue
                
                # 保留符合条件的样本
                filtered_samples.append(sample)
                
                # 检查是否达到目标数量
                if len(filtered_samples) >= target_samples:
                    break
                    
            except json.JSONDecodeError as e:
                print(f"⚠️ 跳过无效JSON行 {line_num}: {e}")
                continue
            except Exception as e:
                print(f"⚠️ 处理行 {line_num} 时出错: {e}")
                continue
    
    # 保存筛选后的样本
    with open(output_file, 'w', encoding='utf-8') as f:
        for sample in filtered_samples:
            f.write(json.dumps(sample, ensure_ascii=False) + '\n')
    
    print(f"\n📊 筛选结果:")
    print(f"✅ 保留样本数: {len(filtered_samples)}")
    print(f"❌ 排除样本数: {excluded_count}")
    print(f"📁 输出文件: {output_file}")
    
    # 统计信息
    if filtered_samples:
        print(f"\n📈 样本统计:")
        print(f"   - 无instruction样本: {sum(1 for s in filtered_samples if not s.get('instruction', '').strip())}")
        print(f"   - 有instruction样本: {sum(1 for s in filtered_samples if s.get('instruction', '').strip())}")
        
        # 显示前几个样本的答案开头
        print(f"\n🔍 前5个样本的答案开头:")
        for i, sample in enumerate(filtered_samples[:5]):
            answer = sample.get("answer", "")[:100]
            print(f"   {i+1}. {answer}...")
    
    if len(filtered_samples) < target_samples:
        print(f"⚠️ 警告: 只找到 {len(filtered_samples)} 个符合条件的样本，少于目标数量 {target_samples}")

if __name__ == "__main__":
    input_file = "evaluate_mrr/alphafin_eval.jsonl"
    output_file = "data/alphafin/alphafin_eval_clean.jsonl"
    target_samples = 100
    
    print("🚀 开始创建新的评估数据集...")
    filter_eval_samples_v2(input_file, output_file, target_samples)
    print("✅ 新评估数据集创建完成！") 