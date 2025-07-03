#!/usr/bin/env python3
"""
分析完整TAT-QA数据集统计脚本
分析训练集和评估集的问题类型分布
"""

import json
import re
from collections import Counter
from typing import List, Dict, Any
import os

def classify_question_type(context: str) -> str:
    """
    根据上下文内容分类问题类型
    """
    # 检查是否包含表格
    has_table = "Table ID:" in context
    
    # 检查是否包含文本段落（非表格内容）
    # 移除表格内容，检查剩余部分是否还有文本
    context_without_table = context
    if has_table:
        # 找到表格开始位置
        table_start = context.find("Table ID:")
        if table_start > 0:
            # 检查表格前是否有文本
            text_before = context[:table_start].strip()
            if len(text_before) > 50:  # 如果有足够长的文本段落
                return "table+text"
    
    # 检查表格后是否有文本
    if has_table:
        # 简单的启发式方法：如果上下文很长且包含表格，可能还有文本
        if len(context) > 2000:  # 如果上下文很长
            return "table+text"
    
    if has_table:
        return "table"
    elif len(context.strip()) > 0:
        return "text"
    else:
        return "unknown"

def analyze_dataset_file(data_file: str) -> Dict[str, Any]:
    """
    分析单个数据文件的统计信息
    """
    print(f"📊 分析数据集文件: {data_file}")
    
    if not os.path.exists(data_file):
        print(f"❌ 文件不存在: {data_file}")
        return {
            "total_samples": 0,
            "type_distribution": {},
            "type_percentages": {},
            "file_exists": False
        }
    
    question_types = []
    total_samples = 0
    
    with open(data_file, 'r', encoding='utf-8') as f:
        for line_num, line in enumerate(f, 1):
            try:
                sample = json.loads(line.strip())
                question_type = classify_question_type(sample["context"])
                question_types.append(question_type)
                total_samples += 1
            except json.JSONDecodeError as e:
                print(f"⚠️ 第{line_num}行JSON解析错误: {e}")
                continue
            except KeyError as e:
                print(f"⚠️ 第{line_num}行缺少字段: {e}")
                continue
    
    # 统计各类型数量
    type_counts = Counter(question_types)
    
    statistics = {
        "total_samples": total_samples,
        "type_distribution": dict(type_counts),
        "type_percentages": {
            qtype: (count / total_samples * 100) if total_samples > 0 else 0
            for qtype, count in type_counts.items()
        },
        "file_exists": True
    }
    
    return statistics

def analyze_full_tatqa_dataset():
    """
    分析完整TAT-QA数据集
    """
    print("🚀 TAT-QA完整数据集分析工具")
    print("="*60)
    
    # 定义要分析的文件路径
    possible_files = [
        "evaluate_mrr/tatqa_eval_enhanced.jsonl",
        "evaluate_mrr/tatqa_train_qc_enhanced.jsonl"
    ]
    
    # 查找存在的文件
    existing_files = []
    for file_path in possible_files:
        if os.path.exists(file_path):
            existing_files.append(file_path)
    
    if not existing_files:
        print("❌ 未找到任何TAT-QA数据文件")
        print("请检查以下可能的路径:")
        for file_path in possible_files:
            print(f"  - {file_path}")
        return
    
    print(f"✅ 找到 {len(existing_files)} 个数据文件:")
    for file_path in existing_files:
        print(f"  - {file_path}")
    
    # 分析每个文件
    all_statistics = {}
    total_combined = {
        "total_samples": 0,
        "table_samples": 0,
        "text_samples": 0,
        "unknown_samples": 0
    }
    
    for file_path in existing_files:
        print(f"\n📊 分析文件: {file_path}")
        stats = analyze_dataset_file(file_path)
        all_statistics[file_path] = stats
        
        if stats["file_exists"]:
            print(f"  总样本数: {stats['total_samples']}")
            for qtype, count in stats['type_distribution'].items():
                percentage = stats['type_percentages'][qtype]
                print(f"  {qtype}: {count} ({percentage:.1f}%)")
                
                # 累计到总计
                total_combined["total_samples"] += count
                if qtype == "table":
                    total_combined["table_samples"] += count
                elif qtype == "text":
                    total_combined["text_samples"] += count
                else:
                    total_combined["unknown_samples"] += count
    
    # 显示总体统计
    print(f"\n📈 总体统计:")
    print(f"总样本数: {total_combined['total_samples']}")
    if total_combined['total_samples'] > 0:
        print(f"表格问题: {total_combined['table_samples']} ({total_combined['table_samples']/total_combined['total_samples']*100:.1f}%)")
        print(f"文本问题: {total_combined['text_samples']} ({total_combined['text_samples']/total_combined['total_samples']*100:.1f}%)")
        if total_combined['unknown_samples'] > 0:
            print(f"表格+文本: {total_combined['unknown_samples']} ({total_combined['unknown_samples']/total_combined['total_samples']*100:.1f}%)")
    
    # 按文件类型分组统计
    print(f"\n📋 按文件类型分组:")
    train_files = [f for f in existing_files if "train" in f.lower()]
    eval_files = [f for f in existing_files if "eval" in f.lower()]
    
    if train_files:
        train_total = sum(all_statistics[f]["total_samples"] for f in train_files if all_statistics[f]["file_exists"])
        print(f"训练集文件数: {len(train_files)}, 总样本数: {train_total}")
    
    if eval_files:
        eval_total = sum(all_statistics[f]["total_samples"] for f in eval_files if all_statistics[f]["file_exists"])
        print(f"评估集文件数: {len(eval_files)}, 总样本数: {eval_total}")
    
    # 保存详细统计结果
    output_file = "tatqa_dataset_statistics.json"
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump({
            "file_statistics": all_statistics,
            "total_combined": total_combined,
            "existing_files": existing_files
        }, f, indent=2, ensure_ascii=False)
    
    print(f"\n✅ 详细统计结果已保存到: {output_file}")

if __name__ == "__main__":
    analyze_full_tatqa_dataset() 