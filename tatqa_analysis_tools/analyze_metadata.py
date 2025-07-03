#!/usr/bin/env python3
"""
分析数据中的元数据情况
"""

import json
import statistics
from pathlib import Path

def analyze_metadata(file_path: str):
    """分析文件中的元数据情况"""
    print(f"🔍 分析文件: {file_path}")
    
    with open(file_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    print(f"📊 总记录数: {len(data)}")
    print(f"📋 字段列表: {list(data[0].keys())}")
    
    # 元数据字段
    metadata_fields = ['company_name', 'stock_code', 'report_date']
    
    # 统计信息
    metadata_stats = {field: {'non_empty': 0, 'non_none': 0, 'values': []} for field in metadata_fields}
    total_with_metadata = 0
    
    # 长度统计
    context_lengths = []
    answer_lengths = []
    question_lengths = []
    
    for record in data:
        has_any_metadata = False
        
        # 分析元数据
        for field in metadata_fields:
            if field in record:
                value = record[field]
                if value is not None:
                    metadata_stats[field]['non_none'] += 1
                    if str(value).strip() and str(value).lower() != 'none':
                        metadata_stats[field]['non_empty'] += 1
                        metadata_stats[field]['values'].append(str(value))
                        has_any_metadata = True
        
        if has_any_metadata:
            total_with_metadata += 1
        
        # 分析长度
        context = record.get('original_context', record.get('context', ''))
        answer = record.get('original_answer', record.get('answer', ''))
        question = record.get('original_question', record.get('query', ''))
        
        context_lengths.append(len(context))
        answer_lengths.append(len(answer))
        question_lengths.append(len(question))
    
    # 输出元数据统计
    print(f"\n📋 元数据统计:")
    print(f"有元数据的记录: {total_with_metadata}/{len(data)} ({total_with_metadata/len(data)*100:.1f}%)")
    
    for field in metadata_fields:
        non_empty = metadata_stats[field]['non_empty']
        non_none = metadata_stats[field]['non_none']
        total = len(data)
        print(f"  {field}:")
        print(f"    非空值: {non_empty}/{total} ({non_empty/total*100:.1f}%)")
        print(f"    非None值: {non_none}/{total} ({non_none/total*100:.1f}%)")
        
        # 显示一些示例值
        if metadata_stats[field]['values']:
            unique_values = list(set(metadata_stats[field]['values'][:10]))
            print(f"    示例值: {unique_values}")
    
    # 输出长度统计
    print(f"\n📏 长度统计:")
    print(f"Context平均长度: {statistics.mean(context_lengths):.1f} 字符")
    print(f"Answer平均长度: {statistics.mean(answer_lengths):.1f} 字符")
    print(f"Question平均长度: {statistics.mean(question_lengths):.1f} 字符")
    print(f"Context长度范围: {min(context_lengths)} - {max(context_lengths)} 字符")
    print(f"Answer长度范围: {min(answer_lengths)} - {max(answer_lengths)} 字符")
    print(f"Question长度范围: {min(question_lengths)} - {max(question_lengths)} 字符")
    
    return {
        'total_records': len(data),
        'metadata_coverage': total_with_metadata/len(data)*100,
        'metadata_stats': metadata_stats,
        'avg_lengths': {
            'context': statistics.mean(context_lengths),
            'answer': statistics.mean(answer_lengths),
            'question': statistics.mean(question_lengths)
        }
    }

def analyze_tatqa_data(file_path: str):
    """分析TatQA数据"""
    print(f"🔍 分析TatQA文件: {file_path}")
    
    data = []
    if file_path.endswith('.jsonl'):
        with open(file_path, 'r', encoding='utf-8') as f:
            for line in f:
                if line.strip():
                    data.append(json.loads(line))
    else:
        with open(file_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
    
    print(f"📊 总记录数: {len(data)}")
    print(f"📋 字段列表: {list(data[0].keys())}")
    
    # 长度统计
    context_lengths = []
    answer_lengths = []
    question_lengths = []
    
    for record in data:
        context = record.get('context', '')
        answer = record.get('answer', '')
        question = record.get('query', record.get('question', ''))
        
        context_lengths.append(len(context))
        answer_lengths.append(len(answer))
        question_lengths.append(len(question))
    
    print(f"\n📏 TatQA长度统计:")
    print(f"Context平均长度: {statistics.mean(context_lengths):.1f} 字符")
    print(f"Answer平均长度: {statistics.mean(answer_lengths):.1f} 字符")
    print(f"Question平均长度: {statistics.mean(question_lengths):.1f} 字符")
    
    return {
        'total_records': len(data),
        'avg_lengths': {
            'context': statistics.mean(context_lengths),
            'answer': statistics.mean(answer_lengths),
            'question': statistics.mean(question_lengths)
        }
    }

def main():
    print("=== AlphaFin数据元数据分析 ===\n")
    
    # 分析AlphaFin数据
    alphafin_stats = analyze_metadata('data/alphafin/alphafin_merged_generated_qa_full_dedup.json')
    
    print("\n=== TatQA数据分析 ===\n")
    
    # 分析TatQA数据
    tatqa_stats = analyze_tatqa_data('evaluate_mrr/tatqa_eval_enhanced.jsonl')
    
    # 生成报告
    print("\n=== 数据概况总结 ===\n")
    print("1.1 原始数据概况:")
    print(f"  中文数据 (AlphaFin): {alphafin_stats['total_records']} 个样本")
    print(f"  英文数据 (TatQA): {tatqa_stats['total_records']} 个样本")
    
    print("\n1.2 LLM自动化数据处理:")
    print(f"  元数据覆盖率: {alphafin_stats['metadata_coverage']:.1f}%")
    print("  核心功能:")
    print("    - 元数据提取器: 自动提取company_name, stock_code, report_date")
    print("    - 问题生成器: 基于Context和Answer生成Question")
    print("    - 摘要生成器: 基于Context生成Summary")
    
    print("\n1.3 处理后数据统计:")
    print("  中文 (QCA):")
    print(f"    样本数量: {alphafin_stats['total_records']}")
    print(f"    平均Context长度: {alphafin_stats['avg_lengths']['context']:.1f} 字符")
    print(f"    平均Answer长度: {alphafin_stats['avg_lengths']['answer']:.1f} 字符")
    print(f"    平均Question长度: {alphafin_stats['avg_lengths']['question']:.1f} 字符")
    
    print("  英文 (QCA):")
    print(f"    样本数量: {tatqa_stats['total_records']}")
    print(f"    平均Context长度: {tatqa_stats['avg_lengths']['context']:.1f} 字符")
    print(f"    平均Answer长度: {tatqa_stats['avg_lengths']['answer']:.1f} 字符")
    print(f"    平均Question长度: {tatqa_stats['avg_lengths']['question']:.1f} 字符")

if __name__ == "__main__":
    main() 