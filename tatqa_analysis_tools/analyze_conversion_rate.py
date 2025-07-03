#!/usr/bin/env python3
"""
分析TatQA数据转换率和质量
"""

import json
from pathlib import Path

def analyze_tatqa_conversion():
    """分析TatQA数据转换情况"""
    print("=== TatQA数据转换分析 ===\n")
    
    # 原始数据统计
    print("1. 原始数据统计:")
    try:
        with open('data/tatqa_dataset_raw/tatqa_dataset_train.json', 'r') as f:
            train_data = json.load(f)
        with open('data/tatqa_dataset_raw/tatqa_dataset_dev.json', 'r') as f:
            dev_data = json.load(f)
        with open('data/tatqa_dataset_raw/tatqa_dataset_test.json', 'r') as f:
            test_data = json.load(f)
        
        total_original = len(train_data) + len(dev_data) + len(test_data)
        print(f"  训练集: {len(train_data)} 个样本")
        print(f"  验证集: {len(dev_data)} 个样本")
        print(f"  测试集: {len(test_data)} 个样本")
        print(f"  总计: {total_original} 个样本")
        print(f"  文件大小: 18M")
        
        # 统计原始问题数量
        total_questions = 0
        for dataset in [train_data, dev_data, test_data]:
            for item in dataset:
                questions = item.get('questions', [])
                total_questions += len(questions)
        
        print(f"  原始问题总数: {total_questions}")
        
    except Exception as e:
        print(f"  读取原始数据失败: {e}")
        return
    
    # 转换后数据统计
    print("\n2. 转换后数据统计:")
    try:
        # 读取训练和评估数据
        with open('evaluate_mrr/tatqa_train_qc_enhanced.jsonl', 'r') as f:
            train_converted = [json.loads(line) for line in f if line.strip()]
        with open('evaluate_mrr/tatqa_eval_enhanced.jsonl', 'r') as f:
            eval_converted = [json.loads(line) for line in f if line.strip()]
        
        total_converted = len(train_converted) + len(eval_converted)
        print(f"  训练集转换后: {len(train_converted)} 个样本")
        print(f"  评估集转换后: {len(eval_converted)} 个样本")
        print(f"  总计转换后: {total_converted} 个样本")
        
        # 统计answer_from分布（使用评估集）
        answer_from_stats = {}
        for item in eval_converted:
            answer_from = item.get('answer_from', 'unknown')
            answer_from_stats[answer_from] = answer_from_stats.get(answer_from, 0) + 1
        
        print(f"  答案来源分布 (评估集):")
        for source, count in answer_from_stats.items():
            percentage = count / len(eval_converted) * 100
            print(f"    {source}: {count} ({percentage:.1f}%)")
        
    except Exception as e:
        print(f"  读取转换后数据失败: {e}")
        return
    
    # 转换率计算
    conversion_rate = total_converted / total_questions * 100
    filtered_rate = 100 - conversion_rate
    
    print(f"\n3. 转换率分析:")
    print(f"  转换率: {conversion_rate:.1f}% ({total_converted}/{total_questions})")
    print(f"  过滤率: {filtered_rate:.1f}% ({total_questions - total_converted}/{total_questions})")
    
    # 分析过滤原因
    print(f"\n4. 过滤原因分析:")
    print("  主要原因:")
    print("    - answer_type=table 但 rel_paragraphs 为空")
    print("    - 表格转换逻辑缺陷导致内容丢失")
    print("    - 问题或答案字段为空")
    print("    - 表格结构过于复杂，无法有效转换")
    
    # 数据质量分析
    print(f"\n5. 数据质量分析:")
    
    # 长度统计（使用评估集）
    context_lengths = [len(item.get('context', '')) for item in eval_converted]
    answer_lengths = [len(item.get('answer', '')) for item in eval_converted]
    question_lengths = [len(item.get('query', '')) for item in eval_converted]
    
    print(f"  Context平均长度: {sum(context_lengths)/len(context_lengths):.1f} 字符")
    print(f"  Answer平均长度: {sum(answer_lengths)/len(answer_lengths):.1f} 字符")
    print(f"  Question平均长度: {sum(question_lengths)/len(question_lengths):.1f} 字符")
    
    # 元数据覆盖率
    doc_id_coverage = sum(1 for item in eval_converted if item.get('doc_id')) / len(eval_converted) * 100
    relevant_doc_coverage = sum(1 for item in eval_converted if item.get('relevant_doc_ids')) / len(eval_converted) * 100
    
    print(f"  doc_id覆盖率: {doc_id_coverage:.1f}%")
    print(f"  relevant_doc_ids覆盖率: {relevant_doc_coverage:.1f}%")

def analyze_alphafin_processing():
    """分析AlphaFin数据处理情况"""
    print("\n=== AlphaFin数据处理分析 ===\n")
    
    # 原始数据
    print("1. 原始数据:")
    try:
        with open('data/alphafin/data.json', 'r') as f:
            raw_data = json.load(f)
        print(f"  原始样本数: {len(raw_data)}")
        print(f"  文件大小: 425M")
        print(f"  字段: {list(raw_data[0].keys())}")
    except Exception as e:
        print(f"  读取原始数据失败: {e}")
        return
    
    # 过滤后数据
    print("\n2. 过滤后数据:")
    try:
        with open('data/alphafin/alphafin_rag_ready_0627.json', 'r') as f:
            filtered_data = json.load(f)
        print(f"  过滤后样本数: {len(filtered_data)}")
        
        # 计算过滤率
        filter_rate = (len(raw_data) - len(filtered_data)) / len(raw_data) * 100
        print(f"  过滤率: {filter_rate:.1f}% ({len(raw_data) - len(filtered_data)}/{len(raw_data)})")
        
    except Exception as e:
        print(f"  读取过滤后数据失败: {e}")
        return
    
    # LLM处理后数据
    print("\n3. LLM处理后数据:")
    try:
        with open('data/alphafin/alphafin_merged_generated_qa_full_dedup.json', 'r') as f:
            processed_data = json.load(f)
        print(f"  LLM处理后样本数: {len(processed_data)}")
        
        # 元数据统计
        metadata_fields = ['company_name', 'stock_code', 'report_date']
        metadata_stats = {}
        for field in metadata_fields:
            count = sum(1 for item in processed_data 
                       if item.get(field) and str(item.get(field)).strip() 
                       and str(item.get(field)).lower() != 'none')
            percentage = count / len(processed_data) * 100
            metadata_stats[field] = (count, percentage)
        
        print(f"  元数据覆盖率:")
        for field, (count, percentage) in metadata_stats.items():
            print(f"    {field}: {count}/{len(processed_data)} ({percentage:.1f}%)")
        
    except Exception as e:
        print(f"  读取LLM处理后数据失败: {e}")

def main():
    analyze_tatqa_conversion()
    analyze_alphafin_processing()
    
    print("\n=== 总结 ===")
    print("1.1 原始数据概况:")
    print("  中文数据 (AlphaFin): 167,362 个样本，425M")
    print("  英文数据 (TatQA): 2,757 个样本，18M")
    
    print("\n1.2 LLM自动化数据处理:")
    print("  核心功能:")
    print("    - 元数据提取器: 自动提取company_name, stock_code, report_date")
    print("    - 问题生成器: 基于Context和Answer生成Question")
    print("    - 摘要生成器: 基于Context生成Summary")
    
    print("\n1.3 处理后数据统计:")
    print("  中文 (QCA): 27,596 个样本")
    print("  英文 (QCA): 16,546 个样本 (训练: 14,883, 评估: 1,663)")
    
    print("\n1.4 TatQA数据转换过程与质量:")
    print("  关键步骤: Table Textualization将表格转换为自然语言")
    print("  转换率: 约60.0% (16,546/27,552)")
    print("  过滤率: 约40.0%")
    print("  主要原因: answer_type=table但rel_paragraphs为空，表格转换逻辑缺陷")

if __name__ == "__main__":
    main() 