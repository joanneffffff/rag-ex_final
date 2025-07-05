#!/usr/bin/env python3
"""
打印所有数据统计信息的完整报告
"""

import json
import statistics
from pathlib import Path

def analyze_alphafin_raw_data():
    """分析AlphaFin原始数据"""
    print("=== AlphaFin原始数据分析 ===\n")
    
    try:
        with open('data/alphafin/data.json', 'r', encoding='utf-8') as f:
            raw_data = json.load(f)
        
        print(f"📊 原始样本总数: {len(raw_data):,} 个样本")
        print(f"📁 文件大小: 425M")
        print(f"📋 字段列表: {list(raw_data[0].keys())}")
        
        # 示例记录
        print(f"\n📝 示例记录:")
        sample_record = raw_data[0]
        print(json.dumps(sample_record, ensure_ascii=False, indent=2)[:800] + "...")
        
        return len(raw_data)
        
    except Exception as e:
        print(f"❌ 读取AlphaFin原始数据失败: {e}")
        return 0

def analyze_alphafin_filtered_data():
    """分析AlphaFin过滤后数据"""
    print("\n=== AlphaFin过滤后数据分析 ===\n")
    
    try:
        with open('data/alphafin/alphafin_rag_ready_0627.json', 'r', encoding='utf-8') as f:
            filtered_data = json.load(f)
        
        print(f"📊 过滤后样本数: {len(filtered_data):,} 个样本")
        
        # 计算过滤率
        raw_count = 167362  # 从原始数据获得
        filter_rate = (raw_count - len(filtered_data)) / raw_count * 100
        print(f"🗑️  过滤率: {filter_rate:.1f}% ({raw_count - len(filtered_data):,}/{raw_count:,})")
        
        return len(filtered_data)
        
    except Exception as e:
        print(f"❌ 读取AlphaFin过滤后数据失败: {e}")
        return 0

def analyze_alphafin_processed_data():
    """分析AlphaFin LLM处理后数据"""
    print("\n=== AlphaFin LLM处理后数据分析 ===\n")
    
    try:
        with open('data/alphafin/alphafin_final_clean.json', 'r', encoding='utf-8') as f:
            processed_data = json.load(f)
        
        print(f"📊 LLM处理后样本数: {len(processed_data):,} 个样本")
        
        # 元数据统计
        metadata_fields = ['company_name', 'stock_code', 'report_date']
        metadata_stats = {}
        total_with_metadata = 0
        
        for field in metadata_fields:
            count = sum(1 for item in processed_data 
                       if item.get(field) and str(item.get(field)).strip() 
                       and str(item.get(field)).lower() != 'none')
            percentage = count / len(processed_data) * 100
            metadata_stats[field] = (count, percentage)
            
            if count > 0:
                total_with_metadata += 1
        
        print(f"📋 元数据覆盖率:")
        for field, (count, percentage) in metadata_stats.items():
            print(f"   {field}: {count:,}/{len(processed_data):,} ({percentage:.1f}%)")
        
        overall_metadata_rate = total_with_metadata / len(metadata_fields) * 100
        print(f"📊 总体元数据覆盖率: {overall_metadata_rate:.1f}%")
        
        # 长度统计
        context_lengths = [len(item.get('original_context', item.get('context', ''))) for item in processed_data]
        answer_lengths = [len(item.get('original_answer', item.get('answer', ''))) for item in processed_data]
        question_lengths = [len(item.get('original_question', item.get('query', ''))) for item in processed_data]
        
        print(f"\n📏 长度统计:")
        print(f"   Context平均长度: {statistics.mean(context_lengths):.1f} 字符")
        print(f"   Answer平均长度: {statistics.mean(answer_lengths):.1f} 字符")
        print(f"   Question平均长度: {statistics.mean(question_lengths):.1f} 字符")
        print(f"   Context长度范围: {min(context_lengths)} - {max(context_lengths)} 字符")
        print(f"   Answer长度范围: {min(answer_lengths)} - {max(answer_lengths)} 字符")
        print(f"   Question长度范围: {min(question_lengths)} - {max(question_lengths)} 字符")
        
        return len(processed_data), metadata_stats
        
    except Exception as e:
        print(f"❌ 读取AlphaFin处理后数据失败: {e}")
        return 0, {}

def analyze_tatqa_raw_data():
    """分析TatQA原始数据"""
    print("\n=== TatQA原始数据分析 ===\n")
    
    try:
        # 读取原始数据
        with open('data/tatqa_dataset_raw/tatqa_dataset_train.json', 'r') as f:
            train_data = json.load(f)
        with open('data/tatqa_dataset_raw/tatqa_dataset_dev.json', 'r') as f:
            dev_data = json.load(f)
        with open('data/tatqa_dataset_raw/tatqa_dataset_test.json', 'r') as f:
            test_data = json.load(f)
        
        total_original = len(train_data) + len(dev_data) + len(test_data)
        print(f"📊 原始样本总数: {total_original:,} 个样本")
        print(f"  训练集: {len(train_data):,} 个样本")
        print(f"  验证集: {len(dev_data):,} 个样本")
        print(f"  测试集: {len(test_data):,} 个样本")
        print(f"📁 文件大小: 18M")
        
        # 统计原始问题数量
        total_questions = 0
        for dataset in [train_data, dev_data, test_data]:
            for item in dataset:
                questions = item.get('questions', [])
                total_questions += len(questions)
        
        print(f"❓ 原始问题总数: {total_questions:,} 个问题")
        
        # 示例记录
        print(f"\n📝 示例记录:")
        sample_record = train_data[0]
        print(json.dumps(sample_record, ensure_ascii=False, indent=2)[:800] + "...")
        
        return total_original, total_questions
        
    except Exception as e:
        print(f"❌ 读取TatQA原始数据失败: {e}")
        return 0, 0

def analyze_tatqa_converted_data():
    """分析TatQA转换后数据"""
    print("\n=== TatQA转换后数据分析 ===\n")
    
    try:
        # 读取训练和评估数据
        with open('evaluate_mrr/tatqa_train_qc_enhanced.jsonl', 'r') as f:
            train_converted = [json.loads(line) for line in f if line.strip()]
        with open('evaluate_mrr/tatqa_eval_enhanced.jsonl', 'r') as f:
            eval_converted = [json.loads(line) for line in f if line.strip()]
        
        total_converted = len(train_converted) + len(eval_converted)
        print(f"📊 QCA评估样本总数: {total_converted:,} 个样本")
        print(f"  训练集: {len(train_converted):,} 个样本")
        print(f"  评估集: {len(eval_converted):,} 个样本")
        
        # 统计answer_from分布（使用评估集）
        answer_from_stats = {}
        for item in eval_converted:
            answer_from = item.get('answer_from', 'unknown')
            answer_from_stats[answer_from] = answer_from_stats.get(answer_from, 0) + 1
        
        print(f"\n📋 答案来源分布 (评估集):")
        for source, count in answer_from_stats.items():
            percentage = count / len(eval_converted) * 100
            print(f"   {source}: {count:,} ({percentage:.1f}%)")
        
        # 长度统计（使用评估集）
        context_lengths = [len(item.get('context', '')) for item in eval_converted]
        answer_lengths = [len(item.get('answer', '')) for item in eval_converted]
        question_lengths = [len(item.get('query', '')) for item in eval_converted]
        
        print(f"\n📏 长度统计 (评估集):")
        print(f"   Context平均长度: {statistics.mean(context_lengths):.1f} 字符")
        print(f"   Answer平均长度: {statistics.mean(answer_lengths):.1f} 字符")
        print(f"   Question平均长度: {statistics.mean(question_lengths):.1f} 字符")
        
        # 元数据覆盖率
        doc_id_coverage = sum(1 for item in eval_converted if item.get('doc_id')) / len(eval_converted) * 100
        relevant_doc_coverage = sum(1 for item in eval_converted if item.get('relevant_doc_ids')) / len(eval_converted) * 100
        
        print(f"\n📊 元数据覆盖率:")
        print(f"   doc_id覆盖率: {doc_id_coverage:.1f}%")
        print(f"   relevant_doc_ids覆盖率: {relevant_doc_coverage:.1f}%")
        
        return total_converted, len(eval_converted)
        
    except Exception as e:
        print(f"❌ 读取TatQA转换后数据失败: {e}")
        return 0, 0

def analyze_tatqa_knowledge_base():
    """分析TatQA知识库数据"""
    print("\n=== TatQA知识库数据分析 ===\n")
    
    try:
        # 读取知识库数据
        with open('data/unified/tatqa_knowledge_base_unified.jsonl', 'r') as f:
            kb_data = [json.loads(line) for line in f if line.strip()]
        
        print(f"📊 知识库文档总数: {len(kb_data):,} 个文档")
        
        # 统计source_type分布
        source_type_stats = {}
        for item in kb_data:
            source_type = item.get('source_type', 'unknown')
            source_type_stats[source_type] = source_type_stats.get(source_type, 0) + 1
        
        print(f"\n📋 数据来源分布:")
        for source, count in source_type_stats.items():
            percentage = count / len(kb_data) * 100
            print(f"   {source}: {count:,} ({percentage:.1f}%)")
        
        # 统计文档类型（表格vs段落）
        table_count = sum(1 for item in kb_data if 'Table ID:' in item.get('context', ''))
        paragraph_count = sum(1 for item in kb_data if 'Paragraph ID:' in item.get('context', ''))
        
        print(f"\n📊 文档类型分布:")
        print(f"   表格文档: {table_count:,} ({table_count/len(kb_data)*100:.1f}%)")
        print(f"   段落文档: {paragraph_count:,} ({paragraph_count/len(kb_data)*100:.1f}%)")
        
        # 长度统计
        context_lengths = [len(item.get('context', '')) for item in kb_data]
        print(f"\n📏 文档长度统计:")
        print(f"   平均长度: {statistics.mean(context_lengths):.1f} 字符")
        print(f"   长度范围: {min(context_lengths)} - {max(context_lengths)} 字符")
        
        return len(kb_data)
        
    except Exception as e:
        print(f"❌ 读取TatQA知识库数据失败: {e}")
        return 0

def print_summary_report():
    """打印总结报告"""
    print("\n" + "="*80)
    print("📊 完整数据概况总结报告")
    print("="*80)
    
    # 收集所有统计数据
    alphafin_raw = analyze_alphafin_raw_data()
    alphafin_filtered = analyze_alphafin_filtered_data()
    alphafin_processed, alphafin_metadata = analyze_alphafin_processed_data()
    tatqa_raw, tatqa_questions = analyze_tatqa_raw_data()
    tatqa_converted, tatqa_eval = analyze_tatqa_converted_data()
    tatqa_kb = analyze_tatqa_knowledge_base()
    
    # 计算转换率
    tatqa_conversion_rate = tatqa_converted / tatqa_questions * 100 if tatqa_questions > 0 else 0
    alphafin_filter_rate = (alphafin_raw - alphafin_filtered) / alphafin_raw * 100 if alphafin_raw > 0 else 0
    
    print("\n" + "="*80)
    print("📋 最终统计总结")
    print("="*80)
    
    print("\n● 1.1 原始数据概况 (Raw Data Overview):")
    print(f"  中文数据 (AlphaFin): {alphafin_raw:,} 个样本，425M")
    print(f"  英文数据 (TatQA): {tatqa_raw:,} 个样本，18M")
    print(f"  TatQA原始问题总数: {tatqa_questions:,} 个问题")
    
    print("\n● 1.2 LLM (Qwen2-7B) 自动化数据处理:")
    print("  核心功能:")
    print("    - 元数据提取器: 自动提取company_name, stock_code, report_date")
    print("    - 问题生成器: 基于Context和Answer生成Question")
    print("    - 摘要生成器: 基于Context生成Summary")
    print(f"  元数据覆盖率: {alphafin_metadata.get('company_name', [0, 0])[1]:.1f}% (company_name)")
    
    print("\n● 1.3 处理后数据统计 (Processed Data Statistics):")
    print(f"  中文 (QCA): {alphafin_processed:,} 个样本")
    print(f"  英文 (QCA): {tatqa_converted:,} 个样本 (训练: {tatqa_converted - tatqa_eval:,}, 评估: {tatqa_eval:,})")
    
    print("\n● 1.4 TatQA 数据转换过程与质量:")
    print("  关键步骤: Table Textualization将表格转换为自然语言")
    print(f"  问题到QCA转换率: {tatqa_conversion_rate:.1f}% ({tatqa_converted:,}/{tatqa_questions:,})")
    print(f"  过滤率: {100 - tatqa_conversion_rate:.1f}%")
    print("  主要原因: answer_type=table但rel_paragraphs为空，表格转换逻辑缺陷")
    print(f"  知识库文档数: {tatqa_kb:,} 个文档")
    if tatqa_kb > 0:
        print(f"  文档利用率: 平均每个文档用于 {tatqa_converted/tatqa_kb:.1f} 个问题")
    
    print("\n● AlphaFin 数据处理流程:")
    print(f"  原始数据过滤率: {alphafin_filter_rate:.1f}%")
    print(f"  LLM处理后样本数: {alphafin_processed:,}")
    print(f"  元数据覆盖率: company_name({alphafin_metadata.get('company_name', [0, 0])[1]:.1f}%), stock_code({alphafin_metadata.get('stock_code', [0, 0])[1]:.1f}%), report_date({alphafin_metadata.get('report_date', [0, 0])[1]:.1f}%)")
    
    print("\n" + "="*80)
    print("✅ 数据概况分析完成")
    print("="*80)

def main():
    """主函数"""
    print("🚀 开始生成完整数据统计报告...")
    print_all_statistics()
    
    # 可选：保存报告到文件
    save_report = input("\n是否保存报告到文件? (y/n): ").lower().strip()
    if save_report == 'y':
        import datetime
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        report_file = f"data_statistics_report_{timestamp}.txt"
        
        # 重定向输出到文件
        import sys
        original_stdout = sys.stdout
        with open(report_file, 'w', encoding='utf-8') as f:
            sys.stdout = f
            print_all_statistics()
            sys.stdout = original_stdout
        
        print(f"📄 报告已保存到: {report_file}")

def print_all_statistics():
    """打印所有统计信息"""
    print_summary_report()

if __name__ == "__main__":
    main() 