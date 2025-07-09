#!/usr/bin/env python3
"""
检查知识库数据中的doc_id字段
"""

import json
from pathlib import Path
from typing import Dict, List, Any

def check_doc_ids_in_file(file_path: str, max_samples: int = 100) -> Dict[str, Any]:
    """检查文件中的doc_id字段"""
    print(f"检查文件: {file_path}")
    
    if not Path(file_path).exists():
        print(f"❌ 文件不存在: {file_path}")
        return {}
    
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            # 检查文件格式
            first_char = f.read(1)
            f.seek(0)
            
            if first_char == '[':
                # JSON数组格式
                data = json.load(f)
                print(f"检测到JSON数组格式，共 {len(data)} 条记录")
            else:
                # JSONL格式
                data = []
                for line in f:
                    if line.strip():
                        data.append(json.loads(line))
                print(f"检测到JSONL格式，共 {len(data)} 条记录")
        
        # 统计doc_id情况
        total_records = len(data)
        records_with_doc_id = 0
        records_without_doc_id = 0
        doc_id_formats = {}
        sample_records = []
        
        for i, record in enumerate(data):
            if i >= max_samples:
                break
                
            doc_id = record.get('doc_id')
            
            if doc_id:
                records_with_doc_id += 1
                # 分析doc_id格式
                if isinstance(doc_id, str):
                    if doc_id.startswith('raw_doc_'):
                        format_type = 'raw_doc_*'
                    elif doc_id.startswith('chunk_'):
                        format_type = 'chunk_*'
                    elif doc_id.startswith('generated_doc_'):
                        format_type = 'generated_doc_*'
                    elif doc_id.startswith('train_optimized_'):
                        format_type = 'train_optimized_*'
                    elif len(doc_id) == 16 and all(c in '0123456789abcdef' for c in doc_id):
                        format_type = 'hash_16'
                    elif len(doc_id) == 32 and all(c in '0123456789abcdef' for c in doc_id):
                        format_type = 'hash_32'
                    else:
                        format_type = 'other'
                else:
                    format_type = 'non_string'
                
                doc_id_formats[format_type] = doc_id_formats.get(format_type, 0) + 1
                
                # 保存样本记录
                if len(sample_records) < 5:
                    sample_records.append({
                        'index': i,
                        'doc_id': doc_id,
                        'format': format_type,
                        'has_context': 'context' in record,
                        'has_question': 'question' in record,
                        'context_preview': record.get('context', '')[:100] + '...' if record.get('context') else 'N/A'
                    })
            else:
                records_without_doc_id += 1
                # 保存没有doc_id的样本
                if len(sample_records) < 5:
                    sample_records.append({
                        'index': i,
                        'doc_id': 'MISSING',
                        'format': 'missing',
                        'has_context': 'context' in record,
                        'has_question': 'question' in record,
                        'context_preview': record.get('context', '')[:100] + '...' if record.get('context') else 'N/A'
                    })
        
        # 计算覆盖率
        coverage_rate = records_with_doc_id / total_records if total_records > 0 else 0
        
        return {
            'file_path': file_path,
            'total_records': total_records,
            'records_with_doc_id': records_with_doc_id,
            'records_without_doc_id': records_without_doc_id,
            'coverage_rate': coverage_rate,
            'doc_id_formats': doc_id_formats,
            'sample_records': sample_records
        }
        
    except Exception as e:
        print(f"❌ 检查文件时出错: {e}")
        return {}

def main():
    """主函数"""
    # 检查配置文件中的数据路径
    config_files = [
        "data/alphafin/alphafin_final_clean.json",  # 中文知识库
        "data/unified/tatqa_knowledge_base_combined.jsonl",  # 英文知识库
        "evaluate_mrr/alphafin_eval.jsonl",  # 评测数据
        "evaluate_mrr/tatqa_eval.jsonl"  # TatQA评测数据
    ]
    
    results = {}
    
    for file_path in config_files:
        print(f"\n{'='*60}")
        result = check_doc_ids_in_file(file_path, max_samples=1000)
        if result:
            results[file_path] = result
    
    # 打印汇总报告
    print(f"\n{'='*60}")
    print("知识库doc_id检查汇总报告")
    print(f"{'='*60}")
    
    for file_path, result in results.items():
        print(f"\n文件: {file_path}")
        print(f"总记录数: {result['total_records']}")
        print(f"有doc_id的记录: {result['records_with_doc_id']}")
        print(f"无doc_id的记录: {result['records_without_doc_id']}")
        print(f"覆盖率: {result['coverage_rate']:.2%}")
        
        if result['doc_id_formats']:
            print("doc_id格式分布:")
            for format_type, count in result['doc_id_formats'].items():
                print(f"  {format_type}: {count}")
        
        print("\n样本记录:")
        for sample in result['sample_records']:
            print(f"  索引 {sample['index']}: {sample['doc_id']} ({sample['format']})")
            print(f"    有context: {sample['has_context']}, 有问题: {sample['has_question']}")
            print(f"    内容预览: {sample['context_preview']}")
            print()

if __name__ == "__main__":
    main() 