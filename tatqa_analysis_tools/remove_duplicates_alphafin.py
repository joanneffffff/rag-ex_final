#!/usr/bin/env python3
"""
检测和删除AlphaFin数据集中的重复数据
"""

import json
import hashlib
from collections import defaultdict
from pathlib import Path
import argparse

def calculate_content_hash(content: str) -> str:
    """计算内容的哈希值，用于检测重复"""
    return hashlib.md5(content.encode('utf-8')).hexdigest()

def load_data(file_path: str) -> list:
    """加载数据文件，支持json和jsonl格式"""
    if file_path.endswith('.jsonl'):
        data = []
        with open(file_path, 'r', encoding='utf-8') as f:
            for line in f:
                if line.strip():
                    data.append(json.loads(line))
    else:
        with open(file_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
    return data

def save_data(data: list, output_path: str):
    """保存数据文件，支持json和jsonl格式"""
    if output_path.endswith('.jsonl'):
        with open(output_path, 'w', encoding='utf-8') as f:
            for item in data:
                f.write(json.dumps(item, ensure_ascii=False) + '\n')
    else:
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(data, f, ensure_ascii=False, indent=2)

def analyze_duplicates(file_path: str) -> tuple[dict, list]:
    """分析文件中的重复数据"""
    print(f"🔍 分析文件: {file_path}")
    
    duplicates = {
        'context_duplicates': defaultdict(list),
        'answer_duplicates': defaultdict(list),
        'full_duplicates': defaultdict(list),
        'total_records': 0,
        'unique_records': 0
    }
    
    data = load_data(file_path)
    
    duplicates['total_records'] = len(data)
    print(f"📊 总记录数: {len(data)}")
    
    # 检测重复
    for i, record in enumerate(data):
        # 获取字段内容
        context = record.get('context', record.get('original_context', ''))
        answer = record.get('answer', record.get('original_answer', ''))
        question = record.get('query', record.get('original_question', ''))
        
        # 计算哈希值
        context_hash = calculate_content_hash(context)
        answer_hash = calculate_content_hash(answer)
        full_content = f"{context}|{answer}|{question}"
        full_hash = calculate_content_hash(full_content)
        
        # 记录重复
        duplicates['context_duplicates'][context_hash].append(i)
        duplicates['answer_duplicates'][answer_hash].append(i)
        duplicates['full_duplicates'][full_hash].append(i)
    
    # 统计重复情况
    context_dups = sum(1 for indices in duplicates['context_duplicates'].values() if len(indices) > 1)
    answer_dups = sum(1 for indices in duplicates['answer_duplicates'].values() if len(indices) > 1)
    full_dups = sum(1 for indices in duplicates['full_duplicates'].values() if len(indices) > 1)
    
    print(f"📋 Context重复组数: {context_dups}")
    print(f"📋 Answer重复组数: {answer_dups}")
    print(f"📋 完全重复组数: {full_dups}")
    
    return duplicates, data

def remove_duplicates(file_path: str, output_path: str, duplicate_type: str = 'full') -> dict:
    """删除重复数据并保存清理后的文件"""
    print(f"🧹 开始删除重复数据...")
    print(f"📁 输入文件: {file_path}")
    print(f"📁 输出文件: {output_path}")
    print(f"🎯 重复类型: {duplicate_type}")
    
    data = load_data(file_path)
    original_count = len(data)
    print(f"📊 原始记录数: {original_count}")
    
    # 检测重复
    duplicates, _ = analyze_duplicates(file_path)
    
    # 根据重复类型选择要保留的记录
    if duplicate_type == 'full':
        duplicate_groups = duplicates['full_duplicates']
    elif duplicate_type == 'context':
        duplicate_groups = duplicates['context_duplicates']
    elif duplicate_type == 'answer':
        duplicate_groups = duplicates['answer_duplicates']
    else:
        raise ValueError(f"不支持的重复类型: {duplicate_type}")
    
    # 找出要保留的记录索引（每组保留第一个）
    keep_indices = set()
    removed_indices = set()
    
    for hash_val, indices in duplicate_groups.items():
        if len(indices) > 1:
            # 保留第一个，删除其余的
            keep_indices.add(indices[0])
            removed_indices.update(indices[1:])
        else:
            # 没有重复，保留
            keep_indices.add(indices[0])
    
    # 创建清理后的数据
    cleaned_data = [data[i] for i in range(len(data)) if i in keep_indices]
    
    # 保存清理后的文件
    save_data(cleaned_data, output_path)
    
    removed_count = len(removed_indices)
    cleaned_count = len(cleaned_data)
    
    print(f"✅ 清理完成!")
    print(f"📊 删除记录数: {removed_count}")
    print(f"📊 保留记录数: {cleaned_count}")
    print(f"📊 重复率: {removed_count/original_count*100:.2f}%")
    
    return {
        'original_count': original_count,
        'cleaned_count': cleaned_count,
        'removed_count': removed_count,
        'duplicate_rate': removed_count/original_count*100
    }

def main():
    parser = argparse.ArgumentParser(description="删除AlphaFin数据集中的重复数据")
    parser.add_argument("--input", type=str, required=True, help="输入文件路径")
    parser.add_argument("--output", type=str, help="输出文件路径")
    parser.add_argument("--type", type=str, default="full", 
                       choices=["full", "context", "answer"], 
                       help="重复检测类型: full(完全重复), context(context重复), answer(answer重复)")
    parser.add_argument("--analyze-only", action="store_true", help="仅分析，不删除")
    
    args = parser.parse_args()
    
    if args.analyze_only:
        # 仅分析重复情况
        duplicates, data = analyze_duplicates(args.input)
        
        # 显示详细的重复信息
        print(f"\n📋 详细重复信息:")
        for dup_type, dup_data in duplicates.items():
            if dup_type.endswith('_duplicates'):
                dup_count = sum(1 for indices in dup_data.values() if len(indices) > 1)
                print(f"   {dup_type}: {dup_count} 组重复")
                
                # 显示前几个重复组的详细信息
                count = 0
                for hash_val, indices in dup_data.items():
                    if len(indices) > 1 and count < 3:
                        print(f"     重复组 {count+1}: {len(indices)} 个记录 (索引: {indices[:5]}{'...' if len(indices) > 5 else ''})")
                        # 显示重复记录的内容示例
                        if count == 0:
                            print(f"       示例记录 {indices[0]}:")
                            record = data[indices[0]]
                            context_preview = record.get('context', record.get('original_context', ''))[:100]
                            answer_preview = record.get('answer', record.get('original_answer', ''))[:50]
                            print(f"         Context: {context_preview}...")
                            print(f"         Answer: {answer_preview}...")
                        count += 1
    else:
        # 删除重复数据
        if not args.output:
            print("❌ 错误: 删除重复数据时需要指定 --output 参数")
            return
        result = remove_duplicates(args.input, args.output, args.type)
        
        # 保存分析报告
        report_path = args.output.replace('.json', '_cleaning_report.json').replace('.jsonl', '_cleaning_report.json')
        with open(report_path, 'w', encoding='utf-8') as f:
            json.dump(result, f, ensure_ascii=False, indent=2)
        print(f"📄 清理报告已保存到: {report_path}")

if __name__ == "__main__":
    main() 