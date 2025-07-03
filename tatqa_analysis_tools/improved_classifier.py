#!/usr/bin/env python3
"""
改进的TAT-QA问题类型分类器
更精确地判断table、text和table+text类型
"""

import re
import json
from typing import Dict, List, Tuple

def improved_classify_question_type(context: str) -> str:
    """
    改进的问题类型分类方法
    """
    # 检查是否包含表格
    has_table = "Table ID:" in context
    
    if not has_table:
        return "text"
    
    # 如果有表格，进一步分析
    table_pattern = r'Table ID: [^\n]+\nHeaders:'
    table_matches = list(re.finditer(table_pattern, context))
    
    if not table_matches:
        return "table"  # 有Table ID但没有标准格式，仍算作table
    
    # 分析表格前后的文本
    table_start = table_matches[0].start()
    table_end = context.rfind('\n')  # 假设表格到文件末尾
    
    # 检查表格前的文本
    text_before = context[:table_start].strip()
    
    # 检查表格后的文本（如果有多个表格，检查最后一个表格后的文本）
    if len(table_matches) > 1:
        last_table_end = table_matches[-1].end()
        text_after = context[last_table_end:].strip()
    else:
        text_after = context[table_end:].strip()
    
    # 判断是否有有意义的文本段落
    has_meaningful_text = False
    
    # 检查表格前的文本
    if len(text_before) > 30:  # 降低阈值，30字符以上算有意义
        # 检查是否包含完整的句子
        sentences = re.split(r'[.!?]+', text_before)
        if any(len(s.strip()) > 20 for s in sentences):
            has_meaningful_text = True
    
    # 检查表格后的文本
    if len(text_after) > 30:
        sentences = re.split(r'[.!?]+', text_after)
        if any(len(s.strip()) > 20 for s in sentences):
            has_meaningful_text = True
    
    # 检查表格中间是否有文本（多个表格之间）
    if len(table_matches) > 1:
        for i in range(len(table_matches) - 1):
            between_text = context[table_matches[i].end():table_matches[i+1].start()].strip()
            if len(between_text) > 30:
                sentences = re.split(r'[.!?]+', between_text)
                if any(len(s.strip()) > 20 for s in sentences):
                    has_meaningful_text = True
                    break
    
    if has_meaningful_text:
        return "table+text"
    else:
        return "table"

def analyze_sample_distribution(data_file: str) -> Dict[str, any]:
    """
    分析样本分布，使用改进的分类器
    """
    print(f"🔍 使用改进分类器分析: {data_file}")
    
    type_counts = {"table": 0, "text": 0, "table+text": 0}
    samples_by_type = {"table": [], "text": [], "table+text": []}
    
    with open(data_file, 'r', encoding='utf-8') as f:
        for line_num, line in enumerate(f, 1):
            try:
                sample = json.loads(line.strip())
                context = sample["context"]
                question_type = improved_classify_question_type(context)
                
                type_counts[question_type] += 1
                samples_by_type[question_type].append({
                    "line": line_num,
                    "query": sample["query"][:100] + "...",
                    "context_preview": context[:200] + "...",
                    "doc_id": sample.get("doc_id", "unknown")
                })
                
            except Exception as e:
                print(f"⚠️ 第{line_num}行处理错误: {e}")
                continue
    
    total = sum(type_counts.values())
    
    print(f"📊 改进分类结果:")
    for qtype, count in type_counts.items():
        percentage = (count / total * 100) if total > 0 else 0
        print(f"  {qtype}: {count} ({percentage:.1f}%)")
    
    # 显示一些table+text的示例
    if samples_by_type["table+text"]:
        print(f"\n📋 table+text示例 (前3个):")
        for i, sample in enumerate(samples_by_type["table+text"][:3]):
            print(f"  示例{i+1}:")
            print(f"    问题: {sample['query']}")
            print(f"    文档ID: {sample['doc_id']}")
            print(f"    Context预览: {sample['context_preview']}")
            print()
    
    return {
        "type_counts": type_counts,
        "samples_by_type": samples_by_type,
        "total_samples": total
    }

def main():
    """主函数"""
    data_file = '../evaluate_mrr/tatqa_eval_enhanced.jsonl'
    
    print("🚀 改进的TAT-QA问题类型分类器")
    print("="*50)
    
    # 分析评估集
    results = analyze_sample_distribution(data_file)
    
    # 保存详细结果
    output_file = "improved_classification_results.json"
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    
    print(f"✅ 详细结果已保存到: {output_file}")

if __name__ == "__main__":
    main() 