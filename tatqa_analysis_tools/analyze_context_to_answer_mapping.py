#!/usr/bin/env python3
"""
深入分析context类型与answer_from类型之间的映射关系
"""

import json
import re
from collections import Counter, defaultdict
from pathlib import Path

def determine_context_type(context):
    """根据context内容判断结构类型"""
    
    # 检查是否包含表格标识
    has_table = "Table ID:" in context
    
    # 检查是否包含明显的文本段落（非表格内容）
    lines = context.split('\n')
    table_lines = 0
    text_lines = 0
    
    for line in lines:
        line = line.strip()
        if line.startswith(('Table ID:', 'Headers:', 'Row', 'Category:')):
            table_lines += 1
        elif line and not line.startswith(('Table ID:', 'Headers:', 'Row', 'Category:')):
            # 检查是否是有效的文本内容（不是空行或纯数字）
            if len(line) > 10 and not re.match(r'^[\d\s\-\.,$%()]+$', line):
                text_lines += 1
    
    # 判断类型
    if has_table and text_lines > 2:
        return "table-text"
    elif has_table:
        return "table"
    else:
        return "text"

def analyze_mapping_relationship(file_path):
    """分析context类型与answer_from的映射关系"""
    
    # 统计映射关系
    mapping_stats = defaultdict(Counter)
    context_answer_examples = defaultdict(list)
    
    with open(file_path, 'r', encoding='utf-8') as f:
        for i, line in enumerate(f):
            data = json.loads(line.strip())
            context = data.get('context', '')
            answer_from = data.get('answer_from', 'unknown')
            query = data.get('query', '')
            answer = data.get('answer', '')
            
            context_type = determine_context_type(context)
            
            # 统计映射关系
            mapping_stats[context_type][answer_from] += 1
            
            # 保存示例（每个映射关系保存前3个示例）
            if len(context_answer_examples[f"{context_type}->{answer_from}"]) < 3:
                example = {
                    "line": i + 1,
                    "query": query,
                    "answer": answer,
                    "context_preview": context[:300] + "..." if len(context) > 300 else context
                }
                context_answer_examples[f"{context_type}->{answer_from}"].append(example)
    
    return mapping_stats, context_answer_examples

def print_mapping_analysis(file_path, mapping_stats, context_answer_examples):
    """打印映射分析结果"""
    
    print(f"\n=== {Path(file_path).name} Context类型 -> Answer_from 映射分析 ===")
    
    total_samples = sum(sum(counter.values()) for counter in mapping_stats.values())
    print(f"总样本数: {total_samples}")
    
    # 分析每个context类型的映射
    for context_type in ["table", "text", "table-text"]:
        if context_type in mapping_stats:
            print(f"\n📊 Context类型: '{context_type}'")
            total_for_type = sum(mapping_stats[context_type].values())
            
            for answer_from, count in mapping_stats[context_type].most_common():
                percentage = (count / total_for_type) * 100
                print(f"  -> answer_from='{answer_from}': {count} ({percentage:.1f}%)")
            
            # 显示示例
            print(f"\n  📝 示例:")
            for answer_from, count in mapping_stats[context_type].most_common():
                key = f"{context_type}->{answer_from}"
                if key in context_answer_examples:
                    for i, example in enumerate(context_answer_examples[key], 1):
                        print(f"    {i}. {context_type} -> {answer_from}:")
                        print(f"       问题: {example['query']}")
                        print(f"       答案: {example['answer']}")
                        print(f"       Context预览: {example['context_preview'][:100]}...")
                        print()

def analyze_decision_rules(mapping_stats):
    """分析决策规则"""
    
    print("\n🎯 Context类型 -> Answer_from 决策规则分析")
    
    rules = {}
    
    for context_type in ["table", "text", "table-text"]:
        if context_type in mapping_stats:
            total = sum(mapping_stats[context_type].values())
            most_common = mapping_stats[context_type].most_common(1)[0]
            confidence = (most_common[1] / total) * 100
            
            rules[context_type] = {
                "most_likely": most_common[0],
                "confidence": confidence,
                "distribution": dict(mapping_stats[context_type])
            }
            
            print(f"\n📋 Context类型 '{context_type}':")
            print(f"  最可能的answer_from: '{most_common[0]}' (置信度: {confidence:.1f}%)")
            print(f"  完整分布: {dict(mapping_stats[context_type])}")
    
    return rules

def generate_decision_algorithm(rules):
    """生成决策算法"""
    
    print("\n🚀 推荐的决策算法:")
    
    algorithm = """
def predict_answer_from_by_context(context):
    \"\"\"
    根据context内容预测answer_from类型
    \"\"\"
    context_type = determine_context_type(context)
    
    # 决策规则（基于实际数据统计）
    if context_type == "text":
        return "text"  # 置信度: 100%
    elif context_type == "table":
        # 需要进一步分析，因为table context可能对应table或table-text
        return "table"  # 置信度: ~52%
    elif context_type == "table-text":
        # 需要进一步分析，因为table-text context主要对应table
        return "table"  # 置信度: ~53%
    else:
        return "unknown"
"""
    
    print(algorithm)
    
    # 提供更精确的决策逻辑
    print("\n🔍 更精确的决策逻辑:")
    print("""
def predict_answer_from_precise(context):
    \"\"\"
    更精确的answer_from预测（需要额外特征）
    \"\"\"
    context_type = determine_context_type(context)
    
    if context_type == "text":
        return "text"  # 100% 确定
    
    elif context_type == "table":
        # 分析表格是否包含需要文本解释的复杂计算
        if has_complex_calculations(context):
            return "table-text"
        else:
            return "table"
    
    elif context_type == "table-text":
        # 分析文本内容的重要性
        if text_content_is_critical(context):
            return "table-text"
        else:
            return "table"
    
    return "unknown"

def has_complex_calculations(context):
    # 检查是否包含复杂的计算说明
    calculation_keywords = ["calculate", "compute", "formula", "percentage", "ratio"]
    return any(keyword in context.lower() for keyword in calculation_keywords)

def text_content_is_critical(context):
    # 检查文本内容是否对答案至关重要
    critical_keywords = ["note", "explanation", "definition", "assumption"]
    return any(keyword in context.lower() for keyword in critical_keywords)
""")

def main():
    """主函数"""
    files_to_analyze = [
        "evaluate_mrr/tatqa_train_qc_enhanced.jsonl",
        "evaluate_mrr/tatqa_eval_enhanced.jsonl"
    ]
    
    all_mapping_stats = defaultdict(Counter)
    
    for file_path in files_to_analyze:
        if Path(file_path).exists():
            mapping_stats, context_answer_examples = analyze_mapping_relationship(file_path)
            print_mapping_analysis(file_path, mapping_stats, context_answer_examples)
            
            # 合并统计
            for context_type, counter in mapping_stats.items():
                for answer_from, count in counter.items():
                    all_mapping_stats[context_type][answer_from] += count
        else:
            print(f"\n文件不存在: {file_path}")
    
    # 总体分析
    if all_mapping_stats:
        print("\n" + "="*80)
        print("📊 总体映射关系分析")
        print("="*80)
        
        total_samples = sum(sum(counter.values()) for counter in all_mapping_stats.values())
        print(f"总样本数: {total_samples}")
        
        for context_type in ["table", "text", "table-text"]:
            if context_type in all_mapping_stats:
                print(f"\n📋 Context类型 '{context_type}':")
                total_for_type = sum(all_mapping_stats[context_type].values())
                
                for answer_from, count in all_mapping_stats[context_type].most_common():
                    percentage = (count / total_for_type) * 100
                    print(f"  -> answer_from='{answer_from}': {count} ({percentage:.1f}%)")
        
        # 生成决策规则
        rules = analyze_decision_rules(all_mapping_stats)
        generate_decision_algorithm(rules)

if __name__ == "__main__":
    main() 