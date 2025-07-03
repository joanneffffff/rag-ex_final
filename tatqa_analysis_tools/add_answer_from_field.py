#!/usr/bin/env python3
"""
为测试数据文件添加answer_from字段
通过分析context内容来判断类型
"""

import json
import re
from pathlib import Path

def determine_answer_from(context):
    """根据context内容判断answer_from类型"""
    
    # 检查是否包含表格标识
    if "Table ID:" in context:
        # 进一步检查是否包含表格结构特征
        if re.search(r'Headers:|Row \d+:|Category:', context):
            # 检查是否包含文本段落（非表格内容）
            # 如果context中除了表格还有明显的文本段落，则为table-text
            lines = context.split('\n')
            table_lines = 0
            text_lines = 0
            
            for line in lines:
                line = line.strip()
                if line.startswith(('Table ID:', 'Headers:', 'Row', 'Category:')):
                    table_lines += 1
                elif line and not line.startswith(('Table ID:', 'Headers:', 'Row', 'Category:')):
                    # 检查是否是表格数据行
                    if re.match(r'^[^:]+:.*is.*;', line) or re.match(r'^[^:]+:.*is.*$', line):
                        table_lines += 1
                    else:
                        text_lines += 1
            
            # 如果文本行数较多，可能是table-text
            if text_lines > 2:  # 阈值可以调整
                return "table-text"
            else:
                return "table"
        else:
            return "table-text"
    else:
        return "text"

def add_answer_from_field(input_file, output_file):
    """为数据文件添加answer_from字段"""
    
    updated_samples = []
    
    with open(input_file, 'r', encoding='utf-8') as f:
        for line in f:
            data = json.loads(line.strip())
            context = data.get('context', '')
            
            # 确定answer_from类型
            answer_from = determine_answer_from(context)
            
            # 添加answer_from字段
            data['answer_from'] = answer_from
            updated_samples.append(data)
    
    # 保存更新后的数据
    with open(output_file, 'w', encoding='utf-8') as f:
        for sample in updated_samples:
            f.write(json.dumps(sample, ensure_ascii=False) + '\n')
    
    # 统计分布
    type_counts = {}
    for sample in updated_samples:
        answer_from = sample['answer_from']
        type_counts[answer_from] = type_counts.get(answer_from, 0) + 1
    
    print(f"成功更新文件: {output_file}")
    print(f"总样本数: {len(updated_samples)}")
    print("\n类型分布:")
    for answer_type, count in type_counts.items():
        percentage = (count / len(updated_samples)) * 100
        print(f"  {answer_type}: {count} ({percentage:.1f}%)")
    
    return updated_samples

def main():
    input_file = "evaluate_mrr/tatqa_test_15_samples.jsonl"
    output_file = "evaluate_mrr/tatqa_test_15_samples.jsonl"  # 覆盖原文件
    
    if not Path(input_file).exists():
        print(f"输入文件不存在: {input_file}")
        return
    
    add_answer_from_field(input_file, output_file)

if __name__ == "__main__":
    main() 