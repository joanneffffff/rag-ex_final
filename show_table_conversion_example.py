#!/usr/bin/env python3
"""
展示表格转换为自然文本的具体示例
"""

import json
import re

def table_to_natural_text(table_dict, caption="", unit_info=""):
    """表格转换为自然文本的函数"""
    if not table_dict:
        return ""
    
    # 处理不同的表格格式
    if isinstance(table_dict, dict):
        rows = table_dict.get("table", [])
        table_uid = table_dict.get("uid", "")
    elif isinstance(table_dict, list):
        rows = table_dict
        table_uid = ""
    else:
        return ""
    
    if not rows:
        return ""
    
    lines = []
    
    # 添加表格标识
    if table_uid:
        lines.append(f"Table ID: {table_uid}")
    if caption:
        lines.append(f"Table Topic: {caption}")

    headers = rows[0] if rows else []
    data_rows = rows[1:] if len(rows) > 1 else []

    # 处理表头
    if headers:
        header_text = " | ".join(str(h).strip() for h in headers if str(h).strip())
        if header_text:
            lines.append(f"Headers: {header_text}")

    # 处理数据行
    for i, row in enumerate(data_rows):
        if not row:
            continue
            
        # 跳过完全空的行
        if all(str(v).strip() == "" for v in row):
            continue

        # 处理分类行（第一列有值，其他列为空）
        if len(row) > 1 and str(row[0]).strip() != "" and all(str(v).strip() == "" for v in row[1:]):
            lines.append(f"Category: {str(row[0]).strip()}")
            continue

        # 处理数据行
        row_name = str(row[0]).strip().replace('.', '') if row[0] else ""
        
        data_descriptions = []
        for h_idx, v in enumerate(row):
            if h_idx == 0:  # 跳过第一列（通常是行名）
                continue
            
            header = headers[h_idx] if h_idx < len(headers) else f"Column {h_idx+1}"
            value = str(v).strip()

            if value:
                # 格式化数值
                if re.match(r'^-?\$?\d{1,3}(?:,\d{3})*(?:\.\d+)?$|^\(\$?\d{1,3}(?:,\d{3})*(?:\.\d+)?\)$', value): 
                    formatted_value = value.replace('$', '')
                    if unit_info:
                        if formatted_value.startswith('(') and formatted_value.endswith(')'):
                             formatted_value = f"(${formatted_value[1:-1]} {unit_info})"
                        else:
                             formatted_value = f"${formatted_value} {unit_info}"
                    else:
                        formatted_value = f"${formatted_value}"
                else:
                    formatted_value = value
                
                data_descriptions.append(f"{header} is {formatted_value}")

        # 构建行描述
        if row_name and data_descriptions:
            lines.append(f"{row_name}: {'; '.join(data_descriptions)}")
        elif data_descriptions:
            lines.append(f"Row {i+1}: {'; '.join(data_descriptions)}")
        elif row_name:
            lines.append(f"Item: {row_name}")

    return "\n".join(lines)

def show_conversion_examples():
    """展示转换示例"""
    
    print("=== 表格转换为自然文本示例 ===\n")
    
    # 示例1：简单的财务表格
    print("示例1：财务表格")
    print("原始表格数据:")
    table1 = {
        "uid": "dc9d58a4e24a74d52f719372c1a16e7f",
        "table": [
            ["Current assets", "As Reported", "Adjustments", "Balances without Adoption of Topic 606"],
            ["Receivables, less allowance for doubtful accounts", "$831.7", "$8.7", "$840.4"],
            ["Inventories", "$1,571.7", "($3.1)", "$1,568.6"],
            ["Prepaid expenses and other current assets", "$93.8", "($16.6)", "$77.2"],
            ["", "", "", ""],
            ["Current liabilities", "", "", ""],
            ["Other accrued liabilities", "$691.6", "($1.1)", "$690.5"],
            ["Other noncurrent liabilities", "$1,951.8", "($2.5)", "$1,949.3"]
        ]
    }
    
    print("表格结构:")
    for i, row in enumerate(table1["table"]):
        print(f"  行{i}: {row}")
    
    print("\n转换后的自然文本:")
    converted1 = table_to_natural_text(table1, "Financial Statement Adjustments", "million USD")
    print(converted1)
    
    print("\n" + "="*80 + "\n")
    
    # 示例2：年度费用表格
    print("示例2：年度费用表格")
    print("原始表格数据:")
    table2 = {
        "uid": "33295076b558d53b86fd6e5537022af6",
        "table": [
            ["Years Ended", "July 27, 2019", "July 28, 2018", "July 29, 2017", "Variance in Dollars", "Variance in Percent"],
            ["Research and development", "$6,577", "$6,332", "$6,059", "$245", "4%"],
            ["Percentage of revenue", "12.7%", "12.8%", "12.6%", "", ""],
            ["Sales and marketing", "$9,571", "$9,242", "$9,184", "$329", "4%"],
            ["Percentage of revenue", "18.4%", "18.7%", "19.1%", "", ""],
            ["General and administrative", "$1,827", "$2,144", "$1,993", "($317)", "(15)%"],
            ["Percentage of revenue", "3.5%", "4.3%", "4.2%", "", ""],
            ["Total", "$17,975", "$17,718", "$17,236", "$257", "1%"],
            ["Percentage of revenue", "34.6%", "35.9%", "35.9%", "", ""]
        ]
    }
    
    print("表格结构:")
    for i, row in enumerate(table2["table"]):
        print(f"  行{i}: {row}")
    
    print("\n转换后的自然文本:")
    converted2 = table_to_natural_text(table2, "Annual Operating Expenses", "million USD")
    print(converted2)
    
    print("\n" + "="*80 + "\n")
    
    # 示例3：简单的数据表格
    print("示例3：简单数据表格")
    print("原始表格数据:")
    table3 = {
        "table": [
            ["Product", "Q1 Sales", "Q2 Sales", "Q3 Sales", "Q4 Sales"],
            ["Product A", "100", "120", "110", "130"],
            ["Product B", "80", "90", "85", "95"],
            ["Product C", "60", "70", "65", "75"]
        ]
    }
    
    print("表格结构:")
    for i, row in enumerate(table3["table"]):
        print(f"  行{i}: {row}")
    
    print("\n转换后的自然文本:")
    converted3 = table_to_natural_text(table3, "Quarterly Sales by Product", "units")
    print(converted3)

def show_real_data_example():
    """展示真实数据的转换示例"""
    
    print("\n=== 真实数据转换示例 ===\n")
    
    # 加载评估数据中的表格样本
    with open('evaluate_mrr/tatqa_eval_enhanced.jsonl', 'r') as f:
        eval_data = [json.loads(line) for line in f]
    
    # 找出表格样本
    table_samples = [sample for sample in eval_data if 'Table ID:' in sample.get('context', '')]
    
    if table_samples:
        sample = table_samples[0]
        print("真实TatQA数据示例:")
        print(f"问题: {sample['query']}")
        print(f"答案: {sample['answer']}")
        print(f"转换后的context:")
        print(sample['context'])
        
        # 分析context结构
        lines = sample['context'].split('\n')
        print(f"\n结构分析:")
        for i, line in enumerate(lines):
            if line.strip():
                if line.startswith('Table ID:'):
                    print(f"  {i+1}. 表格标识: {line}")
                elif line.startswith('Headers:'):
                    print(f"  {i+1}. 表头: {line}")
                elif line.startswith('Category:'):
                    print(f"  {i+1}. 分类: {line}")
                elif ':' in line and not line.startswith('Table'):
                    print(f"  {i+1}. 数据行: {line[:50]}...")
                else:
                    print(f"  {i+1}. 其他: {line}")

if __name__ == "__main__":
    show_conversion_examples()
    show_real_data_example() 