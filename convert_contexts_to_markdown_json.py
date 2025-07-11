#!/usr/bin/env python3
"""
Batch convert each sample's context field to Markdown table string and store as markdown_table in a new JSON file.
"""
import json
import re
from typing import List, Dict

def clean_value(value: str) -> str:
    value = value.rstrip('.')
    if 'a negative ' in value:
        value = '-' + value.replace('a negative ', '')
    elif value.startswith('$(') and value.endswith(')'):
        value = '-' + value[2:-1]
    elif value.startswith('$') and '(' in value and ')' in value:
        match = re.search(r'\(\$?([^)]+)\)', value)
        if match:
            value = '-' + match.group(1)
    value = value.replace(',', '')
    return value

def parse_text_table_to_markdown(context_text: str) -> str:
    table_id_match = re.search(r'Table ID: ([^\n]+)', context_text)
    table_id = table_id_match.group(1) if table_id_match else "Unknown"
    lines = context_text.split('\n')
    data_rows = []
    current_category = ""
    for line in lines:
        line = line.strip()
        if not line:
            continue
        if line.startswith('Category:'):
            current_category = line.replace('Category:', '').strip()
            continue
        if line.startswith('For ') and 'is ' in line:
            parts = line.split(':')
            if len(parts) >= 2:
                item_name = parts[0].replace('For ', '').strip()
                values_part = parts[1].strip()
                value_matches = re.findall(r'is ([^,]+?)(?:,|$)', values_part)
                if len(value_matches) >= 2:
                    value1 = clean_value(value_matches[0].strip())
                    value2 = clean_value(value_matches[1].strip())
                    data_rows.append({
                        'category': current_category,
                        'item': item_name,
                        'value1': value1,
                        'value2': value2
                    })
    markdown = f"# Table ID: {table_id}\n\n"
    markdown += "Financial data for June 30, 2019 and 2018. All monetary amounts are in thousands unless otherwise specified.\n\n"
    markdown += "| Item | June 30, 2019 | June 30, 2018 |\n"
    markdown += "|------|---------------|---------------|\n"
    current_cat = ""
    for row in data_rows:
        item = row['item']
        value1 = row['value1']
        value2 = row['value2']
        category = row['category']
        if category and category != current_cat:
            markdown += f"| **{category}** | | |\n"
            current_cat = category
        markdown += f"| {item} | {value1} | {value2} |\n"
    return markdown

def main():
    input_json = 'evaluate_mrr/tatqa_test_15_samples.json'
    output_json = 'evaluate_mrr/tatqa_test_15_samples_with_markdown.json'
    with open(input_json, 'r', encoding='utf-8') as f:
        data = json.load(f)
    for sample in data:
        context = sample.get('context', '')
        # 只对包含表格结构的 context 进行转换
        if 'Table ID:' in context and 'is ' in context:
            markdown_table = parse_text_table_to_markdown(context)
        else:
            markdown_table = ''
        sample['markdown_table'] = markdown_table
    with open(output_json, 'w', encoding='utf-8') as f:
        json.dump(data, f, ensure_ascii=False, indent=2)
    print(f'✅ Saved with markdown_table to: {output_json}')

if __name__ == '__main__':
    main() 