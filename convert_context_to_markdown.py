#!/usr/bin/env python3
"""
Convert text-based table context to Markdown table format
"""

import re
from typing import Dict, List, Tuple

def parse_text_table_to_markdown(context_text: str) -> str:
    """Convert text-based table to Markdown format"""
    
    # Extract table ID and basic info
    table_id_match = re.search(r'Table ID: ([^\n]+)', context_text)
    table_id = table_id_match.group(1) if table_id_match else "Unknown"
    
    # Parse the data rows
    lines = context_text.split('\n')
    data_rows = []
    current_category = ""
    
    for line in lines:
        line = line.strip()
        if not line:
            continue
            
        # Check for category headers
        if line.startswith('Category:'):
            current_category = line.replace('Category:', '').strip()
            continue
            
        # Check for data rows
        if line.startswith('For ') and 'is ' in line:
            # Parse data row
            parts = line.split(':')
            if len(parts) >= 2:
                item_name = parts[0].replace('For ', '').strip()
                values_part = parts[1].strip()
                
                # Extract values using more precise regex
                value_matches = re.findall(r'is ([^,]+?)(?:,|$)', values_part)
                if len(value_matches) >= 2:
                    value1 = value_matches[0].strip()
                    value2 = value_matches[1].strip()
                    
                    # Clean up values - handle negative numbers properly
                    value1 = clean_value(value1)
                    value2 = clean_value(value2)
                    
                    data_rows.append({
                        'category': current_category,
                        'item': item_name,
                        'value1': value1,
                        'value2': value2
                    })
    
    # Generate Markdown table with categories
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
        
        # Add category header if it changes
        if category and category != current_cat:
            markdown += f"| **{category}** | | |\n"
            current_cat = category
        
        # Format the row
        markdown += f"| {item} | {value1} | {value2} |\n"
    
    return markdown

def clean_value(value: str) -> str:
    """Clean up value formatting"""
    # Remove trailing periods
    value = value.rstrip('.')
    
    # Handle negative numbers in various formats
    if 'a negative ' in value:
        value = '-' + value.replace('a negative ', '')
    elif value.startswith('$(') and value.endswith(')'):
        value = '-' + value[2:-1]  # Remove $( and ) and add minus
    elif value.startswith('$') and '(' in value and ')' in value:
        # Handle $(77,328) format
        match = re.search(r'\(\$?([^)]+)\)', value)
        if match:
            value = '-' + match.group(1)
    
    # Remove commas from numbers
    value = value.replace(',', '')
    
    return value

def main():
    # Example context from the first sample
    context_text = """Table ID: b3d63fb06110ad7e91c9e765227c1d27
Table columns: , June 30,, .
Row 1 data: June 30, is 2019, Value is 2018.
Category: Deferred tax assets.
For Non-capital loss carryforwards: June 30, is 161119, Value is 129436.
For Capital loss carryforwards: June 30, is 155, Value is 417.
For Undeducted scientific research and development expenses: June 30, is 137253, Value is 123114.
For Depreciation and amortization: June 30, is 683777, Value is 829369.
For Restructuring costs and other reserves: June 30, is 17845, Value is 17202.
For Deferred revenue: June 30, is 53254, Value is 62726.
For Other: June 30, is 59584, Value is 57461.
For Total deferred tax asset: June 30, is 1112987, Value is 1219725.
For Valuation Allowance: June 30, is $(77,328), Value is $(80,924).
Category: Deferred tax liabilities.
For Scientific research and development tax credits: June 30, is $(14,482), Value is $(13,342).
For Other: June 30, is a negative 72599, Value is a negative 82668.
For Deferred tax liabilities: June 30, is $(87,081), Value is $(96,010).
For Net deferred tax asset: June 30, is 948578, Value is 1042791.
Category: Comprised of:.
For Long-term assets: June 30, is 1004450, Value is 1122729.
For Long-term liabilities: June 30, is a negative 55872, Value is a negative 79938.
Row 20 data: June 30, is 948578, Value is 1042791."""
    
    markdown_table = parse_text_table_to_markdown(context_text)
    print(markdown_table)
    
    # Save to file
    with open('table_context_markdown.md', 'w', encoding='utf-8') as f:
        f.write(markdown_table)
    print("\n✅ Markdown table saved to: table_context_markdown.md")

if __name__ == "__main__":
    main() 