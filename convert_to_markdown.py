#!/usr/bin/env python3
"""
Convert evaluate_mrr/tatqa_test_15_samples.json to Markdown table format
"""

import json
import sys
from pathlib import Path
from typing import Optional

def convert_json_to_markdown(json_file_path: str, output_file_path: Optional[str] = None):
    """Convert JSON file to Markdown table format"""
    
    # Read JSON file
    try:
        with open(json_file_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
    except FileNotFoundError:
        print(f"❌ File not found: {json_file_path}")
        return
    except json.JSONDecodeError as e:
        print(f"❌ JSON parsing error: {e}")
        return
    
    if not isinstance(data, list):
        print("❌ Data format error: expected JSON array")
        return
    
    # Generate Markdown table
    markdown_content = "# TATQA Test Dataset\n\n"
    markdown_content += "| No. | Question | Answer | Answer Source | Document ID |\n"
    markdown_content += "|-----|----------|--------|---------------|-------------|\n"
    
    for i, sample in enumerate(data, 1):
        # Extract fields
        query = sample.get("query", "")
        answer = sample.get("answer", "")
        answer_from = sample.get("answer_from", "")
        doc_id = sample.get("doc_id", "")
        
        # Handle Markdown special characters
        query = query.replace("|", "\\|").replace("\n", "<br>")
        answer = answer.replace("|", "\\|").replace("\n", "<br>")
        
        # Generate table row
        row = f"| {i} | {query} | {answer} | {answer_from} | {doc_id} |\n"
        markdown_content += row
    
    # Add statistics
    markdown_content += "\n## Dataset Statistics\n\n"
    markdown_content += f"- **Total Samples**: {len(data)}\n"
    
    # Statistics by answer source
    answer_from_counts = {}
    for sample in data:
        source = sample.get("answer_from", "unknown")
        answer_from_counts[source] = answer_from_counts.get(source, 0) + 1
    
    markdown_content += "- **Answer Source Distribution**:\n"
    for source, count in answer_from_counts.items():
        percentage = (count / len(data)) * 100
        markdown_content += f"  - {source}: {count} ({percentage:.1f}%)\n"
    
    # Save to file
    if output_file_path is None:
        output_file_path = json_file_path.replace('.json', '_markdown.md')
    
    try:
        with open(output_file_path, 'w', encoding='utf-8') as f:
            f.write(markdown_content)
        print(f"✅ Markdown table saved to: {output_file_path}")
        print(f"📊 Converted {len(data)} samples")
    except Exception as e:
        print(f"❌ Error saving file: {e}")

def main():
    if len(sys.argv) < 2:
        print("Usage: python convert_to_markdown.py <json_file_path> [output_file_path]")
        sys.exit(1)
    
    json_file_path = sys.argv[1]
    output_file_path = sys.argv[2] if len(sys.argv) > 2 else None
    
    convert_json_to_markdown(json_file_path, output_file_path)

if __name__ == "__main__":
    main() 