#!/usr/bin/env python3
"""
提取摘要数据脚本
从训练和评估数据中提取generated_question、summary和doc_id字段
只使用真正的摘要，不使用完整的上下文
"""

import json
import argparse
from pathlib import Path

def extract_summary_data(input_file, output_file, data_type="train"):
    """
    从输入文件中提取摘要数据
    
    Args:
        input_file: 输入文件路径
        output_file: 输出文件路径
        data_type: 数据类型 ("train" 或 "eval")
    """
    print(f"处理 {data_type} 数据: {input_file}")
    
    extracted_data = []
    skipped_count = 0
    
    with open(input_file, 'r', encoding='utf-8') as f:
        for line_num, line in enumerate(f, 1):
            if line.strip():
                try:
                    item = json.loads(line)
                    
                    # 提取字段
                    generated_question = item.get('generated_question', '')
                    summary = item.get('summary', '')
                    doc_id = item.get('doc_id', '')
                    
                    # 对于训练数据，如果没有generated_question，尝试使用query
                    if not generated_question and data_type == "train":
                        generated_question = item.get('query', '')
                    
                    # 对于训练数据，如果没有summary，尝试使用context（但这里我们只想要真正的摘要）
                    if not summary and data_type == "train":
                        # 跳过没有真正摘要的数据
                        skipped_count += 1
                        continue
                    
                    # 检查必要字段是否存在
                    if generated_question and summary and doc_id:
                        extracted_item = {
                            'generated_question': generated_question,
                            'summary': summary,
                            'doc_id': doc_id
                        }
                        
                        # 对于评估数据，添加answer字段
                        if data_type == "eval":
                            answer = item.get('answer', '')
                            extracted_item['answer'] = answer
                        
                        extracted_data.append(extracted_item)
                    else:
                        skipped_count += 1
                        
                except json.JSONDecodeError as e:
                    print(f"第 {line_num} 行JSON解析错误: {e}")
                    skipped_count += 1
                    continue
    
    # 保存提取的数据
    with open(output_file, 'w', encoding='utf-8') as f:
        for item in extracted_data:
            f.write(json.dumps(item, ensure_ascii=False) + '\n')
    
    print(f"✅ 提取完成:")
    print(f"  - 有效数据: {len(extracted_data)} 条")
    print(f"  - 跳过数据: {skipped_count} 条")
    print(f"  - 输出文件: {output_file}")
    
    return len(extracted_data)

def main():
    parser = argparse.ArgumentParser(description="提取摘要数据")
    parser.add_argument("--train_input", type=str, default="evaluate_mrr/alphafin_train_qc.jsonl",
                       help="训练数据输入文件")
    parser.add_argument("--eval_input", type=str, default="evaluate_mrr/alphafin_eval.jsonl",
                       help="评估数据输入文件")
    parser.add_argument("--train_output", type=str, default="evaluate_mrr/alphafin_train_summary.jsonl",
                       help="训练数据输出文件")
    parser.add_argument("--eval_output", type=str, default="evaluate_mrr/alphafin_eval_summary.jsonl",
                       help="评估数据输出文件")
    
    args = parser.parse_args()
    
    print("🚀 开始提取摘要数据...")
    
    # 处理训练数据
    if Path(args.train_input).exists():
        train_count = extract_summary_data(args.train_input, args.train_output, "train")
    else:
        print(f"❌ 训练数据文件不存在: {args.train_input}")
        train_count = 0
    
    # 处理评估数据
    if Path(args.eval_input).exists():
        eval_count = extract_summary_data(args.eval_input, args.eval_output, "eval")
    else:
        print(f"❌ 评估数据文件不存在: {args.eval_input}")
        eval_count = 0
    
    print(f"\n📊 总结:")
    print(f"  - 训练数据: {train_count} 条")
    print(f"  - 评估数据: {eval_count} 条")
    
    if train_count == 0:
        print("⚠️  警告: 没有提取到有效的训练数据")
        print("   可能原因:")
        print("   1. 训练数据中没有summary字段")
        print("   2. 训练数据格式不正确")
        print("   3. 需要先生成包含summary的训练数据")

if __name__ == "__main__":
    main() 