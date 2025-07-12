#!/usr/bin/env python3
"""
更新 alphafin_eval_samples.jsonl 文件中的指令
为包含特定答案模式但instruction为空的样本添加预测指令
"""

import json
import re
from pathlib import Path

def update_instructions():
    """更新数据集中的指令"""
    
    # 文件路径
    input_file = "data/alphafin/alphafin_eval_samples.jsonl"
    output_file = "data/alphafin/alphafin_eval_samples_updated.jsonl"
    
    # 目标指令
    target_instruction = "请根据下方提供的该股票相关研报与数据，对该股票的下个月的涨跌，进行预测，请给出明确的答案，\"涨\" 或者 \"跌\"。同时给出这个股票下月的涨跌概率，分别是:极大，较大，中上，一般。"
    
    # 匹配模式
    prediction_pattern = re.compile(r"这个股票的下月最终收益结果是")
    
    updated_count = 0
    total_count = 0
    
    print(f"正在处理文件: {input_file}")
    
    with open(input_file, 'r', encoding='utf-8') as f_in, \
         open(output_file, 'w', encoding='utf-8') as f_out:
        
        for line_num, line in enumerate(f_in, 1):
            try:
                item = json.loads(line.strip())
                total_count += 1
                
                answer = item.get("answer", "")
                instruction = item.get("instruction", "")
                
                # 检查是否包含预测模式且instruction为空
                if prediction_pattern.search(answer) and not instruction.strip():
                    item["instruction"] = target_instruction
                    updated_count += 1
                    print(f"第 {line_num} 行: 添加预测指令")
                    print(f"  Answer: {answer[:100]}...")
                elif prediction_pattern.search(answer):
                    print(f"第 {line_num} 行: 包含预测模式但instruction不为空")
                    print(f"  Instruction: {instruction[:50]}...")
                    print(f"  Answer: {answer[:100]}...")
                
                # 写入更新后的数据
                f_out.write(json.dumps(item, ensure_ascii=False) + '\n')
                
            except json.JSONDecodeError as e:
                print(f"第 {line_num} 行: JSON解析错误 - {e}")
                continue
            except Exception as e:
                print(f"第 {line_num} 行: 处理错误 - {e}")
                continue
    
    print(f"\n处理完成!")
    print(f"总样本数: {total_count}")
    print(f"更新样本数: {updated_count}")
    print(f"输出文件: {output_file}")

if __name__ == "__main__":
    update_instructions() 