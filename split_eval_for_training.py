#!/usr/bin/env python3
"""
将评估数据分割为训练集和评估集
这样我们就有包含真正摘要的训练数据了
"""

import json
import random
from pathlib import Path

def split_eval_data(input_file, train_output, eval_output, train_ratio=0.8, seed=42):
    """
    将评估数据分割为训练集和评估集
    
    Args:
        input_file: 输入文件路径
        train_output: 训练数据输出文件
        eval_output: 评估数据输出文件
        train_ratio: 训练集比例
        seed: 随机种子
    """
    print(f"分割评估数据: {input_file}")
    
    # 读取数据
    data = []
    with open(input_file, 'r', encoding='utf-8') as f:
        for line in f:
            if line.strip():
                data.append(json.loads(line))
    
    print(f"总数据量: {len(data)} 条")
    
    # 随机打乱
    random.seed(seed)
    random.shuffle(data)
    
    # 分割
    n_train = int(len(data) * train_ratio)
    train_data = data[:n_train]
    eval_data = data[n_train:]
    
    print(f"分割结果:")
    print(f"  - 训练集: {len(train_data)} 条 ({train_ratio*100:.0f}%)")
    print(f"  - 评估集: {len(eval_data)} 条 ({(1-train_ratio)*100:.0f}%)")
    
    # 保存训练集（只保留训练需要的字段）
    print(f"保存训练集: {train_output}")
    with open(train_output, 'w', encoding='utf-8') as f:
        for item in train_data:
            train_item = {
                'generated_question': item['generated_question'],
                'summary': item['summary'],
                'doc_id': item['doc_id']
            }
            f.write(json.dumps(train_item, ensure_ascii=False) + '\n')
    
    # 保存评估集（保留完整字段）
    print(f"保存评估集: {eval_output}")
    with open(eval_output, 'w', encoding='utf-8') as f:
        for item in eval_data:
            f.write(json.dumps(item, ensure_ascii=False) + '\n')
    
    print("✅ 分割完成！")
    
    return len(train_data), len(eval_data)

def main():
    input_file = "evaluate_mrr/alphafin_eval_summary.jsonl"
    train_output = "evaluate_mrr/alphafin_train_summary.jsonl"
    eval_output = "evaluate_mrr/alphafin_eval_summary_split.jsonl"
    
    if not Path(input_file).exists():
        print(f"❌ 输入文件不存在: {input_file}")
        return
    
    train_count, eval_count = split_eval_data(
        input_file=input_file,
        train_output=train_output,
        eval_output=eval_output,
        train_ratio=0.8,
        seed=42
    )
    
    print(f"\n📊 最终结果:")
    print(f"  - 训练数据: {train_count} 条 (包含真正的摘要)")
    print(f"  - 评估数据: {eval_count} 条")
    print(f"  - 训练文件: {train_output}")
    print(f"  - 评估文件: {eval_output}")

if __name__ == "__main__":
    main() 