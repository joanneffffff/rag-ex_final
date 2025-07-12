#!/usr/bin/env python3
import json
import re

# 加载数据集
with open('data/alphafin/alphafin_eval_samples_updated.jsonl', 'r') as f:
    data = [json.loads(line) for line in f]

print(f"总数据条数: {len(data)}")

# 检查instruction字段
has_instruction = sum(1 for item in data if item.get('instruction', '').strip())
no_instruction = sum(1 for item in data if not item.get('instruction', '').strip())

print(f"有instruction字段: {has_instruction}条")
print(f"无instruction字段: {no_instruction}条")

# 检查筛选逻辑
target_prediction_pattern = re.compile(r"这个股票的下月最终收益结果是:[''''](涨|跌)[''''],?(上涨|下跌)概率:(极大|较大|中上|一般)[。.]?")

filtered_count = 0
skipped_count = 0

for i, item in enumerate(data):
    instruction_content = item.get("instruction", "").strip()
    answer_content = item.get("answer", "").strip()
    
    # 条件1: instruction 为空 (通用问答，如数值抽取、摘要等)
    # 条件2: instruction 不为空，并且 answer 匹配特定预测模式 (预测类问答)
    if not instruction_content or target_prediction_pattern.fullmatch(answer_content):
        filtered_count += 1
    else:
        skipped_count += 1
        print(f"跳过样本 {i}: instruction='{instruction_content[:50]}...', answer='{answer_content[:50]}...'")

print(f"\n筛选后保留: {filtered_count}条")
print(f"筛选后跳过: {skipped_count}条")
print(f"总计: {filtered_count + skipped_count}条") 