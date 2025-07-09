#!/usr/bin/env python3
import json

# 加载结果文件
with open('alphafin_data_process/evaluation_results/raw_results_baseline_20250708_185614.json', 'r') as f:
    baseline = json.load(f)

with open('alphafin_data_process/evaluation_results/raw_results_prefilter_20250708_185614.json', 'r') as f:
    prefilter = json.load(f)

print(f'Baseline结果数: {len(baseline)}')
print(f'Prefilter结果数: {len(prefilter)}')

print('\n前5个查询的对比:')
for i in range(min(5, len(baseline), len(prefilter))):
    print(f'\n查询{i+1}: {baseline[i]["query_text"][:50]}...')
    print(f'Baseline第1位: {baseline[i]["retrieved_doc_ids_ranked"][0]}')
    print(f'Prefilter第1位: {prefilter[i]["retrieved_doc_ids_ranked"][0]}')
    print(f'Ground Truth: {baseline[i]["ground_truth_doc_ids"]}')
    print(f'Baseline命中: {baseline[i]["retrieved_doc_ids_ranked"][0] in baseline[i]["ground_truth_doc_ids"]}')
    print(f'Prefilter命中: {prefilter[i]["retrieved_doc_ids_ranked"][0] in prefilter[i]["ground_truth_doc_ids"]}')

# 统计命中率
baseline_hits = 0
prefilter_hits = 0

for i in range(len(baseline)):
    if baseline[i]["retrieved_doc_ids_ranked"][0] in baseline[i]["ground_truth_doc_ids"]:
        baseline_hits += 1
    if prefilter[i]["retrieved_doc_ids_ranked"][0] in prefilter[i]["ground_truth_doc_ids"]:
        prefilter_hits += 1

print(f'\n总体统计:')
print(f'Baseline命中率: {baseline_hits}/{len(baseline)} = {baseline_hits/len(baseline):.3f}')
print(f'Prefilter命中率: {prefilter_hits}/{len(prefilter)} = {prefilter_hits/len(prefilter):.3f}') 