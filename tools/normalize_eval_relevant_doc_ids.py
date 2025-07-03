import json
from pathlib import Path

in_file = 'evaluate_mrr/tatqa_eval_enhanced.jsonl'
out_file = 'evaluate_mrr/tatqa_eval_enhanced_idnorm.jsonl'

with open(in_file, 'r', encoding='utf-8') as fin, open(out_file, 'w', encoding='utf-8') as fout:
    for line in fin:
        item = json.loads(line)
        if 'relevant_doc_ids' in item:
            item['relevant_doc_ids'] = [docid.replace('-', '').lower() for docid in item['relevant_doc_ids']]
        fout.write(json.dumps(item, ensure_ascii=False) + '\n')

print(f'✅ 已生成归一化ID的评估集: {out_file}') 