#!/usr/bin/env python3
"""
修复知识库数据的doc_id覆盖率问题
"""

import json
import hashlib
from pathlib import Path
from typing import Dict, List, Any

def fix_knowledge_base_doc_ids():
    """修复知识库数据的doc_id覆盖率"""
    
    # 1. 修复中文知识库
    print("🔧 修复中文知识库doc_id...")
    chinese_kb_path = "data/alphafin/alphafin_final_clean.json"
    
    if Path(chinese_kb_path).exists():
        with open(chinese_kb_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        fixed_count = 0
        for i, record in enumerate(data):
            if not record.get('doc_id'):
                # 使用内容哈希作为doc_id
                content = record.get('content', '') or record.get('context', '')
                if content:
                    doc_id = hashlib.md5(content.encode('utf-8')).hexdigest()[:16]
                    record['doc_id'] = doc_id
                    fixed_count += 1
        
        # 保存修复后的数据
        with open(chinese_kb_path, 'w', encoding='utf-8') as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
        
        print(f"✅ 中文知识库修复完成：添加了 {fixed_count} 个doc_id")
    else:
        print(f"❌ 中文知识库文件不存在：{chinese_kb_path}")
    
    # 2. 修复英文知识库
    print("🔧 修复英文知识库doc_id...")
    english_kb_path = "data/tatqa/tatqa_knowledge_base_combined.jsonl"
    
    if Path(english_kb_path).exists():
        fixed_records = []
        fixed_count = 0
        
        with open(english_kb_path, 'r', encoding='utf-8') as f:
            for line in f:
                if line.strip():
                    record = json.loads(line)
                    if not record.get('doc_id'):
                        # 使用内容哈希作为doc_id
                        content = record.get('content', '') or record.get('context', '')
                        if content:
                            doc_id = hashlib.md5(content.encode('utf-8')).hexdigest()[:16]
                            record['doc_id'] = doc_id
                            fixed_count += 1
                    fixed_records.append(record)
        
        # 保存修复后的数据
        with open(english_kb_path, 'w', encoding='utf-8') as f:
            for record in fixed_records:
                f.write(json.dumps(record, ensure_ascii=False) + '\n')
        
        print(f"✅ 英文知识库修复完成：添加了 {fixed_count} 个doc_id")
    else:
        print(f"❌ 英文知识库文件不存在：{english_kb_path}")
    
    # 3. 修复TatQA评测数据
    print("🔧 修复TatQA评测数据doc_id...")
    tatqa_eval_path = "evaluate_mrr/tatqa_eval.jsonl"
    
    if Path(tatqa_eval_path).exists():
        fixed_records = []
        fixed_count = 0
        
        with open(tatqa_eval_path, 'r', encoding='utf-8') as f:
            for line in f:
                if line.strip():
                    record = json.loads(line)
                    if not record.get('doc_id'):
                        # 使用context的哈希作为doc_id
                        context = record.get('context', '')
                        if context:
                            doc_id = hashlib.md5(context.encode('utf-8')).hexdigest()[:16]
                            record['doc_id'] = doc_id
                            fixed_count += 1
                    fixed_records.append(record)
        
        # 保存修复后的数据
        with open(tatqa_eval_path, 'w', encoding='utf-8') as f:
            for record in fixed_records:
                f.write(json.dumps(record, ensure_ascii=False) + '\n')
        
        print(f"✅ TatQA评测数据修复完成：添加了 {fixed_count} 个doc_id")
    else:
        print(f"❌ TatQA评测数据文件不存在：{tatqa_eval_path}")

def verify_fixes():
    """验证修复效果"""
    print("\n🔍 验证修复效果...")
    
    # 检查中文知识库
    chinese_kb_path = "data/alphafin/alphafin_final_clean.json"
    if Path(chinese_kb_path).exists():
        with open(chinese_kb_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        doc_id_count = sum(1 for record in data if record.get('doc_id'))
        total_count = len(data)
        coverage = doc_id_count / total_count * 100 if total_count > 0 else 0
        
        print(f"📊 中文知识库doc_id覆盖率: {coverage:.2f}% ({doc_id_count}/{total_count})")
    
    # 检查英文知识库
    english_kb_path = "data/tatqa/tatqa_knowledge_base_combined.jsonl"
    if Path(english_kb_path).exists():
        doc_id_count = 0
        total_count = 0
        
        with open(english_kb_path, 'r', encoding='utf-8') as f:
            for line in f:
                if line.strip():
                    record = json.loads(line)
                    total_count += 1
                    if record.get('doc_id'):
                        doc_id_count += 1
        
        coverage = doc_id_count / total_count * 100 if total_count > 0 else 0
        print(f"📊 英文知识库doc_id覆盖率: {coverage:.2f}% ({doc_id_count}/{total_count})")
    
    # 检查TatQA评测数据
    tatqa_eval_path = "evaluate_mrr/tatqa_eval.jsonl"
    if Path(tatqa_eval_path).exists():
        doc_id_count = 0
        total_count = 0
        
        with open(tatqa_eval_path, 'r', encoding='utf-8') as f:
            for line in f:
                if line.strip():
                    record = json.loads(line)
                    total_count += 1
                    if record.get('doc_id'):
                        doc_id_count += 1
        
        coverage = doc_id_count / total_count * 100 if total_count > 0 else 0
        print(f"📊 TatQA评测数据doc_id覆盖率: {coverage:.2f}% ({doc_id_count}/{total_count})")

if __name__ == "__main__":
    print("🚀 开始修复知识库数据的doc_id覆盖率问题...")
    
    # 执行修复
    fix_knowledge_base_doc_ids()
    
    # 验证修复效果
    verify_fixes()
    
    print("\n✅ 修复完成！现在可以正常运行评测了。") 