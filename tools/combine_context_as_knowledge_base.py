#!/usr/bin/env python3
"""
将训练数据和评估数据的context合并成知识库
避免数据泄露，同时确保评估数据的context在知识库中
"""

import json
from pathlib import Path

def combine_context_as_knowledge_base():
    """合并context作为知识库"""
    
    # 输入文件
    train_file = "evaluate_mrr/tatqa_train_qc_enhanced.jsonl"
    eval_file = "evaluate_mrr/tatqa_eval_enhanced.jsonl"
    
    # 输出文件
    knowledge_base_file = "evaluate_mrr/tatqa_knowledge_base.jsonl"
    
    print("🔄 合并context作为知识库...")
    
    # 读取训练数据
    train_contexts = []
    with open(train_file, "r", encoding="utf-8") as f:
        for line in f:
            item = json.loads(line)
            train_contexts.append({
                "text": item.get("context", ""),
                "doc_id": item.get("doc_id", ""),
                "source_type": "train"
            })
    
    print(f"✅ 读取训练数据: {len(train_contexts)} 个context")
    
    # 读取评估数据
    eval_contexts = []
    with open(eval_file, "r", encoding="utf-8") as f:
        for line in f:
            item = json.loads(line)
            eval_contexts.append({
                "text": item.get("context", ""),
                "doc_id": item.get("doc_id", ""),
                "source_type": "eval"
            })
    
    print(f"✅ 读取评估数据: {len(eval_contexts)} 个context")
    
    # 合并所有context
    all_contexts = train_contexts + eval_contexts
    
    # 去重（基于doc_id）
    unique_contexts = {}
    for ctx in all_contexts:
        doc_id = ctx["doc_id"]
        if doc_id not in unique_contexts:
            unique_contexts[doc_id] = ctx
        else:
            # 如果已存在，保留训练数据的版本
            if ctx["source_type"] == "train":
                unique_contexts[doc_id] = ctx
    
    print(f"✅ 去重后: {len(unique_contexts)} 个唯一context")
    
    # 写入知识库文件
    with open(knowledge_base_file, "w", encoding="utf-8") as f:
        for ctx in unique_contexts.values():
            f.write(json.dumps(ctx, ensure_ascii=False) + "\n")
    
    print(f"✅ 知识库已生成: {knowledge_base_file}")
    
    # 统计信息
    train_count = sum(1 for ctx in unique_contexts.values() if ctx["source_type"] == "train")
    eval_count = sum(1 for ctx in unique_contexts.values() if ctx["source_type"] == "eval")
    
    print(f"📊 知识库统计:")
    print(f"  训练数据context: {train_count}")
    print(f"  评估数据context: {eval_count}")
    print(f"  总计: {len(unique_contexts)}")

if __name__ == "__main__":
    combine_context_as_knowledge_base() 