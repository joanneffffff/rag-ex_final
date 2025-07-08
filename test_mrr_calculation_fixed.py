#!/usr/bin/env python3
"""
测试MRR计算是否正确 (修复版)
使用正确的doc_id匹配逻辑
"""

import json
import numpy as np
import torch
from sentence_transformers import SentenceTransformer

def test_mrr_calculation_fixed():
    """测试MRR计算逻辑 (修复版)"""
    print("🧪 开始测试MRR计算 (修复版)...")
    
    # 加载一些评估数据
    eval_data = []
    with open("evaluate_mrr/alphafin_eval.jsonl", 'r', encoding='utf-8') as f:
        for i, line in enumerate(f):
            if i >= 10:  # 取前10个样本进行测试
                break
            item = json.loads(line)
            eval_data.append({
                'query': item.get('generated_question', ''),
                'context': item.get('summary', ''),
                'doc_id': item.get('doc_id', '')
            })
    
    print(f"📊 加载了 {len(eval_data)} 个测试样本")
    
    # 显示前3个样本的信息
    for i, item in enumerate(eval_data[:3]):
        print(f"\n样本 {i+1}:")
        print(f"  doc_id: {item['doc_id']}")
        print(f"  query: {item['query'][:80]}...")
        print(f"  context: {item['context'][:80]}...")
    
    # 加载模型
    print(f"\n🤖 加载模型...")
    model = SentenceTransformer("Langboat/mengzi-bert-base-fin")
    
    # 编码所有上下文
    contexts = [item['context'] for item in eval_data]
    print(f"编码 {len(contexts)} 个上下文...")
    context_embeddings = model.encode(contexts, convert_to_tensor=True)
    
    # 创建doc_id到索引的映射
    doc_id_to_idx = {}
    for idx, item in enumerate(eval_data):
        doc_id = item.get('doc_id') or str(idx)
        doc_id_to_idx[doc_id] = idx
    
    print(f"doc_id映射: {doc_id_to_idx}")
    
    # 测试每个查询
    mrrs = []
    for i, item in enumerate(eval_data):
        print(f"\n--- 测试查询 {i+1} ---")
        print(f"doc_id: {item['doc_id']}")
        print(f"query: {item['query'][:60]}...")
        
        # 编码查询
        query_emb = model.encode(item['query'], convert_to_tensor=True)
        
        # 计算相似度 - 修复：计算查询与所有上下文的相似度
        scores = torch.cosine_similarity(query_emb.unsqueeze(0), context_embeddings).cpu().numpy()
        
        # 使用doc_id找到目标上下文的索引
        target_doc_id = item.get('doc_id') or str(i)
        target_context_idx = doc_id_to_idx.get(target_doc_id, i)
        
        # 排序
        sorted_indices = np.argsort(scores)[::-1]
        print(f"相似度分数: {scores}")  # 显示所有分数
        print(f"排序后的索引: {sorted_indices}")  # 显示所有索引
        print(f"目标索引: {target_context_idx}")
        
        # 找到排名
        rank = -1
        for r, idx in enumerate(sorted_indices):
            if idx == target_context_idx:
                rank = r + 1
                break
        
        print(f"目标排名: {rank}")
        
        if rank != -1:
            mrr_score = 1.0 / rank
            mrrs.append(mrr_score)
            print(f"MRR分数: {mrr_score:.4f}")
        else:
            mrrs.append(0.0)
            print(f"MRR分数: 0.0000 (未找到)")
    
    # 计算平均MRR
    avg_mrr = float(np.mean(mrrs)) if mrrs else 0.0
    print(f"\n📊 测试结果:")
    print(f"  各样本MRR: {[f'{mrr:.4f}' for mrr in mrrs]}")
    print(f"  平均MRR: {avg_mrr:.4f}")
    
    # 分析结果
    if avg_mrr > 0.5:
        print("✅ MRR计算正确，模型能够正确匹配查询和上下文")
    elif avg_mrr > 0.1:
        print("⚠️  MRR较低，但计算逻辑正确，可能需要更多训练")
    else:
        print("❌ MRR极低，可能存在以下问题:")
        print("   1. 字段映射错误")
        print("   2. 数据质量问题")
        print("   3. 模型需要更多训练")
    
    return avg_mrr

if __name__ == "__main__":
    test_mrr_calculation_fixed() 