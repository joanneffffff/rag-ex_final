#!/usr/bin/env python3
"""
增强版评估函数
支持多种匹配策略的正确答案匹配
"""

import hashlib
import re
from typing import List, Dict, Any, Optional
from difflib import SequenceMatcher
import numpy as np
import torch
from sentence_transformers import util

from xlm.dto.dto import DocumentWithMetadata


def find_correct_document_rank_enhanced(
    context: str,
    retrieved_docs: List[DocumentWithMetadata],
    sample: Dict[str, Any],
    encoder=None,
    similarity_threshold: float = 0.6  # 降低相似度阈值
) -> int:
    """
    增强版正确答案匹配函数，支持多种匹配策略
    
    Args:
        context: Gold Context (正确答案的上下文内容)
        retrieved_docs: 检索到的文档列表 (DocumentWithMetadata对象)
        sample: 完整的评估样本，包含 relevant_doc_ids 等元数据
        encoder: 编码器实例 (用于相似度计算)
        similarity_threshold: 相似度阈值
        
    Returns:
        正确答案的排名 (1-based)，如果没找到返回0
    """
    
    if not retrieved_docs:
        return 0
    
    # 策略1: relevant_doc_ids匹配（最严格，适用于英文数据）
    if 'relevant_doc_ids' in sample and sample['relevant_doc_ids']:
        relevant_doc_ids = set(sample['relevant_doc_ids'])
        for rank, doc in enumerate(retrieved_docs, 1):
            # 检查文档的metadata中是否有匹配的doc_id
            if hasattr(doc.metadata, 'doc_id') and doc.metadata.doc_id in relevant_doc_ids:
                return rank
    
    # 策略2: ID匹配（适用于中文数据）
    if 'doc_id' in sample and sample['doc_id']:
        target_doc_id = sample['doc_id']
        for rank, doc in enumerate(retrieved_docs, 1):
            if hasattr(doc.metadata, 'doc_id') and doc.metadata.doc_id == target_doc_id:
                return rank
    
    # 策略3: 内容哈希匹配
    context_hash = hashlib.md5(context.strip().encode('utf-8')).hexdigest()
    for rank, doc in enumerate(retrieved_docs, 1):
        doc_hash = hashlib.md5(doc.content.strip().encode('utf-8')).hexdigest()
        if doc_hash == context_hash:
            return rank
    
    # 策略4: 精确文本匹配（原始策略，但更健壮）
    context_clean = _normalize_text(context)
    for rank, doc in enumerate(retrieved_docs, 1):
        doc_content_clean = _normalize_text(doc.content)
        if doc_content_clean == context_clean:
            return rank
    
    # 策略5: 相似度匹配（使用编码器）
    if encoder:
        try:
            # 编码context和所有检索到的文档
            context_embedding = encoder.encode([context])
            doc_contents = [doc.content for doc in retrieved_docs]
            doc_embeddings = encoder.encode(doc_contents)
            
            # 转换为tensor用于相似度计算
            context_embedding_tensor = torch.tensor(context_embedding, dtype=torch.float32)
            doc_embeddings_tensor = torch.tensor(doc_embeddings, dtype=torch.float32)
            
            # 计算相似度
            cos_scores = util.cos_sim(context_embedding_tensor, doc_embeddings_tensor)[0]
            
            for rank, score in enumerate(cos_scores, 1):
                if score >= similarity_threshold:
                    return rank
        except Exception as e:
            print(f"相似度匹配失败: {e}")
    
    # 策略6: 模糊文本匹配
    context_clean = _normalize_text(context)
    best_ratio = 0
    best_rank = 0
    
    for rank, doc in enumerate(retrieved_docs, 1):
        doc_content_clean = _normalize_text(doc.content)
        ratio = SequenceMatcher(None, context_clean, doc_content_clean).ratio()
        if ratio > best_ratio and ratio >= similarity_threshold:
            best_ratio = ratio
            best_rank = rank
    
    if best_rank > 0:
        return best_rank
    
    # 如果所有策略都失败，返回0
    return 0


def _normalize_text(text: str) -> str:
    """
    标准化文本，去除多余空格和标点差异
    """
    if not text:
        return ""
    
    # 去除多余空格
    text = re.sub(r'\s+', ' ', text.strip())
    
    # 标准化标点符号
    text = re.sub(r'[，,]', ',', text)
    text = re.sub(r'[。.]', '.', text)
    text = re.sub(r'[：:]', ':', text)
    text = re.sub(r'[；;]', ';', text)
    
    return text.lower()


def compute_mrr_with_enhanced_matching(
    retrieved_docs_list: List[List[DocumentWithMetadata]],
    eval_data: List[Dict[str, Any]],
    encoder=None,
    similarity_threshold: float = 0.8
) -> Dict[str, float]:
    """
    使用增强版匹配策略计算MRR
    
    Args:
        retrieved_docs_list: 每个样本的检索文档列表
        eval_data: 评估数据
        encoder: 编码器实例
        similarity_threshold: 相似度阈值
        
    Returns:
        评估结果字典
    """
    mrr_scores = []
    hit_at_1 = 0
    hit_at_3 = 0
    hit_at_5 = 0
    hit_at_10 = 0
    found_samples = 0
    
    for i, (retrieved_docs, sample) in enumerate(zip(retrieved_docs_list, eval_data)):
        context = sample.get('context', '')
        if not context:
            mrr_scores.append(0.0)
            continue
        
        found_rank = find_correct_document_rank_enhanced(
            context=context,
            retrieved_docs=retrieved_docs,
            sample=sample,
            encoder=encoder,
            similarity_threshold=similarity_threshold
        )
        
        if found_rank > 0:
            found_samples += 1
            mrr_score = 1.0 / found_rank
            mrr_scores.append(mrr_score)
            
            if found_rank == 1:
                hit_at_1 += 1
            if found_rank <= 3:
                hit_at_3 += 1
            if found_rank <= 5:
                hit_at_5 += 1
            if found_rank <= 10:
                hit_at_10 += 1
        else:
            mrr_scores.append(0.0)
    
    # 计算最终指标
    total_samples = len(eval_data)
    mrr = sum(mrr_scores) / total_samples if total_samples > 0 else 0.0
    hit_at_1_rate = hit_at_1 / total_samples if total_samples > 0 else 0.0
    hit_at_3_rate = hit_at_3 / total_samples if total_samples > 0 else 0.0
    hit_at_5_rate = hit_at_5 / total_samples if total_samples > 0 else 0.0
    hit_at_10_rate = hit_at_10 / total_samples if total_samples > 0 else 0.0
    
    return {
        'mrr': mrr,
        'hit_at_1': hit_at_1_rate,
        'hit_at_3': hit_at_3_rate,
        'hit_at_5': hit_at_5_rate,
        'hit_at_10': hit_at_10_rate,
        'total_samples': total_samples,
        'found_samples': found_samples
    } 