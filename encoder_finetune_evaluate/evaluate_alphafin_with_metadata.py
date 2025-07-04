#!/usr/bin/env python3
"""
AlphaFin数据评估脚本，支持元数据过滤
基于中文评估脚本但适配AlphaFin数据结构
"""

import json
import argparse
import torch
import torch.nn.functional as F
import numpy as np
from pathlib import Path
from tqdm import tqdm
import re
import sys
import os

# 添加项目根目录到Python路径
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

def extract_alphafin_metadata(context_text: str) -> dict:
    """从AlphaFin上下文文本中提取元数据"""
    metadata = {
        "company_name": "",
        "stock_code": "",
        "report_date": "",
        "report_title": ""
    }
    
    # 提取公司名称和股票代码
    company_pattern = r'([^（]+)（([0-9]{6}）)'
    match = re.search(company_pattern, context_text)
    if match:
        metadata["company_name"] = match.group(1).strip()
        metadata["stock_code"] = match.group(2).strip()
    
    # 提取报告日期
    date_pattern = r'(\d{4}-\d{2}-\d{2})'
    dates = re.findall(date_pattern, context_text)
    if dates:
        metadata["report_date"] = dates[0]
    
    # 提取报告标题
    title_pattern = r'研究报告，其标题是："([^"]+)"'
    title_match = re.search(title_pattern, context_text)
    if title_match:
        metadata["report_title"] = title_match.group(1).strip()
    
    return metadata

def filter_corpus_by_metadata(corpus_documents: dict, target_metadata: dict) -> dict:
    """根据元数据过滤检索库"""
    if not target_metadata or not any(target_metadata.values()):
        return corpus_documents
    
    filtered_corpus = {}
    filter_criteria = []
    
    # 构建过滤条件
    if target_metadata.get("company_name"):
        filter_criteria.append(("company_name", target_metadata["company_name"]))
    if target_metadata.get("stock_code"):
        filter_criteria.append(("stock_code", target_metadata["stock_code"]))
    if target_metadata.get("report_date"):
        filter_criteria.append(("report_date", target_metadata["report_date"]))
    
    if not filter_criteria:
        return corpus_documents
    
    print(f"🔍 应用元数据过滤: {filter_criteria}")
    
    for doc_id, content in corpus_documents.items():
        content_metadata = extract_alphafin_metadata(content)
        
        # 检查是否满足所有过滤条件
        matches_all = True
        for field, value in filter_criteria:
            if content_metadata.get(field) != value:
                matches_all = False
                break
        
        if matches_all:
            filtered_corpus[doc_id] = content
    
    print(f"📊 过滤结果: {len(filtered_corpus)}/{len(corpus_documents)} 个文档")
    return filtered_corpus

def calculate_mrr(rankings):
    """计算MRR分数"""
    if not rankings:
        return 0.0
    
    reciprocal_ranks = []
    for rank in rankings:
        if rank > 0:
            reciprocal_ranks.append(1.0 / rank)
        else:
            reciprocal_ranks.append(0.0)
    
    return np.mean(reciprocal_ranks)

def mean_pooling(model_output, attention_mask):
    """平均池化"""
    token_embeddings = model_output[0]
    input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
    return torch.sum(token_embeddings * input_mask_expanded, 1) / torch.clamp(input_mask_expanded.sum(1), min=1e-9)

def load_encoder_model(model_name: str, device: str):
    """加载编码器模型"""
    try:
        from transformers import AutoTokenizer, AutoModel
        print(f"📖 加载编码器模型: {model_name}")
        
        tokenizer = AutoTokenizer.from_pretrained(model_name, cache_dir="cache")
        model = AutoModel.from_pretrained(model_name, cache_dir="cache").to(device)
        
        # 设置特殊token
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
        
        return tokenizer, model
    except Exception as e:
        print(f"❌ 加载编码器模型失败: {e}")
        return None, None

def load_reranker_model(model_name: str, device: str):
    """加载重排序模型"""
    try:
        from transformers import AutoTokenizer, AutoModelForSequenceClassification
        print(f"📖 加载重排序模型: {model_name}")
        
        tokenizer = AutoTokenizer.from_pretrained(model_name, cache_dir="cache")
        model = AutoModelForSequenceClassification.from_pretrained(model_name, cache_dir="cache").to(device)
        
        return tokenizer, model
    except Exception as e:
        print(f"❌ 加载重排序模型失败: {e}")
        return None, None

def main():
    parser = argparse.ArgumentParser(description="AlphaFin数据评估脚本")
    parser.add_argument("--eval_data", type=str, 
                       default="evaluate_mrr/alphafin_eval.jsonl",
                       help="评估数据文件")
    parser.add_argument("--corpus_data", type=str,
                       default="data/alphafin/alphafin_merged_generated_qa_full_dedup.json",
                       help="检索库数据文件")
    parser.add_argument("--encoder_model", type=str,
                       default="microsoft/DialoGPT-medium",
                       help="编码器模型名称")
    parser.add_argument("--reranker_model", type=str,
                       default="microsoft/DialoGPT-medium",
                       help="重排序模型名称")
    parser.add_argument("--top_k_retrieval", type=int, default=100,
                       help="检索top-k")
    parser.add_argument("--top_k_rerank", type=int, default=10,
                       help="重排序top-k")
    parser.add_argument("--max_samples", type=int, default=100,
                       help="最大评估样本数")
    parser.add_argument("--use_metadata_filter", action="store_true",
                       help="是否使用元数据过滤")
    parser.add_argument("--device", type=str, default="cuda",
                       help="设备选择")
    
    args = parser.parse_args()
    
    print("🚀 开始AlphaFin数据评估")
    print(f"📊 配置:")
    print(f"  - 评估数据: {args.eval_data}")
    print(f"  - 检索库数据: {args.corpus_data}")
    print(f"  - 编码器模型: {args.encoder_model}")
    print(f"  - 重排序模型: {args.reranker_model}")
    print(f"  - 最大样本数: {args.max_samples}")
    print(f"  - 元数据过滤: {'启用' if args.use_metadata_filter else '禁用'}")
    
    # 检查文件是否存在
    if not Path(args.eval_data).exists():
        print(f"❌ 评估数据文件不存在: {args.eval_data}")
        return
    
    if not Path(args.corpus_data).exists():
        print(f"❌ 检索库数据文件不存在: {args.corpus_data}")
        return
    
    # 设置设备
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    print(f"🔧 使用设备: {device}")
    
    # 加载模型
    encoder_tokenizer, encoder_model = load_encoder_model(args.encoder_model, device)
    reranker_tokenizer, reranker_model = load_reranker_model(args.reranker_model, device)
    
    if encoder_tokenizer is None or encoder_model is None:
        print("❌ 编码器模型加载失败")
        return
    
    if reranker_tokenizer is None or reranker_model is None:
        print("❌ 重排序模型加载失败")
        return
    
    # 加载检索库数据
    print(f"📖 加载检索库数据: {args.corpus_data}")
    with open(args.corpus_data, 'r', encoding='utf-8') as f:
        corpus_data = json.load(f)
    
    # 构建检索库
    corpus_documents = {}
    for idx, item in enumerate(corpus_data):
        doc_id = str(idx)
        context = item.get('original_context', '')
        if context:
            corpus_documents[doc_id] = context
    
    print(f"✅ 构建了 {len(corpus_documents)} 个检索库文档")
    
    # 加载评估数据
    print(f"📖 加载评估数据: {args.eval_data}")
    eval_data = []
    with open(args.eval_data, 'r', encoding='utf-8') as f:
        for i, line in enumerate(f):
            if i >= args.max_samples:
                break
            eval_data.append(json.loads(line))
    
    print(f"✅ 加载了 {len(eval_data)} 个评估样本")
    
    # 生成检索库嵌入
    print("🔄 生成检索库嵌入...")
    corpus_ids = list(corpus_documents.keys())
    corpus_texts = [corpus_documents[doc_id] for doc_id in corpus_ids]
    corpus_embeddings = []
    
    batch_size = 4
    max_length = 512
    
    for i in tqdm(range(0, len(corpus_texts), batch_size), desc="生成检索库嵌入"):
        batch_texts = corpus_texts[i:i + batch_size]
        with torch.no_grad():
            encoded_input = encoder_tokenizer(
                batch_texts,
                padding='max_length',
                truncation=True,
                max_length=max_length,
                return_tensors='pt'
            ).to(device)
            
            model_output = encoder_model(**encoded_input)
            embeddings = mean_pooling(model_output, encoded_input['attention_mask'])
            embeddings = F.normalize(embeddings, p=2, dim=1)
            corpus_embeddings.append(embeddings.cpu())
    
    corpus_embeddings = torch.cat(corpus_embeddings, dim=0).to(device)
    print(f"✅ 生成了 {corpus_embeddings.shape[0]} 个检索库嵌入")
    
    # 开始评估
    all_retrieval_ranks = []
    all_rerank_ranks = []
    skipped_queries_count = 0
    
    print("🚀 开始评估...")
    for item in tqdm(eval_data, desc="评估查询"):
        query_text = item.get('query', '').strip()
        ground_truth_context = item.get('context', '')
        ground_truth_answer = item.get('answer', '')
        
        if not query_text or not ground_truth_context:
            skipped_queries_count += 1
            continue
        
        # 应用元数据过滤
        if args.use_metadata_filter:
            query_metadata = extract_alphafin_metadata(ground_truth_context)
            filtered_corpus = filter_corpus_by_metadata(corpus_documents, query_metadata)
            
            if not filtered_corpus:
                print(f"⚠️  查询 '{query_text[:50]}...' 无匹配的过滤文档，使用完整检索库")
                filtered_corpus = corpus_documents
        else:
            filtered_corpus = corpus_documents
        
        # 找到正确答案的文档ID
        ground_truth_doc_id = None
        for doc_id, content in filtered_corpus.items():
            if content == ground_truth_context:
                ground_truth_doc_id = doc_id
                break
        
        if ground_truth_doc_id is None:
            skipped_queries_count += 1
            continue
        
        # 1. 检索阶段
        with torch.no_grad():
            query_encoded = encoder_tokenizer(
                query_text,
                padding='max_length',
                truncation=True,
                max_length=max_length,
                return_tensors='pt'
            ).to(device)
            
            model_output = encoder_model(**query_encoded)
            embeddings = mean_pooling(model_output, query_encoded['attention_mask'])
            query_embedding = F.normalize(embeddings, p=2, dim=1)
            
            # 计算相似度
            similarities = torch.matmul(query_embedding, corpus_embeddings.transpose(0, 1))
            
            # 获取top-k检索结果
            top_k_values, top_k_indices = torch.topk(
                similarities, 
                min(args.top_k_retrieval, len(corpus_documents)), 
                dim=1
            )
            
            retrieved_doc_ids_and_scores = []
            for i, idx in enumerate(top_k_indices[0]):
                doc_id = corpus_ids[idx.item()]
                if doc_id in filtered_corpus:  # 只考虑过滤后的文档
                    score = top_k_values[0][i].item()
                    retrieved_doc_ids_and_scores.append((doc_id, score))
        
        # 计算检索排名
        retrieval_rank = 0
        for rank, (doc_id, _) in enumerate(retrieved_doc_ids_and_scores, 1):
            if doc_id == ground_truth_doc_id:
                retrieval_rank = rank
                break
        all_retrieval_ranks.append(retrieval_rank)
        
        # 2. 重排序阶段
        if reranker_model and retrieved_doc_ids_and_scores:
            rerank_data = []
            for doc_id, _ in retrieved_doc_ids_and_scores[:args.top_k_rerank]:
                doc_text = filtered_corpus.get(doc_id, "")
                if doc_text:
                    # 构建重排序输入
                    rerank_input = f"Query: {query_text}\nDocument: {doc_text}"
                    rerank_data.append((rerank_input, doc_id))
            
            if rerank_data:
                reranked_results = []
                reranker_batch_size = 4
                
                for j in range(0, len(rerank_data), reranker_batch_size):
                    batch_inputs = [item[0] for item in rerank_data[j:j + reranker_batch_size]]
                    batch_doc_ids = [item[1] for item in rerank_data[j:j + reranker_batch_size]]
                    
                    with torch.no_grad():
                        encoded_input = reranker_tokenizer(
                            batch_inputs,
                            padding='max_length',
                            truncation=True,
                            max_length=max_length,
                            return_tensors='pt'
                        ).to(device)
                        
                        outputs = reranker_model(**encoded_input)
                        scores = torch.softmax(outputs.logits, dim=1)[:, 1].tolist()
                        
                        for k, score in enumerate(scores):
                            reranked_results.append({
                                'doc_id': batch_doc_ids[k],
                                'score': score
                            })
                
                # 按分数排序
                reranked_results.sort(key=lambda x: x['score'], reverse=True)
                
                # 计算重排序排名
                rerank_rank = 0
                for rank, res in enumerate(reranked_results, 1):
                    if res['doc_id'] == ground_truth_doc_id:
                        rerank_rank = rank
                        break
                all_rerank_ranks.append(rerank_rank)
            else:
                all_rerank_ranks.append(0)
        else:
            all_rerank_ranks.append(0)
    
    # 计算MRR分数
    mrr_retrieval = calculate_mrr(all_retrieval_ranks)
    mrr_rerank = calculate_mrr(all_rerank_ranks)
    
    # 输出结果
    print("\n" + "="*50)
    print("📊 评估结果")
    print("="*50)
    print(f"总查询数: {len(eval_data)}")
    print(f"跳过的查询数: {skipped_queries_count}")
    print(f"有效查询数: {len(eval_data) - skipped_queries_count}")
    print(f"检索MRR @{args.top_k_retrieval}: {mrr_retrieval:.4f}")
    print(f"重排序MRR @{args.top_k_rerank}: {mrr_rerank:.4f}")
    
    if args.use_metadata_filter:
        print(f"✅ 元数据过滤已启用")
    else:
        print(f"ℹ️  元数据过滤未启用")

if __name__ == "__main__":
    main() 