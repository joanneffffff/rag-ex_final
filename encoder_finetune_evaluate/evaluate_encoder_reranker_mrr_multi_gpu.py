#!/usr/bin/env python3
"""
多GPU版本的TatQA数据集MRR评估脚本
支持Encoder + FAISS + Reranker组合评估
使用多个GPU并行处理，大幅提升速度
"""

import json
import argparse
from pathlib import Path
from tqdm import tqdm
from sentence_transformers import SentenceTransformer, util
import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent))
from xlm.components.retriever.reranker import QwenReranker
from xlm.dto.dto import DocumentWithMetadata, DocumentMetadata
import torch
import numpy as np
import os
import re
from typing import List, Dict, Any, Optional
from enhanced_evaluation_functions import find_correct_document_rank_enhanced

def get_question_or_query(item_dict):
    """兼容性函数：获取question或query字段"""
    if "question" in item_dict:
        return item_dict["question"]
    elif "query" in item_dict:
        return item_dict["query"]
    return None

def extract_unit_from_paragraph(paragraphs):
    """从段落中提取数值单位"""
    for para in paragraphs:
        text = para.get("text", "") if isinstance(para, dict) else para
        match = re.search(r'dollars in (millions|billions)|in (millions|billions)', text, re.IGNORECASE)
        if match:
            unit = match.group(1) or match.group(2)
            if unit:
                return unit.lower().replace('s', '') + " USD"
    return ""

def table_to_natural_text(table_dict, caption="", unit_info=""):
    """将表格转换为自然语言文本"""
    rows = table_dict.get("table", [])
    lines = []

    if caption:
        lines.append(f"Table Topic: {caption}.")

    if not rows:
        return ""

    headers = rows[0]
    data_rows = rows[1:]

    for i, row in enumerate(data_rows):
        if not row or all(str(v).strip() == "" for v in row):
            continue

        if len(row) > 1 and str(row[0]).strip() != "" and all(str(v).strip() == "" for v in row[1:]):
            lines.append(f"Table Category: {str(row[0]).strip()}.")
            continue

        row_name = str(row[0]).strip().replace('.', '')

        data_descriptions = []
        for h_idx, v in enumerate(row):
            if h_idx == 0:
                continue
            header = headers[h_idx] if h_idx < len(headers) else f"Column {h_idx+1}"
            value = str(v).strip()
            if value:
                if re.match(r'^-?\$?\d{1,3}(?:,\d{3})*(?:\.\d+)?$|^\(\$?\d{1,3}(?:,\d{3})*(?:\.\d+)?\)$', value):
                    formatted_value = value.replace('$', '')
                    if unit_info:
                        if formatted_value.startswith('(') and formatted_value.endswith(')'):
                             formatted_value = f"(${formatted_value[1:-1]} {unit_info})"
                        else:
                             formatted_value = f"${formatted_value} {unit_info}"
                    else:
                        formatted_value = f"${formatted_value}"
                else:
                    formatted_value = value
                data_descriptions.append(f"{header} is {formatted_value}")
        if row_name and data_descriptions:
            lines.append(f"Details for item {row_name}: {'; '.join(data_descriptions)}.")
        elif data_descriptions:
            lines.append(f"Other data item: {'; '.join(data_descriptions)}.")
        elif row_name:
            lines.append(f"Data item: {row_name}.")
    return "\n".join(lines)

def process_tatqa_to_qca_for_corpus(input_paths):
    """处理TatQA原始数据，构建检索库"""
    all_chunks = []
    global_doc_counter = 0

    for input_path in input_paths:
        if not Path(input_path).exists():
            print(f"警告：文件不存在，跳过: {input_path}")
            continue
            
        with open(input_path, "r", encoding="utf-8") as f:
            data = json.load(f)
        
        if not isinstance(data, list):
            print(f"警告：文件 {Path(input_path).name} 的顶层结构不是列表，尝试作为单个文档处理。")
            data = [data]

        for i, item in tqdm(enumerate(data), desc=f"Processing docs from {Path(input_path).name} for corpus"):
            if not isinstance(item, dict):
                print(f"警告：文件 {Path(input_path).name} 中发现非字典项，跳过。项内容：{item}")
                continue
            
            doc_id = item.get("doc_id")
            if doc_id is None:
                doc_id = f"generated_doc_{global_doc_counter}_{Path(input_path).stem}_{i}"
                global_doc_counter += 1

            paragraphs = item.get("paragraphs", [])
            tables = item.get("tables", [])

            unit_info = extract_unit_from_paragraph(paragraphs)

            for p_idx, para in enumerate(paragraphs):
                para_text = para.get("text", "") if isinstance(para, dict) else para
                if para_text.strip():
                    all_chunks.append({
                        "doc_id": doc_id,
                        "chunk_id": f"para_{p_idx}",
                        "text": para_text.strip(),
                        "source_type": "paragraph"
                    })
            
            for t_idx, table in enumerate(tables):
                table_text = table_to_natural_text(table, table.get("caption", ""), unit_info)
                if table_text.strip():
                    all_chunks.append({
                        "doc_id": doc_id,
                        "chunk_id": f"table_{t_idx}",
                        "text": table_text.strip(),
                        "source_type": "table"
                    })
    return all_chunks

def compute_mrr_with_reranker_multi_gpu(encoder_model_name, reranker_model_name, eval_data, corpus_chunks, 
                                       top_k_retrieval=100, top_k_rerank=10, batch_size=32, num_gpus=2):
    """
    多GPU版本的MRR计算函数 - 使用更简单的方法避免CUDA多进程问题
    """
    print(f"使用 {num_gpus} 个GPU进行并行处理")
    
    # 检查可用GPU数量
    available_gpus = torch.cuda.device_count()
    if available_gpus < num_gpus:
        print(f"警告：请求使用 {num_gpus} 个GPU，但只有 {available_gpus} 个可用")
        num_gpus = available_gpus
    
    # 将数据分割成多个批次
    total_samples = len(eval_data)
    samples_per_gpu = total_samples // num_gpus
    remainder = total_samples % num_gpus
    
    batches = []
    start_idx = 0
    for gpu_id in range(num_gpus):
        # 分配样本，确保均匀分布
        batch_size_for_gpu = samples_per_gpu + (1 if gpu_id < remainder else 0)
        end_idx = start_idx + batch_size_for_gpu
        batch_data = eval_data[start_idx:end_idx]
        batches.append((batch_data, gpu_id))
        start_idx = end_idx
    
    print(f"数据分割完成：")
    for i, (batch_data, gpu_id) in enumerate(batches):
        print(f"  GPU {gpu_id}: {len(batch_data)} 个样本")
    
    # 使用单进程多GPU处理（避免CUDA多进程问题）
    all_mrr_scores = []
    
    for batch_idx, (batch_data, gpu_id) in enumerate(batches):
        print(f"\n{'='*50}")
        print(f"处理 GPU {gpu_id} 的批次 {batch_idx + 1}/{len(batches)}")
        print(f"{'='*50}")
        
        # 设置当前GPU
        device = f"cuda:{gpu_id}"
        torch.cuda.set_device(gpu_id)
        print(f"设置CUDA设备为: {device}")
        print(f"当前CUDA设备: {torch.cuda.current_device()}")
        
        # 加载模型到指定GPU
        print(f"GPU {gpu_id}: 加载模型...")
        encoder_model = SentenceTransformer(encoder_model_name, device=device)
        
        reranker_model = QwenReranker(
            model_name=reranker_model_name,
            device=device,
            use_quantization=True,
            quantization_type="4bit"
        )
        
        # 确保模型在正确的GPU上
        encoder_model.to(device)
        
        # 验证模型设备
        encoder_device = next(encoder_model.parameters()).device
        reranker_device = reranker_model.model.device
        print(f"GPU {gpu_id}: Encoder模型在 {encoder_device}")
        print(f"GPU {gpu_id}: Reranker模型在 {reranker_device}")
        
        # 强制设置当前设备
        torch.cuda.set_device(gpu_id)
        print(f"GPU {gpu_id}: 当前CUDA设备设置为 {torch.cuda.current_device()}")
        
        # 显示GPU内存使用情况
        gpu_memory_allocated = torch.cuda.memory_allocated(gpu_id) / 1024**3
        gpu_memory_reserved = torch.cuda.memory_reserved(gpu_id) / 1024**3
        print(f"GPU {gpu_id}: 内存使用 - 已分配: {gpu_memory_allocated:.2f}GB, 已保留: {gpu_memory_reserved:.2f}GB")
        
        # 编码检索库（一次性完成）
        print(f"GPU {gpu_id}: 编码检索库...")
        corpus_texts = [chunk["text"] for chunk in corpus_chunks]
        corpus_embeddings = encoder_model.encode(
            corpus_texts, 
            show_progress_bar=False, 
            convert_to_tensor=True,
            batch_size=batch_size
        )
        
        batch_results = []
        
        # 处理当前批次的样本
        for i, sample in enumerate(tqdm(batch_data, desc=f"GPU {gpu_id} 处理样本")):
            query_text = get_question_or_query(sample)
            correct_chunk_content = sample["context"]
            
            if not query_text or not correct_chunk_content:
                batch_results.append(0.0)
                continue
            
            try:
                # 1. 编码查询
                query_embedding = encoder_model.encode(
                    [query_text], 
                    show_progress_bar=False, 
                    convert_to_tensor=True,
                    batch_size=1
                )
                
                # 2. 初步检索 (Encoder)
                cos_scores = util.cos_sim(query_embedding, corpus_embeddings)[0]
                top_retrieved_indices = torch.topk(cos_scores, k=min(top_k_retrieval, len(corpus_chunks)))[1].tolist()
                
                # 准备 Reranker 的输入
                retrieved_chunks = []
                docs_for_rerank = []
                for idx in top_retrieved_indices:
                    chunk = corpus_chunks[idx]
                    retrieved_chunks.append(chunk)
                    docs_for_rerank.append(chunk["text"])

                if not docs_for_rerank:
                    batch_results.append(0.0)
                    continue

                # 3. 重排序 (Reranker)
                try:
                    reranked_results = reranker_model.rerank(
                        query=query_text,
                        documents=docs_for_rerank,
                        batch_size=4
                    )
                    reranker_scores = [score for doc, score in reranked_results]
                except Exception as e:
                    print(f"GPU {gpu_id} 重排序失败: {e}")
                    reranker_scores = [1.0] * len(docs_for_rerank)
                
                # 4. 创建DocumentWithMetadata列表
                scored_retrieved_chunks_with_scores = []
                for j, score in enumerate(reranker_scores):
                    chunk = retrieved_chunks[j]
                    scored_retrieved_chunks_with_scores.append({
                        "chunk": chunk, 
                        "score": score
                    })

                # 按 Reranker 分数降序排序
                scored_retrieved_chunks_with_scores.sort(key=lambda x: x["score"], reverse=True)
                
                # 根据排序后的分数创建DocumentWithMetadata列表
                final_retrieved_docs_for_ranking: List[DocumentWithMetadata] = []
                for item in scored_retrieved_chunks_with_scores[:top_k_rerank]:
                    chunk = item["chunk"]
                    final_retrieved_docs_for_ranking.append(DocumentWithMetadata(
                        content=chunk['text'],
                        metadata=DocumentMetadata(
                            doc_id=chunk.get('doc_id'),
                            source=chunk.get('source_type', 'unknown'),
                            created_at="",
                            author="",
                            language="english"
                        )
                    ))

                # 5. 找到正确答案的排名（使用增强版函数）
                found_rank = find_correct_document_rank_enhanced(
                    context=correct_chunk_content,
                    retrieved_docs=final_retrieved_docs_for_ranking,
                    sample=sample,
                    encoder=encoder_model
                )
                
                if found_rank > 0:
                    mrr_score = 1.0 / found_rank
                    batch_results.append(mrr_score)
                else:
                    batch_results.append(0.0)
                    
            except Exception as e:
                print(f"GPU {gpu_id} 样本 {i} 处理失败: {e}")
                batch_results.append(0.0)
        
        all_mrr_scores.extend(batch_results)
        print(f"GPU {gpu_id}: 完成处理 {len(batch_data)} 个样本")
        
        # 清理GPU内存
        del encoder_model, reranker_model, corpus_embeddings
        torch.cuda.empty_cache()
    
    # 计算最终指标
    mrr = np.mean(all_mrr_scores)
    hit_at_1 = sum(1 for score in all_mrr_scores if score == 1.0)
    hit_at_3 = sum(1 for score in all_mrr_scores if score >= 1.0/3)
    hit_at_5 = sum(1 for score in all_mrr_scores if score >= 1.0/5)
    hit_at_10 = sum(1 for score in all_mrr_scores if score >= 1.0/10)
    
    total_samples = len(all_mrr_scores)
    hit_at_1_rate = hit_at_1 / total_samples
    hit_at_3_rate = hit_at_3 / total_samples
    hit_at_5_rate = hit_at_5 / total_samples
    hit_at_10_rate = hit_at_10 / total_samples
    
    print(f"\n=== 多GPU评估结果 ===")
    print(f"总样本数: {total_samples}")
    print(f"MRR: {mrr:.4f}")
    print(f"Hit@1: {hit_at_1_rate:.4f} ({hit_at_1}/{total_samples})")
    print(f"Hit@3: {hit_at_3_rate:.4f} ({hit_at_3}/{total_samples})")
    print(f"Hit@5: {hit_at_5_rate:.4f} ({hit_at_5}/{total_samples})")
    print(f"Hit@10: {hit_at_10_rate:.4f} ({hit_at_10}/{total_samples})")
    
    return mrr

def main():
    parser = argparse.ArgumentParser(description="多GPU版本的TatQA数据集MRR评估")
    parser.add_argument("--encoder_model_name", type=str, 
                       default="models/finetuned_finbert_tatqa",
                       help="Encoder模型路径")
    parser.add_argument("--reranker_model_name", type=str, 
                       default="Qwen/Qwen3-Reranker-0.6B",
                       help="Reranker模型路径或Hugging Face ID")
    parser.add_argument("--eval_jsonl", type=str, 
                       default="evaluate_mrr/tatqa_eval_enhanced.jsonl",
                       help="评估数据JSONL文件路径")
    parser.add_argument("--base_raw_data_path", type=str, 
                       default="data/tatqa_dataset_raw/",
                       help="TatQA原始数据路径")
    parser.add_argument("--top_k_retrieval", type=int, default=100, 
                       help="Encoder检索的top-k数量")
    parser.add_argument("--top_k_rerank", type=int, default=10, 
                       help="Reranker重排序后的top-k数量")
    parser.add_argument("--batch_size", type=int, default=32, 
                       help="批处理大小")
    parser.add_argument("--num_gpus", type=int, default=2, 
                       help="使用的GPU数量")
    parser.add_argument("--max_eval_samples", type=int, default=None, 
                       help="最大评估样本数（用于快速测试）")
    
    args = parser.parse_args()

    # 检查GPU可用性
    if not torch.cuda.is_available():
        print("❌ CUDA不可用，无法使用GPU")
        return
    
    available_gpus = torch.cuda.device_count()
    print(f"✅ 检测到 {available_gpus} 个GPU")
    for i in range(available_gpus):
        gpu_name = torch.cuda.get_device_name(i)
        gpu_memory = torch.cuda.get_device_properties(i).total_memory / 1024**3
        print(f"  GPU {i}: {gpu_name} ({gpu_memory:.1f} GB)")

    # ==================== 加载评估数据 ====================
    print(f"\n1. 加载评估数据: {args.eval_jsonl}")
    if not Path(args.eval_jsonl).exists():
        print(f"❌ 评估数据文件不存在: {args.eval_jsonl}")
        return
        
    raw_eval_data = []
    with open(args.eval_jsonl, "r", encoding="utf-8") as f:
        for line in f:
            raw_eval_data.append(json.loads(line))
    
    eval_data = []
    for item in raw_eval_data:
        is_valid_item = isinstance(item, dict) and "context" in item and \
                        (get_question_or_query(item) is not None)
        
        if is_valid_item:
            eval_data.append(item)
        else:
            print(f"警告：跳过无效样本: {item}")
    
    if args.max_eval_samples:
        eval_data = eval_data[:args.max_eval_samples]
        print(f"限制评估样本数为: {args.max_eval_samples}")
    
    print(f"✅ 加载了 {len(eval_data)} 个有效评估样本")

    if not eval_data:
        print("❌ 没有找到任何有效的评估样本")
        return

    # ==================== 构建检索库 ====================
    print(f"\n2. 构建检索库: {args.base_raw_data_path}")
    corpus_input_paths = [
        Path(args.base_raw_data_path) / "tatqa_dataset_train.json",
        Path(args.base_raw_data_path) / "tatqa_dataset_dev.json",
        Path(args.base_raw_data_path) / "tatqa_dataset_test_gold.json"
    ]
    
    # 检查文件是否存在
    existing_paths = [p for p in corpus_input_paths if p.exists()]
    if not existing_paths:
        print(f"❌ 没有找到任何TatQA原始数据文件")
        print(f"期望的文件:")
        for p in corpus_input_paths:
            print(f"  - {p}")
        return
    
    corpus_chunks = process_tatqa_to_qca_for_corpus(existing_paths)
    print(f"✅ 构建了 {len(corpus_chunks)} 个Chunk作为检索库")

    # ==================== 多GPU计算 MRR ====================
    print(f"\n3. 开始多GPU评估...")
    mrr = compute_mrr_with_reranker_multi_gpu(
        encoder_model_name=args.encoder_model_name,
        reranker_model_name=args.reranker_model_name,
        eval_data=eval_data,
        corpus_chunks=corpus_chunks,
        top_k_retrieval=args.top_k_retrieval,
        top_k_rerank=args.top_k_rerank,
        batch_size=args.batch_size,
        num_gpus=args.num_gpus
    )
    
    print(f"\n🎯 最终结果: MRR = {mrr:.4f}")

if __name__ == "__main__":
    main() 