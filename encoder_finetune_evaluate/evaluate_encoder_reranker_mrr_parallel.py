#!/usr/bin/env python3
"""
真正的并行多GPU版本MRR评估
使用多进程实现真正的并行处理
"""

import json
import argparse
import numpy as np
import torch
from pathlib import Path
from tqdm import tqdm
from sentence_transformers import SentenceTransformer, util
from multiprocessing import Process, Queue, Manager
import time

from xlm.components.retriever.reranker import QwenReranker
from xlm.utils.enhanced_evaluation import find_correct_document_rank_enhanced
from xlm.utils.data_structures import DocumentWithMetadata, DocumentMetadata

def get_question_or_query(item_dict):
    """获取问题或查询文本"""
    return item_dict.get("query") or item_dict.get("question")

def gpu_worker(gpu_id, batch_data, corpus_chunks, encoder_model_name, reranker_model_name, 
               top_k_retrieval, top_k_rerank, batch_size, result_queue):
    """
    单个GPU工作进程
    """
    try:
        # 设置GPU设备
        device = f"cuda:{gpu_id}"
        torch.cuda.set_device(gpu_id)
        
        print(f"GPU {gpu_id}: 开始处理 {len(batch_data)} 个样本")
        
        # 加载模型
        encoder_model = SentenceTransformer(encoder_model_name, device=device)
        reranker_model = QwenReranker(
            model_name=reranker_model_name,
            device=device,
            use_quantization=True,
            quantization_type="4bit"
        )
        
        # 编码检索库
        corpus_texts = [chunk["text"] for chunk in corpus_chunks]
        corpus_embeddings = encoder_model.encode(
            corpus_texts, 
            show_progress_bar=False, 
            convert_to_tensor=True,
            batch_size=batch_size
        )
        
        batch_results = []
        
        # 处理样本
        for i, sample in enumerate(batch_data):
            query_text = get_question_or_query(sample)
            correct_chunk_content = sample["context"]
            
            if not query_text or not correct_chunk_content:
                batch_results.append(0.0)
                continue
            
            try:
                # 编码查询
                query_embedding = encoder_model.encode(
                    [query_text], 
                    show_progress_bar=False, 
                    convert_to_tensor=True,
                    batch_size=1
                )
                
                # 初步检索
                cos_scores = util.cos_sim(query_embedding, corpus_embeddings)[0]
                top_retrieved_indices = torch.topk(cos_scores, k=min(top_k_retrieval, len(corpus_chunks)))[1].tolist()
                
                # 准备重排序
                retrieved_chunks = []
                docs_for_rerank = []
                for idx in top_retrieved_indices:
                    chunk = corpus_chunks[idx]
                    retrieved_chunks.append(chunk)
                    docs_for_rerank.append(chunk["text"])

                if not docs_for_rerank:
                    batch_results.append(0.0)
                    continue

                # 重排序
                reranked_results = reranker_model.rerank(
                    query=query_text,
                    documents=docs_for_rerank,
                    batch_size=4
                )
                reranker_scores = [score for doc, score in reranked_results]
                
                # 创建文档列表
                scored_retrieved_chunks_with_scores = []
                for j, score in enumerate(reranker_scores):
                    chunk = retrieved_chunks[j]
                    scored_retrieved_chunks_with_scores.append({
                        "chunk": chunk, 
                        "score": score
                    })

                scored_retrieved_chunks_with_scores.sort(key=lambda x: x["score"], reverse=True)
                
                final_retrieved_docs_for_ranking = []
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

                # 找到正确答案排名
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
        
        # 发送结果
        result_queue.put((gpu_id, batch_results))
        print(f"GPU {gpu_id}: 完成处理，发送结果")
        
        # 清理
        del encoder_model, reranker_model, corpus_embeddings
        torch.cuda.empty_cache()
        
    except Exception as e:
        print(f"GPU {gpu_id} 工作进程失败: {e}")
        result_queue.put((gpu_id, [0.0] * len(batch_data)))

def compute_mrr_with_reranker_parallel(encoder_model_name, reranker_model_name, eval_data, corpus_chunks, 
                                      top_k_retrieval=100, top_k_rerank=10, batch_size=32, num_gpus=2):
    """
    真正的并行多GPU版本MRR计算
    """
    print(f"使用 {num_gpus} 个GPU进行并行处理")
    
    # 检查可用GPU数量
    available_gpus = torch.cuda.device_count()
    if available_gpus < num_gpus:
        print(f"警告：请求使用 {num_gpus} 个GPU，但只有 {available_gpus} 个可用")
        num_gpus = available_gpus
    
    # 分割数据
    total_samples = len(eval_data)
    samples_per_gpu = total_samples // num_gpus
    remainder = total_samples % num_gpus
    
    batches = []
    start_idx = 0
    for gpu_id in range(num_gpus):
        batch_size_for_gpu = samples_per_gpu + (1 if gpu_id < remainder else 0)
        end_idx = start_idx + batch_size_for_gpu
        batch_data = eval_data[start_idx:end_idx]
        batches.append((batch_data, gpu_id))
        start_idx = end_idx
    
    print(f"数据分割完成：")
    for i, (batch_data, gpu_id) in enumerate(batches):
        print(f"  GPU {gpu_id}: {len(batch_data)} 个样本")
    
    # 创建结果队列
    result_queue = Queue()
    
    # 启动并行进程
    processes = []
    start_time = time.time()
    
    for batch_data, gpu_id in batches:
        p = Process(target=gpu_worker, args=(
            gpu_id, batch_data, corpus_chunks, encoder_model_name, reranker_model_name,
            top_k_retrieval, top_k_rerank, batch_size, result_queue
        ))
        processes.append(p)
        p.start()
    
    # 收集结果
    all_mrr_scores = []
    for _ in range(len(processes)):
        gpu_id, batch_results = result_queue.get()
        all_mrr_scores.extend(batch_results)
        print(f"收到 GPU {gpu_id} 的结果: {len(batch_results)} 个分数")
    
    # 等待所有进程完成
    for p in processes:
        p.join()
    
    end_time = time.time()
    print(f"并行处理总耗时: {end_time - start_time:.2f}秒")
    
    # 计算指标
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
    
    print(f"\n=== 并行多GPU评估结果 ===")
    print(f"总样本数: {total_samples}")
    print(f"MRR: {mrr:.4f}")
    print(f"Hit@1: {hit_at_1_rate:.4f} ({hit_at_1}/{total_samples})")
    print(f"Hit@3: {hit_at_3_rate:.4f} ({hit_at_3}/{total_samples})")
    print(f"Hit@5: {hit_at_5_rate:.4f} ({hit_at_5}/{total_samples})")
    print(f"Hit@10: {hit_at_10_rate:.4f} ({hit_at_10}/{total_samples})")
    
    return mrr

# 其他函数保持不变...
def extract_unit_from_paragraph(paragraphs):
    import re
    for para in paragraphs:
        text = para.get("text", "") if isinstance(para, dict) else para
        match = re.search(r'dollars in (millions|billions)|in (millions|billions)', text, re.IGNORECASE)
        if match:
            unit = match.group(1) or match.group(2)
            if unit:
                return unit.lower().replace('s', '') + " USD"
    return ""

def table_to_natural_text(table_dict, caption="", unit_info=""):
    import re
    if not table_dict:
        return ""
    
    if isinstance(table_dict, dict):
        rows = table_dict.get("table", [])
        table_uid = table_dict.get("uid", "")
    elif isinstance(table_dict, list):
        rows = table_dict
        table_uid = ""
    else:
        return ""
    
    if not rows:
        return ""
    
    lines = []
    
    if table_uid:
        lines.append(f"Table ID: {table_uid}")
    if caption:
        lines.append(f"Table Topic: {caption}")

    headers = rows[0] if rows else []
    data_rows = rows[1:] if len(rows) > 1 else []

    if headers:
        header_text = " | ".join(str(h).strip() for h in headers if str(h).strip())
        if header_text:
            lines.append(f"Headers: {header_text}")

    for i, row in enumerate(data_rows):
        if not row or all(str(v).strip() == "" for v in row):
            continue

        if len(row) > 1 and str(row[0]).strip() != "" and all(str(v).strip() == "" for v in row[1:]):
            lines.append(f"Category: {str(row[0]).strip()}")
            continue

        row_name = str(row[0]).strip().replace('.', '') if row[0] else ""
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
            lines.append(f"{row_name}: {'; '.join(data_descriptions)}")
        elif data_descriptions:
            lines.append(f"Row {i+1}: {'; '.join(data_descriptions)}")
        elif row_name:
            lines.append(f"Item: {row_name}")

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

        for i, item in enumerate(tqdm(data, desc=f"Processing docs from {Path(input_path).name} for corpus")):
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

def main():
    parser = argparse.ArgumentParser(description="真正的并行多GPU版本TatQA数据集MRR评估")
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

    # 加载评估数据
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

    # 构建检索库
    print(f"\n2. 构建检索库: {args.base_raw_data_path}")
    raw_data_paths = [
        Path(args.base_raw_data_path) / "tatqa_dataset_train.json",
        Path(args.base_raw_data_path) / "tatqa_dataset_dev.json",
        Path(args.base_raw_data_path) / "tatqa_dataset_test_gold.json"
    ]
    
    corpus_chunks = process_tatqa_to_qca_for_corpus(raw_data_paths)
    print(f"✅ 构建了 {len(corpus_chunks)} 个Chunk作为检索库")

    # 开始并行多GPU评估
    print(f"\n3. 开始并行多GPU评估...")
    mrr = compute_mrr_with_reranker_parallel(
        encoder_model_name=args.encoder_model_name,
        reranker_model_name=args.reranker_model_name,
        eval_data=eval_data,
        corpus_chunks=corpus_chunks,
        top_k_retrieval=args.top_k_retrieval,
        top_k_rerank=args.top_k_rerank,
        batch_size=args.batch_size,
        num_gpus=args.num_gpus
    )
    
    print(f"\n🎉 并行多GPU评估完成！最终MRR: {mrr:.4f}")

if __name__ == "__main__":
    main() 