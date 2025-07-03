#!/usr/bin/env python3
"""
修复版多GPU并行RAG系统真实检索逻辑MRR评估
使用增强版训练数据作为检索库，确保与评估数据匹配
"""

import json
import argparse
import torch
import numpy as np
from pathlib import Path
import multiprocessing as mp
from multiprocessing import Process
from tqdm import tqdm
import sys
import os

# 添加项目根目录到Python路径
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# 导入必要的模块
try:
    from xlm.dto.dto import DocumentWithMetadata, DocumentMetadata
    from xlm.components.encoder.finbert import FinbertEncoder
    from xlm.components.retriever.bilingual_retriever import BilingualRetriever
    from xlm.components.retriever.reranker import QwenReranker
    from encoder_finetune_evaluate.enhanced_evaluation_functions import find_correct_document_rank_enhanced
except ImportError as e:
    print(f"❌ 导入错误: {e}")
    print("请确保在项目根目录下运行此脚本")
    sys.exit(1)

def get_question_or_query(item_dict):
    """提取问题或查询"""
    if "question" in item_dict:
        return item_dict["question"]
    elif "query" in item_dict:
        return item_dict["query"]
    return None

def load_enhanced_corpus(corpus_file: str) -> list:
    """加载增强版训练数据作为检索库"""
    print(f"🔄 加载增强版检索库: {corpus_file}")
    
    if not Path(corpus_file).exists():
        print(f"❌ 检索库文件不存在: {corpus_file}")
        return []
    
    corpus_chunks = []
    with open(corpus_file, "r", encoding="utf-8") as f:
        for line in f:
            chunk = json.loads(line)
            # 使用relevant_doc_ids中的哈希值作为doc_id
            relevant_doc_ids = chunk.get("relevant_doc_ids", [])
            if relevant_doc_ids:
                doc_id = relevant_doc_ids[0]  # 使用第一个哈希值
            else:
                doc_id = chunk.get("doc_id", "")
            
            # 知识库文件使用"text"字段，评估数据使用"context"字段
            content = chunk.get("text", chunk.get("context", ""))
            corpus_chunks.append({
                "text": content,
                "doc_id": doc_id,
                "source_type": chunk.get("source_type", "enhanced")
            })
    
    print(f"✅ 加载了 {len(corpus_chunks)} 个增强版chunk")
    return corpus_chunks

def gpu_worker_rag_system_fixed(gpu_id, data_queue, result_queue, encoder_model_name, reranker_model_name, 
                               corpus_chunks, top_k_retrieval, top_k_rerank, batch_size):
    """修复版GPU工作进程函数"""
    try:
        # 设置CUDA设备
        torch.cuda.set_device(gpu_id)
        device = torch.device(f'cuda:{gpu_id}')
        print(f"GPU {gpu_id}: 开始工作，设备: {device}")
        
        # 加载模型
        print(f"GPU {gpu_id}: 加载模型...")
        
        # 加载Encoder模型
        encoder_en = FinbertEncoder(
            model_name=encoder_model_name,
            cache_dir="cache"
        )
        print(f"GPU {gpu_id}: Encoder模型加载完成")
        
        # 将corpus_chunks转换为DocumentWithMetadata格式
        print(f"GPU {gpu_id}: 转换检索库格式...")
        corpus_documents = []
        for chunk in corpus_chunks:
            doc = DocumentWithMetadata(
                content=chunk["text"],
                metadata=DocumentMetadata(
                    doc_id=chunk.get('doc_id'),
                    source=chunk.get('source_type', 'enhanced'),
                    created_at="",
                    author="",
                    language="english"
                )
            )
            corpus_documents.append(doc)
        print(f"GPU {gpu_id}: 转换了 {len(corpus_documents)} 个文档")
        
        # 初始化BilingualRetriever（只使用英文部分）
        print(f"GPU {gpu_id}: 初始化BilingualRetriever...")
        # 创建一个虚拟的中文编码器（因为BilingualRetriever需要两个编码器）
        encoder_ch = FinbertEncoder(
            model_name=encoder_model_name,  # 使用相同的模型
            cache_dir="cache"
        )
        
        retriever = BilingualRetriever(
            encoder_en=encoder_en,
            encoder_ch=encoder_ch,  # 使用虚拟中文编码器
            corpus_documents_en=corpus_documents,
            corpus_documents_ch=[],  # 空的中文文档列表
            use_faiss=False,  # 不使用FAISS，避免索引问题
            use_gpu=True,
            batch_size=8,  # 减小编码批次大小
            cache_dir="cache",
            use_existing_embedding_index=False  # 强制重新计算嵌入
        )
        print(f"GPU {gpu_id}: BilingualRetriever初始化完成")
        
        # 初始化Reranker
        print(f"GPU {gpu_id}: 加载Reranker模型...")
        reranker = QwenReranker(
            model_name=reranker_model_name,
            device=f"cuda:{gpu_id}",
            cache_dir="cache",
            use_quantization=True,
            quantization_type="4bit"
        )
        print(f"GPU {gpu_id}: Reranker模型加载完成")
        
        # 处理数据
        batch_results = []
        while True:
            try:
                # 从队列获取数据
                batch_data = data_queue.get(timeout=1)
                if batch_data is None:  # 结束信号
                    break
                
                print(f"GPU {gpu_id}: 处理批次，样本数: {len(batch_data)}")
                
                # 添加tqdm进度条
                from tqdm import tqdm
                for i, sample in enumerate(tqdm(batch_data, desc=f"GPU {gpu_id} 处理样本", leave=False)):
                    try:
                        query_text = get_question_or_query(sample)
                        correct_chunk_content = sample.get("context", "")
                        
                        if not query_text or not correct_chunk_content:
                            batch_results.append(0.0)
                            continue
                        
                        # 1. 使用RAG系统的真实检索逻辑
                        retrieved_result = retriever.retrieve(
                            text=query_text,
                            top_k=top_k_retrieval,
                            return_scores=True,
                            language="english"
                        )
                        
                        if isinstance(retrieved_result, tuple):
                            retrieved_docs, retrieval_scores = retrieved_result
                        else:
                            retrieved_docs = retrieved_result
                            retrieval_scores = []
                        
                        if not retrieved_docs:
                            batch_results.append(0.0)
                            continue
                        
                        # 2. 使用Reranker重排序
                        try:
                            # 准备重排序的文档
                            docs_for_rerank = []
                            for doc in retrieved_docs:
                                docs_for_rerank.append({
                                    'content': doc.content,
                                    'metadata': doc.metadata.__dict__
                                })
                            
                            # 执行重排序
                            reranked_results = reranker.rerank_with_metadata(
                                query=query_text,
                                documents_with_metadata=docs_for_rerank,
                                batch_size=2  # 小批次避免内存不足
                            )
                            
                            # 提取重排序后的文档和分数
                            final_retrieved_docs_for_ranking = []
                            for result in reranked_results[:top_k_rerank]:
                                doc = DocumentWithMetadata(
                                    content=result['content'],
                                    metadata=DocumentMetadata(**result['metadata'])
                                )
                                final_retrieved_docs_for_ranking.append(doc)
                            
                        except Exception as e:
                            print(f"GPU {gpu_id} 重排序失败: {e}")
                            # 如果重排序失败，使用原始检索结果
                            final_retrieved_docs_for_ranking = retrieved_docs[:top_k_rerank]
                        
                        # 3. 找到正确答案的排名
                        found_rank = find_correct_document_rank_enhanced(
                            context=correct_chunk_content,
                            retrieved_docs=final_retrieved_docs_for_ranking,
                            sample=sample,
                            encoder=encoder_en
                        )
                        
                        if found_rank > 0:
                            mrr_score = 1.0 / found_rank
                            batch_results.append(mrr_score)
                        else:
                            batch_results.append(0.0)
                            
                    except Exception as e:
                        print(f"GPU {gpu_id} 样本 {i} 处理失败: {e}")
                        batch_results.append(0.0)
                
                print(f"GPU {gpu_id}: 批次处理完成，结果数: {len(batch_results)}")
                
            except Exception as e:
                print(f"GPU {gpu_id} 队列处理失败: {e}")
                break
        
        # 发送结果
        result_queue.put((gpu_id, batch_results))
        print(f"GPU {gpu_id}: 工作完成，结果已发送")
        
    except Exception as e:
        print(f"GPU {gpu_id} 工作进程失败: {e}")
        result_queue.put((gpu_id, []))

def compute_mrr_with_rag_system_multi_gpu_fixed(encoder_model_name, reranker_model_name, eval_data, corpus_chunks, 
                                               top_k_retrieval=100, top_k_rerank=10, batch_size=32, num_gpus=2):
    """修复版多GPU并行RAG系统MRR计算"""
    
    # 数据分割
    total_samples = len(eval_data)
    samples_per_gpu = total_samples // num_gpus
    remainder = total_samples % num_gpus
    
    print(f"使用 {num_gpus} 个GPU进行并行处理")
    print(f"数据分割完成：")
    
    start_idx = 0
    gpu_data = []
    for gpu_id in range(num_gpus):
        # 分配样本，处理余数
        current_batch_size = samples_per_gpu + (1 if gpu_id < remainder else 0)
        end_idx = start_idx + current_batch_size
        gpu_samples = eval_data[start_idx:end_idx]
        gpu_data.append(gpu_samples)
        
        print(f"  GPU {gpu_id}: {len(gpu_samples)} 个样本")
        start_idx = end_idx
    
    # 创建进程间通信队列
    data_queues = [mp.Queue() for _ in range(num_gpus)]
    result_queue = mp.Queue()
    
    # 启动GPU工作进程
    processes = []
    for gpu_id in range(num_gpus):
        p = Process(target=gpu_worker_rag_system_fixed, args=(
            gpu_id, data_queues[gpu_id], result_queue, encoder_model_name, reranker_model_name,
            corpus_chunks, top_k_retrieval, top_k_rerank, batch_size
        ))
        p.start()
        processes.append(p)
        print(f"GPU {gpu_id} 进程已启动")
    
    # 发送数据到各个GPU
    for gpu_id in range(num_gpus):
        data_queues[gpu_id].put(gpu_data[gpu_id])
        data_queues[gpu_id].put(None)  # 结束信号
        print(f"数据已发送到 GPU {gpu_id}")
    
    # 收集结果
    all_results = []
    from tqdm import tqdm
    for _ in tqdm(range(num_gpus), desc="收集GPU结果"):
        gpu_id, results = result_queue.get()
        all_results.extend(results)
        print(f"收到 GPU {gpu_id} 的结果，样本数: {len(results)}")
    
    # 等待所有进程完成
    for p in processes:
        p.join()
    
    # 计算最终指标
    mrr = np.mean(all_results)
    hit_at_1 = sum(1 for score in all_results if score == 1.0)
    hit_at_3 = sum(1 for score in all_results if score >= 1.0/3)
    hit_at_5 = sum(1 for score in all_results if score >= 1.0/5)
    hit_at_10 = sum(1 for score in all_results if score >= 1.0/10)
    
    total_samples = len(all_results)
    hit_at_1_rate = hit_at_1 / total_samples
    hit_at_3_rate = hit_at_3 / total_samples
    hit_at_5_rate = hit_at_5 / total_samples
    hit_at_10_rate = hit_at_10 / total_samples
    
    print(f"\n=== 修复版多GPU并行RAG系统评估结果 ===")
    print(f"总样本数: {total_samples}")
    print(f"MRR: {mrr:.4f}")
    print(f"Hit@1: {hit_at_1_rate:.4f} ({hit_at_1}/{total_samples})")
    print(f"Hit@3: {hit_at_3_rate:.4f} ({hit_at_3}/{total_samples})")
    print(f"Hit@5: {hit_at_5_rate:.4f} ({hit_at_5}/{total_samples})")
    print(f"Hit@10: {hit_at_10_rate:.4f} ({hit_at_10}/{total_samples})")
    
    return mrr

def main():
    parser = argparse.ArgumentParser(description="修复版多GPU并行RAG系统真实检索逻辑MRR评估")
    parser.add_argument("--encoder_model_name", type=str, 
                       default="models/finetuned_finbert_tatqa",
                       help="Encoder模型路径")
    parser.add_argument("--reranker_model_name", type=str, 
                       default="Qwen/Qwen3-Reranker-0.6B",
                       help="Reranker模型路径或Hugging Face ID")
    parser.add_argument("--eval_jsonl", type=str, 
                       default="evaluate_mrr/tatqa_eval_enhanced.jsonl",
                       help="评估数据JSONL文件路径")
    parser.add_argument("--corpus_jsonl", type=str, 
                       default="evaluate_mrr/tatqa_knowledge_base.jsonl",
                       help="知识库JSONL文件路径")
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

    if not eval_data:
        print("❌ 没有找到任何有效的评估样本")
        return

    # 加载增强版检索库
    print(f"\n2. 加载增强版检索库: {args.corpus_jsonl}")
    corpus_chunks = load_enhanced_corpus(args.corpus_jsonl)
    
    if not corpus_chunks:
        print("❌ 没有找到有效的检索库数据")
        return

    # 多GPU并行RAG系统计算 MRR
    print(f"\n3. 开始修复版多GPU并行RAG系统评估...")
    mrr = compute_mrr_with_rag_system_multi_gpu_fixed(
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
    # 设置多进程启动方法
    try:
        mp.set_start_method('spawn', force=True)
    except RuntimeError:
        pass  # 如果已经设置过了，就忽略
    main() 