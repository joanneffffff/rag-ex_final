#!/usr/bin/env python3
"""
使用RAG系统真实检索逻辑的TatQA数据集MRR评估
包括FAISS索引、BilingualRetriever和Qwen3-Reranker
"""

import json
import argparse
import numpy as np
import torch
from pathlib import Path
from typing import List, Dict, Any, Optional
from tqdm import tqdm
import sys

# 添加项目根目录到路径
sys.path.append(str(Path(__file__).parent.parent))

from xlm.dto.dto import DocumentWithMetadata, DocumentMetadata
from xlm.components.encoder.finbert import FinbertEncoder
from xlm.components.retriever.bilingual_retriever import BilingualRetriever
from xlm.components.retriever.reranker import QwenReranker
from xlm.utils.dual_language_loader import DualLanguageLoader
from enhanced_evaluation_functions import find_correct_document_rank_enhanced

def get_question_or_query(item_dict):
    """从样本中提取问题或查询"""
    if "query" in item_dict:
        return item_dict["query"]
    elif "question" in item_dict:
        return item_dict["question"]
    else:
        return None

def extract_unit_from_paragraph(paragraphs):
    """从段落中提取单位信息"""
    unit_info = ""
    for para in paragraphs:
        if isinstance(para, dict):
            text = para.get("text", "")
        else:
            text = para
        if "million" in text.lower() or "thousand" in text.lower():
            if "million" in text.lower():
                unit_info = "million USD"
            elif "thousand" in text.lower():
                unit_info = "thousand USD"
            break
    return unit_info

def table_to_natural_text(table_dict, caption="", unit_info=""):
    """将表格转换为自然语言文本"""
    import re
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

        for i, item in enumerate(data):
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

def compute_mrr_with_rag_system(encoder_model_name, reranker_model_name, eval_data, corpus_chunks, 
                               top_k_retrieval=100, top_k_rerank=10, batch_size=32):
    """使用RAG系统真实检索逻辑计算MRR"""
    
    print("🚀 初始化RAG系统组件...")
    
    # 1. 初始化编码器
    print("1. 加载Encoder模型...")
    encoder_en = FinbertEncoder(
        model_name=encoder_model_name,
        cache_dir="cache"
    )
    print("✅ Encoder模型加载完成")
    
    # 2. 将corpus_chunks转换为DocumentWithMetadata格式
    print("2. 转换检索库格式...")
    corpus_documents = []
    for chunk in corpus_chunks:
        doc = DocumentWithMetadata(
            content=chunk["text"],
            metadata=DocumentMetadata(
                doc_id=chunk.get('doc_id'),
                source=chunk.get('source_type', 'unknown'),
                created_at="",
                author="",
                language="english"
            )
        )
        corpus_documents.append(doc)
    print(f"✅ 转换了 {len(corpus_documents)} 个文档")
    
    # 3. 初始化BilingualRetriever（只使用英文部分）
    print("3. 初始化BilingualRetriever...")
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
        use_faiss=True,  # 使用FAISS索引
        use_gpu=True,
        batch_size=batch_size,
        cache_dir="cache",
        use_existing_embedding_index=False  # 强制重新计算嵌入
    )
    print("✅ BilingualRetriever初始化完成")
    
    # 4. 初始化Reranker
    print("4. 加载Reranker模型...")
    reranker = QwenReranker(
        model_name=reranker_model_name,
        device="cuda:0",
        cache_dir="cache",
        use_quantization=True,
        quantization_type="4bit"
    )
    print("✅ Reranker模型加载完成")
    
    # 5. 开始评估
    print("5. 开始MRR评估...")
    all_mrr_scores = []
    
    for i, sample in enumerate(tqdm(eval_data, desc="评估进度")):
        try:
            query_text = get_question_or_query(sample)
            correct_chunk_content = sample.get("context", "")
            
            if not query_text or not correct_chunk_content:
                all_mrr_scores.append(0.0)
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
                all_mrr_scores.append(0.0)
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
                print(f"重排序失败: {e}")
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
                all_mrr_scores.append(mrr_score)
            else:
                all_mrr_scores.append(0.0)
                
        except Exception as e:
            print(f"样本 {i} 处理失败: {e}")
            all_mrr_scores.append(0.0)
    
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
    
    print(f"\n=== RAG系统真实检索逻辑评估结果 ===")
    print(f"总样本数: {total_samples}")
    print(f"MRR: {mrr:.4f}")
    print(f"Hit@1: {hit_at_1_rate:.4f} ({hit_at_1}/{total_samples})")
    print(f"Hit@3: {hit_at_3_rate:.4f} ({hit_at_3}/{total_samples})")
    print(f"Hit@5: {hit_at_5_rate:.4f} ({hit_at_5}/{total_samples})")
    print(f"Hit@10: {hit_at_10_rate:.4f} ({hit_at_10}/{total_samples})")
    
    return mrr

def main():
    parser = argparse.ArgumentParser(description="使用RAG系统真实检索逻辑的TatQA数据集MRR评估")
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
    parser.add_argument("--max_eval_samples", type=int, default=None, 
                       help="最大评估样本数（用于快速测试）")
    
    args = parser.parse_args()

    # 检查GPU可用性
    if not torch.cuda.is_available():
        print("❌ CUDA不可用，无法使用GPU")
        return
    
    print(f"✅ 检测到GPU: {torch.cuda.get_device_name(0)}")

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

    # 构建检索库
    print(f"\n2. 构建检索库: {args.base_raw_data_path}")
    corpus_input_paths = [
        Path(args.base_raw_data_path) / "tatqa_dataset_train.json",
        Path(args.base_raw_data_path) / "tatqa_dataset_dev.json",
        Path(args.base_raw_data_path) / "tatqa_dataset_test_gold.json"
    ]
    
    existing_paths = [p for p in corpus_input_paths if p.exists()]
    if not existing_paths:
        print(f"❌ 没有找到任何TatQA原始数据文件")
        return
    
    corpus_chunks = process_tatqa_to_qca_for_corpus(existing_paths)
    print(f"✅ 构建了 {len(corpus_chunks)} 个Chunk作为检索库")

    # 使用RAG系统真实检索逻辑计算 MRR
    print(f"\n3. 开始RAG系统真实检索逻辑评估...")
    mrr = compute_mrr_with_rag_system(
        encoder_model_name=args.encoder_model_name,
        reranker_model_name=args.reranker_model_name,
        eval_data=eval_data,
        corpus_chunks=corpus_chunks,
        top_k_retrieval=args.top_k_retrieval,
        top_k_rerank=args.top_k_rerank,
        batch_size=args.batch_size
    )
    
    print(f"\n🎯 最终结果: MRR = {mrr:.4f}")

if __name__ == "__main__":
    main() 