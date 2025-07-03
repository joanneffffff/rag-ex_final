#!/usr/bin/env python3
"""
测试检索质量 - MRR评估 (CPU版本)
使用evaluate_mrr/alphafin_eval.jsonl和tatqa_eval_enhanced.jsonl作为测试集
支持两种模式：
1. 评估数据context加入知识库 - 测试真实检索能力
2. 评估数据context不加入知识库 - 避免数据泄露

改进的匹配策略：
1. relevant_doc_ids匹配（最严格，适用于英文数据）
2. ID匹配（适用于中文数据）
3. 内容哈希匹配
4. 相似度匹配
5. 模糊文本匹配

评估逻辑已更新为 Encoder + FAISS + Reranker 组合
"""

import sys
import os
import json
import numpy as np
import hashlib
from pathlib import Path
from typing import List, Dict, Any, Tuple, Optional
from tqdm import tqdm

# 确保导入路径正确，根据你的项目结构调整
sys.path.append(str(Path(__file__).parent.parent)) # 假设 xlm 模块在上一级目录

# 导入必要的类型和组件
from xlm.dto.dto import DocumentWithMetadata, DocumentMetadata
from xlm.components.retriever.bilingual_retriever import BilingualRetriever
from xlm.components.encoder.finbert import FinbertEncoder # 你的Encoder类
from xlm.components.retriever.reranker import Reranker # 你的Reranker类
from xlm.utils.optimized_data_loader import OptimizedDataLoader # 你的数据加载器
from config.parameters import Config # 你的配置类

# 导入增强版评估函数 (假设它存在并能够处理多种匹配策略)
# 假设 evaluate_mrr/enhanced_evaluation_functions.py 在同一目录或可访问路径
from enhanced_evaluation_functions import find_correct_document_rank_enhanced 


def load_eval_data(eval_file: str) -> List[Dict[str, Any]]:
    """加载评估数据"""
    data = []
    with open(eval_file, 'r', encoding='utf-8') as f:
        for line in f:
            if line.strip():
                data.append(json.loads(line))
    return data

def calculate_mrr(ranks: List[int]) -> float:
    """计算MRR (Mean Reciprocal Rank)"""
    if not ranks:
        return 0.0
    reciprocal_ranks = [1.0 / rank if rank > 0 else 0.0 for rank in ranks]
    return float(np.mean(reciprocal_ranks))

def calculate_hit_rate(ranks: List[int], k: int = 1) -> float:
    """计算Hit@k"""
    if not ranks:
        return 0.0
    hits = [1 if rank <= k and rank > 0 else 0 for rank in ranks]
    return float(np.mean(hits))

def evaluate_dataset(eval_data: List[Dict[str, Any]], 
                     retriever: BilingualRetriever, 
                     encoder: FinbertEncoder, 
                     language: str, 
                     dataset_name: str,
                     reranker: Optional[Reranker] = None) -> Dict[str, float]:
    """
    评估单个数据集的检索质量 (Encoder + FAISS + Reranker)
    
    Args:
        eval_data: 评估数据列表
        retriever: BilingualRetriever 实例 (负责初步的Encoder + FAISS检索)
        encoder: 对应语言的编码器 (用于 find_correct_document_rank_enhanced 中的相似度计算)
        language: 语言 ('zh' 或 'en')
        dataset_name: 数据集名称
        reranker: Reranker 实例 (可选，用于重排序)
    
    Returns:
        评估结果字典
    """
    print(f"开始评估 {dataset_name} 数据集 ({len(eval_data)} 个样本) - 使用 Encoder + FAISS{' + Reranker' if reranker else ''}...")
    
    mrr_scores = []
    hit_at_1 = 0
    hit_at_3 = 0
    hit_at_5 = 0
    hit_at_10 = 0
    found_samples = 0
    
    for i, sample in enumerate(tqdm(eval_data, desc=f"评估 {dataset_name}")):
        query = sample.get('query', sample.get('question', '')) # 兼容TatQA和AlphaFin
        gold_context_content = sample.get('context', '') # 这是Gold Context，用于匹配

        if not query or not gold_context_content:
            continue
        
        try:
            # 1. 执行初步检索 (Encoder + FAISS)
            retrieved_result = retriever.retrieve(
                text=query, 
                top_k=50, # 初始检索K值可以设高一些，给Reranker更多选择
                return_scores=True, 
                language=language
            )
            
            initial_retrieved_docs: List[DocumentWithMetadata] # 明确类型
            if isinstance(retrieved_result, tuple):
                initial_retrieved_docs, initial_scores = retrieved_result
            else:
                initial_retrieved_docs = retrieved_result
                initial_scores = []
            
            # -----------------------------------------------------------
            # 2. 执行重排序 (Reranker)
            final_retrieved_docs_for_ranking: List[DocumentWithMetadata]
            if reranker and initial_retrieved_docs:
                # 获取文档内容列表用于rerank
                docs_content_for_rerank = [doc.content for doc in initial_retrieved_docs]
                
                # 调用reranker，返回排序后的 (doc_text, score) 元组列表
                reranked_items = reranker.rerank(
                    query=query, 
                    documents=docs_content_for_rerank, 
                    batch_size=4
                )

                # 根据reranked_items的顺序，重建 DocumentWithMetadata 列表
                # 需要一个从content到原始DocumentWithMetadata的映射，以保留原始ID等元数据
                content_to_original_doc_map = {doc.content: doc for doc in initial_retrieved_docs}
                
                temp_reranked_docs = []
                for doc_text, score in reranked_items[:20]:  # 取前20个结果
                    original_doc = content_to_original_doc_map.get(doc_text)
                    if original_doc:
                        # 确保DocWithMetadata对象被添加到列表中
                        temp_reranked_docs.append(original_doc)
                
                final_retrieved_docs_for_ranking = temp_reranked_docs
            else:
                final_retrieved_docs_for_ranking = initial_retrieved_docs # 如果没有reranker，使用初步检索结果
            # -----------------------------------------------------------

            # 3. 检查正确答案的排名
            found_rank = find_correct_document_rank_enhanced(
                context=gold_context_content, # Gold Context，来自评估样本
                retrieved_docs=final_retrieved_docs_for_ranking, # 经过重排序的文档列表
                sample=sample, # 完整的评估样本，包含 relevant_doc_ids 等
                encoder=encoder # 对应语言的编码器 (用于相似度等匹配)
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
                mrr_scores.append(0.0) # 如果没找到，MRR贡献为0
                
        except Exception as e:
            print(f"  样本 {i} ({query[:30]}...) 处理失败: {e}")
            # traceback.print_exc() # 打印详细错误信息，调试时使用
            mrr_scores.append(0.0) # 失败的样本MRR贡献为0
    
    # 计算指标
    total_samples = len(eval_data)
    mrr = sum(mrr_scores) / total_samples if total_samples > 0 else 0.0
    hit_at_1_rate = hit_at_1 / total_samples if total_samples > 0 else 0.0
    hit_at_3_rate = hit_at_3 / total_samples if total_samples > 0 else 0.0
    hit_at_5_rate = hit_at_5 / total_samples if total_samples > 0 else 0.0
    hit_at_10_rate = hit_at_10 / total_samples if total_samples > 0 else 0.0
    
    print(f"  {dataset_name} 评估完成:")
    print(f"    MRR: {mrr:.4f}")
    print(f"    Hit@1: {hit_at_1_rate:.4f} ({hit_at_1}/{total_samples})")
    print(f"    Hit@3: {hit_at_3_rate:.4f} ({hit_at_3}/{total_samples})")
    print(f"    Hit@5: {hit_at_5_rate:.4f} ({hit_at_5}/{total_samples})")
    print(f"    Hit@10: {hit_at_10_rate:.4f} ({hit_at_10}/{total_samples})")
    print(f"    找到正确答案的样本数: {found_samples}/{total_samples}") # 找到的样本数

    return {
        'mrr': mrr,
        'hit_at_1': hit_at_1_rate,
        'hit_at_3': hit_at_3_rate,
        'hit_at_5': hit_at_5_rate,
        'hit_at_10': hit_at_10_rate,
        'total_samples': total_samples,
        'found_samples': found_samples
    }

def test_retrieval_with_eval_context(include_eval_data: bool = True):
    """测试检索质量 - 可选择是否包含评估数据到知识库"""
    mode = "包含评估数据" if include_eval_data else "不包含评估数据"
    print("=" * 60)
    print(f"测试检索质量 - MRR评估 ({mode}) - CPU版本")
    print("=" * 60)
    
    try:
        config = Config()
        
        print("1. 加载编码器（CPU模式）...")
        encoder_ch = FinbertEncoder(
            model_name="./models/finetuned_alphafin_zh_optimized",
            cache_dir=config.encoder.cache_dir,
            device="cpu"  # 强制使用CPU
        )
        encoder_en = FinbertEncoder(
            model_name="models/finetuned_finbert_tatqa",
            cache_dir=config.encoder.cache_dir,
            device="cpu"  # 强制使用CPU
        )
        print("   ✅ 编码器加载成功（CPU模式）")
        
        print("\n2. 加载Reranker模型（CPU模式）...") # 新增：加载Reranker到此函数
        reranker_model = Reranker(
            model_name=config.reranker.model_name, 
            cache_dir=config.reranker.cache_dir, 
            device="cpu"
        )
        print("   ✅ Reranker加载成功（CPU模式）")
        
        print("\n3. 加载训练数据（知识库）...")
        data_loader = OptimizedDataLoader(
            data_dir="data",
            max_samples=-1,  # 加载所有数据
            chinese_document_level=True,
            english_chunk_level=True,
            include_eval_data=include_eval_data  # 直接控制是否包含评估数据
        )
        
        chinese_chunks = data_loader.chinese_docs
        english_chunks = data_loader.english_docs
        
        print(f"   ✅ 训练数据加载成功:")
        print(f"      中文chunks: {len(chinese_chunks)}")
        print(f"      英文chunks: {len(english_chunks)}")
        
        print("\n4. 加载评估数据...")
        alphafin_eval = load_eval_data("evaluate_mrr/alphafin_eval.jsonl")
        tatqa_eval = load_eval_data("evaluate_mrr/tatqa_eval_enhanced.jsonl")
        
        print(f"   ✅ 评估数据加载成功:")
        print(f"      AlphaFin评估样本: {len(alphafin_eval)}")
        print(f"      TatQA增强版评估样本: {len(tatqa_eval)}")
        
        print("\n5. 创建检索器（CPU模式）...")
        retriever = BilingualRetriever(
            encoder_en=encoder_en,
            encoder_ch=encoder_ch,
            corpus_documents_en=english_chunks,
            corpus_documents_ch=chinese_chunks,
            use_faiss=True,
            use_gpu=False,  # 强制不使用GPU
            batch_size=4,   # 减小batch_size以适应CPU
            cache_dir=config.encoder.cache_dir
        )
        print("   ✅ 检索器创建成功（CPU模式）")
        
        # 评估中文数据
        print(f"\n--- 评估中文数据 (AlphaFin) ---")
        chinese_results = evaluate_dataset(
            eval_data=alphafin_eval,
            retriever=retriever,
            encoder=encoder_ch,
            language='zh',
            dataset_name="AlphaFin",
            reranker=reranker_model # 传递Reranker实例
        )
        
        # 评估英文数据
        print(f"\n--- 评估英文数据 (TatQA增强版) ---")
        english_results = evaluate_dataset(
            eval_data=tatqa_eval,
            retriever=retriever,
            encoder=encoder_en,
            language='en',
            dataset_name="TatQA",
            reranker=reranker_model # 传递Reranker实例
        )
        
        # 汇总结果
        print("\n" + "=" * 60)
        print("评估结果汇总")
        print("=" * 60)
        
        print(f"中文数据 (AlphaFin):")
        print(f"  MRR: {chinese_results['mrr']:.4f}")
        print(f"  Hit@1: {chinese_results['hit_at_1']:.4f}")
        print(f"  Hit@3: {chinese_results['hit_at_3']:.4f}")
        print(f"  Hit@5: {chinese_results['hit_at_5']:.4f}")
        print(f"  Hit@10: {chinese_results['hit_at_10']:.4f}")
        print(f"  总样本数: {chinese_results['total_samples']}")
        print(f"  找到正确答案的样本数: {chinese_results['found_samples']}")
        
        print(f"\n英文数据 (TatQA增强版):")
        print(f"  MRR: {english_results['mrr']:.4f}")
        print(f"  Hit@1: {english_results['hit_at_1']:.4f}")
        print(f"  Hit@3: {english_results['hit_at_3']:.4f}")
        print(f"  Hit@5: {english_results['hit_at_5']:.4f}")
        print(f"  Hit@10: {english_results['hit_at_10']:.4f}")
        print(f"  总样本数: {english_results['total_samples']}")
        print(f"  找到正确答案的样本数: {english_results['found_samples']}")
        
        # 计算总体指标
        total_samples = chinese_results['total_samples'] + english_results['total_samples']
        total_found = chinese_results['found_samples'] + english_results['found_samples']
        all_mrr_scores = chinese_results['mrr_scores_raw'] + english_results['mrr_scores_raw'] # 假设evaluate_dataset返回mrr_scores_raw
        overall_mrr = sum(all_mrr_scores) / total_samples if total_samples > 0 else 0.0

        # 需要重新计算总体的Hit@K，因为不能简单求平均
        overall_hit1 = (chinese_results['hit_at_1'] * chinese_results['total_samples'] + english_results['hit_at_1'] * english_results['total_samples']) / total_samples if total_samples > 0 else 0.0
        overall_hit_at_3 = (chinese_results['hit_at_3'] * chinese_results['total_samples'] + english_results['hit_at_3'] * english_results['total_samples']) / total_samples if total_samples > 0 else 0.0
        overall_hit_at_5 = (chinese_results['hit_at_5'] * chinese_results['total_samples'] + english_results['hit_at_5'] * english_results['total_samples']) / total_samples if total_samples > 0 else 0.0
        overall_hit_at_10 = (chinese_results['hit_at_10'] * chinese_results['total_samples'] + english_results['hit_at_10'] * english_results['total_samples']) / total_samples if total_samples > 0 else 0.0

        print(f"\n总体检索:")
        print(f"  总样本数: {total_samples}")
        print(f"  总体MRR: {overall_mrr:.4f}")
        print(f"  总体Hit@1: {overall_hit1:.4f}")
        print(f"  总体Hit@3: {overall_hit_at_3:.4f}")
        print(f"  总体Hit@5: {overall_hit_at_5:.4f}")
        print(f"  总体Hit@10: {overall_hit_at_10:.4f}")
        print(f"  总找到数: {total_found}")
        print(f"  总体召回率: {total_found/total_samples:.4f}")
        
        return {
            'chinese': chinese_results,
            'english': english_results,
            'overall': {
                'mrr': overall_mrr,
                'hit_at_1': overall_hit1,
                'hit_at_3': overall_hit_at_3,
                'hit_at_5': overall_hit_at_5,
                'hit_at_10': overall_hit_at_10,
                'total_samples': total_samples,
                'found_samples': total_found
            }
        }
        
    except Exception as e:
        print(f"❌ 评估失败: {e}")
        import traceback
        traceback.print_exc()
        return None

def compare_retrieval_modes():
    """比较不同检索模式的效果"""
    print("=" * 60)
    print("比较不同检索模式的效果")
    print("=" * 60)
    
    # 测试包含评估数据的模式
    print("\n1. 测试包含评估数据的模式...")
    results_with_eval = test_retrieval_with_eval_context(include_eval_data=True)
    
    # 测试不包含评估数据的模式
    print("\n2. 测试不包含评估数据的模式...")
    results_without_eval = test_retrieval_with_eval_context(include_eval_data=False)
    
    # 比较结果
    print("\n" + "=" * 60)
    print("模式比较结果")
    print("=" * 60)
    
    if results_with_eval and results_without_eval:
        print("包含评估数据 vs 不包含评估数据:")
        print(f"中文MRR: {results_with_eval['chinese']['mrr']:.4f} vs {results_without_eval['chinese']['mrr']:.4f}")
        print(f"英文MRR: {results_with_eval['english']['mrr']:.4f} vs {results_without_eval['english']['mrr']:.4f}")
        print(f"总体MRR: {results_with_eval['overall']['mrr']:.4f} vs {results_without_eval['overall']['mrr']:.4f}")
    else:
        print("❌ 比较失败")

def test_retrieval_quality():
    """测试检索质量"""
    print("=" * 60)
    print("测试检索质量")
    print("=" * 60)
    
    # 默认测试不包含评估数据的模式
    results = test_retrieval_with_eval_context(include_eval_data=False)
    
    if results:
        print("\n🎉 测试完成！")
        print(f"总体MRR: {results['overall']['mrr']:.4f}")
        print(f"总体Hit@1: {results['overall']['hit_at_1']:.4f}")
    else:
        print("❌ 测试失败")

def evaluate_retrieval_quality(include_eval_data=True, max_eval_samples=None):
    """
    完整评估检索质量 (CPU版本)
    
    Args:
        include_eval_data: 是否包含评估数据到知识库
        max_eval_samples: 最大评估样本数
    """
    print("=" * 60)
    print(f"完整评估检索质量 (CPU版本)")
    print(f"包含评估数据: {include_eval_data}")
    print(f"最大样本数: {max_eval_samples if max_eval_samples else '所有'}")
    print("=" * 60)
    
    try:
        config = Config()
        
        print("1. 加载编码器（CPU模式）...")
        encoder_ch = FinbertEncoder(
            model_name="./models/finetuned_alphafin_zh_optimized",
            cache_dir=config.encoder.cache_dir,
            device="cpu"
        )
        encoder_en = FinbertEncoder(
            model_name="models/finetuned_finbert_tatqa",
            cache_dir=config.encoder.cache_dir,
            device="cpu"
        )
        print("   ✅ 编码器加载成功")
        
        print("\n2. 加载Reranker模型（CPU模式）...") 
        reranker_model = Reranker(
            model_name=config.reranker.model_name, 
            cache_dir=config.reranker.cache_dir, 
            device="cpu"
        )
        print("   ✅ Reranker加载成功")
        
        print("\n3. 加载数据...")
        data_loader = OptimizedDataLoader(
            data_dir="data",
            max_samples=-1,
            chinese_document_level=True,
            english_chunk_level=True,
            include_eval_data=include_eval_data
        )
        
        chinese_chunks = data_loader.chinese_docs
        english_chunks = data_loader.english_docs
        
        print(f"   ✅ 数据加载成功:")
        print(f"      中文chunks: {len(chinese_chunks)}")
        print(f"      英文chunks: {len(english_chunks)}")
        
        print("\n4. 加载评估数据...")
        alphafin_eval = load_eval_data("evaluate_mrr/alphafin_eval.jsonl")
        tatqa_eval = load_eval_data("evaluate_mrr/tatqa_eval_enhanced.jsonl") 
        
        if max_eval_samples:
            alphafin_eval = alphafin_eval[:max_eval_samples]
            tatqa_eval = tatqa_eval[:max_eval_samples]
        
        print(f"   ✅ 评估数据加载成功:")
        print(f"      AlphaFin评估样本: {len(alphafin_eval)}")
        print(f"      TatQA增强版评估样本: {len(tatqa_eval)}")
        
        print("\n5. 创建检索器（CPU模式）...")
        retriever = BilingualRetriever(
            encoder_en=encoder_en,
            encoder_ch=encoder_ch,
            corpus_documents_en=english_chunks,
            corpus_documents_ch=chinese_chunks,
            use_faiss=True,
            use_gpu=False,
            batch_size=4,
            cache_dir=config.encoder.cache_dir
        )
        print("   ✅ 检索器创建成功")
        
        # 评估中文数据
        print(f"\n--- 评估中文数据 (AlphaFin) ---")
        chinese_results = evaluate_dataset(
            eval_data=alphafin_eval,
            retriever=retriever,
            encoder=encoder_ch,
            language='zh',
            dataset_name="AlphaFin",
            reranker=reranker_model # 传递Reranker实例
        )
        
        # 评估英文数据
        print(f"\n--- 评估英文数据 (TatQA增强版) ---")
        english_results = evaluate_dataset(
            eval_data=tatqa_eval,
            retriever=retriever,
            encoder=encoder_en,
            language='en',
            dataset_name="TatQA",
            reranker=reranker_model # 传递Reranker实例
        )
        
        # 汇总结果
        print(f"\n=== 评估结果汇总 ===")
        print(f"中文数据 (AlphaFin):")
        print(f"  MRR: {chinese_results['mrr']:.4f}")
        print(f"  Hit@1: {chinese_results['hit_at_1']:.4f}")
        print(f"  Hit@3: {chinese_results['hit_at_3']:.4f}")
        print(f"  Hit@5: {chinese_results['hit_at_5']:.4f}")
        print(f"  Hit@10: {chinese_results['hit_at_10']:.4f}")
        print(f"  总样本数: {chinese_results['total_samples']}")
        print(f"  找到正确答案的样本数: {chinese_results['found_samples']}")
        
        print(f"\n英文数据 (TatQA增强版):")
        print(f"  MRR: {english_results['mrr']:.4f}")
        print(f"  Hit@1: {english_results['hit_at_1']:.4f}")
        print(f"  Hit@3: {english_results['hit_at_3']:.4f}")
        print(f"  Hit@5: {english_results['hit_at_5']:.4f}")
        print(f"  Hit@10: {english_results['hit_at_10']:.4f}")
        print(f"  总样本数: {english_results['total_samples']}")
        print(f"  找到正确答案的样本数: {english_results['found_samples']}")
        
        # 计算总体指标
        # 这里的总体指标计算方式已调整为对各个样本的MRR和Hit@K进行汇总，而非简单平均
        all_mrr_scores = chinese_results['mrr_scores_raw'] + english_results['mrr_scores_raw'] 
        all_hit_at_1_raw = chinese_results['hit_at_1_raw'] + english_results['hit_at_1_raw']
        all_hit_at_3_raw = chinese_results['hit_at_3_raw'] + english_results['hit_at_3_raw']
        all_hit_at_5_raw = chinese_results['hit_at_5_raw'] + english_results['hit_at_5_raw']
        all_hit_at_10_raw = chinese_results['hit_at_10_raw'] + english_results['hit_at_10_raw']

        total_samples_overall = len(all_mrr_scores) # 这里的total_samples_overall就是所有样本的总数
        
        overall_mrr = sum(all_mrr_scores) / total_samples_overall if total_samples_overall > 0 else 0.0
        overall_hit_at_1 = sum(all_hit_at_1_raw) / total_samples_overall if total_samples_overall > 0 else 0.0
        overall_hit_at_3 = sum(all_hit_at_3_raw) / total_samples_overall if total_samples_overall > 0 else 0.0
        overall_hit_at_5 = sum(all_hit_at_5_raw) / total_samples_overall if total_samples_overall > 0 else 0.0
        overall_hit_at_10 = sum(all_hit_at_10_raw) / total_samples_overall if total_samples_overall > 0 else 0.0
        
        total_found_overall = chinese_results['found_samples'] + english_results['found_samples']

        print(f"\n总体检索:")
        print(f"  总样本数: {total_samples_overall}")
        print(f"  总体MRR: {overall_mrr:.4f}")
        print(f"  总体Hit@1: {overall_hit_at_1:.4f}")
        print(f"  总体Hit@3: {overall_hit_at_3:.4f}")
        print(f"  总体Hit@5: {overall_hit_at_5:.4f}")
        print(f"  总体Hit@10: {overall_hit_at_10:.4f}")
        print(f"  总找到数: {total_found_overall}")
        print(f"  总体召回率: {total_found_overall/total_samples_overall:.4f}")
        
        return {
            'chinese': chinese_results,
            'english': english_results,
            'overall': {
                'mrr': overall_mrr,
                'hit_at_1': overall_hit_at_1,
                'hit_at_3': overall_hit_at_3,
                'hit_at_5': overall_hit_at_5,
                'hit_at_10': overall_hit_at_10,
                'total_samples': total_samples_overall,
                'found_samples': total_found_overall
            }
        }
        
    except Exception as e:
        print(f"❌ 评估失败: {e}")
        import traceback
        traceback.print_exc()
        return None

def compare_retrieval_modes():
    """比较不同检索模式的效果"""
    print("=" * 60)
    print("比较不同检索模式的效果")
    print("=" * 60)
    
    # 测试包含评估数据的模式
    print("\n1. 测试包含评估数据的模式...")
    results_with_eval = evaluate_retrieval_quality(include_eval_data=True) # 调用统一的评估函数
    
    # 测试不包含评估数据的模式
    print("\n2. 测试不包含评估数据的模式...")
    results_without_eval = evaluate_retrieval_quality(include_eval_data=False) # 调用统一的评估函数
    
    # 比较结果
    print("\n" + "=" * 60)
    print("模式比较结果")
    print("=" * 60)
    
    if results_with_eval and results_without_eval:
        print("包含评估数据 vs 不包含评估数据:")
        print(f"中文MRR: {results_with_eval['chinese']['mrr']:.4f} vs {results_without_eval['chinese']['mrr']:.4f}")
        print(f"英文MRR: {results_with_eval['english']['mrr']:.4f} vs {results_without_eval['english']['mrr']:.4f}")
        print(f"总体MRR: {results_with_eval['overall']['mrr']:.4f} vs {results_without_eval['overall']['mrr']:.4f}")
    else:
        print("❌ 比较失败")

def test_retrieval_quality():
    """测试检索质量 (旧版函数，推荐使用 evaluate_retrieval_quality)"""
    print("=" * 60)
    print("测试检索质量")
    print("=" * 60)
    
    # 默认测试不包含评估数据的模式
    results = evaluate_retrieval_quality(include_eval_data=False)
    
    if results:
        print("\n🎉 测试完成！")
        print(f"总体MRR: {results['overall']['mrr']:.4f}")
        print(f"总体Hit@1: {results['overall']['hit_at_1']:.4f}")
    else:
        print("❌ 测试失败")

# --- 主执行逻辑 ---
if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="评估检索质量（CPU版本）")
    parser.add_argument("--include_eval_data", action="store_true", 
                        help="是否将评估数据包含在知识库中")
    parser.add_argument("--max_samples", type=int, default=None,
                        help="最大评估样本数，None表示评估所有样本")
    parser.add_argument("--test_mode", action="store_true",
                        help="测试模式，只评估少量样本（已废弃，请使用 --max_samples）")
    parser.add_argument("--compare_modes", action="store_true",
                        help="比较不同检索模式（知识库是否包含评估数据）")
    parser.add_argument("--full_eval", action="store_true",
                        help="执行完整评估模式（默认），计算所有指标")
    
    args = parser.parse_args()
    
    if args.compare_modes:
        compare_retrieval_modes()
    elif args.full_eval: # 新增的完整评估模式
        evaluate_retrieval_quality(
            include_eval_data=args.include_eval_data,
            max_eval_samples=args.max_samples
        )
    else: # 兼容旧的 test_mode 或其他默认行为，但推荐使用 --full_eval
        # 如果既没有 --compare_modes 也没有 --full_eval，则执行默认的 test_retrieval_quality
        # （可能只评估少量样本或旧逻辑）
        print("Warning: Running default or old test mode. Consider using --full_eval for comprehensive assessment.")
        test_retrieval_quality() # 默认或旧的简化测试