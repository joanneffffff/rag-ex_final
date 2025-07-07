import json
import argparse
from pathlib import Path
from typing import List, Dict, Any
from alphafin_data_process.multi_stage_retrieval_final import MultiStageRetrievalSystem

# 检索适配器函数
def get_ranked_documents_for_evaluation(
    query: str,
    retrieval_system: MultiStageRetrievalSystem,
    metadata: dict = {},   # 必须为dict
    top_k: int = 10,
    mode: str = "baseline",  # "baseline", "prefilter", "reranker"
    use_prefilter: bool = True  # 添加预过滤开关参数
) -> List[Dict[str, Any]]:
    """
    统一检索接口，支持三种模式。
    返回格式：每条为dict，含index, faiss_score, combined_score, doc_id, summary等字段。
    """
    if retrieval_system is None:
        raise ValueError("必须提供retrieval_system实例")
    
    if mode == "baseline":
        # baseline: 根据预过滤开关决定是否使用元数据过滤
        if use_prefilter:
            # 使用元数据过滤的baseline
            company_name = metadata.get('company_name') if metadata else None
            stock_code = metadata.get('stock_code') if metadata else None
            report_date = metadata.get('report_date') if metadata else None
            candidate_indices = retrieval_system.pre_filter(company_name, stock_code, report_date)
        else:
            # 不使用元数据过滤的baseline（真正的baseline）
            candidate_indices = list(range(len(retrieval_system.data)))
        
        results = retrieval_system.faiss_search(query, candidate_indices, top_k)
        # 只返回faiss分数
        return [
            {
                'index': idx,
                'faiss_score': score,
                'doc_id': retrieval_system.data[idx].get('doc_id', str(idx)),
                'summary': retrieval_system.data[idx].get('summary', '')
            }
            for idx, score in results
        ]
    elif mode == "prefilter":
        # 先元数据过滤，再faiss_search
        company_name = metadata.get('company_name') if metadata else None
        stock_code = metadata.get('stock_code') if metadata else None
        report_date = metadata.get('report_date') if metadata else None
        candidate_indices = retrieval_system.pre_filter(company_name, stock_code, report_date)
        results = retrieval_system.faiss_search(query, candidate_indices, top_k)
        return [
            {
                'index': idx,
                'faiss_score': score,
                'doc_id': retrieval_system.data[idx].get('doc_id', str(idx)),
                'summary': retrieval_system.data[idx].get('summary', '')
            }
            for idx, score in results
        ]
    elif mode == "reranker":
        # 先元数据过滤，再faiss_search，再rerank
        company_name = metadata.get('company_name') if metadata else None
        stock_code = metadata.get('stock_code') if metadata else None
        report_date = metadata.get('report_date') if metadata else None
        candidate_indices = retrieval_system.pre_filter(company_name, stock_code, report_date)
        faiss_results = retrieval_system.faiss_search(query, candidate_indices, top_k=50)  # 先召回较多
        rerank_results = retrieval_system.rerank(query, faiss_results, top_k)
        return [
            {
                'index': idx,
                'faiss_score': faiss_score,
                'combined_score': combined_score,
                'doc_id': retrieval_system.data[idx].get('doc_id', str(idx)),
                'summary': retrieval_system.data[idx].get('summary', '')
            }
            for idx, faiss_score, combined_score in rerank_results
        ]
    else:
        raise ValueError(f"未知检索模式: {mode}")

# 评测指标计算函数
def evaluate_mrr_and_hitk(
    dataset: List[Dict],
    retrieval_system: MultiStageRetrievalSystem,
    mode: str = "baseline",
    top_k: int = 10,
    use_prefilter: bool = True  # 添加预过滤开关参数
):
    """
    用于批量评测MRR和Hit@k，所有输出均为中文。
    """
    mrr_total = 0.0
    hitk_total = 0
    total = 0
    for sample in dataset:
        query = sample.get('generated_question')
        if query is None:
            query = ""
        gold_doc_id = sample.get('doc_id')
        # 可选元数据
        metadata = {
            'company_name': sample.get('company_name'),
            'stock_code': sample.get('stock_code'),
            'report_date': sample.get('report_date')
        }
        results = get_ranked_documents_for_evaluation(
            query=str(query),
            retrieval_system=retrieval_system,
            metadata=metadata,
            top_k=top_k,
            mode=mode,
            use_prefilter=use_prefilter  # 传递预过滤开关
        )
        # 计算MRR和Hit@k
        rank = None
        for i, doc in enumerate(results):
            if doc['doc_id'] == gold_doc_id:
                rank = i + 1
                break
        if rank:
            mrr_total += 1.0 / rank
            hitk_total += 1
        total += 1
    mrr = mrr_total / total if total > 0 else 0.0
    hitk = hitk_total / total if total > 0 else 0.0
    print(f"\n评测模式: {mode}")
    print(f"预过滤开关: {'开启' if use_prefilter else '关闭'}")
    print(f"总样本数: {total}")
    print(f"MRR: {mrr:.4f}")
    print(f"Hit@{top_k}: {hitk:.4f}")
    return mrr, hitk

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="AlphaFin检索系统评测脚本（所有输出均为中文）")
    parser.add_argument('--data_path', type=str, default='data/alphafin/alphafin_eval_samples.json', help='评测数据集路径（JSON），包含generated_question和doc_id字段')
    parser.add_argument('--retrieval_data_path', type=str, default='data/alphafin/alphafin_final_clean.json', help='检索系统数据文件路径（AlphaFin完整数据集）')
    parser.add_argument('--mode', type=str, default='baseline', choices=['baseline', 'prefilter', 'reranker'], help='检索模式：baseline(仅FAISS), prefilter(元数据过滤+FAISS), reranker(元数据过滤+FAISS+重排序)')
    parser.add_argument('--top_k', type=int, default=10, help='检索返回top_k')
    parser.add_argument('--use_prefilter', action='store_true', default=True, help='是否使用预过滤（默认True）')
    parser.add_argument('--no_prefilter', dest='use_prefilter', action='store_false', help='关闭预过滤')
    args = parser.parse_args()

    print("加载评测数据集...")
    with open(args.data_path, 'r', encoding='utf-8') as f:
        dataset = json.load(f)
    print(f"评测样本数: {len(dataset)}")

    print("初始化检索系统...")
    retrieval_system = MultiStageRetrievalSystem(Path(args.retrieval_data_path))

    print("开始评测...")
    evaluate_mrr_and_hitk(dataset, retrieval_system, mode=args.mode, top_k=args.top_k, use_prefilter=args.use_prefilter) 