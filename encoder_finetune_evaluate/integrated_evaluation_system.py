#!/usr/bin/env python3
"""
集成评估系统 - 支持多种检索和重排序方法的对比实验
整合Encoder、FAISS、Reranker、元数据过滤等所有模块
支持多GPU并行处理
使用与真实RAG系统相同的组件和逻辑
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
import time
import multiprocessing as mp
from multiprocessing import Process, Queue
from typing import List, Dict, Tuple, Optional
from dataclasses import dataclass
from enum import Enum

# 添加项目根目录到Python路径
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# 导入真实RAG系统的组件
from xlm.components.encoder.finbert import FinbertEncoder
from xlm.components.retriever.bilingual_retriever import BilingualRetriever
from xlm.components.retriever.reranker import QwenReranker
from xlm.dto.dto import DocumentWithMetadata, DocumentMetadata

class RetrievalMethod(Enum):
    """检索方法枚举"""
    ENCODER_ONLY = "encoder_only"
    ENCODER_FAISS = "encoder_faiss"
    ENCODER_RERANKER = "encoder_reranker"
    ENCODER_FAISS_RERANKER = "encoder_faiss_reranker"

@dataclass
class EvaluationConfig:
    """评估配置"""
    eval_data_path: str
    corpus_data_path: str
    encoder_model: str
    reranker_model: str
    retrieval_method: RetrievalMethod
    top_k_retrieval: int = 100
    top_k_rerank: int = 10
    max_samples: int = 100
    use_metadata_filter: bool = False
    use_faiss: bool = False
    device: str = "cuda"
    batch_size: int = 4
    max_length: int = 512
    num_gpus: int = 2  # 新增：GPU数量

@dataclass
class EvaluationResult:
    """评估结果"""
    method: str
    mrr_retrieval: float
    mrr_rerank: float
    avg_retrieval_time: float
    avg_rerank_time: float
    total_queries: int
    valid_queries: int
    skipped_queries: int

class MetadataExtractor:
    """元数据提取器"""
    
    @staticmethod
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

class MetadataFilter:
    """元数据过滤器"""
    
    def __init__(self, dataset_type: str = "alphafin"):
        self.dataset_type = dataset_type
        self.extractor = MetadataExtractor()
    
    def filter_corpus(self, corpus_documents: List[DocumentWithMetadata], target_metadata: dict) -> List[DocumentWithMetadata]:
        """根据元数据过滤检索库"""
        if not target_metadata or not any(target_metadata.values()):
            return corpus_documents
        
        filtered_corpus = []
        filter_criteria = []
        
        # 构建过滤条件
        for field, value in target_metadata.items():
            if value:
                filter_criteria.append((field, value))
        
        if not filter_criteria:
            return corpus_documents
        
        print(f"🔍 应用元数据过滤: {filter_criteria}")
        
        for doc in corpus_documents:
            content_metadata = self.extractor.extract_alphafin_metadata(doc.content)
            
            # 检查是否满足所有过滤条件
            matches_all = True
            for field, value in filter_criteria:
                if content_metadata.get(field) != value:
                    matches_all = False
                    break
            
            if matches_all:
                filtered_corpus.append(doc)
        
        print(f"📊 过滤结果: {len(filtered_corpus)}/{len(corpus_documents)} 个文档")
        return filtered_corpus

class RAGSystemManager:
    """RAG系统管理器 - 使用真实RAG系统组件"""
    
    def __init__(self, config: EvaluationConfig, gpu_id: int = 0):
        self.config = config
        self.gpu_id = gpu_id
        self.device = torch.device(f"cuda:{gpu_id}" if torch.cuda.is_available() else "cpu")
        
        # RAG系统组件
        self.encoder_en = None
        self.encoder_ch = None
        self.retriever = None
        self.reranker = None
        
    def load_encoders(self):
        """加载编码器"""
        try:
            print(f"📖 GPU {self.gpu_id}: 加载编码器模型: {self.config.encoder_model}")
            
            # 加载英文编码器
            self.encoder_en = FinbertEncoder(
                model_name=self.config.encoder_model,
                cache_dir="cache"
            )
            
            # 加载中文编码器（使用相同模型或指定中文模型）
            self.encoder_ch = FinbertEncoder(
                model_name=self.config.encoder_model,  # 可以根据需要指定不同的中文模型
                cache_dir="cache"
            )
            
            print(f"✅ GPU {self.gpu_id}: 编码器加载完成")
            return True
        except Exception as e:
            print(f"❌ GPU {self.gpu_id}: 加载编码器失败: {e}")
            return False
    
    def load_reranker(self):
        """加载重排序器"""
        try:
            print(f"📖 GPU {self.gpu_id}: 加载重排序模型: {self.config.reranker_model}")
            
            self.reranker = QwenReranker(
                model_name=self.config.reranker_model,
                device=f"cuda:{self.gpu_id}" if torch.cuda.is_available() else "cpu",
                cache_dir="cache",
                use_quantization=True,
                quantization_type="4bit"
            )
            
            print(f"✅ GPU {self.gpu_id}: 重排序模型加载完成")
            return True
        except Exception as e:
            print(f"❌ GPU {self.gpu_id}: 加载重排序模型失败: {e}")
            return False
    
    def create_retriever(self, corpus_documents_en: List[DocumentWithMetadata], 
                        corpus_documents_ch: Optional[List[DocumentWithMetadata]] = None):
        """创建检索器"""
        try:
            print(f"🔧 GPU {self.gpu_id}: 创建BilingualRetriever...")
            
            if corpus_documents_ch is None:
                corpus_documents_ch = []
            
            if self.encoder_en is None or self.encoder_ch is None:
                print(f"❌ GPU {self.gpu_id}: 编码器未加载")
                return False
            
            self.retriever = BilingualRetriever(
                encoder_en=self.encoder_en,
                encoder_ch=self.encoder_ch,
                corpus_documents_en=corpus_documents_en,
                corpus_documents_ch=corpus_documents_ch,
                use_faiss=self.config.use_faiss,
                use_gpu=True,
                batch_size=self.config.batch_size,
                cache_dir="cache",
                use_existing_embedding_index=False
            )
            
            print(f"✅ GPU {self.gpu_id}: BilingualRetriever创建完成")
            return True
        except Exception as e:
            print(f"❌ GPU {self.gpu_id}: 创建检索器失败: {e}")
            return False
    
    def retrieve_documents(self, query: str, top_k: int, language: str = "english") -> Tuple[List[DocumentWithMetadata], List[float]]:
        """检索文档 - 使用真实RAG系统逻辑"""
        if not self.retriever:
            return [], []
        
        try:
            # 使用BilingualRetriever的retrieve方法
            result = self.retriever.retrieve(
                text=query,
                top_k=top_k,
                return_scores=True,
                language=language
            )
            
            if isinstance(result, tuple):
                documents, scores = result
            else:
                documents = result
                scores = []
            
            return documents, scores
        except Exception as e:
            print(f"❌ GPU {self.gpu_id}: 检索失败: {e}")
            return [], []
    
    def rerank_documents(self, query: str, documents: List[DocumentWithMetadata], top_k: int) -> List[DocumentWithMetadata]:
        """重排序文档 - 使用QwenReranker"""
        if not self.reranker or not documents:
            return documents[:top_k]
        
        try:
            # 提取文档内容
            doc_texts = [doc.content for doc in documents]
            
            # 使用QwenReranker重排序
            reranked_items = self.reranker.rerank(
                query=query,
                documents=doc_texts,
                batch_size=self.config.batch_size
            )
            
            # 根据重排序结果重新排列文档
            content_to_doc_map = {doc.content: doc for doc in documents}
            reranked_docs = []
            
            for doc_text, score in reranked_items[:top_k]:
                if doc_text in content_to_doc_map:
                    reranked_docs.append(content_to_doc_map[doc_text])
            
            return reranked_docs
        except Exception as e:
            print(f"❌ GPU {self.gpu_id}: 重排序失败: {e}")
            return documents[:top_k]

class IntegratedEvaluator:
    """集成评估器 - 使用真实RAG系统组件"""
    
    def __init__(self, config: EvaluationConfig):
        self.config = config
    
    def load_data(self) -> Tuple[List[Dict], List[DocumentWithMetadata], List[DocumentWithMetadata]]:
        """加载数据"""
        print("📖 加载评估数据...")
        
        # 加载评估数据
        eval_data = []
        with open(self.config.eval_data_path, 'r', encoding='utf-8') as f:
            for line in f:
                if line.strip():
                    eval_data.append(json.loads(line))
        
        if self.config.max_samples:
            eval_data = eval_data[:self.config.max_samples]
        
        print(f"✅ 加载了 {len(eval_data)} 个评估样本")
        
        # 加载检索库数据
        print("📖 加载检索库数据...")
        corpus_documents_en = []
        corpus_documents_ch = []
        
        with open(self.config.corpus_data_path, 'r', encoding='utf-8') as f:
            for line in f:
                if line.strip():
                    item = json.loads(line)
                    doc_id = item.get('doc_id', str(len(corpus_documents_en)))
                    content = item.get('context', item.get('content', ''))
                    
                    if content:
                        # 创建DocumentWithMetadata对象
                        metadata = DocumentMetadata(
                            source=item.get('source', 'unknown'),
                            created_at=item.get('created_at', ''),
                            author=item.get('author', ''),
                            language=item.get('language', 'english')
                        )
                        
                        doc = DocumentWithMetadata(
                            content=content,
                            metadata=metadata
                        )
                        
                        # 根据语言分配到不同语料库
                        if item.get('language', 'english') == 'chinese':
                            corpus_documents_ch.append(doc)
                        else:
                            corpus_documents_en.append(doc)
        
        print(f"✅ 加载了 {len(corpus_documents_en)} 个英文文档, {len(corpus_documents_ch)} 个中文文档")
        
        return eval_data, corpus_documents_en, corpus_documents_ch
    
    def evaluate_single_query(self, query_item: Dict, rag_manager: RAGSystemManager) -> Tuple[int, int, float, float]:
        """评估单个查询"""
        query = query_item.get('query', query_item.get('question', ''))
        correct_context = query_item.get('context', '')
        
        if not query or not correct_context:
            return 0, 0, 0.0, 0.0
        
        # 检测语言
        try:
            from langdetect import detect
            language = detect(query)
            language = 'zh' if language.startswith('zh') else 'en'
        except:
            language = 'en'  # 默认英文
        
        start_time = time.time()
        
        # 1. 初始检索
        retrieved_docs, retrieval_scores = rag_manager.retrieve_documents(
            query=query, 
            top_k=self.config.top_k_retrieval,
            language=language
        )
        
        retrieval_time = time.time() - start_time
        
        # 找到正确答案的排名
        retrieval_rank = 0
        for rank, doc in enumerate(retrieved_docs, 1):
            if doc.content == correct_context:
                retrieval_rank = rank
                break
        
        # 2. 重排序（如果需要）
        rerank_rank = 0
        rerank_time = 0.0
        
        if self.config.retrieval_method in [RetrievalMethod.ENCODER_RERANKER, RetrievalMethod.ENCODER_FAISS_RERANKER]:
            if retrieved_docs:
                start_time = time.time()
                
                # 重排序
                reranked_docs = rag_manager.rerank_documents(
                    query=query,
                    documents=retrieved_docs,
                    top_k=self.config.top_k_rerank
                )
                
                # 找到正确答案的排名
                for rank, doc in enumerate(reranked_docs, 1):
                    if doc.content == correct_context:
                        rerank_rank = rank
                        break
                
                rerank_time = time.time() - start_time
        
        return retrieval_rank, rerank_rank, retrieval_time, rerank_time
    
    def evaluate(self) -> Optional[EvaluationResult]:
        """执行评估"""
        # 加载数据
        eval_data, corpus_documents_en, corpus_documents_ch = self.load_data()
        
        # 创建RAG系统管理器
        rag_manager = RAGSystemManager(self.config)
        
        # 加载编码器
        if not rag_manager.load_encoders():
            return None
        
        # 加载重排序器（如果需要）
        if self.config.retrieval_method in [RetrievalMethod.ENCODER_RERANKER, RetrievalMethod.ENCODER_FAISS_RERANKER]:
            if not rag_manager.load_reranker():
                return None
        
        # 创建检索器
        if not rag_manager.create_retriever(corpus_documents_en, corpus_documents_ch):
            return None
        
        # 元数据过滤
        if self.config.use_metadata_filter:
            metadata_filter = MetadataFilter()
            # 这里简化处理，实际应用中需要根据查询提取目标元数据
            target_metadata = {}  # 可以根据需要设置
            corpus_documents_en = metadata_filter.filter_corpus(corpus_documents_en, target_metadata)
            corpus_documents_ch = metadata_filter.filter_corpus(corpus_documents_ch, target_metadata)
        
        # 评估
        all_retrieval_ranks = []
        all_rerank_ranks = []
        all_retrieval_times = []
        all_rerank_times = []
        skipped_queries = 0
        
        for query_item in tqdm(eval_data, desc="评估查询"):
            try:
                retrieval_rank, rerank_rank, retrieval_time, rerank_time = self.evaluate_single_query(
                    query_item, rag_manager
                )
                
                all_retrieval_ranks.append(retrieval_rank)
                all_rerank_ranks.append(rerank_rank)
                all_retrieval_times.append(retrieval_time)
                all_rerank_times.append(rerank_time)
                
            except Exception as e:
                print(f"❌ 查询评估失败: {e}")
                skipped_queries += 1
                all_retrieval_ranks.append(0)
                all_rerank_ranks.append(0)
                all_retrieval_times.append(0.0)
                all_rerank_times.append(0.0)
        
        # 计算指标
        mrr_retrieval = self.calculate_mrr(all_retrieval_ranks)
        mrr_rerank = self.calculate_mrr(all_rerank_ranks)
        avg_retrieval_time = float(np.mean(all_retrieval_times)) if all_retrieval_times else 0.0
        avg_rerank_time = float(np.mean(all_rerank_times)) if all_rerank_times else 0.0
        
        return EvaluationResult(
            method=self.config.retrieval_method.value,
            mrr_retrieval=mrr_retrieval,
            mrr_rerank=mrr_rerank,
            avg_retrieval_time=avg_retrieval_time,
            avg_rerank_time=avg_rerank_time,
            total_queries=len(eval_data),
            valid_queries=len(eval_data) - skipped_queries,
            skipped_queries=skipped_queries
        )
    
    def calculate_mrr(self, rankings: List[int]) -> float:
        """计算MRR分数"""
        if not rankings:
            return 0.0
        
        reciprocal_ranks = []
        for rank in rankings:
            if rank > 0:
                reciprocal_ranks.append(1.0 / rank)
            else:
                reciprocal_ranks.append(0.0)
        
        return float(np.mean(reciprocal_ranks))

def gpu_worker_evaluation(gpu_id: int, data_queue: Queue, result_queue: Queue, config: EvaluationConfig):
    """GPU工作进程函数"""
    try:
        # 设置CUDA设备
        torch.cuda.set_device(gpu_id)
        print(f"🚀 GPU {gpu_id}: 开始工作，设备: cuda:{gpu_id}")
        
        # 创建GPU特定的配置
        gpu_config = EvaluationConfig(
            eval_data_path=config.eval_data_path,
            corpus_data_path=config.corpus_data_path,
            encoder_model=config.encoder_model,
            reranker_model=config.reranker_model,
            retrieval_method=config.retrieval_method,
            top_k_retrieval=config.top_k_retrieval,
            top_k_rerank=config.top_k_rerank,
            max_samples=config.max_samples,
            use_metadata_filter=config.use_metadata_filter,
            use_faiss=config.use_faiss,
            device=f"cuda:{gpu_id}",
            batch_size=config.batch_size,
            max_length=config.max_length,
            num_gpus=1  # 单个GPU
        )
        
        # 创建评估器
        evaluator = IntegratedEvaluator(gpu_config)
        
        # 处理数据
        while True:
            try:
                # 从队列获取数据
                batch_data = data_queue.get(timeout=1)
                if batch_data is None:  # 结束信号
                    break
                
                print(f"📊 GPU {gpu_id}: 处理批次，样本数: {len(batch_data)}")
                
                # 执行评估
                result = evaluator.evaluate()
                
                # 发送结果
                result_queue.put((gpu_id, result))
                print(f"✅ GPU {gpu_id}: 评估完成")
                
            except Exception as e:
                print(f"❌ GPU {gpu_id}: 处理失败: {e}")
                result_queue.put((gpu_id, None))
                break
        
    except Exception as e:
        print(f"❌ GPU {gpu_id}: 工作进程失败: {e}")
        result_queue.put((gpu_id, None))

def run_multi_gpu_evaluation(config: EvaluationConfig) -> Optional[EvaluationResult]:
    """运行多GPU评估"""
    # 检查GPU可用性
    if not torch.cuda.is_available():
        print("❌ CUDA不可用，回退到单GPU模式")
        config.num_gpus = 1
        config.device = "cpu"
    
    available_gpus = torch.cuda.device_count()
    if config.num_gpus > available_gpus:
        print(f"⚠️ 请求的GPU数量 ({config.num_gpus}) 超过可用GPU数量 ({available_gpus})")
        config.num_gpus = available_gpus
    
    print(f"🚀 使用 {config.num_gpus} 个GPU进行并行评估")
    
    # 加载数据
    print("📖 加载数据...")
    evaluator = IntegratedEvaluator(config)
    eval_data, corpus_documents_en, corpus_documents_ch = evaluator.load_data()
    
    # 数据分割
    total_samples = len(eval_data)
    samples_per_gpu = total_samples // config.num_gpus
    remainder = total_samples % config.num_gpus
    
    print(f"📊 数据分割:")
    start_idx = 0
    gpu_data = []
    for gpu_id in range(config.num_gpus):
        current_batch_size = samples_per_gpu + (1 if gpu_id < remainder else 0)
        end_idx = start_idx + current_batch_size
        gpu_samples = eval_data[start_idx:end_idx]
        gpu_data.append(gpu_samples)
        print(f"  GPU {gpu_id}: {len(gpu_samples)} 个样本")
        start_idx = end_idx
    
    # 创建进程间通信队列
    data_queues = [Queue() for _ in range(config.num_gpus)]
    result_queue = Queue()
    
    # 启动GPU工作进程
    processes = []
    for gpu_id in range(config.num_gpus):
        p = Process(target=gpu_worker_evaluation, args=(
            gpu_id, data_queues[gpu_id], result_queue, config
        ))
        p.start()
        processes.append(p)
        print(f"🚀 GPU {gpu_id} 进程已启动")
    
    # 发送数据到各个GPU
    for gpu_id in range(config.num_gpus):
        data_queues[gpu_id].put(gpu_data[gpu_id])
        data_queues[gpu_id].put(None)  # 结束信号
        print(f"📤 数据已发送到 GPU {gpu_id}")
    
    # 收集结果
    all_results = []
    for _ in tqdm(range(config.num_gpus), desc="收集GPU结果"):
        gpu_id, result = result_queue.get()
        if result:
            all_results.append(result)
            print(f"📥 收到 GPU {gpu_id} 的结果")
        else:
            print(f"❌ GPU {gpu_id} 返回空结果")
    
    # 等待所有进程完成
    for p in processes:
        p.join()
    
    # 合并结果
    if not all_results:
        print("❌ 没有收到任何有效结果")
        return None
    
    # 简单合并策略：取第一个有效结果
    # 在实际应用中，可能需要更复杂的合并策略
    final_result = all_results[0]
    print(f"✅ 多GPU评估完成，使用 GPU 0 的结果")
    
    return final_result

class ComparisonExperiment:
    """对比实验管理器"""
    
    def __init__(self, base_config: EvaluationConfig):
        self.base_config = base_config
        self.results = []
    
    def run_comparison(self, methods: List[RetrievalMethod]) -> List[EvaluationResult]:
        """运行对比实验"""
        print("🔬 开始对比实验")
        print("="*60)
        
        for method in methods:
            print(f"\n📊 测试方法: {method.value}")
            print("-" * 40)
            
            # 创建配置
            config = EvaluationConfig(
                eval_data_path=self.base_config.eval_data_path,
                corpus_data_path=self.base_config.corpus_data_path,
                encoder_model=self.base_config.encoder_model,
                reranker_model=self.base_config.reranker_model,
                retrieval_method=method,
                top_k_retrieval=self.base_config.top_k_retrieval,
                top_k_rerank=self.base_config.top_k_rerank,
                max_samples=self.base_config.max_samples,
                use_metadata_filter=self.base_config.use_metadata_filter,
                use_faiss=(method in [RetrievalMethod.ENCODER_FAISS, RetrievalMethod.ENCODER_FAISS_RERANKER]),
                device=self.base_config.device,
                batch_size=self.base_config.batch_size,
                max_length=self.base_config.max_length,
                num_gpus=self.base_config.num_gpus
            )
            
            # 执行评估
            if config.num_gpus > 1:
                result = run_multi_gpu_evaluation(config)
            else:
                evaluator = IntegratedEvaluator(config)
                result = evaluator.evaluate()
            
            if result:
                self.results.append(result)
                self.print_result(result)
            else:
                print(f"❌ 方法 {method.value} 评估失败")
        
        return self.results
    
    def print_result(self, result: EvaluationResult):
        """打印单个结果"""
        print(f"📈 结果:")
        print(f"  - 检索MRR @{self.base_config.top_k_retrieval}: {result.mrr_retrieval:.4f}")
        print(f"  - 重排序MRR @{self.base_config.top_k_rerank}: {result.mrr_rerank:.4f}")
        print(f"  - 平均检索时间: {result.avg_retrieval_time:.3f}s")
        print(f"  - 平均重排序时间: {result.avg_rerank_time:.3f}s")
        print(f"  - 有效查询数: {result.valid_queries}/{result.total_queries}")
    
    def print_comparison_summary(self):
        """打印对比总结"""
        if not self.results:
            print("❌ 没有可比较的结果")
            return
        
        print("\n" + "="*60)
        print("📊 对比实验总结")
        print("="*60)
        
        # 按检索MRR排序
        sorted_results = sorted(self.results, key=lambda x: x.mrr_retrieval, reverse=True)
        
        print(f"{'方法':<25} {'检索MRR':<10} {'重排序MRR':<12} {'检索时间':<10} {'重排序时间':<12}")
        print("-" * 80)
        
        for result in sorted_results:
            print(f"{result.method:<25} {result.mrr_retrieval:<10.4f} {result.mrr_rerank:<12.4f} "
                  f"{result.avg_retrieval_time:<10.3f} {result.avg_rerank_time:<12.3f}")
        
        # 找出最佳方法
        best_retrieval = max(self.results, key=lambda x: x.mrr_retrieval)
        best_rerank = max(self.results, key=lambda x: x.mrr_rerank)
        
        print(f"\n🏆 最佳检索方法: {best_retrieval.method} (MRR: {best_retrieval.mrr_retrieval:.4f})")
        print(f"🏆 最佳重排序方法: {best_rerank.method} (MRR: {best_rerank.mrr_rerank:.4f})")

def main():
    parser = argparse.ArgumentParser(description="集成评估系统 - 使用真实RAG系统组件")
    parser.add_argument("--eval_data", type=str, 
                       default="evaluate_mrr/alphafin_eval.jsonl",
                       help="评估数据文件")
    parser.add_argument("--corpus_data", type=str,
                       default="data/alphafin/alphafin_merged_generated_qa_full_dedup.json",
                       help="检索库数据文件")
    parser.add_argument("--encoder_model", type=str,
                       default="models/finetuned_finbert_tatqa",
                       help="编码器模型名称")
    parser.add_argument("--reranker_model", type=str,
                       default="Qwen/Qwen3-Reranker-0.6B",
                       help="重排序模型名称")
    parser.add_argument("--method", type=str,
                       choices=["encoder_only", "encoder_faiss", "encoder_reranker", "encoder_faiss_reranker", "all"],
                       default="all",
                       help="评估方法")
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
    parser.add_argument("--num_gpus", type=int, default=2,
                       help="使用的GPU数量")
    
    args = parser.parse_args()
    
    print("🚀 集成评估系统 - 使用真实RAG系统组件")
    print(f"📊 配置:")
    print(f"  - 评估数据: {args.eval_data}")
    print(f"  - 检索库数据: {args.corpus_data}")
    print(f"  - 编码器模型: {args.encoder_model}")
    print(f"  - 重排序模型: {args.reranker_model}")
    print(f"  - 评估方法: {args.method}")
    print(f"  - 最大样本数: {args.max_samples}")
    print(f"  - 元数据过滤: {'启用' if args.use_metadata_filter else '禁用'}")
    print(f"  - GPU数量: {args.num_gpus}")
    
    # 检查文件是否存在
    if not Path(args.eval_data).exists():
        print(f"❌ 评估数据文件不存在: {args.eval_data}")
        return
    
    if not Path(args.corpus_data).exists():
        print(f"❌ 检索库数据文件不存在: {args.corpus_data}")
        return
    
    # 创建基础配置
    base_config = EvaluationConfig(
        eval_data_path=args.eval_data,
        corpus_data_path=args.corpus_data,
        encoder_model=args.encoder_model,
        reranker_model=args.reranker_model,
        retrieval_method=RetrievalMethod.ENCODER_ONLY,  # 占位符
        top_k_retrieval=args.top_k_retrieval,
        top_k_rerank=args.top_k_rerank,
        max_samples=args.max_samples,
        use_metadata_filter=args.use_metadata_filter,
        device=args.device,
        num_gpus=args.num_gpus
    )
    
    # 确定要测试的方法
    if args.method == "all":
        methods = [
            RetrievalMethod.ENCODER_ONLY,
            RetrievalMethod.ENCODER_FAISS,
            RetrievalMethod.ENCODER_RERANKER,
            RetrievalMethod.ENCODER_FAISS_RERANKER
        ]
    else:
        methods = [RetrievalMethod(args.method)]
    
    # 运行对比实验
    experiment = ComparisonExperiment(base_config)
    results = experiment.run_comparison(methods)
    
    # 打印总结
    experiment.print_comparison_summary()
    
    print(f"\n🎉 评估完成！")

if __name__ == "__main__":
    # 设置多进程启动方法
    try:
        mp.set_start_method('spawn', force=True)
    except RuntimeError:
        pass  # 如果已经设置过了，就忽略
    main() 