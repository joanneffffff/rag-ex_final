"""
RAG系统适配器 - 基于run_optimized_ui的实际工作流程
用于AlphaFin检索评测的统一接口
"""

import sys
import hashlib
import json
from pathlib import Path
from typing import List, Dict, Any, Optional, Tuple
import torch

# 添加项目根目录到路径
sys.path.append(str(Path(__file__).parent.parent))

from xlm.ui.optimized_rag_ui import OptimizedRagUI
# --- 修改开始 ---
# 移除对 parameters_cuda1_conservative 的导入，因为它不再是默认的备用配置
# from config.parameters_cuda1_conservative import config as conservative_config 
from config.parameters import Config, config as default_main_config # 导入 Config 类和 parameters.py 中的默认 config 对象
# --- 修改结束 ---

class RagSystemAdapter:
    """
    RAG系统适配器 - 基于run_optimized_ui的实际工作流程
    直接使用OptimizedRagUI实例进行检索评测
    """
    
    def __init__(self, config: Optional[Config] = None):
        """
        初始化RAG系统适配器 - 使用与run_optimized_ui相同的初始化方式
        
        Args:
            config: 配置对象，如果为None则使用 config/parameters.py 中的默认配置
        """
        # --- 修改开始 ---
        # 如果外部没有传入 config 对象，则默认使用 config/parameters.py 中的 config
        self.config = config or default_main_config 
        # --- 修改结束 ---
        self.ui = None
        
        # 使用与run_optimized_ui相同的初始化方式
        self._init_ui()
    
    def _init_ui(self):
        """初始化UI实例 - 完全复制run_optimized_ui的初始化逻辑"""
        print("初始化RAG系统适配器...")
        print("使用与run_optimized_ui相同的初始化方式")
        
        # 创建UI实例，使用与run_optimized_ui相同的参数
        self.ui = OptimizedRagUI(
            cache_dir=self.config.cache_dir,
            use_faiss=True,
            enable_reranker=True,
            window_title="Financial Explainable RAG System",
            title="Financial Explainable RAG System"
        )
        
        print("RAG系统适配器初始化完成")
    
    def get_ranked_documents_for_evaluation(
        self,
        query: str,
        top_k: int = 10,
        mode: str = "baseline",  # "baseline", "prefilter", "reranker"
        use_prefilter: Optional[bool] = None  # 如果为None，则根据mode和配置文件自动设置
    ) -> List[Dict[str, Any]]:
        """
        统一检索接口 - 基于OptimizedRagUI的实际工作流程
        
        Args:
            query: 查询文本
            top_k: 返回的文档数量
            mode: 检索模式（baseline=仅FAISS, prefilter=元数据过滤+FAISS, reranker=元数据过滤+FAISS+重排序, reranker_no_prefilter=FAISS+重排序）
            use_prefilter: 是否使用预过滤（如果为None，则根据mode和配置文件自动设置，使用时会自动启用映射功能）
            
        Returns:
            检索结果列表，每条包含doc_id、content、metadata等信息
        """
        if self.ui is None:
            raise ValueError("UI系统未初始化")
        
        # 根据mode自动设置use_prefilter
        if mode == "baseline":
            # baseline模式：纯FAISS检索，不使用预过滤
            use_prefilter = False
        elif mode == "prefilter":
            # prefilter模式：强制使用预过滤
            use_prefilter = True
        elif mode == "reranker":
            # reranker模式：强制使用预过滤
            use_prefilter = True
        elif mode == "reranker_no_prefilter":
            # 新的reranker模式：不使用预过滤，但使用重排序
            use_prefilter = False
        else:
            # 其他模式：使用传入的参数或配置文件设置
            if use_prefilter is None:
                use_prefilter = self.ui.config.retriever.use_prefilter
        
        print(f"开始检索评测...")
        print(f"查询: {query}")
        print(f"检索模式: {mode}")
        print(f"预过滤开关: {'开启' if use_prefilter else '关闭'}")
        if use_prefilter:
            print("预过滤模式下自动启用股票代码和公司名称映射")
        print(f"Top-K: {top_k}")
        
        # 根据mode设置reranker_checkbox
        reranker_checkbox = (mode == 'reranker' or mode == 'reranker_no_prefilter')
        print(f"DEBUG: mode={mode}, reranker_checkbox={reranker_checkbox}")
        
        # 检测语言
        try:
            from langdetect import detect
            lang = detect(query)
            # 检查是否包含中文字符
            chinese_chars = sum(1 for char in query if '\u4e00' <= char <= '\u9fff')
            total_chars = len([char for char in query if char.isalpha() or '\u4e00' <= char <= '\u9fff'])
            
            # 如果包含中文字符且中文比例超过30%，或者langdetect检测为中文，则认为是中文
            if chinese_chars > 0 and (chinese_chars / total_chars > 0.3 or lang.startswith('zh')):
                language = 'zh'
            else:
                language = 'en'
        except ImportError: # Changed from generic 'except' to specific 'ImportError' for clarity
            # If langdetect fails, use character detection
            chinese_chars = sum(1 for char in query if '\u4e00' <= char <= '\u9fff')
            language = 'zh' if chinese_chars > 0 else 'en'
        except Exception as e: # Catch other potential exceptions from langdetect
            print(f"Language detection failed: {e}. Falling back to character-based detection.")
            chinese_chars = sum(1 for char in query if '\u4e00' <= char <= '\u9fff')
            language = 'zh' if chinese_chars > 0 else 'en'
            
        print(f"检测到的语言: {language}")
        
        # 直接使用UI的组件进行检索，不调用_unified_rag_processing方法
        print(f"DEBUG: 开始检索，mode={mode}, reranker_checkbox={reranker_checkbox}, use_prefilter={use_prefilter}")
        results = self._extract_retrieval_results_from_ui(query, language, reranker_checkbox, top_k, use_prefilter)
        
        print(f"检索完成，返回 {len(results)} 个文档")
        return results
    
    def _extract_retrieval_results_from_ui(
        self,
        query: str,
        language: str,
        reranker_checkbox: bool,
        top_k: int,
        use_prefilter: bool = True
    ) -> List[Dict[str, Any]]:
        """
        从UI的检索流程中提取文档结果
        这个方法模拟UI的检索过程，但不生成答案，只返回检索结果
        """
        if self.ui is None:
            raise ValueError("UI系统未初始化")
            
        results = []
        
        # 1. 中文查询：关键词提取 -> 元数据过滤 -> FAISS检索 -> chunk重排序
        if language == 'zh' and self.ui.chinese_retrieval_system:
            print("检测到中文查询，尝试使用元数据过滤...")
            try:
                # 1.1 提取关键词
                from xlm.utils.stock_info_extractor import extract_stock_info, extract_stock_info_with_mapping, extract_report_date
                # 使用映射优先的提取函数
                company_name, stock_code = extract_stock_info_with_mapping(query)
                report_date = extract_report_date(query)
                
                # 1.2 元数据过滤 - 根据use_prefilter参数决定是否使用
                if use_prefilter:
                    print("启用元数据预过滤（自动启用股票代码和公司名称映射）...")
                    candidate_indices = self.ui.chinese_retrieval_system.pre_filter(
                        company_name=company_name,
                        stock_code=stock_code,
                        report_date=report_date
                    )
                else:
                    print("跳过元数据预过滤，使用全量检索...")
                    candidate_indices = list(range(len(self.ui.chinese_retrieval_system.data)))
                
                if candidate_indices:
                    print(f"候选文档数量: {len(candidate_indices)}")
                    
                    # 1.3 使用已有的FAISS索引在过滤后的文档中进行检索
                    faiss_results = self.ui.chinese_retrieval_system.faiss_search(
                        query=query,
                        candidate_indices=candidate_indices,
                        top_k=self.ui.config.retriever.retrieval_top_k
                    )
                    
                    if faiss_results:
                        print(f"FAISS检索成功，找到 {len(faiss_results)} 个相关文档")
                        
                        # 1.4 转换为DocumentWithMetadata格式（content是chunk）
                        unique_docs = []
                        for doc_idx, faiss_score in faiss_results:
                            original_doc = self.ui.chinese_retrieval_system.data[doc_idx]
                            chunks = self.ui.chinese_retrieval_system.doc_to_chunks_mapping.get(doc_idx, [])
                            if chunks:
                                content = chunks[0]  # 使用chunk作为content
                                # 使用原始数据文件的doc_id，而不是索引号
                                original_doc_id = original_doc.get('doc_id', str(doc_idx))
                                from xlm.dto.dto import DocumentWithMetadata, DocumentMetadata
                                doc = DocumentWithMetadata(
                                    content=content,
                                    metadata=DocumentMetadata(
                                        source=str(original_doc.get('company_name', '')),
                                        created_at="",
                                        author="",
                                        language="chinese",
                                        doc_id=str(original_doc_id),
                                        origin_doc_id=str(original_doc_id)
                                    )
                                )
                                unique_docs.append((doc, faiss_score))
                        
                        # 1.5 对chunk应用重排序器（如果启用reranker模式）
                        print(f"DEBUG: reranker_checkbox={reranker_checkbox}, self.ui.reranker={self.ui.reranker is not None}")
                        if reranker_checkbox and self.ui.reranker:
                            print("对chunk应用重排序器...")
                            
                            # 使用chinese_retrieval_system的rerank方法，它正确处理索引
                            # 将unique_docs转换为candidate_results格式 [(doc_idx, faiss_score), ...]
                            candidate_results = []
                            for doc, faiss_score in unique_docs:
                                # 从doc_id获取原始索引
                                doc_id = getattr(doc.metadata, 'doc_id', None)
                                if doc_id:
                                    # 在原始数据中找到对应的索引
                                    for idx, record in enumerate(self.ui.chinese_retrieval_system.data):
                                        if record.get('doc_id') == doc_id:
                                            candidate_results.append((idx, faiss_score))
                                            break
                            
                            if candidate_results:
                                # 使用chinese_retrieval_system的rerank方法
                                reranked_results = self.ui.chinese_retrieval_system.rerank(
                                    query=query,
                                    candidate_results=candidate_results,
                                    top_k=self.ui.config.retriever.rerank_top_k
                                )
                                
                                # 将重排序结果转换回DocumentWithMetadata格式
                                unique_docs = []
                                for doc_idx, faiss_score, combined_score in reranked_results:
                                    if doc_idx < len(self.ui.chinese_retrieval_system.data):
                                        original_doc = self.ui.chinese_retrieval_system.data[doc_idx]
                                        chunks = self.ui.chinese_retrieval_system.doc_to_chunks_mapping.get(doc_idx, [])
                                        if chunks:
                                            content = chunks[0]
                                            doc_id = original_doc.get('doc_id', str(doc_idx))
                                            from xlm.dto.dto import DocumentWithMetadata, DocumentMetadata
                                            doc = DocumentWithMetadata(
                                                content=content,
                                                metadata=DocumentMetadata(
                                                    source=str(original_doc.get('company_name', '')),
                                                    created_at="",
                                                    author="",
                                                    language="chinese",
                                                    doc_id=str(doc_id),
                                                    origin_doc_id=str(doc_id)
                                                )
                                            )
                                            unique_docs.append((doc, combined_score))
                                
                                print(f"chunk重排序完成，保留前 {len(unique_docs)} 个文档")
                            else:
                                print("无法找到候选结果进行重排序")
                                unique_docs = unique_docs[:top_k]
                        else:
                            print("跳过重排序器...")
                            unique_docs = unique_docs[:top_k]
                        
                        # 格式化返回结果
                        for i, (doc, score) in enumerate(unique_docs):
                            result = {
                                'index': i,
                                'faiss_score': score,
                                'doc_id': getattr(doc.metadata, 'doc_id', str(i)),
                                'content': doc.content,
                                'source': getattr(doc.metadata, 'source', ''),
                                'language': getattr(doc.metadata, 'language', ''),
                                'summary': getattr(doc.metadata, 'summary', '')
                            }
                            results.append(result)
                        
                        print(f"中文完整流程检索完成，返回 {len(results)} 个文档")
                        return results
                    else:
                        print("FAISS检索未找到相关文档，回退到统一FAISS检索...")
                else:
                    print("元数据过滤未找到候选文档，回退到统一FAISS检索...")
                    
            except Exception as e:
                print(f"中文处理流程失败: {e}，回退到统一RAG处理")
        
        # 2. 使用统一的检索器进行FAISS检索 - 与UI完全相同的逻辑
        # 中文使用summary，英文使用chunk
        retrieval_result = self.ui.retriever.retrieve(
            text=query, 
            top_k=self.ui.config.retriever.retrieval_top_k,
            return_scores=True,
            language=language
        )
        
        # 处理返回结果
        if isinstance(retrieval_result, tuple):
            retrieved_documents, retriever_scores = retrieval_result
        else:
            retrieved_documents = retrieval_result
            retriever_scores = [1.0] * len(retrieved_documents)  # 默认分数
        
        print(f"FAISS召回数量: {len(retrieved_documents)}")
        if not retrieved_documents:
            return []
        
        # 3. 可选的重排序（如果启用reranker模式） - 修复映射问题
        print(f"DEBUG: 统一RAG流程 reranker_checkbox={reranker_checkbox}, self.ui.reranker={self.ui.reranker is not None}")
        if reranker_checkbox and self.ui.reranker:
            print(f"应用重排序器... 输入数量: {len(retrieved_documents)}")
            
            # 检测查询语言 - 修复语言检测逻辑
            try:
                from langdetect import detect
                query_language = detect(query)
                is_chinese_query = query_language.startswith('zh')
                print(f"DEBUG: 语言检测结果: {query_language}, is_chinese_query: {is_chinese_query}")
            except ImportError:
                is_chinese_query = any('\u4e00' <= char <= '\u9fff' for char in query)
                print(f"DEBUG: 使用字符检测, is_chinese_query: {is_chinese_query}")
            except Exception as e:
                print(f"Language detection failed: {e}. Falling back to character-based detection.")
                is_chinese_query = any('\u4e00' <= char <= '\u9fff' for char in query)
                print(f"DEBUG: 回退到字符检测, is_chinese_query: {is_chinese_query}")
            
            # 强制中文查询使用中文数据
            if any('\u4e00' <= char <= '\u9fff' for char in query):
                is_chinese_query = True
                print(f"DEBUG: 检测到中文字符，强制使用中文数据")
            
            # 提取文档内容和索引信息
            doc_texts = []
            doc_indices = []  # 保存原始索引
            doc_ids = []      # 保存doc_id
            
            for i, doc in enumerate(retrieved_documents):
                # 获取文档内容
                if hasattr(doc, 'content'):
                    content = doc.content
                else:
                    content = str(doc)
                
                # 限制内容长度
                if len(content) > 2000:
                    content = content[:2000] + "..."
                
                doc_texts.append(content)
                doc_indices.append(i)
                doc_id = getattr(doc.metadata, 'doc_id', f'doc_{i}')
                doc_ids.append(doc_id)
            
            # 使用QwenReranker的rerank_with_indices方法
            reranked_items = self.ui.reranker.rerank_with_indices(
                query=query,
                documents=doc_texts,
                doc_indices=doc_indices,
                doc_ids=doc_ids,
                batch_size=self.ui.config.reranker.batch_size
            )
            
            # 将重排序结果映射回文档（直接使用索引）
            print(f"DEBUG: 重排序器返回 {len(reranked_items)} 个结果")
            
            # 分析重排序分数
            if reranked_items:
                scores = [score for _, _, score in reranked_items]
                score_range = max(scores) - min(scores)
                print(f"DEBUG: 重排序分数范围: {min(scores):.4f} - {max(scores):.4f}, 差异: {score_range:.4f}")
                
                if score_range < 0.01:
                    print("DEBUG: ⚠️ 分数差异很小，排序可能不会改变")
                elif score_range < 0.1:
                    print("DEBUG: ⚠️ 分数差异较小，排序变化可能不明显")
                else:
                    print("DEBUG: ✅ 分数差异较大，排序应该有明显变化")
            
            reranked_docs = []
            reranked_scores = []
            for original_index, doc_id, rerank_score in reranked_items:
                if original_index < len(retrieved_documents):
                    original_doc = retrieved_documents[original_index]
                    reranked_docs.append(original_doc)
                    reranked_scores.append(rerank_score)
                    print(f"DEBUG: ✅ 成功映射文档 (doc_id: {doc_id}, index: {original_index})，重排序分数: {rerank_score:.4f}")
                else:
                    print(f"DEBUG: ❌ 索引超出范围: {original_index} >= {len(retrieved_documents)}")
            
            # 按重排序分数排序
            if reranked_docs:
                sorted_pairs = sorted(zip(reranked_docs, reranked_scores), key=lambda x: x[1], reverse=True)
                retrieved_documents = [doc for doc, _ in sorted_pairs[:self.ui.config.retriever.rerank_top_k]]
                retriever_scores = [score for _, score in sorted_pairs[:self.ui.config.retriever.rerank_top_k]]
                print(f"重排序后数量: {len(retrieved_documents)}")
                
                # 显示重排序后的前3个文档
                print("DEBUG: 重排序后前3个文档:")
                for i, (doc, score) in enumerate(sorted_pairs[:3]):
                    doc_id = getattr(doc.metadata, 'doc_id', 'N/A')
                    print(f"  {i+1}. {doc_id} - 分数: {score:.4f}")
            else:
                print("DEBUG: 重排序映射失败，保持原始排序")
        else:
            print("跳过重排序器...")
        
        # 4. 去重处理 - 与UI完全相同的逻辑
        unique_docs = []
        seen_hashes = set()
        
        for doc, score in zip(retrieved_documents, retriever_scores):
            if hasattr(doc, 'content'):
                content = doc.content
            else:
                content = str(doc)
            h = hashlib.md5(content.encode('utf-8')).hexdigest()
            if h not in seen_hashes:
                unique_docs.append((doc, score))
                seen_hashes.add(h)
            if len(unique_docs) >= self.ui.config.retriever.rerank_top_k:
                break
        
        # 5. 格式化返回结果
        for i, (doc, score) in enumerate(unique_docs[:top_k]):
            result = {
                'index': i,
                'faiss_score': score,
                'doc_id': getattr(doc.metadata, 'doc_id', str(i)),
                'content': doc.content,
                'source': getattr(doc.metadata, 'source', ''),
                'language': getattr(doc.metadata, 'language', ''),
                'summary': getattr(doc.metadata, 'summary', '')
            }
            results.append(result)
        
        print(f"统一RAG检索完成，返回 {len(results)} 个文档")
        return results
    
    def evaluate_retrieval_performance(
        self,
        eval_dataset: List[Dict],
        top_k: int = 10,
        mode: str = "baseline",
        use_prefilter: Optional[bool] = None
    ) -> List[Dict[str, Any]]:
        """
        执行检索并返回原始结果 - 优化版本，不计算指标
        
        Args:
            eval_dataset: 评测数据集，支持多种格式：
                - AlphaFin格式: 包含generated_question和doc_id
                - TatQA格式: 包含generated_question和relevant_doc_ids
                - 通用格式: 支持question/query和id/document_id/target_id等字段
            top_k: 检索返回的文档数量（实际检索深度）
            mode: 检索模式
            use_prefilter: 是否使用预过滤（如果为None，则根据mode和配置文件自动设置，使用时会自动启用映射功能）
            
        Returns:
            原始检索结果列表，每个元素包含：
            - query_text: str - 查询文本
            - ground_truth_doc_ids: List[str] - 正确答案的文档ID列表
            - retrieved_doc_ids_ranked: List[str] - 检索到的文档ID列表（按排序结果）
        """
        # 根据mode和配置文件自动设置use_prefilter（如果未指定）
        if use_prefilter is None:
            if self.ui is None:
                use_prefilter = True  # 默认值，理论上不会发生
            elif mode == "baseline":
                # baseline模式：根据配置文件决定是否使用预过滤
                use_prefilter = self.ui.config.retriever.use_prefilter
            elif mode == "prefilter":
                use_prefilter = True  # prefilter模式强制使用预过滤
            elif mode == "reranker":
                use_prefilter = True  # reranker模式强制使用预过滤
            else:
                use_prefilter = self.ui.config.retriever.use_prefilter  # 默认使用配置文件设置
        
        print(f"开始执行检索（优化版本）...")
        print(f"评测样本数: {len(eval_dataset)}")
        print(f"检索模式: {mode}")
        print(f"预过滤开关: {'开启' if use_prefilter else '关闭'}")
        if use_prefilter:
            print("预过滤模式下自动启用股票代码和公司名称映射")
        print(f"检索深度: {top_k}")
        
        raw_results = []
        
        for i, sample in enumerate(eval_dataset):
            # 兼容不同数据集的查询字段
            query = sample.get('generated_question', '') or sample.get('question', '') or sample.get('query', '')
            
            # 兼容不同数据集的目标文档ID字段
            target_doc_ids = []
            
            # 1. 优先使用relevant_doc_ids（TatQA数据集）
            if 'relevant_doc_ids' in sample and sample['relevant_doc_ids']:
                target_doc_ids = sample['relevant_doc_ids']
                if isinstance(target_doc_ids, str):
                    # 如果是字符串，尝试解析为列表
                    try:
                        target_doc_ids = json.loads(target_doc_ids)
                    except json.JSONDecodeError: # Added specific exception for clarity
                        target_doc_ids = [target_doc_ids]
                elif not isinstance(target_doc_ids, list):
                    target_doc_ids = [target_doc_ids]
            
            # 2. 如果没有relevant_doc_ids，使用doc_id（AlphaFin数据集）
            if not target_doc_ids and 'doc_id' in sample:
                doc_id = sample['doc_id']
                if doc_id:
                    target_doc_ids = [doc_id] if isinstance(doc_id, str) else doc_id
            
            # 3. 如果都没有，尝试其他可能的字段
            if not target_doc_ids:
                for field in ['id', 'document_id', 'target_id']:
                    if field in sample and sample[field]:
                        target_doc_ids = [sample[field]]
                        break
            
            if not query or not target_doc_ids:
                print(f"跳过样本 {i+1}: 缺少查询或目标文档ID")
                continue
            
            print(f"\n处理样本 {i+1}/{len(eval_dataset)}")
            print(f"查询: {query[:100]}...")
            print(f"目标文档IDs: {target_doc_ids}")
            
            try:
                # 使用统一的检索接口
                results = self.get_ranked_documents_for_evaluation(
                    query=query,
                    top_k=top_k,
                    mode=mode,
                    use_prefilter=use_prefilter
                )
                
                # 提取检索到的文档ID列表（按排序结果）
                retrieved_doc_ids = []
                for result in results:
                    doc_id = result.get('doc_id')
                    if doc_id:
                        retrieved_doc_ids.append(str(doc_id))
                
                # 构建原始结果
                raw_result = {
                    'query_text': query,
                    'ground_truth_doc_ids': [str(doc_id) for doc_id in target_doc_ids],
                    'retrieved_doc_ids_ranked': retrieved_doc_ids
                }
                
                raw_results.append(raw_result)
                print(f"✅ 检索完成，返回 {len(retrieved_doc_ids)} 个文档")
                
            except Exception as e:
                print(f"处理样本时出错: {e}")
                # 即使出错也要添加结果，避免索引错位
                raw_result = {
                    'query_text': query,
                    'ground_truth_doc_ids': [str(doc_id) for doc_id in target_doc_ids],
                    'retrieved_doc_ids_ranked': []
                }
                raw_results.append(raw_result)
                continue
        
        print(f"\n检索完成，共处理 {len(raw_results)} 个样本")
        return raw_results

def main():
    """主函数 - 演示如何使用适配器"""
    import json
    import argparse
    
    parser = argparse.ArgumentParser(description="RAG系统适配器评测脚本")
    parser.add_argument('--eval_data_path', type=str, default='data/alphafin/alphafin_eval_samples.jsonl', 
                        help='评测数据集路径')
    parser.add_argument('--mode', type=str, default='baseline', 
                        choices=['baseline', 'prefilter', 'reranker'], 
                        help='检索模式')
    parser.add_argument('--top_k', type=int, default=10, 
                        help='检索返回的文档数量')
    
    args = parser.parse_args()
    
    # 初始化适配器
    adapter = RagSystemAdapter()
    
    # 加载评测数据
    print(f"加载评测数据: {args.eval_data_path}")
    with open(args.eval_data_path, 'r', encoding='utf-8') as f:
        eval_dataset = json.load(f)
    
    # 执行评测
    results = adapter.evaluate_retrieval_performance(
        eval_dataset=eval_dataset,
        top_k=args.top_k,
        mode=args.mode
    )
    
    print("评测完成！")

if __name__ == "__main__":
    main()