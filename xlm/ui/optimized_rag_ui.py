#!/usr/bin/env python3
"""
Optimized RAG UI with FAISS support
"""

import os
import sys
import re
from pathlib import Path
from typing import List, Optional, Tuple
import gradio as gr
import numpy as np
import torch
import faiss
from langdetect import detect, LangDetectException
import hashlib

# Add parent directory to path
sys.path.append(str(Path(__file__).parent.parent.parent))

from xlm.dto.dto import DocumentWithMetadata, DocumentMetadata, RagOutput
from xlm.components.rag_system.rag_system import RagSystem
from xlm.components.generator.generator import Generator
from xlm.components.retriever.retriever import Retriever
from xlm.components.retriever.reranker import QwenReranker
from xlm.components.retriever.bilingual_retriever import BilingualRetriever
from xlm.components.encoder.finbert import FinbertEncoder
from xlm.utils.dual_language_loader import DualLanguageLoader
from xlm.utils.visualizer import Visualizer
from xlm.registry.retriever import load_enhanced_retriever
from xlm.registry.generator import load_generator
from config.parameters import Config, EncoderConfig, RetrieverConfig, ModalityConfig, EMBEDDING_CACHE_DIR, RERANKER_CACHE_DIR
from xlm.components.prompt_templates.template_loader import template_loader
from xlm.utils.stock_info_extractor import extract_stock_info, extract_report_date

# 尝试导入多阶段检索系统
try:
    from alphafin_data_process.multi_stage_retrieval_final import MultiStageRetrievalSystem
    MULTI_STAGE_AVAILABLE = True
except ImportError:
    print("警告: 多阶段检索系统不可用，将使用传统检索")
    MULTI_STAGE_AVAILABLE = False

def try_load_qwen_reranker(model_name, cache_dir=None):
    """尝试加载Qwen重排序器，支持GPU 0和CPU回退"""
    try:
        import torch
        from transformers import AutoTokenizer, AutoModelForSequenceClassification
        
        # 确保cache_dir是有效的字符串
        if cache_dir is None:
            cache_dir = RERANKER_CACHE_DIR
        
        print(f"尝试使用8bit量化加载QwenReranker...")
        print(f"加载重排序器模型: {model_name}")
        
        # 首先尝试GPU 0
        if torch.cuda.is_available() and torch.cuda.device_count() > 0:
            device = "cuda:0"  # 明确指定GPU 0
            print(f"- 设备: {device}")
            print(f"- 缓存目录: {cache_dir}")
            print(f"- 量化: True (8bit)")
            print(f"- Flash Attention: False")
            
            try:
                # 检查GPU 0的可用内存
                gpu_memory = torch.cuda.get_device_properties(0).total_memory
                allocated_memory = torch.cuda.memory_allocated(0)
                free_memory = gpu_memory - allocated_memory
                
                print(f"- GPU 0 总内存: {gpu_memory / 1024**3:.1f}GB")
                print(f"- GPU 0 已用内存: {allocated_memory / 1024**3:.1f}GB")
                print(f"- GPU 0 可用内存: {free_memory / 1024**3:.1f}GB")
                
                # 如果可用内存少于2GB，回退到CPU
                if free_memory < 2 * 1024**3:  # 2GB
                    print("- GPU 0 内存不足，回退到CPU")
                    device = "cpu"
                else:
                    # 尝试在GPU 0上加载
                    tokenizer = AutoTokenizer.from_pretrained(model_name, cache_dir=cache_dir)
                    model = AutoModelForSequenceClassification.from_pretrained(
                        model_name,
                        cache_dir=cache_dir,
                        torch_dtype=torch.float16,
                        device_map="auto",
                        load_in_8bit=True
                    )
                    print("量化模型已自动设置到设备，跳过手动移动")
                    print("重排序器模型加载完成")
                    print("量化加载成功！")
                    return QwenReranker(model_name, device=device, cache_dir=cache_dir)
                    
            except Exception as e:
                print(f"- GPU 0 加载失败: {e}")
                print("- 回退到CPU")
                device = "cpu"
        
        # CPU回退
        if device == "cpu" or not torch.cuda.is_available():
            device = "cpu"
            print(f"- 设备: {device}")
            print(f"- 缓存目录: {cache_dir}")
            print(f"- 量化: False (CPU模式)")
            
            tokenizer = AutoTokenizer.from_pretrained(model_name, cache_dir=cache_dir)
            model = AutoModelForSequenceClassification.from_pretrained(
                model_name,
                cache_dir=cache_dir,
                torch_dtype=torch.float32
            )
            model = model.to(device)
            print("重排序器模型加载完成")
            print("CPU加载成功！")
            return QwenReranker(model_name, device=device, cache_dir=cache_dir)
            
    except Exception as e:
        print(f"加载重排序器失败: {e}")
        return None

class OptimizedRagUI:
    def __init__(
        self,
        # encoder_model_name: str = "sentence-transformers/all-MiniLM-L6-v2",
        encoder_model_name: str = "paraphrase-multilingual-MiniLM-L12-v2",
        # generator_model_name: str = "facebook/opt-125m",
        # generator_model_name: str = "Qwen/Qwen2-1.5B-Instruct",
        # generator_model_name: str = "SUFE-AIFLM-Lab/Fin-R1",  # 使用金融专用Fin-R1模型
        cache_dir: Optional[str] = None,
        use_faiss: bool = True,
        enable_reranker: bool = True,
        use_existing_embedding_index: Optional[bool] = None,  # 从config读取，None表示使用默认值
        max_alphafin_chunks: Optional[int] = None,  # 从config读取，None表示使用默认值
        window_title: str = "RAG System with FAISS",
        title: str = "RAG System with FAISS",
        examples: Optional[List[List[str]]] = None,
    ):
        # 使用config中的平台感知配置
        self.config = Config()
        self.cache_dir = EMBEDDING_CACHE_DIR if (not cache_dir or not isinstance(cache_dir, str)) else cache_dir
        self.encoder_model_name = encoder_model_name
        # 从config读取生成器模型名称，而不是硬编码
        self.generator_model_name = self.config.generator.model_name
        self.use_faiss = use_faiss
        self.enable_reranker = enable_reranker
        # 从config读取参数，如果传入None则使用config默认值
        self.use_existing_embedding_index = use_existing_embedding_index if use_existing_embedding_index is not None else self.config.retriever.use_existing_embedding_index
        self.max_alphafin_chunks = max_alphafin_chunks if max_alphafin_chunks is not None else self.config.retriever.max_alphafin_chunks
        self.window_title = window_title
        self.title = title
        self.examples = examples or [
            ["什么是股票投资？"],
            ["请解释债券的基本概念"],
            ["基金投资与股票投资有什么区别？"],
            ["What is stock investment?"],
            ["Explain the basic concepts of bonds"],
            ["What are the differences between fund investment and stock investment?"]
        ]
        
        # Set environment variables for model caching
        os.environ['TRANSFORMERS_CACHE'] = os.path.join(self.cache_dir, 'transformers')
        os.environ['HF_HOME'] = self.cache_dir
        os.environ['HF_DATASETS_CACHE'] = os.path.join(self.cache_dir, 'datasets')
        
        # Initialize system components
        self._init_components()
        
        # Create Gradio interface
        self.interface = self._create_interface()
    
    def _init_components(self):
        """Initialize RAG system components"""
        print("\nStep 1. Loading bilingual retriever with dual encoders...")
        
        # 使用config中的平台感知配置
        config = Config()
        
        # 清理GPU内存
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            print(f"GPU内存清理完成")
        
        # 初始化多阶段检索系统
        print("\nStep 1.0. Initializing Multi-Stage Retrieval System for Chinese queries...")
        if MULTI_STAGE_AVAILABLE:
            try:
                # 使用配置文件中的中文数据路径
                chinese_data_path = Path(config.data.chinese_data_path)
                
                if chinese_data_path.exists():
                    print("✅ 初始化中文多阶段检索系统...")
                    self.chinese_retrieval_system = MultiStageRetrievalSystem(
                        data_path=chinese_data_path,
                        dataset_type="chinese",
                        use_existing_config=True
                    )
                    print("✅ 中文多阶段检索系统初始化完成")
                else:
                    print(f"❌ 中文数据文件不存在: {chinese_data_path}")
                    self.chinese_retrieval_system = None
                
                # 英文数据使用传统RAG系统，不初始化多阶段检索
                print("ℹ️ 英文数据使用传统RAG系统，跳过多阶段检索初始化")
                self.english_retrieval_system = None
                
            except Exception as e:
                print(f"❌ 多阶段检索系统初始化失败: {e}")
                self.chinese_retrieval_system = None
                self.english_retrieval_system = None
        
        print("\nStep 1.1. Loading data with optimized chunking...")
        # 加载双语言数据 - 使用配置文件中的路径
        data_loader = DualLanguageLoader()
        
        # 分别加载中文和英文数据
        chinese_docs = []
        english_docs = []
        
        # 加载中文数据
        if config.data.chinese_data_path:
            print(f"加载中文数据: {config.data.chinese_data_path}")
            if config.data.chinese_data_path.endswith('.json'):
                chinese_docs = data_loader.load_alphafin_data(config.data.chinese_data_path)
            elif config.data.chinese_data_path.endswith('.jsonl'):
                chinese_docs = data_loader.load_jsonl_data(config.data.chinese_data_path, 'chinese')
        
        # 加载英文数据（使用context-only方法）
        if config.data.english_data_path:
            print(f"加载英文数据: {config.data.english_data_path}")
            english_docs = data_loader.load_tatqa_context_only(config.data.english_data_path)
        
        print(f"数据加载完成: {len(chinese_docs)} 个中文文档, {len(english_docs)} 个英文文档")
        
        # 清理内存
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        
        print("\nStep 2. Loading Chinese encoder...")
        print(f"Step 2. Loading Chinese encoder ({config.encoder.chinese_model_path})...")
        self.encoder_ch = FinbertEncoder(
            model_name=config.encoder.chinese_model_path,
            cache_dir=config.encoder.cache_dir,
            device=config.encoder.device  # 使用配置文件中的设备设置
        )
        
        print("\nStep 3. Loading English encoder...")
        print(f"Step 3. Loading English encoder ({config.encoder.english_model_path})...")
        self.encoder_en = FinbertEncoder(
            model_name=config.encoder.english_model_path,
            cache_dir=config.encoder.cache_dir,
            device=config.encoder.device  # 使用配置文件中的设备设置
        )
        
        # 清理内存
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        
        # 强制重新计算嵌入
        if self.use_existing_embedding_index is False:
            print("Forcing to recompute embeddings (ignoring existing cache)...")
        else:
            print("Using existing embedding index (if available)...")
        
        print(f"[UI DEBUG] self.use_existing_embedding_index={self.use_existing_embedding_index}")
        
        print("=== BEFORE BilingualRetriever ===")
        self.retriever = BilingualRetriever(
            encoder_en=self.encoder_en,
            encoder_ch=self.encoder_ch,
            corpus_documents_en=english_docs,
            corpus_documents_ch=chinese_docs,
            use_faiss=self.use_faiss,
            use_gpu=True,
            batch_size=32,
            cache_dir=config.encoder.cache_dir,
            use_existing_embedding_index=self.use_existing_embedding_index
        )
        print("=== AFTER BilingualRetriever ===")
        print(f"[UI DEBUG] BilingualRetriever created with use_existing_embedding_index={self.use_existing_embedding_index}")
        
        # 清理内存
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        
        print("\nStep 3.1. Initializing FAISS index...")
        self._init_faiss()
        
        print("\nStep 4. Loading reranker...")
        if self.enable_reranker:
            self.reranker = try_load_qwen_reranker(
                model_name=config.reranker.model_name,
                cache_dir=config.reranker.cache_dir
            )
            if self.reranker is None:
                print("⚠️ 重排序器加载失败，将禁用重排序功能")
                self.enable_reranker = False
        else:
            self.reranker = None
        
        # 清理内存
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        
        print("\nStep 5. Loading generator...")
        # 现在GPU内存充足，使用GPU 1加载生成器
        try:
            print("GPU内存充足，使用GPU 1加载生成器...")
            self.generator = load_generator(
                generator_model_name=config.generator.model_name,
                use_local_llm=True,
                use_gpu=True,  # 使用GPU
                gpu_device="cuda:1",  # 使用GPU 1
                cache_dir=config.generator.cache_dir
            )
            print("✅ 生成器GPU模式加载成功")
            
        except Exception as e:
            print(f"❌ 生成器GPU模式加载失败: {e}")
            print("回退到CPU模式...")
            try:
                self.generator = load_generator(
                    generator_model_name=config.generator.model_name,
                    use_local_llm=True,
                    use_gpu=False,  # 回退到CPU
                    cache_dir=config.generator.cache_dir
                )
                print("✅ 生成器CPU模式加载成功")
            except Exception as e2:
                print(f"❌ 生成器CPU模式也失败: {e2}")
                raise e2
        
        # 清理内存
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        
        print("\nStep 6. Initializing RAG system...")
        self.rag_system = RagSystem(
            retriever=self.retriever,
            generator=self.generator,
            retriever_top_k=20
        )
        
        print("\nStep 7. Loading visualizer...")
        self.visualizer = Visualizer(show_mid_features=True)
    
    def _init_faiss(self):
        """Initialize FAISS index"""
        # BilingualRetriever已经处理了FAISS索引，这里不需要额外的FAISS初始化
        print("FAISS索引已在BilingualRetriever中处理，跳过UI层的FAISS初始化")
        self.index = None
    
    def _create_interface(self) -> gr.Blocks:
        """Create optimized Gradio interface"""
        with gr.Blocks(
            title=self.window_title
        ) as interface:
            # 标题
            gr.Markdown(f"# {self.title}")
            
            # 输入区域
            with gr.Row():
                with gr.Column(scale=4):
                    datasource = gr.Radio(
                        choices=["TatQA", "AlphaFin", "Both"],
                        value="Both",
                        label="Data Source"
                    )
                    
            with gr.Row():
                with gr.Column(scale=4):
                    question_input = gr.Textbox(
                        show_label=False,
                        placeholder="Enter your question",
                        label="Question",
                        lines=3
                    )
            
            # 控制按钮区域
            with gr.Row():
                with gr.Column(scale=1):
                    reranker_checkbox = gr.Checkbox(
                        label="Enable Reranker",
                        value=True,
                        interactive=True
                    )
                with gr.Column(scale=1):
                    submit_btn = gr.Button("Submit")
            
            # 使用标签页分离显示
            with gr.Tabs():
                # 回答标签页
                with gr.TabItem("Answer"):
                    answer_output = gr.Textbox(
                        show_label=False,
                        interactive=False,
                        label="Generated Response",
                        lines=5
                    )
                
                # 解释标签页
                with gr.TabItem("Explanation"):
                    # 使用HTML组件来显示可点击的上下文
                    context_html_output = gr.HTML(
                        label="Retrieved Contexts (Click to expand)",
                        value="<p>No contexts retrieved yet.</p>"
                    )
                    
                    # 保留原有的DataFrame作为备用
                    context_output = gr.Dataframe(
                        headers=["Score", "Context"],
                        datatype=["number", "str"],
                        label="Retrieved Contexts (Table View)",
                        interactive=False,
                        visible=False  # 默认隐藏
                    )

            # 添加示例问题
            gr.Examples(
                examples=self.examples,
                inputs=[question_input],
                label="Example Questions"
            )

            # 绑定事件
            submit_btn.click(
                self._process_question,
                inputs=[question_input, datasource, reranker_checkbox],
                outputs=[answer_output, context_html_output]
            )
            
            return interface
    
    def _process_question(
        self,
        question: str,
        datasource: str,
        reranker_checkbox: bool
    ) -> tuple[str, str]:
        if not question.strip():
            return "请输入问题", ""
        
        # 检测语言
        try:
            lang = detect(question)
            language = 'zh' if lang.startswith('zh') else 'en'
        except:
            language = 'en'
        
        # 统一使用相同的RAG系统处理
        return self._unified_rag_processing(question, language, reranker_checkbox)
    
    def _unified_rag_processing(self, question: str, language: str, reranker_checkbox: bool) -> tuple[str, str]:
        """
        统一的RAG处理流程 - 中文和英文使用相同的FAISS、重排序器和生成器
        """
        print(f"开始统一RAG检索...")
        print(f"查询: {question}")
        print(f"语言: {language}")
        print(f"使用FAISS: {self.use_faiss}")
        print(f"启用重排序器: {reranker_checkbox}")
        
                # 1. 中文查询：关键词提取 -> 元数据过滤 -> FAISS检索 -> chunk重排序
        if language == 'zh' and self.chinese_retrieval_system:
            print("检测到中文查询，尝试使用元数据过滤...")
            try:
                # 1.1 提取关键词
                company_name, stock_code = extract_stock_info(question)
                report_date = extract_report_date(question)
                if company_name:
                    print(f"提取到公司名称: {company_name}")
                if stock_code:
                    print(f"提取到股票代码: {stock_code}")
                if report_date:
                    print(f"提取到报告日期: {report_date}")
                
                # 1.2 元数据过滤
                candidate_indices = self.chinese_retrieval_system.pre_filter(
                    company_name=company_name,
                    stock_code=stock_code,
                    report_date=report_date,
                    max_candidates=1000
                )
                
                if candidate_indices:
                    print(f"元数据过滤成功，找到 {len(candidate_indices)} 个候选文档")
                    
                    # 1.3 使用已有的FAISS索引在过滤后的文档中进行检索
                    faiss_results = self.chinese_retrieval_system.faiss_search(
                        query=question,
                        candidate_indices=candidate_indices,
                        top_k=self.config.retriever.retrieval_top_k  # 使用配置的检索数量
                    )
                    
                    if faiss_results:
                        print(f"FAISS检索成功，找到 {len(faiss_results)} 个相关文档")
                        
                        # 1.4 转换为DocumentWithMetadata格式（content是chunk）
                        unique_docs = []
                        for doc_idx, faiss_score in faiss_results:
                            original_doc = self.chinese_retrieval_system.data[doc_idx]
                            chunks = self.chinese_retrieval_system.doc_to_chunks_mapping.get(doc_idx, [])
                            if chunks:
                                content = chunks[0]  # 使用chunk作为content
                                doc = DocumentWithMetadata(
                                    content=content,
                                    metadata=DocumentMetadata(
                                        source=str(original_doc.get('company_name', '')),
                                        created_at="",
                                        author="",
                                        language="chinese",
                                        doc_id=str(doc_idx)
                                    )
                                )
                                unique_docs.append((doc, faiss_score))
                        
                        # 1.5 对chunk应用重排序器
                        if reranker_checkbox and self.reranker:
                            print("对chunk应用重排序器...")
                            reranked_docs = []
                            reranked_scores = []
                            
                            # 提取文档内容
                            doc_texts = [doc.content for doc, _ in unique_docs]
                            
                            # 使用QwenReranker的rerank方法
                            reranked_items = self.reranker.rerank(
                                query=question,
                                documents=doc_texts,
                                batch_size=4
                            )
                            
                            # 将重排序结果映射回文档
                            content_to_doc_map = {doc.content: doc for doc, _ in unique_docs}
                            for doc_text, rerank_score in reranked_items:
                                if doc_text in content_to_doc_map:
                                    reranked_docs.append(content_to_doc_map[doc_text])
                                    reranked_scores.append(rerank_score)
                            
                            try:
                                sorted_pairs = sorted(zip(reranked_docs, reranked_scores), key=lambda x: x[1], reverse=True)
                                unique_docs = [(doc, score) for doc, score in sorted_pairs[:self.config.retriever.rerank_top_k]]
                                print(f"chunk重排序完成，保留前 {len(unique_docs)} 个文档")
                            except Exception as e:
                                print(f"重排序异常: {e}")
                                unique_docs = []
                        else:
                            print("跳过重排序器...")
                            unique_docs = unique_docs[:10]
                        
                        # 1.6 使用chunk生成答案
                        answer = self._generate_answer_with_context(question, unique_docs)
                        return self._format_and_return_result(answer, unique_docs, reranker_checkbox, "中文完整流程")
                    else:
                        print("FAISS检索未找到相关文档，回退到统一FAISS检索...")
                else:
                    print("元数据过滤未找到候选文档，回退到统一FAISS检索...")
                    
            except Exception as e:
                print(f"中文处理流程失败: {e}，回退到统一RAG处理")
        
        # 2. 使用统一的检索器进行FAISS检索
        # 中文使用summary，英文使用chunk
        retrieved_documents, retriever_scores = self.retriever.retrieve(
            text=question, 
            top_k=self.config.retriever.retrieval_top_k,  # 使用配置的检索数量
            return_scores=True,
            language=language
        )
        
        print(f"FAISS召回数量: {len(retrieved_documents)}")
        if not retrieved_documents:
            return "未找到相关文档", ""
        
        # 3. 可选的重排序（如果启用）
        if reranker_checkbox and self.reranker:
            print(f"应用重排序器... 输入数量: {len(retrieved_documents)}")
            reranked_docs = []
            reranked_scores = []
            
            # 提取文档内容
            doc_texts = [doc.content if hasattr(doc, 'content') else str(doc) for doc in retrieved_documents]
            
            # 使用QwenReranker的rerank方法
            reranked_items = self.reranker.rerank(
                query=question,
                documents=doc_texts,
                batch_size=1  # 减小到1以避免GPU内存不足
            )
            
            # 将重排序结果映射回文档
            content_to_doc_map = {doc.content if hasattr(doc, 'content') else str(doc): doc for doc in retrieved_documents}
            for doc_text, rerank_score in reranked_items:
                if doc_text in content_to_doc_map:
                    reranked_docs.append(content_to_doc_map[doc_text])
                    reranked_scores.append(rerank_score)
            
            # 按重排序分数排序
            sorted_pairs = sorted(zip(reranked_docs, reranked_scores), key=lambda x: x[1], reverse=True)
            retrieved_documents = [doc for doc, _ in sorted_pairs[:self.config.retriever.rerank_top_k]]  # 使用配置的重排序top-k
            retriever_scores = [score for _, score in sorted_pairs[:self.config.retriever.rerank_top_k]]
            print(f"重排序后数量: {len(retrieved_documents)}")
        else:
            print("跳过重排序器...")
        
        # 4. 去重处理
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
            if len(unique_docs) >= self.config.retriever.rerank_top_k:
                break
        
        # 5. 使用统一的生成器生成答案
        answer = self._generate_answer_with_context(question, unique_docs)
        
        # 6. 打印结果并返回
        return self._format_and_return_result(answer, unique_docs, reranker_checkbox, "统一RAG")
    
    def _generate_answer_with_context(self, question: str, unique_docs: List[Tuple[DocumentWithMetadata, float]]) -> str:
        """使用上下文生成答案"""
        # 构建上下文
        context_parts = []
        for doc, _ in unique_docs:
            if hasattr(doc, 'content'):
                content = doc.content
            else:
                content = str(doc)
            
            if not isinstance(content, str):
                if isinstance(content, dict):
                    content = content.get('context', content.get('content', str(content)))
                else:
                    content = str(content)
            context_parts.append(content)
        
        context_str = "\n\n".join(context_parts)
        
        # 使用生成器生成答案
        print("使用生成器生成答案...")
        
        # 根据查询语言选择prompt模板
        try:
            from langdetect import detect
            query_language = detect(question)
            is_chinese_query = query_language.startswith('zh')
        except:
            # 如果语言检测失败，根据查询内容判断
            is_chinese_query = any('\u4e00' <= char <= '\u9fff' for char in question)
        
        if is_chinese_query:
            # 中文查询使用中文prompt
            prompt = f"基于以下上下文回答问题：\n\n{context_str}\n\n问题：{question}\n\n回答："
        else:
            # 英文查询使用英文prompt模板
            try:
                from xlm.components.prompt_templates.template_loader import template_loader
                prompt = template_loader.format_template(
                    "rag_english_template",
                    context=context_str,
                    question=question
                )
                if prompt is None:
                    # 回退到简单英文prompt
                    prompt = f"Context: {context_str}\nQuestion: {question}\nAnswer:"
            except Exception as e:
                print(f"英文模板加载失败: {e}，使用简单英文prompt")
                prompt = f"Context: {context_str}\nQuestion: {question}\nAnswer:"
        
        try:
            answer = self.generator.generate(texts=[prompt])[0]
        except Exception as e:
            print(f"生成器调用失败: {e}")
            answer = "生成器调用失败"
        
        return answer
    
    def _format_and_return_result(self, answer: str, unique_docs: List[Tuple[DocumentWithMetadata, float]], 
                                 reranker_checkbox: bool, method: str) -> tuple[str, str]:
        """格式化并返回结果"""
        # 打印检索结果
        print(f"\n=== 检索到的原始上下文 ({method}) ===")
        print(f"检索到 {len(unique_docs)} 个唯一文档")
        for i, (doc, score) in enumerate(unique_docs[:5]):
            if hasattr(doc, 'content'):
                content = doc.content
            else:
                content = str(doc)
            
            if not isinstance(content, str):
                if isinstance(content, dict):
                    content = content.get('context', content.get('content', str(content)))
                else:
                    content = str(content)
            
            display_content = content[:300] + "..." if len(content) > 300 else content
            print(f"文档 {i+1} (分数: {score:.4f}): {display_content}")
        
        if len(unique_docs) > 5:
            print(f"... 还有 {len(unique_docs) - 5} 个文档")
        
        # 打印LLM响应
        print(f"\n=== LLM响应 ===")
        print(f"生成答案: {answer}")
        print(f"检索文档数: {len(unique_docs)}")
        
        # 添加重排序器信息
        if reranker_checkbox and self.reranker:
            answer = f"[Reranker: Enabled] {answer}"
        else:
            answer = f"[Reranker: Disabled] {answer}"
        
        # 生成HTML格式的可点击上下文
        html_content = self._generate_clickable_context_html(unique_docs)
        
        print(f"=== 查询处理完成 ===\n")
        return answer, html_content
    
    def _fallback_retrieval(self, question: str, language: str) -> tuple[str, str]:
        """
        回退到传统RAG系统处理英文查询
        """
        print(f"开始传统RAG检索...")
        print(f"查询: {question}")
        print(f"语言: {language}")
        print(f"使用FAISS: {self.use_faiss}")
        print(f"启用重排序器: {self.enable_reranker}")
        
        rag_output = self.rag_system.run(user_input=question, language=language)
        
        # 去重处理
        unique_docs = []
        seen_hashes = set()
        
        for doc, score in zip(rag_output.retrieved_documents, rag_output.retriever_scores):
            content = doc.content
            h = hashlib.md5(content.encode('utf-8')).hexdigest()
            if h not in seen_hashes:
                unique_docs.append((doc, score))
                seen_hashes.add(h)
            if len(unique_docs) >= 20:
                break
        
        answer = rag_output.generated_responses[0] if rag_output.generated_responses else "Unable to generate answer"
        
        # 打印检索到的原始上下文
        print(f"\n=== 检索到的原始上下文 ===")
        print(f"检索到 {len(unique_docs)} 个唯一文档")
        for i, (doc, score) in enumerate(unique_docs[:5]):  # 只显示前5个
            content = doc.content
            if not isinstance(content, str):
                if isinstance(content, dict):
                    content = content.get('context', content.get('content', str(content)))
                else:
                    content = str(content)
            
            # 截断显示
            display_content = content[:300] + "..." if len(content) > 300 else content
            print(f"文档 {i+1} (分数: {score:.4f}): {display_content}")
        
        if len(unique_docs) > 5:
            print(f"... 还有 {len(unique_docs) - 5} 个文档")
        
        # 打印LLM响应
        print(f"\n=== LLM响应 ===")
        print(f"生成答案: {answer}")
        print(f"检索文档数: {len(unique_docs)}")
        
        # Add reranker info to answer if used
        # 传统RAG系统不支持重排序，所以这里不添加
        answer = f"[Reranker: Disabled] {answer}"
        
        # 生成HTML格式的可点击上下文
        html_content = self._generate_clickable_context_html(unique_docs)
        
        print(f"=== 查询处理完成 ===\n")
        return answer, html_content
    
    def _generate_clickable_context_html(self, unique_docs: List[Tuple[DocumentWithMetadata, float]]) -> str:
        if not unique_docs:
            return "<p>没有检索到相关文档。</p>"

        html_parts = []
        html_parts.append("""
        <div style='font-family: Arial, sans-serif;'>
        <style>
        .expand-btn, .collapse-btn {
            margin-top: 10px;
            border: none;
            padding: 8px 16px;
            border-radius: 4px;
            cursor: pointer;
            font-size: 12px;
            color: white;
            transition: background-color 0.3s ease;
        }
        .expand-btn { 
            background-color: #4caf50; 
        }
        .expand-btn:hover { 
            background-color: #45a049; 
        }
        .collapse-btn { 
            background-color: #757575; 
        }
        .collapse-btn:hover { 
            background-color: #616161; 
        }
        .content-section { 
            margin-bottom: 20px; 
            border: 1px solid #ddd; 
            border-radius: 8px; 
            padding: 15px; 
            background-color: #f9f9f9; 
            transition: box-shadow 0.3s ease;
        }
        .content-section:hover {
            box-shadow: 0 2px 8px rgba(0,0,0,0.1);
        }
        .header { 
            display: flex; 
            justify-content: space-between; 
            align-items: center; 
            margin-bottom: 10px; 
        }
        .score { 
            background-color: #ff9800; 
            color: #fff; 
            padding: 4px 8px; 
            border-radius: 4px; 
            font-size: 12px; 
            font-weight: bold;
        }
        .short-content, .full-content { 
            margin: 0; 
            line-height: 1.6; 
        }
        .short-content { 
            color: #555; 
        }
        .full-content { 
            color: #333; 
        }
        .full-content p {
            white-space: pre-wrap;
            font-family: 'Courier New', monospace;
            background-color: #f8f9fa;
            padding: 15px;
            border-radius: 6px;
            border: 1px solid #e9ecef;
            margin: 10px 0;
            font-size: 13px;
            line-height: 1.5;
            overflow-x: auto;
            max-height: 500px;
            overflow-y: auto;
        }
        </style>
        """)

        for i, (doc, score) in enumerate(unique_docs):
            # 获取完整的原始内容
            content = doc.content
            if not isinstance(content, str):
                if isinstance(content, dict):
                    content = content.get('context', content.get('content', str(content)))
                else:
                    content = str(content)
            
            # 确保内容不为空
            if not content or not content.strip():
                content = "内容为空"
            
            # 短内容预览（前300字符）
            short_content = content[:300] + "..." if len(content) > 300 else content
            
            # 完整内容，保持原始格式
            # 使用HTML实体转义，保持换行和格式
            full_content = content.replace('&', '&amp;').replace('<', '&lt;').replace('>', '&gt;')
            full_content = full_content.replace('\n', '<br>').replace('  ', '&nbsp;&nbsp;')

            html_parts.append(f"""
            <div class='content-section'>
                <div class='header'>
                    <strong style='color: #333;'>文档 {i+1}</strong>
                    <span class='score'>score: {score:.4f}</span>
                </div>
                <div class='short-content' id='short_{i}'>
                    <p>{short_content}</p>
                    <button class='expand-btn' onclick='document.getElementById("short_{i}").style.display="none"; document.getElementById("full_{i}").style.display="block";'>
                        Read more
                    </button>
                </div>
                <div class='full-content' id='full_{i}' style='display: none;'>
                    <p>{full_content}</p>
                    <button class='collapse-btn' onclick='document.getElementById("full_{i}").style.display="none"; document.getElementById("short_{i}").style.display="block";'>
                        Show less 
                    </button>
                </div>
            </div>
            """)
        html_parts.append("</div>")
        return ''.join(html_parts)
    
    def _detect_data_source(self, question: str, language: str) -> str:
        """检测数据源类型"""
        if language == 'zh':
            return "AlphaFin"
        else:
            return "TAT_QA"
    
    def launch(self, share: bool = False):
        """Launch UI interface"""
        self.interface.launch(share=share)
    
    def _chunk_documents(self, documents: List[DocumentWithMetadata], chunk_size: int = 512, overlap: int = 50) -> List[DocumentWithMetadata]:
        """
        将文档分割成更小的chunks
        
        Args:
            documents: 原始文档列表
            chunk_size: chunk大小（字符数）
            overlap: 重叠字符数
            
        Returns:
            分块后的文档列表
        """
        chunked_docs = []
        
        for doc in documents:
            content = doc.content
            if len(content) <= chunk_size:
                # 文档较短，不需要分块
                chunked_docs.append(doc)
            else:
                # 文档较长，需要分块
                start = 0
                chunk_id = 0
                
                while start < len(content):
                    end = start + chunk_size
                    
                    # 确保不在单词中间截断
                    if end < len(content):
                        # 尝试在句号、逗号或空格处截断
                        for i in range(end, max(start + chunk_size - 100, start), -1):
                            if content[i] in '.。，, ':
                                end = i + 1
                                break
                    
                    chunk_content = content[start:end].strip()
                    
                    if chunk_content:  # 确保chunk不为空
                        # 创建新的文档元数据
                        chunk_metadata = DocumentMetadata(
                            source=f"{doc.metadata.source}_chunk_{chunk_id}",
                            created_at=doc.metadata.created_at,
                            author=doc.metadata.author,
                            language=doc.metadata.language
                        )
                        
                        # 创建新的文档对象
                        chunk_doc = DocumentWithMetadata(
                            content=chunk_content,
                            metadata=chunk_metadata
                        )
                        
                        chunked_docs.append(chunk_doc)
                        chunk_id += 1
                    
                    # 移动到下一个chunk，考虑重叠
                    start = end - overlap
                    if start >= len(content):
                        break
        
        return chunked_docs 
    
    def _chunk_documents_advanced(self, documents: List[DocumentWithMetadata]) -> List[DocumentWithMetadata]:
        """
        使用finetune_chinese_encoder.py中的高级chunk逻辑处理中文文档
        并集成finetune_encoder.py中的表格文本化处理
        """
        import re
        import json
        import ast
        
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
            """将表格转换为自然语言描述"""
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
        
        def convert_json_context_to_natural_language_chunks(json_str_context, company_name="公司"):
            chunks = []
            if not json_str_context or not json_str_context.strip():
                return chunks
            processed_str_context = json_str_context.replace("\\n", "\n")
            cleaned_initial = re.sub(re.escape("【问题】:"), "", processed_str_context)
            cleaned_initial = re.sub(re.escape("【答案】:"), "", cleaned_initial).strip()
            cleaned_initial = cleaned_initial.replace('，', ',')
            cleaned_initial = cleaned_initial.replace('：', ':')
            cleaned_initial = cleaned_initial.replace('【', '') 
            cleaned_initial = cleaned_initial.replace('】', '') 
            cleaned_initial = cleaned_initial.replace('\u3000', ' ')
            cleaned_initial = cleaned_initial.replace('\xa0', ' ').strip()
            cleaned_initial = re.sub(r'\s+', ' ', cleaned_initial).strip()
            
            # 处理研报格式
            report_match = re.match(
                r"这是以(.+?)为题目,在(\d{4}-\d{2}-\d{2}(?: \d{2}:\d{2}:\d{2})?)日期发布的研究报告。研报内容如下: (.+)", 
                cleaned_initial, 
                re.DOTALL
            )
            if report_match:
                report_title_full = report_match.group(1).strip()
                report_date = report_match.group(2).strip()
                report_raw_content = report_match.group(3).strip() 
                content_after_second_title_match = re.match(r"研报题目是:(.+)", report_raw_content, re.DOTALL)
                if content_after_second_title_match:
                    report_content_preview = content_after_second_title_match.group(1).strip()
                else:
                    report_content_preview = report_raw_content 
                report_content_preview = re.sub(re.escape("【问题】:"), "", report_content_preview)
                report_content_preview = re.sub(re.escape("【答案】:"), "", report_content_preview).strip()
                report_content_preview = re.sub(r'\s+', ' ', report_content_preview).strip() 
                company_stock_match = re.search(r"(.+?)（(\d{6}\.\w{2})）", report_title_full)
                company_info = ""
                if company_stock_match:
                    report_company_name = company_stock_match.group(1).strip()
                    report_stock_code = company_stock_match.group(2).strip()
                    company_info = f"，公司名称：{report_company_name}，股票代码：{report_stock_code}"
                    report_title_main = re.sub(r"（\d{6}\.\w{2}）", "", report_title_full).strip()
                else:
                    report_title_main = report_title_full
                chunk_text = f"一份发布日期为 {report_date} 的研究报告，其标题是：\"{report_title_main}\"{company_info}。报告摘要内容：{report_content_preview.rstrip('...') if report_content_preview.endswith('...') else report_content_preview}。"
                chunks.append(chunk_text)
                return chunks 

            # 处理字典格式
            extracted_dict_str = None
            parsed_data = None 
            temp_dict_search_str = re.sub(r"Timestamp\(['\"](.*?)['\"]\)", r"'\1'", cleaned_initial) 
            all_dict_matches = re.findall(r"(\{.*?\})", temp_dict_search_str, re.DOTALL) 
            for potential_dict_str in all_dict_matches:
                cleaned_potential_dict_str = potential_dict_str.strip()
                json_compatible_str_temp = cleaned_potential_dict_str.replace("'", '"')
                try:
                    parsed_data_temp = json.loads(json_compatible_str_temp)
                    if isinstance(parsed_data_temp, dict):
                        extracted_dict_str = cleaned_potential_dict_str
                        parsed_data = parsed_data_temp
                        break 
                except json.JSONDecodeError:
                    pass 
                fixed_for_ast_eval_temp = re.sub(
                    r"(?<!['\"\w.])\b(0[1-9]\d*)\b(?![\d.]|['\"\w.])", 
                    r"'\1'", 
                    cleaned_potential_dict_str
                )
                try:
                    parsed_data_temp = ast.literal_eval(fixed_for_ast_eval_temp)
                    if isinstance(parsed_data_temp, dict):
                        extracted_dict_str = cleaned_potential_dict_str
                        parsed_data = parsed_data_temp
                        break 
                except (ValueError, SyntaxError):
                    pass 

            if extracted_dict_str is not None and isinstance(parsed_data, dict):
                for metric_name, time_series_data in parsed_data.items():
                    if not isinstance(metric_name, str):
                        metric_name = str(metric_name)
                    cleaned_metric_name = re.sub(r'（.*?）', '', metric_name).strip()
                    if not isinstance(time_series_data, dict):
                        if time_series_data is not None and str(time_series_data).strip():
                            chunks.append(f"{company_name}的{cleaned_metric_name}数据为：{time_series_data}。")
                        continue
                    if not time_series_data:
                        continue
                    try:
                        sorted_dates = sorted(time_series_data.keys(), key=str)
                    except TypeError:
                        sorted_dates = [str(k) for k in time_series_data.keys()]
                    description_parts = []
                    for date in sorted_dates:
                        value = time_series_data[date]
                        if isinstance(value, (int, float)):
                            formatted_value = f"{value:.4f}".rstrip('0').rstrip('.') if isinstance(value, float) else str(value)
                        else:
                            formatted_value = str(value)
                        description_parts.append(f"在{date}为{formatted_value}")
                    if description_parts:
                        if len(description_parts) <= 3:
                            full_description = f"{company_name}的{cleaned_metric_name}数据: " + "，".join(description_parts) + "。"
                        else:
                            first_part = "，".join(description_parts[:3])
                            last_part = "，".join(description_parts[-3:])
                            if len(sorted_dates) > 6:
                                full_description = f"{company_name}的{cleaned_metric_name}数据从{sorted_dates[0]}到{sorted_dates[-1]}，主要变化为：{first_part}，...，{last_part}。"
                            else:
                                full_description = f"{company_name}的{cleaned_metric_name}数据: " + "，".join(description_parts) + "。"
                        chunks.append(full_description)
                return chunks 

            # 处理纯文本
            pure_text = cleaned_initial
            pure_text = re.sub(r"^\d{4}-\d{2}-\d{2}( \d{2}:\d{2}:\d{2})?[_;]?", "", pure_text, 1).strip()
            pure_text = re.sub(r"^[\u4e00-\u9fa5]+(?:/[\u4e00-\u9fa5]+)?\d{4}年\d{2}月\d{2}日\d{2}:\d{2}:\d{2}(?:据[\u4e00-\u9fa5]+?,)?\d{1,2}月\d{1,2}日,?", "", pure_text).strip()
            pure_text = re.sub(r"^(?:市场资金进出)?截至周[一二三四五六日]收盘,?", "", pure_text).strip()
            pure_text = re.sub(r"^[\u4e00-\u9fa5]+?中期净利预减\d+%-?\d*%(?:[\u4e00-\u9fa5]+?\d{1,2}月\d{1,2}日晚间公告,)?", "", pure_text).strip()

            if pure_text: 
                chunks.append(pure_text)
            else:
                chunks.append(f"原始格式，解析失败或无有效结构：{json_str_context.strip()[:100]}...")
            return chunks
        
        chunked_docs = []
        for doc in documents:
            # 检查是否包含表格数据
            content = doc.content
            
            # 尝试解析为JSON，检查是否包含表格结构
            try:
                parsed_content = json.loads(content)
                if isinstance(parsed_content, dict) and 'tables' in parsed_content:
                    # 处理包含表格的文档
                    paragraphs = parsed_content.get('paragraphs', [])
                    tables = parsed_content.get('tables', [])
                    
                    # 提取单位信息
                    unit_info = extract_unit_from_paragraph(paragraphs)
                    
                    # 处理段落
                    for p_idx, para in enumerate(paragraphs):
                        para_text = para.get("text", "") if isinstance(para, dict) else para
                        if para_text.strip():
                            chunk_metadata = DocumentMetadata(
                                source=f"{doc.metadata.source}_table_para_{p_idx}",
                                created_at=doc.metadata.created_at,
                                author=doc.metadata.author,
                                language=doc.metadata.language
                            )
                            chunk_doc = DocumentWithMetadata(
                                content=para_text.strip(),
                                metadata=chunk_metadata
                            )
                            chunked_docs.append(chunk_doc)
                    
                    # 处理表格
                    for t_idx, table in enumerate(tables):
                        table_text = table_to_natural_text(table, table.get("caption", ""), unit_info)
                        if table_text.strip():
                            chunk_metadata = DocumentMetadata(
                                source=f"{doc.metadata.source}_table_text_{t_idx}",
                                created_at=doc.metadata.created_at,
                                author=doc.metadata.author,
                                language=doc.metadata.language
                            )
                            chunk_doc = DocumentWithMetadata(
                                content=table_text.strip(),
                                metadata=chunk_metadata
                            )
                            chunked_docs.append(chunk_doc)
                    
                    continue  # 已处理表格数据，跳过后续处理
                    
            except (json.JSONDecodeError, TypeError):
                pass  # 不是JSON格式，继续使用原有的chunk逻辑
            
            # 使用原有的高级chunk逻辑处理
            chunks = convert_json_context_to_natural_language_chunks(content)
            
            for i, chunk_content in enumerate(chunks):
                if chunk_content.strip():
                    chunk_metadata = DocumentMetadata(
                        source=f"{doc.metadata.source}_advanced_chunk_{i}",
                        created_at=doc.metadata.created_at,
                        author=doc.metadata.author,
                        language=doc.metadata.language
                    )
                    
                    chunk_doc = DocumentWithMetadata(
                        content=chunk_content,
                        metadata=chunk_metadata
                    )
                    
                    chunked_docs.append(chunk_doc)
        
        return chunked_docs
    
    def _chunk_documents_simple(self, documents: List[DocumentWithMetadata], chunk_size: int = 512, overlap: int = 50) -> List[DocumentWithMetadata]:
        """
        简单的文档分块方法，用于英文文档
        """
        chunked_docs = []
        
        for doc in documents:
            content = doc.content
            if len(content) <= chunk_size:
                # 文档较短，不需要分块
                chunked_docs.append(doc)
            else:
                # 文档较长，需要分块
                start = 0
                chunk_id = 0
                
                while start < len(content):
                    end = start + chunk_size
                    
                    # 确保不在单词中间截断
                    if end < len(content):
                        # 尝试在句号、逗号或空格处截断
                        for i in range(end, max(start + chunk_size - 100, start), -1):
                            if content[i] in '.。，, ':
                                end = i + 1
                                break
                    
                    chunk_content = content[start:end].strip()
                    
                    if chunk_content:  # 确保chunk不为空
                        # 创建新的文档元数据
                        chunk_metadata = DocumentMetadata(
                            source=f"{doc.metadata.source}_simple_chunk_{chunk_id}",
                            created_at=doc.metadata.created_at,
                            author=doc.metadata.author,
                            language=doc.metadata.language
                        )
                        
                        # 创建新的文档对象
                        chunk_doc = DocumentWithMetadata(
                            content=chunk_content,
                            metadata=chunk_metadata
                        )
                        
                        chunked_docs.append(chunk_doc)
                        chunk_id += 1
                    
                    # 移动到下一个chunk，考虑重叠
                    start = end - overlap
                    if start >= len(content):
                        break
        
        return chunked_docs 