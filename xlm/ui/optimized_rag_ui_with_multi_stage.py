#!/usr/bin/env python3
"""
Optimized RAG UI with Multi-Stage Retrieval System Integration
"""

import os
import sys
import re
import hashlib
from pathlib import Path
from typing import List, Optional, Tuple
import gradio as gr
import numpy as np
import torch
import faiss
from langdetect import detect, LangDetectException

# 添加项目根目录到路径
project_root = Path(__file__).parent.parent.parent
sys.path.append(str(project_root))

from xlm.dto.dto import DocumentWithMetadata, DocumentMetadata, RagOutput
from xlm.components.rag_system.rag_system import RagSystem
from xlm.components.generator.generator import Generator
from xlm.components.retriever.retriever import Retriever
from xlm.components.retriever.reranker import QwenReranker
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

class OptimizedRagUIWithMultiStage:
    def __init__(
        self,
        cache_dir: Optional[str] = None,
        use_faiss: bool = True,
        enable_reranker: bool = True,
        use_existing_embedding_index: Optional[bool] = None,
        max_alphafin_chunks: Optional[int] = None,
        window_title: str = "Financial Explainable RAG System with Multi-Stage Retrieval",
        title: str = "Financial Explainable RAG System with Multi-Stage Retrieval",
        examples: Optional[List[List[str]]] = None,
    ):
        # 使用config中的平台感知配置
        self.config = Config()
        self.cache_dir = EMBEDDING_CACHE_DIR if (not cache_dir or not isinstance(cache_dir, str)) else cache_dir
        self.use_faiss = use_faiss
        self.enable_reranker = enable_reranker
        self.use_existing_embedding_index = use_existing_embedding_index if use_existing_embedding_index is not None else self.config.retriever.use_existing_embedding_index
        self.max_alphafin_chunks = max_alphafin_chunks if max_alphafin_chunks is not None else self.config.retriever.max_alphafin_chunks
        self.window_title = window_title
        self.title = title
        self.examples = examples or [
            ["德赛电池(000049)的下一季度收益预测如何？"],
            ["用友网络2019年的每股经营活动产生的现金流量净额是多少？"],
            ["下月股价能否上涨?"],
            ["How was internally developed software capitalised?"],
            ["Why did the Operating revenues decreased from 2018 to 2019?"],
            ["Why did the Operating costs decreased from 2018 to 2019?"]
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
        """Initialize RAG system components with multi-stage retrieval"""
        print("\nStep 1. Initializing Multi-Stage Retrieval System...")
        
        # 初始化传统RAG系统作为回退
        print("Step 2. Initializing Traditional RAG System as fallback...")
        try:
            # 加载检索器
            self.retriever = load_enhanced_retriever(
                config=self.config
            )
            
            # 加载生成器
            self.generator = load_generator(
                generator_model_name=self.config.generator.model_name,
                use_local_llm=True,
                use_gpu=True,
                gpu_device="cuda:1",
                cache_dir=self.config.generator.cache_dir
            )
            
            # 初始化RAG系统
            self.rag_system = RagSystem(
                retriever=self.retriever,
                generator=self.generator,
                retriever_top_k=self.config.retriever.retrieval_top_k  # 使用配置中的设置
            )
            print("✅ 传统RAG系统初始化完成")
        except Exception as e:
            print(f"❌ 传统RAG系统初始化失败: {e}")
            self.rag_system = None
        
        # 初始化多阶段检索系统
        if MULTI_STAGE_AVAILABLE:
            try:
                # 中文数据路径
                chinese_data_path = Path("data/alphafin/alphafin_merged_generated_qa.json")
                
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
                
                # 英文数据路径（如果有的话）
                english_data_path = Path("data/tatqa/processed_data.json")  # 需要预处理
                if english_data_path.exists():
                    print("✅ 初始化英文多阶段检索系统...")
                    self.english_retrieval_system = MultiStageRetrievalSystem(
                        data_path=english_data_path,
                        dataset_type="english",
                        use_existing_config=True
                    )
                    print("✅ 英文多阶段检索系统初始化完成")
                else:
                    print(f"⚠️ 英文数据文件不存在: {english_data_path}")
                    self.english_retrieval_system = None
                
            except Exception as e:
                print(f"❌ 多阶段检索系统初始化失败: {e}")
                self.chinese_retrieval_system = None
                self.english_retrieval_system = None
        else:
            print("❌ 多阶段检索系统不可用，回退到传统检索")
            self.chinese_retrieval_system = None
            self.english_retrieval_system = None
        
        print("\nStep 3. Loading visualizer...")
        self.visualizer = Visualizer(show_mid_features=True)
        
        print("✅ 所有组件初始化完成")
    
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
                    context_output = gr.Dataframe(
                        headers=["Score", "Context"],
                        datatype=["number", "str"],
                        label="Retrieved Contexts",
                        interactive=False
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
                outputs=[answer_output, context_output]
            )
            
            return interface
    
    def _process_question(
        self,
        question: str,
        datasource: str,
        reranker_checkbox: bool
    ) -> tuple[str, List[List[str]]]:
        if not question.strip():
            return "请输入问题", []
        
        # 检测语言
        try:
            lang = detect(question)
            language = 'zh' if lang.startswith('zh') else 'en'
        except:
            language = 'en'
        
        # 根据语言选择检索系统
        if language == 'zh' and self.chinese_retrieval_system:
            return self._process_chinese_with_multi_stage(question, reranker_checkbox)
        elif language == 'en' and self.english_retrieval_system:
            return self._process_english_with_multi_stage(question, reranker_checkbox)
        else:
            return self._fallback_retrieval(question, language)
    
    def _process_chinese_with_multi_stage(self, question: str, reranker_checkbox: bool) -> tuple[str, List[List[str]]]:
        """使用多阶段检索系统处理中文查询"""
        if not self.chinese_retrieval_system:
            return self._fallback_retrieval(question, 'zh')
        
        try:
            print(f"🔍 开始中文多阶段检索...")
            print(f"📋 查询: {question}")
            company_name, stock_code = extract_stock_info(question)
            report_date = extract_report_date(question)
            print(f"🏢 公司名称: {company_name}")
            print(f"📈 股票代码: {stock_code}")
            print(f"📅 报告日期: {report_date}")
            print(f"⚙️ 配置参数: retrieval_top_k={self.config.retriever.retrieval_top_k}, rerank_top_k={self.config.retriever.rerank_top_k}")
            
            results = self.chinese_retrieval_system.search(
                query=question,
                company_name=company_name,
                stock_code=stock_code,
                report_date=report_date,
                top_k=self.config.retriever.rerank_top_k  # 使用配置中的重排序top-k
            )
            
            # 转换为DocumentWithMetadata格式
            retrieved_documents = []
            retriever_scores = []
            
            # 检查results的格式
            print(f"📊 检索结果类型: {type(results)}")
            if isinstance(results, dict) and 'retrieved_documents' in results:
                documents = results['retrieved_documents']
                llm_answer = results.get('llm_answer', '')
                print(f"📄 检索到 {len(documents)} 个文档")
                print(f"🤖 LLM答案: {'已生成' if llm_answer else '未生成'}")
                for result in documents:
                    doc = DocumentWithMetadata(
                        content=result.get('original_context', result.get('summary', '')),
                        metadata=DocumentMetadata(
                            source=result.get('company_name', 'Unknown'),
                            created_at="",
                            author="",
                            language="chinese"
                        )
                    )
                    retrieved_documents.append(doc)
                    retriever_scores.append(result.get('combined_score', 0.0))
                
                # 如果多阶段检索系统已经生成了答案，直接使用
                if llm_answer:
                    context_data = []
                    for doc, score in zip(retrieved_documents[:self.config.retriever.rerank_top_k], retriever_scores[:self.config.retriever.rerank_top_k]):
                        context_data.append([f"{score:.4f}", doc.content[:500] + "..." if len(doc.content) > 500 else doc.content])
                    answer = f"[Multi-Stage Retrieval: ZH] {llm_answer}"
                    return answer, context_data
            else:
                for result in results:
                    doc = DocumentWithMetadata(
                        content=result.get('original_context', result.get('summary', '')),
                        metadata=DocumentMetadata(
                            source=result.get('company_name', 'Unknown'),
                            created_at="",
                            author="",
                            language="chinese"
                        )
                    )
                    retrieved_documents.append(doc)
                    retriever_scores.append(result.get('combined_score', 0.0))
            
            if retrieved_documents:
                context_str = "\n\n".join([doc.content for doc in retrieved_documents[:10]])
                
                # 根据查询语言动态选择prompt模板
                try:
                    from langdetect import detect
                    query_language = detect(question)
                    is_chinese_query = query_language.startswith('zh')
                except:
                    # 如果语言检测失败，根据查询内容判断
                    is_chinese_query = any('\u4e00' <= char <= '\u9fff' for char in question)
                
                if is_chinese_query:
                    # 中文查询使用中文prompt模板
                    summary = context_str[:200] + "..." if len(context_str) > 200 else context_str
                    prompt = template_loader.format_template(
                        "multi_stage_chinese_template",
                        summary=summary,
                        context=context_str,
                        query=question
                    )
                    if prompt is None:
                        # 回退到简单中文prompt
                        prompt = f"基于以下上下文回答问题：\n\n{context_str}\n\n问题：{question}\n\n回答："
                else:
                    # 英文查询使用英文prompt模板
                    prompt = template_loader.format_template(
                        "rag_english_template",
                        context=context_str,
                        question=question
                    )
                    if prompt is None:
                        # 回退到简单英文prompt
                        prompt = f"Context: {context_str}\nQuestion: {question}\nAnswer:"
                
                generated_responses = self.generator.generate(texts=[prompt])
                answer = generated_responses[0] if generated_responses else "Unable to generate answer"
                context_data = []
                for doc, score in zip(retrieved_documents[:self.config.retriever.rerank_top_k], retriever_scores[:self.config.retriever.rerank_top_k]):
                    context_data.append([f"{score:.4f}", doc.content[:500] + "..." if len(doc.content) > 500 else doc.content])
                answer = f"[Multi-Stage Retrieval: ZH] {answer}"
                return answer, context_data
            else:
                return "No relevant documents found.", []
                
        except Exception as e:
            return self._fallback_retrieval(question, 'zh')
    
    def _process_english_with_multi_stage(self, question: str, reranker_checkbox: bool) -> tuple[str, List[List[str]]]:
        """使用多阶段检索系统处理英文查询"""
        if not self.english_retrieval_system:
            return self._fallback_retrieval(question, 'en')
        
        try:
            print(f"🔍 开始英文多阶段检索...")
            print(f"📋 查询: {question}")
            print(f"⚙️ 配置参数: retrieval_top_k={self.config.retriever.retrieval_top_k}, rerank_top_k={self.config.retriever.rerank_top_k}")
            
            # 执行多阶段检索
            results = self.english_retrieval_system.search(
                query=question,
                top_k=self.config.retriever.rerank_top_k  # 使用配置中的重排序top-k
            )
            
            # 转换为DocumentWithMetadata格式
            retrieved_documents = []
            retriever_scores = []
            
            for result in results:
                doc = DocumentWithMetadata(
                    content=result.get('context', result.get('content', '')),
                    metadata=DocumentMetadata(
                        source=result.get('source', 'Unknown'),
                        created_at="",
                        author="",
                        language="english"
                    )
                )
                retrieved_documents.append(doc)
                retriever_scores.append(result.get('combined_score', 0.0))
            
            if retrieved_documents:
                context_str = "\n\n".join([doc.content for doc in retrieved_documents[:10]])
                
                # 根据查询语言动态选择prompt模板
                try:
                    from langdetect import detect
                    query_language = detect(question)
                    is_chinese_query = query_language.startswith('zh')
                except:
                    # 如果语言检测失败，根据查询内容判断
                    is_chinese_query = any('\u4e00' <= char <= '\u9fff' for char in question)
                
                if is_chinese_query:
                    # 中文查询使用中文prompt模板
                    summary = context_str[:200] + "..." if len(context_str) > 200 else context_str
                    prompt = template_loader.format_template(
                        "multi_stage_chinese_template",
                        summary=summary,
                        context=context_str,
                        query=question
                    )
                    if prompt is None:
                        # 回退到简单中文prompt
                        prompt = f"基于以下上下文回答问题：\n\n{context_str}\n\n问题：{question}\n\n回答："
                else:
                    # 英文查询使用英文prompt模板
                    prompt = template_loader.format_template(
                        "rag_english_template",
                        context=context_str, 
                        question=question
                    )
                    if prompt is None:
                        # 回退到简单英文prompt
                        prompt = f"Context: {context_str}\nQuestion: {question}\nAnswer:"
                
                generated_responses = self.generator.generate(texts=[prompt])
                answer = generated_responses[0] if generated_responses else "Unable to generate answer"
                context_data = []
                for doc, score in zip(retrieved_documents[:self.config.retriever.rerank_top_k], retriever_scores[:self.config.retriever.rerank_top_k]):
                    context_data.append([f"{score:.4f}", doc.content[:500] + "..." if len(doc.content) > 500 else doc.content])
                answer = f"[Multi-Stage Retrieval: EN] {answer}"
                return answer, context_data
            else:
                return "No relevant documents found.", []
                
        except Exception as e:
            return self._fallback_retrieval(question, 'en')
    
    def _fallback_retrieval(self, question: str, language: str) -> tuple[str, List[List[str]]]:
        """回退到传统检索"""
        if self.rag_system is None:
            return "传统RAG系统未初始化，无法处理查询", []
        
        try:
            # 运行RAG系统
            rag_output = self.rag_system.run(user_input=question, language=language)
            
            # 生成答案
            if rag_output.retrieved_documents:
                # 构建上下文
                context_str = "\n\n".join([doc.content for doc in rag_output.retrieved_documents[:10]])
                
                # 根据查询语言动态选择prompt模板
                try:
                    from langdetect import detect
                    query_language = detect(question)
                    is_chinese_query = query_language.startswith('zh')
                except:
                    # 如果语言检测失败，根据查询内容判断
                    is_chinese_query = any('\u4e00' <= char <= '\u9fff' for char in question)
                
                if is_chinese_query:
                    # 中文查询使用中文prompt模板
                    summary = context_str[:200] + "..." if len(context_str) > 200 else context_str
                    prompt = template_loader.format_template(
                        "multi_stage_chinese_template",
                        summary=summary,
                        context=context_str,
                        query=question
                    )
                    if prompt is None:
                        # 回退到简单中文prompt
                        prompt = f"基于以下上下文回答问题：\n\n{context_str}\n\n问题：{question}\n\n回答："
                else:
                    # 英文查询使用英文prompt模板
                    prompt = template_loader.format_template(
                        "rag_english_template",
                        context=context_str, 
                        question=question
                    )
                    if prompt is None:
                        # 回退到简单英文prompt
                        prompt = f"Context: {context_str}\nQuestion: {question}\nAnswer:"
                
                # 生成答案
                generated_responses = self.generator.generate(texts=[prompt])
                answer = generated_responses[0] if generated_responses else "Unable to generate answer"
                
                # 准备上下文数据
                context_data = []
                for doc, score in zip(rag_output.retrieved_documents[:self.config.retriever.rerank_top_k], rag_output.retriever_scores[:self.config.retriever.rerank_top_k]):
                    # 统一只显示content字段，不显示question和answer
                    content = doc.content
                    # 确保content是字符串类型
                    if not isinstance(content, str):
                        if isinstance(content, dict):
                            # 如果是字典，尝试提取context或content字段
                            content = content.get('context', content.get('content', str(content)))
                        else:
                            content = str(content)
                    
                    # 截断过长的内容
                    display_content = content[:500] + "..." if len(content) > 500 else content
                    context_data.append([f"{score:.4f}", display_content])
                
                # 添加检索系统信息
                answer = f"[Multi-Stage Retrieval: {language.upper()}] {answer}"
                
                return answer, context_data
            else:
                return "No relevant documents found.", []
                
        except Exception as e:
            return f"检索失败: {str(e)}", []
    
    def launch(self, share: bool = False):
        """Launch UI interface"""
        self.interface.launch(share=share) 