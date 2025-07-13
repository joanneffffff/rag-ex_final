#!/usr/bin/env python3
"""
Optimized RAG UI with Multi-Stage Retrieval System Integration
"""

import os
import sys
import re
import hashlib
import json
import logging
from pathlib import Path
from typing import List, Optional, Tuple, Dict, Any
import gradio as gr
import numpy as np
import torch
import faiss
from langdetect import detect, LangDetectException

# 添加项目根目录到Python路径
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from xlm.dto.dto import DocumentWithMetadata, DocumentMetadata, RagOutput
from xlm.components.rag_system.rag_system import RagSystem
from xlm.components.generator.generator import Generator
from xlm.components.retriever.retriever import Retriever
from xlm.components.retriever.reranker import QwenReranker
from xlm.utils.visualizer import Visualizer
from xlm.registry.retriever import load_enhanced_retriever
from xlm.registry.generator import load_generator
from config.parameters import Config, EncoderConfig, RetrieverConfig, ModalityConfig, EMBEDDING_CACHE_DIR, RERANKER_CACHE_DIR, config
from xlm.components.prompt_templates.template_loader import template_loader
from xlm.utils.stock_info_extractor import extract_stock_info, extract_stock_info_with_mapping, extract_report_date

# 尝试导入多阶段检索系统
try:
    from alphafin_data_process.multi_stage_retrieval_final import MultiStageRetrievalSystem
    MULTI_STAGE_AVAILABLE = True
except ImportError:
    print("警告: 多阶段检索系统不可用，将使用传统检索")
    MULTI_STAGE_AVAILABLE = False

# 配置日志
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def build_smart_context(summary: str, context: str, query: str) -> str:
    """
    智能构建context，使用与chinese_llm_evaluation.py相同的逻辑
    这个函数负责将原始的 `context` 字符串进行处理，避免过度截断
    """
    processed_context = context
    try:
        # 尝试将 context 解析为字典，如果是则格式化为可读的JSON
        # 注意：这里使用 json.loads() 代替 eval() 更安全，但需要先替换单引号为双引号
        context_data = json.loads(context.replace("'", '"')) 
        if isinstance(context_data, dict):
            processed_context = json.dumps(context_data, ensure_ascii=False, indent=2)
            logger.debug("✅ Context识别为字典字符串并已格式化为JSON。")
    except (json.JSONDecodeError, TypeError):
        logger.debug("⚠️ Context非JSON字符串格式，直接使用原始context。")
        pass

    # 使用与chinese_llm_evaluation.py相同的长度限制：3500字符
    max_processed_context_length = 3500 # 字符长度，作为粗略限制
    if len(processed_context) > max_processed_context_length:
        logger.warning(f"⚠️ 处理后的Context长度过长 ({len(processed_context)}字符)，进行截断。")
        processed_context = processed_context[:max_processed_context_length] + "..."

    return processed_context

def _load_template_content_from_file(template_file_name: str) -> str:
    """
    从文件加载模板内容
    """
    template_path = Path("data/prompt_templates") / template_file_name
    if not template_path.exists():
        logger.error(f"❌ 模板文件不存在: {template_path}")
        return ""
    
    try:
        with open(template_path, 'r', encoding='utf-8') as f:
            return f.read()
    except Exception as e:
        logger.error(f"❌ 读取模板文件失败: {e}")
        return ""

def get_messages_for_test(summary: str, context: str, query: str, 
                          template_file_name: str = "multi_stage_chinese_template_with_fewshot.txt") -> List[Dict[str, str]]:
    """
    构建用于测试的 messages 列表，从指定模板文件加载内容，并将 item_instruction 融入 Prompt。
    Args:
        summary (str): LLM Qwen2-7B 生成的摘要。
        context (str): 完整上下文（已包含摘要）。
        query (str): 用户问题。
        template_file_name (str): 要加载的模板文件名。
    Returns:
        List[Dict[str, str]]: 构建好的 messages 列表。
    """
    template_full_string = _load_template_content_from_file(template_file_name)

    messages = []
    # 使用正则表达式分割所有部分，并保留分隔符内容
    parts = re.split(r'(===SYSTEM===|===USER===|===ASSISTANT===)', template_full_string, flags=re.DOTALL)

    # 移除第一个空字符串（如果存在）和多余的空白
    parts = [p.strip() for p in parts if p.strip()]

    current_role = None
    current_content = []

    for part in parts:
        if part in ["===SYSTEM===", "===USER===", "===ASSISTANT==="]:
            if current_role is not None:
                messages.append({"role": current_role.lower().replace("===", ""), "content": "\n".join(current_content).strip()})
            current_role = part
            current_content = []
        else:
            current_content.append(part)

    # 添加最后一个部分的 message
    if current_role is not None:
        messages.append({"role": current_role.lower().replace("===", ""), "content": "\n".join(current_content).strip()})

    # 替换占位符
    for message in messages:
        if message["role"] == "user":
            modified_content = message["content"]
            modified_content = modified_content.replace('{query}', query)
            modified_content = modified_content.replace('{summary}', summary)
            modified_content = modified_content.replace('{context}', context)
            message["content"] = modified_content

    logger.debug(f"构建的 messages: {messages}")
    return messages

def _convert_messages_to_chatml(messages: List[Dict[str, str]]) -> str:
    """
    将 messages 列表转换为 Fin-R1 (Qwen2.5 based) 期望的ChatML格式字符串。
    Qwen系列标准应该是 `im_end`
    """
    if not messages:
        return ""

    formatted_prompt = ""
    for message in messages:
        role = message.get("role", "")
        content = message.get("content", "")

        if role == "system":
            formatted_prompt += f"<|im_start|>system\n{content.strip()}<|im_end|>\n"
        elif role == "user":
            formatted_prompt += f"<|im_start|>user\n{content.strip()}<|im_end|>\n"
        elif role == "assistant":
            # 这里的 assistant 角色通常是 Few-shot 示例的一部分
            formatted_prompt += f"<|im_start|>assistant\n{content.strip()}<|im_end|>\n"

    # 在最后追加一个 <|im_start|>assistant\n，表示希望模型开始生成新的 assistant 回复
    formatted_prompt += "<|im_start|>assistant\n"

    logger.debug(f"转换后的 ChatML Prompt (前500字符):\n{formatted_prompt[:500]}...")
    return formatted_prompt

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
                    stock_prediction_checkbox = gr.Checkbox(
                        label="stock prediction (only for chinese query)",
                        value=False,
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
                inputs=[question_input, datasource, reranker_checkbox, stock_prediction_checkbox],
                outputs=[answer_output, context_output]
            )
            
            return interface
    
    def _process_question(
        self,
        question: str,
        datasource: str,
        reranker_checkbox: bool,
        stock_prediction_checkbox: bool
    ) -> tuple[str, List[List[str]]]:
        if not question.strip():
            return "please input your question", []
        
        # 检测语言
        try:
            lang = detect(question)
            language = 'zh' if lang.startswith('zh') else 'en'
        except:
            language = 'en'
        
        # 股票预测模式处理
        if stock_prediction_checkbox and language == 'zh':
            return self._process_stock_prediction(question, reranker_checkbox)
        
        # 根据语言选择检索系统
        if language == 'zh' and self.chinese_retrieval_system:
            return self._process_chinese_with_multi_stage(question, reranker_checkbox)
        elif language == 'en' and self.english_retrieval_system:
            return self._process_english_with_multi_stage(question, reranker_checkbox)
        else:
            return self._fallback_retrieval(question, language)
    
    def _process_stock_prediction(self, question: str, reranker_checkbox: bool) -> tuple[str, List[List[str]]]:
        """
        处理股票预测模式 - 仅适用于中文查询
        检索使用原始query，生成使用instruction
        """
        print(f"🔍 [股票预测模式] 开始处理...")
        print(f"📝 [股票预测模式] 原始查询: '{question}'")
        
        # 构建股票预测的instruction
        instruction = self._build_stock_prediction_instruction(question)
        print(f"📋 [股票预测模式] 生成的instruction:")
        print(f"   {instruction}")
        print(f"🔄 [股票预测模式] 查询转换完成:")
        print(f"   - 检索使用: '{question}'")
        print(f"   - 生成使用: '{instruction[:100]}{'...' if len(instruction) > 100 else ''}'")
        
        # 使用原始query进行检索，instruction用于生成
        return self._process_chinese_with_multi_stage_with_instruction(question, instruction, reranker_checkbox)
    
    def _process_chinese_with_multi_stage_with_instruction(self, query: str, instruction: str, reranker_checkbox: bool) -> tuple[str, List[List[str]]]:
        """
        使用多阶段检索系统处理中文查询（分离模式）
        使用query进行检索，instruction进行生成
        """
        print(f"🚀 [多阶段分离模式] 开始处理...")
        print(f"🔍 [多阶段分离模式] 检索查询: '{query}'")
        print(f"📝 [多阶段分离模式] 生成instruction: '{instruction[:100]}{'...' if len(instruction) > 100 else ''}'")
        print(f"📊 [多阶段分离模式] 处理策略:")
        print(f"   - 检索阶段: 使用原始query '{query}' 进行文档检索")
        print(f"   - 生成阶段: 使用instruction进行答案生成")
        
        if not self.chinese_retrieval_system:
            return self._fallback_retrieval(query, 'zh')
        
        try:
            print(f"🔍 [多阶段分离模式] 开始中文多阶段检索...")
            company_name, stock_code = extract_stock_info_with_mapping(query)
            report_date = extract_report_date(query)
            print(f"🏢 [多阶段分离模式] 公司名称: {company_name}")
            print(f"📈 [多阶段分离模式] 股票代码: {stock_code}")
            print(f"📅 [多阶段分离模式] 报告日期: {report_date}")
            
            # 使用query进行检索
            results = self.chinese_retrieval_system.search(
                query=query,
                company_name=company_name,
                stock_code=stock_code,
                report_date=report_date,
                top_k=self.config.retriever.rerank_top_k
            )
            
            # 转换为DocumentWithMetadata格式
            retrieved_documents = []
            retriever_scores = []
            
            # 检查results的格式
            print(f"📊 [多阶段分离模式] 检索结果类型: {type(results)}")
            if isinstance(results, dict) and 'retrieved_documents' in results:
                documents = results['retrieved_documents']
                print(f"📄 [多阶段分离模式] 检索到 {len(documents)} 个文档")
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
                # 使用build_smart_context处理上下文，避免过度截断
                context_parts = []
                summary_parts = []
                
                for doc in retrieved_documents[:10]:
                    content = doc.content
                    if not isinstance(content, str):
                        if isinstance(content, dict):
                            content = content.get('context', content.get('content', str(content)))
                        else:
                            content = str(content)
                    
                    # 使用build_smart_context处理上下文
                    processed_context = build_smart_context("", content, instruction)
                    context_parts.append(processed_context)
                
                context_str = "\n\n".join(context_parts)
                summary = context_str[:200] + "..." if len(context_str) > 200 else context_str
                
                # 使用与chinese_llm_evaluation.py相同的prompt生成逻辑
                chinese_template = getattr(self.config.data, 'chinese_prompt_template', 'multi_stage_chinese_template_with_fewshot.txt')
                print(f"使用配置的中文模板: {chinese_template}")
                
                # 使用get_messages_for_test和_convert_messages_to_chatml
                messages = get_messages_for_test(summary, context_str, instruction, chinese_template)
                prompt = _convert_messages_to_chatml(messages)
                
                print(f"🤖 [多阶段分离模式] 使用instruction生成答案...")
                
                # 生成答案
                try:
                    generated_responses = self.generator.generate(texts=[prompt])
                    answer = generated_responses[0] if generated_responses else "Unable to generate answer"
                    print(f"✅ [多阶段分离模式] 答案生成完成")
                except Exception as e:
                    print(f"❌ [多阶段分离模式] 答案生成失败: {e}")
                    answer = f"[多阶段分离模式] 答案生成失败: {e}"
                
                # 清理股票预测答案，移除"注意："及其后面的文字
                answer = self._clean_stock_prediction_answer(answer)
                
                # 准备上下文数据
                context_data = []
                for doc, score in zip(retrieved_documents[:self.config.retriever.rerank_top_k], retriever_scores[:self.config.retriever.rerank_top_k]):
                    context_data.append([f"{score:.4f}", doc.content[:500] + "..." if len(doc.content) > 500 else doc.content])
                
                return answer, context_data
            else:
                print(f"❌ [多阶段分离模式] 未找到相关文档")
                return "未找到相关文档", []
                
        except Exception as e:
            print(f"❌ [多阶段分离模式] 处理失败: {e}")
            return self._fallback_retrieval(query, 'zh')
    
    def _build_stock_prediction_instruction(self, question: str) -> str:
        """
        构建股票预测的instruction
        """
        # 使用与chinese_llm_evaluation.py相同的instruction格式
        return f"请根据下方提供的该股票相关研报与数据，对该股票的下个月的涨跌，进行预测，请给出明确的答案，\"涨\" 或者 \"跌\"。同时给出这个股票下月的涨跌概率，分别是:极大，较大，中上，一般。\n\n问题：{question}"
    
    def _clean_stock_prediction_answer(self, answer: str) -> str:
        """
        清理股票预测答案，移除"注意："及其后面的文字
        """
        if not answer:
            return answer
        
        # 查找"注意："的位置
        notice_index = answer.find("注意：")
        if notice_index != -1:
            # 移除"注意："及其后面的所有文字
            cleaned_answer = answer[:notice_index].strip()
            print(f"🔧 清理股票预测答案:")
            print(f"   原始答案: {answer}")
            print(f"   清理后答案: {cleaned_answer}")
            return cleaned_answer
        
        return answer
    
    def _process_chinese_with_multi_stage(self, question: str, reranker_checkbox: bool) -> tuple[str, List[List[str]]]:
        """使用多阶段检索系统处理中文查询"""
        if not self.chinese_retrieval_system:
            return self._fallback_retrieval(question, 'zh')
        
        try:
            print(f"🔍 开始中文多阶段检索...")
            print(f"📋 查询: {question}")
            company_name, stock_code = extract_stock_info_with_mapping(question)
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
                # 使用build_smart_context处理上下文，避免过度截断
                context_parts = []
                summary_parts = []
                
                for doc in retrieved_documents[:10]:
                    content = doc.content
                    if not isinstance(content, str):
                        if isinstance(content, dict):
                            content = content.get('context', content.get('content', str(content)))
                        else:
                            content = str(content)
                    
                    # 使用build_smart_context处理上下文
                    processed_context = build_smart_context("", content, question)
                    context_parts.append(processed_context)
                
                context_str = "\n\n".join(context_parts)
                
                # 根据查询语言动态选择prompt模板
                try:
                    from langdetect import detect
                    query_language = detect(question)
                    is_chinese_query = query_language.startswith('zh')
                except:
                    # 如果语言检测失败，根据查询内容判断
                    is_chinese_query = any('\u4e00' <= char <= '\u9fff' for char in question)
                
                if is_chinese_query:
                    # 中文查询使用与chinese_llm_evaluation.py相同的prompt生成逻辑
                    summary = context_str[:200] + "..." if len(context_str) > 200 else context_str
                    chinese_template = getattr(self.config.data, 'chinese_prompt_template', 'multi_stage_chinese_template_with_fewshot.txt')
                    print(f"使用配置的中文模板: {chinese_template}")
                    
                    # 使用get_messages_for_test和_convert_messages_to_chatml
                    messages = get_messages_for_test(summary, context_str, question, chinese_template)
                    prompt = _convert_messages_to_chatml(messages)
                else:
                    # 英文查询：使用配置的英文模板
                    try:
                        # 导入RAG系统的英文prompt处理函数
                        from xlm.components.rag_system.rag_system import get_final_prompt_messages_english, _convert_messages_to_chatml
                        
                        # 使用配置的英文模板
                        english_template = getattr(self.config.data, 'english_prompt_template', 'unified_english_template_no_think.txt')
                        messages = get_final_prompt_messages_english(context_str, question, english_template)
                        prompt = _convert_messages_to_chatml(messages)
                        print(f"使用配置的英文模板: {english_template}")
                    except Exception as e:
                        print(f"英文模板加载失败: {e}，使用简单英文prompt")
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