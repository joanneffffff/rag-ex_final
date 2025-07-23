#!/usr/bin/env python3
"""
Optimized RAG UI with FAISS support
"""

import os
import sys
import re
from pathlib import Path
from typing import List, Optional, Tuple, Dict, Any
import gradio as gr
import numpy as np
import torch
import faiss
from langdetect import detect, LangDetectException
import hashlib
import json
import logging

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
from xlm.utils.stock_info_extractor import extract_stock_info, extract_stock_info_with_mapping, extract_report_date

# Try to import multi-stage retrieval system
try:
    from alphafin_data_process.multi_stage_retrieval_final import MultiStageRetrievalSystem
    MULTI_STAGE_AVAILABLE = True
except ImportError:
    print("Warning: Multi-stage retrieval system not available, using traditional retrieval.")
    MULTI_STAGE_AVAILABLE = False

# Set environment variables
ENHANCED_ENGLISH_AVAILABLE = True

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def build_smart_context(summary: str, context: str, query: str) -> str:
    """
    Intelligently build context, using the same logic as chinese_llm_evaluation.py. 
    This function processes the original 'context' string to avoid excessive truncation.
    """
    processed_context = context
    try:
        # Try to parse context as a dictionary, if so, format as readable JSON
        # Note: Use json.loads() instead of eval() for safety, but need to replace single quotes with double quotes first
        context_data = json.loads(context.replace("'", '"')) 
        if isinstance(context_data, dict):
            processed_context = json.dumps(context_data, ensure_ascii=False, indent=2)
            logger.debug("Context recognized as dictionary string and formatted as JSON.")
    except (json.JSONDecodeError, TypeError):
        logger.debug("Context is not a JSON string, using original context.")
        pass

    # Use the same length limit as chinese_llm_evaluation.py: 3500 characters
    max_processed_context_length = 3500
    if len(processed_context) > max_processed_context_length:
        logger.warning(f"Processed context is too long ({len(processed_context)} characters), truncating.")
        processed_context = processed_context[:max_processed_context_length] + "..."

    return processed_context

def try_load_qwen_reranker(model_name, cache_dir=None, device=None):
    """Try to load Qwen reranker, supports specifying device and fallback strategy"""
    try:
        import torch
        from transformers import AutoTokenizer, AutoModelForSequenceClassification
        
        # Ensure cache_dir is a valid string
        if cache_dir is None:
            cache_dir = RERANKER_CACHE_DIR
        
        print(f"Trying to load QwenReranker with 8bit quantization...")
        print(f"Loading reranker model: {model_name}")
        
        # Use the specified device, if not specified use GPU 0
        if device is None:
            device = "cuda:0"  # Default use GPU 0
        
        print(f"- device: {device}")
        print(f"- cache_dir: {cache_dir}")
        print(f"- quantization: True (8bit)")
        print(f"- Flash Attention: False")
        
        # Check device type
        if device.startswith("cuda"):
            try:
                # Parse GPU ID
                gpu_id = int(device.split(":")[1]) if ":" in device else 0
                
                # Check GPU memory
                gpu_memory = torch.cuda.get_device_properties(gpu_id).total_memory
                allocated_memory = torch.cuda.memory_allocated(gpu_id)
                free_memory = gpu_memory - allocated_memory
                
                print(f"- GPU {gpu_id} total memory: {gpu_memory / 1024**3:.1f}GB")
                print(f"- GPU {gpu_id} allocated memory: {allocated_memory / 1024**3:.1f}GB")
                print(f"- GPU {gpu_id} free memory: {free_memory / 1024**3:.1f}GB")
                
                # If free memory is less than 2GB, fallback to CPU
                if free_memory < 2 * 1024**3:  # 2GB
                    print(f"- GPU {gpu_id} has insufficient memory, falling back to CPU")
                    device = "cpu"
                else:
                    # Try to load on the specified GPU
                    tokenizer = AutoTokenizer.from_pretrained(model_name, cache_dir=cache_dir)
                    model = AutoModelForSequenceClassification.from_pretrained(
                        model_name,
                        cache_dir=cache_dir,
                        torch_dtype=torch.float16,
                        device_map="auto",
                        load_in_8bit=True
                    )
                    print("Quantized model automatically set to device, skipping manual move")
                    print("Reranker model loaded successfully")
                    print("Quantized loading successful!")
                    return QwenReranker(model_name, device=device, cache_dir=cache_dir)
                    
            except Exception as e:
                print(f"- GPU {gpu_id} loading failed: {e}")
                print("- Fallback to CPU")
                device = "cpu"
        
        # CPU fallback
        device = "cpu"  # Ensure device variable is always defined
        if device == "cpu" or not torch.cuda.is_available():
            print(f"- device: {device}")
            print(f"- cache_dir: {cache_dir}")
            print(f"- quantization: False (CPU mode)")
            
            tokenizer = AutoTokenizer.from_pretrained(model_name, cache_dir=cache_dir)
            model = AutoModelForSequenceClassification.from_pretrained(
                model_name,
                cache_dir=cache_dir,
                torch_dtype=torch.float32
            )
            model = model.to(device)
            print("Reranker model loaded successfully")
            print("CPU loading successful!")
            return QwenReranker(model_name, device=device, cache_dir=cache_dir)
            
    except Exception as e:
        print(f"Failed to load reranker: {e}")
        return None

class OptimizedRagUI:
    def __init__(
        self,
        # encoder_model_name: str = "sentence-transformers/all-MiniLM-L6-v2",
        encoder_model_name: str = "paraphrase-multilingual-MiniLM-L12-v2",
        # generator_model_name: str = "facebook/opt-125m",
        # generator_model_name: str = "Qwen/Qwen2-1.5B-Instruct",
        # generator_model_name: str = "SUFE-AIFLM-Lab/Fin-R1", 
        cache_dir: Optional[str] = None,
        use_faiss: bool = True,
        enable_reranker: bool = True,
        use_existing_embedding_index: Optional[bool] = None,
        max_alphafin_chunks: Optional[int] = None,
        window_title: str = "Financial Explainable RAG System",
        title: str = "Financial Explainable RAG System",
        examples: Optional[List[List[str]]] = None,
    ):
        # Use platform-aware configuration from config
        self.config = Config()
        self.cache_dir = EMBEDDING_CACHE_DIR if (not cache_dir or not isinstance(cache_dir, str)) else cache_dir
        self.encoder_model_name = encoder_model_name
        # Read generator model name from config, not hardcoded
        self.generator_model_name = self.config.generator.model_name
        self.use_faiss = use_faiss
        self.enable_reranker = enable_reranker
        # Read parameters from config, if None use config default value
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
        self.docid2context = self._load_docid2context(self.config.data.chinese_data_path)

    def _build_stock_prediction_instruction(self, question: str) -> str:
        """
        Build instruction for stock prediction
        """
        # Use the same instruction format as chinese_llm_evaluation.py, explicitly require output format
        return f"请根据下方提供的该股票相关研报与数据，对该股票的下个月的涨跌，进行预测，请给出明确的答案，\"涨\" 或者 \"跌\"。同时给出这个股票下月的涨跌概率，分别是:极大，较大，中上，一般。\n\n请严格按照以下格式输出：\n这个股票的下月最终收益结果是:'涨/跌',上涨/下跌概率:极大/较大/中上/一般\n\n问题：{question}"

    def _load_docid2context(self, data_path):
        import json
        docid2context = {}
        try:
            with open(data_path, "r", encoding="utf-8") as f:
                data = json.load(f)
                for item in data:
                    doc_id = str(item.get("doc_id", ""))
                    # Compare the length of original_content and context, use the longer one
                    original_content = item.get("original_content", "")
                    context_content = item.get("context", "")
                    context = original_content if len(original_content) > len(context_content) else context_content
                    if doc_id and context:  # Only add valid mappings
                        docid2context[doc_id] = context
            print(f"Successfully loaded {len(docid2context)} doc_id to context mappings")
        except Exception as e:
            print(f"Failed to load doc_id to context mappings: {e}")
        return docid2context

    def _init_components(self):
        """Initialize RAG system components"""
        print("\nStep 1. Loading bilingual retriever with dual encoders...")
        
        # Use platform-aware configuration from config
        config = Config()
        
        # Clean up GPU memory
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            print(f"GPU memory cleaned up")
        
        # Initialize multi-stage retrieval system
        print("\nStep 1.0. Initializing Multi-Stage Retrieval System for Chinese queries...")
        if MULTI_STAGE_AVAILABLE:
            try:
                # Use Chinese data path from config
                chinese_data_path = Path(config.data.chinese_data_path)
                
                if chinese_data_path.exists():
                    print("Initializing Chinese multi-stage retrieval system...")
                    self.chinese_retrieval_system = MultiStageRetrievalSystem(
                        data_path=chinese_data_path,
                        dataset_type="chinese",
                        use_existing_config=True
                    )
                    print("Chinese multi-stage retrieval system initialized successfully")
                else:
                    print(f"Chinese data file does not exist: {chinese_data_path}")
                    self.chinese_retrieval_system = None
                
                # English data uses traditional RAG system, no multi-stage retrieval initialization
                print("English data uses traditional RAG system, skipping multi-stage retrieval initialization")
                self.english_retrieval_system = None
                
            except Exception as e:
                print(f"Multi-stage retrieval system initialization failed: {e}")
                self.chinese_retrieval_system = None
                self.english_retrieval_system = None
        
        print("\nStep 1.1. Loading data with optimized chunking...")
        # Load bilingual data - use path from config
        data_loader = DualLanguageLoader()
        
        # Load Chinese and English data separately
        chinese_docs = []
        english_docs = []
        
        # Load Chinese data
        if config.data.chinese_data_path:
            print(f"Loading Chinese data: {config.data.chinese_data_path}")
            if config.data.chinese_data_path.endswith('.json'):
                chinese_docs = data_loader.load_alphafin_data(config.data.chinese_data_path)
            elif config.data.chinese_data_path.endswith('.jsonl'):
                chinese_docs = data_loader.load_jsonl_data(config.data.chinese_data_path, 'chinese')
        
        # Load English data (using context-only method)
        if config.data.english_data_path:
            print(f"Loading English data: {config.data.english_data_path}")
            english_docs = data_loader.load_tatqa_context_only(config.data.english_data_path)
        
        print(f"Data loading completed: {len(chinese_docs)} Chinese documents, {len(english_docs)} English documents")
        
        # Clean up memory
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        
        print("\nStep 2. Loading Chinese encoder...")
        print(f"Step 2. Loading Chinese encoder ({config.encoder.chinese_model_path})...")
        self.encoder_ch = FinbertEncoder(
            model_name=config.encoder.chinese_model_path,
            cache_dir=config.encoder.cache_dir,
            device=config.encoder.device  # Use device from config
        )
        
        print("\nStep 3. Loading English encoder...")
        print(f"Step 3. Loading English encoder ({config.encoder.english_model_path})...")
        self.encoder_en = FinbertEncoder(
            model_name=config.encoder.english_model_path,
            cache_dir=config.encoder.cache_dir,
            device=config.encoder.device  # Use device from config
        )
        
        # Clean up memory
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        
        # Force recompute embeddings
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
        
        # Clean up memory
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        
        print("\nStep 3.1. Initializing FAISS index...")
        self._init_faiss()
        
        print("\nStep 4. Loading reranker...")
        if self.enable_reranker:
            self.reranker = try_load_qwen_reranker(
                model_name=config.reranker.model_name,
                cache_dir=config.reranker.cache_dir,
                device=config.reranker.device  # Use device from config
            )
            if self.reranker is None:
                print("Reranker loading failed, disabling reranker functionality")
                self.enable_reranker = False
        else:
            self.reranker = None
        
        # Clean up memory
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        
        print("\nStep 5. Loading generator...")
        # Try to use shared resource manager
        try:
            from xlm.utils.shared_resource_manager import shared_resource_manager
            
            # Try to get LLM generator from shared resource manager
            self.generator = shared_resource_manager.get_llm_generator(
                model_name=config.generator.model_name,
                cache_dir=config.generator.cache_dir,
                device=config.generator.device,
                use_quantization=config.generator.use_quantization,
                quantization_type=config.generator.quantization_type
            )
            
            if self.generator:
                print("Using shared generator")
            else:
                print("Shared generator retrieval failed, falling back to independent loading")
                # Fall back to independent loading
                try:
                    print("GPU memory is sufficient, loading generator with GPU 1...")
                    self.generator = load_generator(
                        generator_model_name=config.generator.model_name,
                        use_local_llm=True,
                        use_gpu=True,  # Use GPU
                        gpu_device="cuda:1",  # Use GPU 1
                        cache_dir=config.generator.cache_dir
                    )
                    print("Generator GPU mode loaded successfully")
                    
                except Exception as e:
                    print(f"Generator GPU mode loading failed: {e}")
                    print("Falling back to CPU mode...")
                    try:
                        self.generator = load_generator(
                            generator_model_name=config.generator.model_name,
                            use_local_llm=True,
                            use_gpu=False,  # Fall back to CPU
                            cache_dir=config.generator.cache_dir
                        )
                        print("Generator CPU mode loaded successfully")
                    except Exception as e2:
                        print(f"Generator CPU mode also failed: {e2}")
                        raise e2
                        
        except ImportError:
            print("Shared resource manager not available, using independent loading")
            # Fall back to independent loading
            try:
                print("GPU memory is sufficient, loading generator with GPU 1...")
                self.generator = load_generator(
                    generator_model_name=config.generator.model_name,
                    use_local_llm=True,
                    use_gpu=True,  # Use GPU
                    gpu_device="cuda:1",  # Use GPU 1
                    cache_dir=config.generator.cache_dir
                )
                print("Generator GPU mode loaded successfully")
                
            except Exception as e:
                print(f"Generator GPU mode loading failed: {e}")
                print("Falling back to CPU mode...")
                try:
                    self.generator = load_generator(
                        generator_model_name=config.generator.model_name,
                        use_local_llm=True,
                        use_gpu=False,  # Fall back to CPU
                        cache_dir=config.generator.cache_dir
                    )
                    print("Generator CPU mode loaded successfully")
                except Exception as e2:
                    print(f"Generator CPU mode also failed: {e2}")
                    raise e2
        
        # Clean up memory
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
        # BilingualRetriever already handles FAISS index, no additional FAISS initialization needed
        print("FAISS index is already handled in BilingualRetriever, skipping UI-level FAISS initialization")
        self.index = None
    
    def _create_interface(self) -> gr.Blocks:
        """Create optimized Gradio interface"""
        with gr.Blocks(
            title=self.window_title
        ) as interface:
            # Title
            gr.Markdown(f"# {self.title}")
            
            # Input area
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
            
            # Control button area
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
            
            # Use tabs to separate display
            with gr.Tabs():
                # Answer tab
                with gr.TabItem("Answer"):
                    answer_output = gr.Textbox(
                        show_label=False,
                        interactive=False,
                        label="Generated Response",
                        lines=5
                    )
                
                # Explanation tab
                with gr.TabItem("Explanation"):
                    # Use HTML component to display clickable contexts
                    context_html_output = gr.HTML(
                        label="Retrieved Contexts (Click to expand)",
                        value="<p>No contexts retrieved yet.</p>"
                    )
                    
                    # Keep original DataFrame as backup
                    context_output = gr.Dataframe(
                        headers=["Score", "Context"],
                        datatype=["number", "str"],
                        label="Retrieved Contexts (Table View)",
                        interactive=False,
                        visible=False  # Default hidden
                    )

            # Add example questions
            gr.Examples(
                examples=self.examples,
                inputs=[question_input],
                label="Example Questions"
            )

            # Bind events
            submit_btn.click(
                self._process_question,
                inputs=[question_input, datasource, reranker_checkbox, stock_prediction_checkbox],
                outputs=[answer_output, context_html_output]
            )
            
            return interface
    
    def _process_question(
        self,
        question: str,
        datasource: str,
        reranker_checkbox: bool,
        stock_prediction_checkbox: bool
    ) -> tuple[str, str]:
        if not question.strip():
            return "Please enter a question", ""
        
        # Detect language
        try:
            lang = detect(question)
            # Check if contains Chinese characters
            chinese_chars = sum(1 for char in question if '\u4e00' <= char <= '\u9fff')
            total_chars = len([char for char in question if char.isalpha() or '\u4e00' <= char <= '\u9fff'])
            
            # If contains Chinese characters and Chinese ratio exceeds 30%, or langdetect detects Chinese, then it is Chinese
            if chinese_chars > 0 and (chinese_chars / total_chars > 0.3 or lang.startswith('zh')):
                language = 'zh'
            else:
                language = 'en'
        except:
            # If langdetect fails, use character detection
            chinese_chars = sum(1 for char in question if '\u4e00' <= char <= '\u9fff')
            language = 'zh' if chinese_chars > 0 else 'en'
        
        # Based on language and stock prediction checkbox, select processing method
        if language == 'zh':
            # All Chinese queries use built-in multi-stage retrieval system
            print("Chinese query detected, using built-in multi-stage retrieval system...")
            return self._unified_rag_processing_with_prompt(question, language, reranker_checkbox, stock_prediction_checkbox)
        else:
            # English query: use traditional RAG processing
            return self._unified_rag_processing(question, language, reranker_checkbox, stock_prediction_checkbox)

    def _unified_rag_processing_with_prompt(self, question: str, language: str, reranker_checkbox: bool, stock_prediction_checkbox: bool) -> tuple[str, str]:
        """
        Unified RAG processing flow - support stock prediction prompt switching
        """
        print(f"Starting unified RAG retrieval...")
        print(f"Query: {question}")
        print(f"Language: {language}")
        print(f"Use FAISS: {self.use_faiss}")
        print(f"Enable reranker: {reranker_checkbox}")
        print(f"Stock prediction mode: {stock_prediction_checkbox}")
        
        # Determine the prompt for generation
        if stock_prediction_checkbox:
            generation_prompt = "请根据下方提供的该股票相关研报与数据，对该股票的下个月的涨跌，进行预测，请给出明确的答案，\"涨\" 或者 \"跌\"。同时给出这个股票下月的涨跌概率，分别是:极大，较大，中上，一般。\n\n请严格按照以下格式输出：\n这个股票的下月最终收益结果是:'涨/跌',上涨/下跌概率:极大/较大/中上/一般"
            print(f"Stock prediction mode activated, generating prompt: {generation_prompt[:100]}...")
        else:
            generation_prompt = question
            print(f"Using original query as generation prompt")
        
        # 1. Chinese query: keyword extraction -> metadata filtering -> FAISS retrieval -> chunk reranking
        if language == 'zh' and self.chinese_retrieval_system:
            print("Chinese query detected, trying to use metadata filtering...")
            try:
                # 1.1 Extract keywords (using original query)
                company_name, stock_code = extract_stock_info_with_mapping(question)
                report_date = extract_report_date(question)
                if company_name:
                    print(f"Extracted company name: {company_name}")
                if stock_code:
                    print(f"Extracted stock code: {stock_code}")
                if report_date:
                    print(f"Extracted report date: {report_date}")
                
                # 1.2 Metadata filtering
                candidate_indices = self.chinese_retrieval_system.pre_filter(
                    company_name=company_name,
                    stock_code=stock_code,
                    report_date=report_date,
                    max_candidates=1000
                )
                
                if candidate_indices:
                    print(f"Metadata filtering successful, found {len(candidate_indices)} candidate documents")
                    
                    # 1.3 Use existing FAISS index to retrieve documents in filtered documents (using original query)
                    faiss_results = self.chinese_retrieval_system.faiss_search(
                        query=question,
                        candidate_indices=candidate_indices,
                        top_k=self.config.retriever.retrieval_top_k
                    )
                    
                    if faiss_results:
                        print(f"FAISS retrieval successful, found {len(faiss_results)} related documents")
                        
                        # 1.4 Convert to DocumentWithMetadata format (content is chunk)
                        unique_docs = []
                        for doc_idx, faiss_score in faiss_results:
                            original_doc = self.chinese_retrieval_system.data[doc_idx]
                            chunks = self.chinese_retrieval_system.doc_to_chunks_mapping.get(doc_idx, [])
                            if chunks:
                                content = chunks[0]  # Use chunk as content
                                # Use doc_id from original data file, not index number
                                original_doc_id = original_doc.get('doc_id', str(doc_idx))
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
                        
                        # 1.5 Apply reranker to chunks (using original query)
                        if reranker_checkbox and self.reranker:
                            print("Applying reranker to chunks...")
                            reranked_docs = []
                            reranked_scores = []
                            
                            # Extract document content
                            doc_texts = []
                            doc_id_to_original_map = {}
                            for doc, _ in unique_docs:
                                doc_id = getattr(doc.metadata, 'doc_id', None)
                                if doc_id is None:
                                    doc_id = hashlib.md5(doc.content.encode('utf-8')).hexdigest()[:16]
                                
                                if hasattr(doc, 'metadata') and hasattr(doc.metadata, 'language') and doc.metadata.language == 'chinese':
                                    summary = ""
                                    if hasattr(doc.metadata, 'summary') and doc.metadata.summary:
                                        summary = doc.metadata.summary
                                    else:
                                        summary = doc.content[:200] + "..." if len(doc.content) > 200 else doc.content
                                    
                                    combined_text = f"Summary: {summary}\n\nDetailed content: {doc.content}"
                                    if len(combined_text) > 4000:
                                        combined_text = f"Summary: {summary}\n\nDetailed content: {doc.content[:3500]}..."
                                    doc_texts.append(combined_text)
                                    doc_id_to_original_map[doc_id] = doc
                                else:
                                    doc_texts.append(doc.content)
                                    doc_id_to_original_map[doc_id] = doc
                            
                            # Use original query for reranking
                            reranked_items = self.reranker.rerank(
                                query=question,
                                documents=doc_texts,
                                batch_size=4
                            )
                            
                            # Map reranking results back to documents
                            for i, (doc_text, rerank_score) in enumerate(reranked_items):
                                if i < len(unique_docs):
                                    doc_id = getattr(unique_docs[i][0].metadata, 'doc_id', None)
                                    if doc_id is None:
                                        doc_id = hashlib.md5(unique_docs[i][0].content.encode('utf-8')).hexdigest()[:16]
                                    
                                    if doc_id in doc_id_to_original_map:
                                        reranked_docs.append(doc_id_to_original_map[doc_id])
                                        reranked_scores.append(rerank_score)
                            
                            try:
                                sorted_pairs = sorted(zip(reranked_docs, reranked_scores), key=lambda x: x[1], reverse=True)
                                unique_docs = [(doc, score) for doc, score in sorted_pairs[:self.config.retriever.rerank_top_k]]
                                print(f"Chunk reranking completed, keeping top {len(unique_docs)} documents")
                            except Exception as e:
                                print(f"Reranking exception: {e}")
                                unique_docs = []
                        else:
                            print("Skipping reranker...")
                            unique_docs = unique_docs[:10]
                        
                        # 1.6 Use generation_prompt to generate answer
                        answer = self._generate_answer_with_context(generation_prompt, unique_docs, stock_prediction_checkbox)
                        return self._format_and_return_result(answer, unique_docs, reranker_checkbox, "中文完整流程")
                    else:
                        print("FAISS retrieval failed to find related documents, falling back to unified FAISS retrieval...")
                else:
                    print("Metadata filtering failed to find candidate documents, falling back to unified FAISS retrieval...")
                    
            except Exception as e:
                print(f"Chinese processing flow failed: {e}, falling back to unified RAG processing")
        
        # 2. Use unified retriever for FAISS retrieval
        retrieval_result = self.retriever.retrieve(
            text=question, 
            top_k=self.config.retriever.retrieval_top_k,
            return_scores=True,
            language=language
        )
        
        # Process return result
        if isinstance(retrieval_result, tuple):
            retrieved_documents, retriever_scores = retrieval_result
        else:
            retrieved_documents = retrieval_result
            retriever_scores = [1.0] * len(retrieved_documents)
        
        print(f"FAISS retrieval successful, found {len(retrieved_documents)} related documents")
        if not retrieved_documents:
            return "No related documents found", ""
        
        # 3. Optional reranking (if enabled)
        if reranker_checkbox and self.reranker:
            print(f"Applying reranker... input number: {len(retrieved_documents)}")
            reranked_docs = []
            reranked_scores = []
            
            # Detect query language
            try:
                from langdetect import detect
                query_language = detect(question)
                is_chinese_query = query_language.startswith('zh')
            except:
                is_chinese_query = any('\u4e00' <= char <= '\u9fff' for char in question)
            
            # Extract document content
            doc_texts = []
            doc_id_to_original_map = {}
            for doc in retrieved_documents:
                doc_id = getattr(doc.metadata, 'doc_id', None)
                if doc_id is None:
                    doc_id = hashlib.md5(doc.content.encode('utf-8')).hexdigest()[:16]
                
                if is_chinese_query and hasattr(doc, 'metadata') and hasattr(doc.metadata, 'language') and doc.metadata.language == 'chinese':
                    summary = ""
                    if hasattr(doc.metadata, 'summary') and doc.metadata.summary:
                        summary = doc.metadata.summary
                    else:
                        summary = doc.content[:200] + "..." if len(doc.content) > 200 else doc.content
                    
                    combined_text = f"Summary: {summary}\n\nDetailed content: {doc.content}"
                    if len(combined_text) > 4000:
                        combined_text = f"Summary: {summary}\n\nDetailed content: {doc.content[:3500]}..."
                    doc_texts.append(combined_text)
                    doc_id_to_original_map[doc_id] = doc
                else:
                    doc_texts.append(doc.content)
                    doc_id_to_original_map[doc_id] = doc
            
            # Use original query for reranking
            reranked_items = self.reranker.rerank(
                query=question,
                documents=doc_texts,
                batch_size=4
            )
            
            # Map reranking results back to documents
            for i, (doc_text, rerank_score) in enumerate(reranked_items):
                if i < len(retrieved_documents):
                    doc_id = getattr(retrieved_documents[i].metadata, 'doc_id', None)
                    if doc_id is None:
                        doc_id = hashlib.md5(retrieved_documents[i].content.encode('utf-8')).hexdigest()[:16]
                    
                    if doc_id in doc_id_to_original_map:
                        reranked_docs.append(doc_id_to_original_map[doc_id])
                        reranked_scores.append(rerank_score)
            
            try:
                sorted_pairs = sorted(zip(reranked_docs, reranked_scores), key=lambda x: x[1], reverse=True)
                retrieved_documents = [doc for doc, _ in sorted_pairs[:self.config.retriever.rerank_top_k]]
                retriever_scores = [score for _, score in sorted_pairs[:self.config.retriever.rerank_top_k]]
                print(f"Reranking completed, keeping top {len(retrieved_documents)} documents")
            except Exception as e:
                print(f"Reranking exception: {e}")
        
        # 4. Deduplication processing
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
        
        # 5. Use generation_prompt to generate answer
        answer = self._generate_answer_with_context(generation_prompt, unique_docs, stock_prediction_checkbox)
        
        # 6. Print result and return
        return self._format_and_return_result(answer, unique_docs, reranker_checkbox, "统一RAG")
    
    def _unified_rag_processing(self, question: str, language: str, reranker_checkbox: bool, stock_prediction_checkbox: bool = False) -> tuple[str, str]:
        """
        Unified RAG processing flow - Chinese and English use the same FAISS, reranker and generator
        """
        print(f"Starting unified RAG retrieval...")
        print(f"Query: {question}")
        print(f"Language: {language}")
        print(f"Use FAISS: {self.use_faiss}")
        print(f"Enable reranker: {reranker_checkbox}")

        
        # English query-specific processing flow
        if language == 'zh':
            print("Chinese query detected, but this system only supports English queries, falling back to unified RAG processing")
        
        # 2. Use unified retriever for FAISS retrieval
        # Chinese uses summary, English uses chunk
        retrieval_result = self.retriever.retrieve(
            text=question, 
            top_k=self.config.retriever.retrieval_top_k,  # Use configuration retrieval number
            return_scores=True,
            language=language
        )
        
        # Process return result
        if isinstance(retrieval_result, tuple):
            retrieved_documents, retriever_scores = retrieval_result
        else:
            retrieved_documents = retrieval_result
            retriever_scores = [1.0] * len(retrieved_documents)  # Default score
        
        print(f"FAISS retrieval successful, found {len(retrieved_documents)} related documents")
        if not retrieved_documents:
            return "No related documents found", ""
        
        # 3. Optional reranking (if enabled)
        if reranker_checkbox and self.reranker:
            print(f"Applying reranker... input number: {len(retrieved_documents)}")
            reranked_docs = []
            reranked_scores = []
            

            
            # Extract document content (only Chinese query uses smart content selection)
            doc_texts = []
            doc_id_to_original_map = {}  # Use doc_id for mapping
            for doc in retrieved_documents:
                # Get doc_id
                doc_id = getattr(doc.metadata, 'doc_id', None)
                if doc_id is None:
                    # If no doc_id, use hash of content as unique identifier
                    doc_id = hashlib.md5(doc.content.encode('utf-8')).hexdigest()[:16]
                
                # English data: only use context
                doc_texts.append(doc.content if hasattr(doc, 'content') else str(doc))
                doc_id_to_original_map[doc_id] = doc  # Use doc_id for mapping
            
            # Use QwenReranker's rerank_with_doc_ids method
            doc_ids = []
            for doc in retrieved_documents:
                doc_id = getattr(doc.metadata, 'doc_id', None)
                if doc_id is None:
                    doc_id = hashlib.md5(doc.content.encode('utf-8')).hexdigest()[:16]
                doc_ids.append(doc_id)
            
            reranked_items = self.reranker.rerank_with_doc_ids(
                query=question,
                documents=doc_texts,
                doc_ids=doc_ids,
                batch_size=self.config.reranker.batch_size  # Use batch size from configuration file
            )
            
            # Map reranking results back to documents (reranker directly returns doc_id, no complex mapping)
            for doc_text, rerank_score, doc_id in reranked_items:
                if doc_id in doc_id_to_original_map:
                    reranked_docs.append(doc_id_to_original_map[doc_id])
                    reranked_scores.append(rerank_score)
                    print(f"DEBUG: Successfully mapped document (doc_id: {doc_id}), reranking score: {rerank_score:.4f}")
                else:
                    print(f"DEBUG: doc_id not in mapping: {doc_id}")
            
            # Sort by reranking score
            sorted_pairs = sorted(zip(reranked_docs, reranked_scores), key=lambda x: x[1], reverse=True)
            retrieved_documents = [doc for doc, _ in sorted_pairs[:self.config.retriever.rerank_top_k]]  # Use configuration reranking top-k
            retriever_scores = [score for _, score in sorted_pairs[:self.config.retriever.rerank_top_k]]
            print(f"Reranking completed, keeping top {len(retrieved_documents)} documents")
        else:
            print("Skipping reranker...")
        
        # 4. Deduplication processing
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
        
        # 5. Use unified generator to generate answer
        answer = self._generate_answer_with_context(question, unique_docs, stock_prediction_checkbox)
        
        # 6. Print result and return
        return self._format_and_return_result(answer, unique_docs, reranker_checkbox, "统一RAG")
    
    def _generate_answer_with_context(self, question: str, unique_docs: List[Tuple[DocumentWithMetadata, float]], stock_prediction_checkbox: bool = False) -> str:
        """Use context to generate answer"""
        # Build context and extract summary
        context_parts = []
        summary_parts = []
        
        # Select prompt template based on query language
        try:
            from langdetect import detect
            query_language = detect(question)
            is_chinese_query = query_language.startswith('zh')
        except:
            # If language detection fails, determine based on query content
            is_chinese_query = any('\u4e00' <= char <= '\u9fff' for char in question)
        
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
            
            if is_chinese_query:
                # Chinese query: use smart content selection
                if hasattr(doc, 'metadata') and hasattr(doc.metadata, 'language') and doc.metadata.language == 'chinese':
                    # Chinese data: try combining summary and context
                    summary = ""
                    if hasattr(doc.metadata, 'summary') and doc.metadata.summary:
                        summary = doc.metadata.summary
                    else:
                        # If no summary, use first 200 characters of context as summary
                        summary = doc.content[:200] + "..." if len(doc.content) > 200 else doc.content
                    
                    # Use build_smart_context to process context, avoid excessive truncation
                    combined_text = f"Summary: {summary}\n\nDetailed content: {content}"
                    processed_context = build_smart_context(summary, combined_text, question)
                    
                    context_parts.append(processed_context)
                    summary_parts.append(summary)
                else:
                    # Non-Chinese data: only use context
                    processed_context = build_smart_context("", content, question)
                    context_parts.append(processed_context)
            else:
                # English query: only use context
                processed_context = build_smart_context("", content, question)
                context_parts.append(processed_context)
        
        context_str = "\n\n".join(context_parts)
        summary_str = "\n\n".join(summary_parts) if summary_parts else None
        
        # Use generator to generate answer
        print("Using generator to generate answer...")
        
        # Determine prompt for generation
        if stock_prediction_checkbox and is_chinese_query:
            question_for_prompt = "请根据下方提供的该股票相关研报与数据，对该股票的下个月的涨跌，进行预测，请给出明确的答案，\"涨\" 或者 \"跌\"。同时给出这个股票下月的涨跌概率，分别是:极大，较大，中上，一般。\n\n请严格按照以下格式输出：\n这个股票的下月最终收益结果是:'涨/跌',上涨/下跌概率:极大/较大/中上/一般"
            print(f"Stock prediction mode activated, using instruction: {question_for_prompt[:100]}...")
        else:
            question_for_prompt = question
            print(f"Using original query as prompt")
        
        if is_chinese_query:
            # Chinese query: use Chinese prompt template, provide summary and context
            try:
                from xlm.components.prompt_templates.template_loader import template_loader
                prompt = template_loader.format_template(
                    "multi_stage_chinese_template",
                    summary=summary_str if summary_str else "No summary information",
                    context=context_str,
                    query=question_for_prompt
                )
                if prompt is None:
                    # Fall back to simple Chinese prompt
                    if summary_str:
                        prompt = f"Summary: {summary_str}\n\nFull context: {context_str}\n\nQuestion: {question_for_prompt}\n\nAnswer:"
                    else:
                        prompt = f"Based on the following context, answer the question: \n\n{context_str}\n\nQuestion: {question_for_prompt}\n\nAnswer:"
            except Exception as e:
                print(f"Chinese template loading failed: {e}, using simple Chinese prompt")
                if summary_str:
                    prompt = f"Summary: {summary_str}\n\nFull context: {context_str}\n\nQuestion: {question_for_prompt}\n\nAnswer:"
                else:
                    prompt = f"Based on the following context, answer the question: \n\n{context_str}\n\nQuestion: {question_for_prompt}\n\nAnswer:"
        else:
            # English query: use configured English template
            try:
                # Import English prompt processing function from RAG system
                from xlm.components.rag_system.rag_system import get_final_prompt_messages_english, _convert_messages_to_chatml
                
                # Use configured English template
                english_template = getattr(self.config.data, 'english_prompt_template', 'unified_english_template_no_think.txt')
                messages = get_final_prompt_messages_english(context_str, question_for_prompt, english_template)
                prompt = _convert_messages_to_chatml(messages)
                print(f"Using configured English template: {english_template}")
            except Exception as e:
                print(f"English template loading failed: {e}, using simple English prompt")
                prompt = f"Context: {context_str}\nQuestion: {question_for_prompt}\nAnswer:"
        
        try:
            # Use generator directly, no mixed decision
            if is_chinese_query:
                # Chinese query: use configured Chinese template
                chinese_template = getattr(self.config.data, 'chinese_prompt_template', 'multi_stage_chinese_template_with_fewshot.txt')
                print(f"Using configured Chinese template: {chinese_template}")
                
                # Chinese query: use prompt directly
                generated_responses = self.generator.generate(texts=[prompt])
                answer = generated_responses[0] if generated_responses else "Unable to generate answer"
            else:
                # English query: use prompt directly
                generated_responses = self.generator.generate(texts=[prompt])
                answer = generated_responses[0] if generated_responses else "Unable to generate answer"
                
                # English query: extract answer
                try:
                    from xlm.components.rag_system.rag_system import extract_final_answer_from_tag
                    extracted_answer = extract_final_answer_from_tag(answer)
                    if extracted_answer and extracted_answer.strip():
                        answer = extracted_answer
                        print(f"Answer extraction successful: {extracted_answer[:100]}...")
                    else:
                        print("Answer extraction failed, using original response")
                except Exception as e:
                    print(f"Answer extraction process failed: {e}, using original response")
        except Exception as e:
            print(f"Generator call failed: {e}")
            answer = "Generator call failed"
        
        # If stock prediction mode is enabled, remove "注意：" and everything after it
        if stock_prediction_checkbox and is_chinese_query:
            answer = self._clean_stock_prediction_answer(answer)
        
        return answer
    
    def _clean_stock_prediction_answer(self, answer: str) -> str:
        """
        Clean stock prediction answer, remove any '注意' and everything after it (including variations like 【注意】, [注意], 注意:, 注意：, etc.)
        """
        import re
        if not answer:
            return answer

        # Support various 'Notice' patterns
        match = re.search(r'[【\[]?注意[】\]]?[:：]', answer)
        if match:
            cleaned_answer = answer[:match.start()].strip()
            print(f"Cleaning stock prediction answer:")
            print(f"    Original answer: {answer}")
            print(f"    Cleaned answer: {cleaned_answer}")
            return cleaned_answer

        return answer
    
    def _format_and_return_result(self, answer: str, unique_docs: List[Tuple[DocumentWithMetadata, float]], 
                                 reranker_checkbox: bool, method: str) -> tuple[str, str]:
        """Format and return result"""
        # Print retrieval result
        if method == "统一RAG":
            method = "Unified RAG"
        print(f"\n=== Retrieved original context ({method}) ===")
        print(f"Retrieved {len(unique_docs)} unique documents")
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
            
            display_content = content[:800] + "..." if len(content) > 800 else content
            print(f"Document {i+1} (score: {score:.4f}): {display_content}")
        
        if len(unique_docs) > 5:
            print(f"... there are {len(unique_docs) - 5} more documents")
        
        # Print LLM response
        print(f"\n=== LLM response ===")
        print(f"Generated answer: {answer}")
        print(f"Retrieved documents: {len(unique_docs)}")
        
        # Add reranker information
        if reranker_checkbox and self.reranker:
            answer = f"[Reranker: Enabled] {answer}"
        else:
            answer = f"[Reranker: Disabled] {answer}"
        
        # Build UI-specific structure (only affects display, not RAG main process)
        ui_docs = []
        seen_ui_hashes = set()  # Add UI-level deduplication
        seen_table_ids = set()  # Add Table ID deduplication
        seen_paragraph_ids = set()  # Add Paragraph ID deduplication
        
        for doc, score in unique_docs:
            if getattr(doc.metadata, 'language', '') == 'chinese':
                doc_id = str(getattr(doc.metadata, 'origin_doc_id', '') or getattr(doc.metadata, 'doc_id', '')).strip()
                raw_context = self.docid2context.get(doc_id, "")
                if not raw_context:
                    raw_context = doc.content
                    print(f"[UI DEBUG] doc_id not hit: {doc_id}, using document content")
            else:
                raw_context = doc.content
            
            # Check content type and apply deduplication logic
            has_table_id = "Table ID:" in raw_context
            has_paragraph_id = "Paragraph ID:" in raw_context
            
            if has_table_id:
                # Table content or table+text content: use Table ID deduplication
                import re
                table_id_match = re.search(r'Table ID:\s*([a-f0-9-]+)', raw_context)
                if table_id_match:
                    table_id = table_id_match.group(1)
                    if table_id in seen_table_ids:
                        print(f"[UI DEBUG] Skip duplicate Table ID: {table_id}, content first 50 characters: {raw_context[:50]}...")
                        continue
                    seen_table_ids.add(table_id)
                    print(f"[UI DEBUG] Keep Table ID: {table_id}")
            elif has_paragraph_id:
                # Pure text content: use Paragraph ID deduplication
                import re
                paragraph_id_match = re.search(r'Paragraph ID:\s*([a-f0-9-]+)', raw_context)
                if paragraph_id_match:
                    paragraph_id = paragraph_id_match.group(1)
                    if paragraph_id in seen_paragraph_ids:
                        print(f"[UI DEBUG] Skip duplicate Paragraph ID: {paragraph_id}, content first 50 characters: {raw_context[:50]}...")
                        continue
                    seen_paragraph_ids.add(paragraph_id)
                    print(f"[UI DEBUG] Keep Paragraph ID: {paragraph_id}")
            
            # Check raw_context for deduplication
            context_hash = hash(raw_context)
            if context_hash in seen_ui_hashes:
                print(f"[UI DEBUG] Skip duplicate UI document, content first 50 characters: {raw_context[:50]}...")
                continue
            
            seen_ui_hashes.add(context_hash)
            preview_content = raw_context[:200] + "..." if len(raw_context) > 200 else raw_context
            ui_docs.append((doc, score, preview_content, raw_context))
        html_content = self._generate_clickable_context_html(ui_docs)
        
        print(f"=== Query processing completed ===\n")
        return answer, html_content

    def _generate_clickable_context_html(self, ui_docs):
        # ui_docs: List[Tuple[DocumentWithMetadata, float, str, str]]
        if not ui_docs:
            return "<p>No relevant documents retrieved.</p>"

        # Final deduplication check, ensure HTML has no duplicate content
        final_ui_docs = []
        seen_final_hashes = set()
        seen_final_table_ids = set()  # Add Table ID deduplication
        seen_final_paragraph_ids = set()  # Add Paragraph ID deduplication
        
        for doc, score, preview_content, raw_context in ui_docs:
            # Check content type and apply deduplication logic
            has_table_id = "Table ID:" in raw_context
            has_paragraph_id = "Paragraph ID:" in raw_context
            
            if has_table_id:
                # Table content or table+text content: use Table ID deduplication
                import re
                table_id_match = re.search(r'Table ID:\s*([a-f0-9-]+)', raw_context)
                if table_id_match:
                    table_id = table_id_match.group(1)
                    if table_id in seen_final_table_ids:
                        print(f"[HTML DEBUG] Skip duplicate Table ID: {table_id}, content first 50 characters: {raw_context[:50]}...")
                        continue
                    seen_final_table_ids.add(table_id)
                    print(f"[HTML DEBUG] Keep Table ID: {table_id}")
            elif has_paragraph_id:
                # Pure text content: use Paragraph ID deduplication
                import re
                paragraph_id_match = re.search(r'Paragraph ID:\s*([a-f0-9-]+)', raw_context)
                if paragraph_id_match:
                    paragraph_id = paragraph_id_match.group(1)
                    if paragraph_id in seen_final_paragraph_ids:
                        print(f"[HTML DEBUG] Skip duplicate Paragraph ID: {paragraph_id}, content first 50 characters: {raw_context[:50]}...")
                        continue
                    seen_final_paragraph_ids.add(paragraph_id)
                    print(f"[HTML DEBUG] Keep Paragraph ID: {paragraph_id}")
            
            # Use hash of raw_context for final deduplication
            context_hash = hash(raw_context)
            if context_hash in seen_final_hashes:
                print(f"[HTML DEBUG] Skip duplicate HTML document, content first 50 characters: {raw_context[:50]}...")
                continue
            
            seen_final_hashes.add(context_hash)
            final_ui_docs.append((doc, score, preview_content, raw_context))

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
        
        for i, (doc, score, preview_content, raw_context) in enumerate(final_ui_docs):
            short_content = preview_content
            full_content = raw_context
            def html_escape(text):
                return text.replace('&', '&amp;').replace('<', '&lt;').replace('>', '&gt;').replace('\n', '<br>').replace('  ', '&nbsp;&nbsp;')
            html_parts.append(f"""
            <div class='content-section'>
                <div class='header'>
                    <strong style='color: #333;'>Document {i+1}</strong>
                    <span class='score'>score: {score:.4f}</span>
                </div>
                <div class='short-content' id='short_{i}'>
                    <p>{html_escape(short_content)}</p>
                    <button class='expand-btn' onclick='document.getElementById(\"short_{i}\").style.display=\"none\"; document.getElementById(\"full_{i}\").style.display=\"block\";'>
                        Read more
                    </button>
                </div>
                <div class='full-content' id='full_{i}' style='display: none;'>
                    <p>{html_escape(full_content)}</p>
                    <button class='collapse-btn' onclick='document.getElementById(\"full_{i}\").style.display=\"none\"; document.getElementById(\"short_{i}\").style.display=\"block\";'>
                        Show less 
                    </button>
                </div>
            </div>
            """)
        html_parts.append("</div>")
        return ''.join(html_parts)
    
    def _detect_data_source(self, question: str, language: str) -> str:
        """Detect data source type"""
        if language == 'zh':
            return "AlphaFin"
        else:
            return "TAT_QA"
    
    def launch(self, share: bool = False):
        """Launch UI interface"""
        self.interface.launch(share=share)
    
    def _chunk_documents(self, documents: List[DocumentWithMetadata], chunk_size: int = 512, overlap: int = 50) -> List[DocumentWithMetadata]:
        """
        Split documents into smaller chunks
        
        Args:
            documents: Original document list
            chunk_size: Chunk size (number of characters)
            overlap: Overlap characters
            
        Returns:
            Chunked document list
        """
        chunked_docs = []
        
        for doc in documents:
            content = doc.content
            if len(content) <= chunk_size:
                # Document is too short, no need to chunk
                chunked_docs.append(doc)
            else:
                # Document is too long, need to chunk
                start = 0
                chunk_id = 0
                
                while start < len(content):
                    end = start + chunk_size
                    
                    # Ensure not to break in the middle of a word
                    if end < len(content):
                        # Try to break at a period, comma, or space
                        for i in range(end, max(start + chunk_size - 100, start), -1):
                            if content[i] in '.。，, ':
                                end = i + 1
                                break
                    
                    chunk_content = content[start:end].strip()
                    
                    if chunk_content:  # Ensure chunk is not empty
                        # Create new document metadata
                        chunk_metadata = DocumentMetadata(
                            source=f"{doc.metadata.source}_chunk_{chunk_id}",
                            created_at=doc.metadata.created_at,
                            author=doc.metadata.author,
                            language=doc.metadata.language,
                            origin_doc_id=getattr(doc.metadata, 'doc_id', None) if doc.metadata.language == 'chinese' else None
                        )
                        
                        # Create new document object
                        chunk_doc = DocumentWithMetadata(
                            content=chunk_content,
                            metadata=chunk_metadata
                        )
                        
                        chunked_docs.append(chunk_doc)
                        chunk_id += 1
                    
                    # Move to next chunk, consider overlap
                    start = end - overlap
                    if start >= len(content):
                        break
        
        return chunked_docs 
    
    def _chunk_documents_advanced(self, documents: List[DocumentWithMetadata]) -> List[DocumentWithMetadata]:
        """
        Use advanced chunk logic in finetune_chinese_encoder.py to process Chinese documents
        and integrate table text processing in finetune_encoder.py
        """
        import re
        import json
        import ast
        
        def extract_unit_from_paragraph(paragraphs):
            """Extract numeric units from paragraphs"""
            for para in paragraphs:
                text = para.get("text", "") if isinstance(para, dict) else para
                match = re.search(r'dollars in (millions|billions)|in (millions|billions)', text, re.IGNORECASE)
                if match:
                    unit = match.group(1) or match.group(2)
                    if unit:
                        return unit.lower().replace('s', '') + " USD"
            return ""

        def table_to_natural_text(table_dict, caption="", unit_info=""):
            """Convert table to natural language description"""
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
            
            # Process report format
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

            # Process dictionary format
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

            # Process pure text
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
            # Check if contains table data
            content = doc.content
            
            # Try to parse as JSON, check if contains table structure
            try:
                parsed_content = json.loads(content)
                if isinstance(parsed_content, dict) and 'tables' in parsed_content:
                    # Process document containing tables
                    paragraphs = parsed_content.get('paragraphs', [])
                    tables = parsed_content.get('tables', [])
                    
                    # Extract unit information
                    unit_info = extract_unit_from_paragraph(paragraphs)
                    
                    # Process paragraphs
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
                    
                    # Process tables
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
                    
                    continue  # Processed table data, skip subsequent processing
                    
            except (json.JSONDecodeError, TypeError):
                pass  # Not JSON format, continue using original chunk logic
            
            # Use original advanced chunk logic
            chunks = convert_json_context_to_natural_language_chunks(content)
            
            for i, chunk_content in enumerate(chunks):
                if chunk_content.strip():
                    chunk_metadata = DocumentMetadata(
                        source=f"{doc.metadata.source}_advanced_chunk_{i}",
                        created_at=doc.metadata.created_at,
                        author=doc.metadata.author,
                        language=doc.metadata.language,
                        origin_doc_id=getattr(doc.metadata, 'doc_id', None) if doc.metadata.language == 'chinese' else None
                    )
                    
                    chunk_doc = DocumentWithMetadata(
                        content=chunk_content,
                        metadata=chunk_metadata
                    )
                    
                    chunked_docs.append(chunk_doc)
        
        return chunked_docs
    
    def _chunk_documents_simple(self, documents: List[DocumentWithMetadata], chunk_size: int = 512, overlap: int = 50) -> List[DocumentWithMetadata]:
        """
        Simple document chunking method, for English documents
        """
        chunked_docs = []
        
        for doc in documents:
            content = doc.content
            if len(content) <= chunk_size:
                # Document is too short, no need to chunk
                chunked_docs.append(doc)
            else:
                # Document is too long, need to chunk
                start = 0
                chunk_id = 0
                
                while start < len(content):
                    end = start + chunk_size
                    
                    # Ensure not to break in the middle of a word
                    if end < len(content):
                        # Try to break at a period, comma, or space
                        for i in range(end, max(start + chunk_size - 100, start), -1):
                            if content[i] in '.。，, ':
                                end = i + 1
                                break
                    
                    chunk_content = content[start:end].strip()
                    
                    if chunk_content:  # Ensure chunk is not empty
                        # Create new document metadata
                        chunk_metadata = DocumentMetadata(
                            source=f"{doc.metadata.source}_simple_chunk_{chunk_id}",
                            created_at=doc.metadata.created_at,
                            author=doc.metadata.author,
                            language=doc.metadata.language
                        )
                        
                        # Create new document object
                        chunk_doc = DocumentWithMetadata(
                            content=chunk_content,
                            metadata=chunk_metadata
                        )
                        
                        chunked_docs.append(chunk_doc)
                        chunk_id += 1
                    
                    # Move to next chunk, consider overlap
                    start = end - overlap
                    if start >= len(content):
                        break
        
        return chunked_docs 