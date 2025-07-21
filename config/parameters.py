"""
Configuration parameters for the RAG system.
"""

import os
import platform
import torch
from dataclasses import dataclass, field
from typing import Dict, Optional, List

# --- Platform-Aware Path Configuration ---
# Set the default Hugging Face cache directory based on the operating system.
# You can modify the Windows path here if needed (e.g., "D:/AI/huggingface").
WINDOWS_CACHE_DIR = "M:/huggingface"
LINUX_CACHE_DIR = "/users/sgjfei3/data/huggingface"

DEFAULT_CACHE_DIR = WINDOWS_CACHE_DIR if platform.system() == "Windows" else LINUX_CACHE_DIR

# Set the embedding cache directory to models/embedding_cache
PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
EMBEDDING_CACHE_DIR = os.path.abspath(os.path.join(PROJECT_ROOT, "../models/embedding_cache"))
GENERATOR_CACHE_DIR = DEFAULT_CACHE_DIR
RERANKER_CACHE_DIR = DEFAULT_CACHE_DIR

@dataclass
class EncoderConfig:
    # Path to the fine-tuned Chinese model
    chinese_model_path: str = "models/alphafin_encoder_finetuned_1epoch"
    # Path to the fine-tuned English model
    english_model_path: str = "models/finetuned_tatqa_mixed_enhanced"
    cache_dir: str = EMBEDDING_CACHE_DIR
    device: Optional[str] = "cuda:0"  # Encoder uses cuda:0
    batch_size: int = 64  # Increased from 32 to 64 to utilize sufficient GPU memory for faster processing
    max_length: int = 512

@dataclass
class RerankerConfig:
    model_name: str = "Qwen/Qwen3-Reranker-0.6B"
    cache_dir: str = RERANKER_CACHE_DIR
    device: Optional[str] = "cuda:0"  # Reranker uses cuda:1 to avoid conflict with encoder
    use_quantization: bool = True
    quantization_type: str = "4bit"  # Changed to 4bit quantization to save GPU memory
    use_flash_attention: bool = False  # Disable Flash Attention optimization
    batch_size: int = 1  # Reduced to 1 to avoid out-of-memory
    enabled: bool = True  # Whether to enable reranker

@dataclass
class RetrieverConfig:
    use_faiss: bool = True  # Changed default to True for efficiency
    num_threads: int = 4
    batch_size: int = 64  # Increased from 32 to 64 to utilize sufficient GPU memory for faster processing
    use_gpu: bool = torch.cuda.is_available() # Dynamically set default based on hardware
    max_context_length: int = 100
    # Rerank related config
    retrieval_top_k: int = 100  # FAISS retrieval top-k chinese:20/english:100
    rerank_top_k: int = 10      # Top-k after rerank, increased from 10 to 20
    # Prefilter config
    use_prefilter: bool = True  # Whether to use prefiltering (automatically enables stock code and company name mapping)
    # Embedding cache control
    use_existing_embedding_index: bool = True  # True=use existing cache, False=force recompute embedding
    max_alphafin_chunks: int = 1000000  # Limit the number of AlphaFin data chunks

@dataclass
class DataConfig:
    data_dir: str = "data"  # Unified root data directory
    max_samples: int = -1  # -1 means load all data, 500 means limit sample count
    # Data path config
    chinese_data_path: str = "data/alphafin/alphafin_final_clean.json"  # Chinese data path
    english_data_path: str = "data/unified/tatqa_knowledge_base_combined.jsonl"     # English data path
    prompt_template_dir: str = "data/prompt_templates"  # Prompt template directory
    
    # Language-specific config
    chinese_prompt_template: str = "multi_stage_chinese_template_with_fewshot.txt"  # Chinese prompt template
    english_prompt_template: str = "unified_english_template_no_think.txt"  # English prompt template

@dataclass
class ModalityConfig:
    text_weight: float = 1.0
    table_weight: float = 1.0
    time_series_weight: float = 1.0
    combine_method: str = "weighted_sum"  # or "concatenate", "attention"

@dataclass
class SystemConfig:
    memory_limit: int = 16  # in GB
    log_level: str = "INFO"
    temp_dir: str = "temp"

@dataclass
class GeneratorConfig:
    model_name: str = "SUFE-AIFLM-Lab/Fin-R1"  # Shanghai University of Finance and Economics financial reasoning LLM, optimized for finance
    cache_dir: str = GENERATOR_CACHE_DIR
    device: Optional[str] = "cuda:1"
    
    # Model-specific config - consistent with test_clean.py
    use_quantization: bool = True  # Enable quantization
    quantization_type: str = "4bit"  # Use 4bit quantization to save memory
    use_flash_attention: bool = False  # Enable Flash Attention optimization
    max_new_tokens: int = 8196  # Increased to 4096 for more complete answers
    # For Fin-R1 model, these parameters are ignored, but kept for compatibility with other models
    temperature: float = 0.1  # Not used by Fin-R1
    top_p: float = 0.7  # Not used by Fin-R1
    do_sample: bool = False  # Use deterministic generation, consistent with test_clean.py
    repetition_penalty: float = 1.1  # Consistent with test_clean.py
    pad_token_id: int = 0  # Padding token ID
    eos_token_id: int = 151645  # End token ID for Fin-R1
    
    # Sentence completion detection config
    enable_sentence_completion: bool = False  # Temporarily disable sentence completion detection to solve stalling issue
    max_completion_attempts: int = 2  # Max retry attempts
    token_increment: int = 100  # Token increment per retry
    max_total_tokens: int = 5000  # Increased to 5000 to support longer generation
    
    # Language consistency check config
    enable_language_consistency_check: bool = True  # Whether to enable language consistency check
    force_chinese_company_names: bool = True  # Force to keep Chinese company names
    enable_company_name_correction: bool = True  # Whether to enable company name correction
    
    # Answer length control
    max_response_chars: int = 1200  # Increase max answer characters to support longer answers
    enable_response_length_limit: bool = False  # Whether to enable answer length limit, set to False to test answer quality
    
    # Performance optimization config
    enable_fast_mode: bool = True  # Enable fast mode, reduce generation parameters
    enable_performance_monitoring: bool = True  # Enable performance monitoring

    # Timeout and performance config
    max_generation_time: int = 300  # Generation timeout (seconds), increased to 5 minutes

@dataclass
class Config:
    cache_dir: str = DEFAULT_CACHE_DIR # Global cache directory - 改为DEFAULT_CACHE_DIR
    encoder: EncoderConfig = field(default_factory=EncoderConfig)
    reranker: RerankerConfig = field(default_factory=RerankerConfig)
    retriever: RetrieverConfig = field(default_factory=RetrieverConfig)
    data: DataConfig = field(default_factory=DataConfig)
    modality: ModalityConfig = field(default_factory=ModalityConfig)
    system: SystemConfig = field(default_factory=SystemConfig)
    generator: GeneratorConfig = field(default_factory=GeneratorConfig)
    prompt_template_dir: str = "data/prompt_templates"  # prompt模板目录

    def __post_init__(self):
        # Propagate the global cache_dir to other configs if they have it
        if hasattr(self.encoder, 'cache_dir'):
            self.encoder.cache_dir = EMBEDDING_CACHE_DIR  # Encoder uses embedding cache directory
        if hasattr(self.reranker, 'cache_dir'):
            self.reranker.cache_dir = RERANKER_CACHE_DIR  # Reranker uses DEFAULT_CACHE_DIR
        if hasattr(self.generator, 'cache_dir'):
            self.generator.cache_dir = GENERATOR_CACHE_DIR  # Generator uses DEFAULT_CACHE_DIR

    @classmethod
    def load_environment_config(cls) -> 'Config':
        """Load configuration based on environment"""
        # Example of environment-based config loading
        if os.getenv("PRODUCTION") == "1":
            return cls(
                encoder=EncoderConfig(
                    chinese_model_path="models/finetuned_alphafin_zh_optimized",
                    english_model_path="models/finetuned_finbert_tatqa",
                    batch_size=64
                ),
                retriever=RetrieverConfig(
                    use_faiss=True,
                    num_threads=8,
                    use_gpu=True
                ),
                reranker=RerankerConfig(
                    enabled=True,
                    use_quantization=True
                ),
                system=SystemConfig(
                    memory_limit=32,
                    log_level="WARNING"
                )
            )
        return cls()  # Default development config

# Default configuration instance
config = Config() 