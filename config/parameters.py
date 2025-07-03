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
    # 中文微调模型路径
    chinese_model_path: str = "models/finetuned_alphafin_zh_optimized"
    # 英文微调模型路径
    english_model_path: str = "models/finetuned_finbert_tatqa"
    cache_dir: str = EMBEDDING_CACHE_DIR
    device: Optional[str] = "cuda:0"  # 编码器使用cuda:0
    batch_size: int = 32
    max_length: int = 512

@dataclass
class RerankerConfig:
    model_name: str = "Qwen/Qwen3-Reranker-0.6B"
    cache_dir: str = RERANKER_CACHE_DIR
    device: Optional[str] = "cuda:0"  # 重排序器使用cuda:0
    use_quantization: bool = True
    quantization_type: str = "4bit"  # 改为4bit量化以节省GPU内存
    batch_size: int = 4
    enabled: bool = True  # 是否启用重排序器

@dataclass
class RetrieverConfig:
    use_faiss: bool = True  # Changed default to True for efficiency
    num_threads: int = 4
    batch_size: int = 32
    use_gpu: bool = torch.cuda.is_available() # Dynamically set default based on hardware
    max_context_length: int = 100
    # 重排序相关配置
    retrieval_top_k: int = 100  # FAISS检索的top-k
    rerank_top_k: int = 10      # 重排序后的top-k，从10增加到20
    # 新增参数data/alphafin/alphafin_merged_generated_qa.json
    use_existing_embedding_index: bool = True  # 强制重新计算embedding，确保生成中文embedding
    max_alphafin_chunks: int = 1000000  # 限制AlphaFin数据chunk数量

@dataclass
class DataConfig:
    data_dir: str = "data"  # Unified root data directory
    max_samples: int = -1  # -1表示加载所有数据，500表示限制样本数
    # 数据路径配置
    chinese_data_path: str = "data/alphafin/alphafin_merged_generated_qa.json"  # 中文数据路径
    english_data_path: str = "evaluate_mrr/tatqa_train_qc.jsonl"     # 英文数据路径
    prompt_template_dir: str = "data/prompt_templates"  # prompt模板目录

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
    # 可选的生成器模型
    # model_name: str = "Qwen/Qwen2-1.5B-Instruct"  # 原始小模型
    # model_name: str = "Qwen/Qwen3-8B"  # Qwen3-8B基础版本，更大的模型，替代Fin-R1
    model_name: str = "SUFE-AIFLM-Lab/Fin-R1"  # 上海财经大学金融推理大模型，专门针对金融领域优化
    cache_dir: str = GENERATOR_CACHE_DIR
    device: Optional[str] = "cuda:1"  # 改为cuda:1，避免与其他组件冲突
    
    # 模型特定配置 - 与test_clean.py保持一致
    use_quantization: bool = True  # 是否使用量化
    quantization_type: str = "4bit"  # 使用4bit量化以节省内存
    max_new_tokens: int = 150  # 与test_clean.py一致
    # 对于Fin-R1模型，这些参数会被忽略，但保留在配置中以防其他模型使用
    temperature: float = 0.1  # Fin-R1不使用此参数
    top_p: float = 0.7  # Fin-R1不使用此参数
    do_sample: bool = False  # 使用确定性生成，与test_clean.py一致
    repetition_penalty: float = 1.1  # 与test_clean.py一致
    pad_token_id: int = 0  # 填充token ID
    eos_token_id: int = 151645  # Fin-R1的结束token ID
    
    # 句子完整性检测配置
    enable_sentence_completion: bool = False  # 暂时禁用句子完整性检测以解决停滞问题
    max_completion_attempts: int = 2  # 最大重试次数
    token_increment: int = 100  # 每次重试增加的token数量
    max_total_tokens: int = 1000  # 从1000增加到1500
    
    # 语言一致性检查配置
    enable_language_consistency_check: bool = True  # 是否启用语言一致性检查
    force_chinese_company_names: bool = True  # 强制保持中文公司名称
    enable_company_name_correction: bool = True  # 是否启用公司名称修正
    
    # 回答长度控制
    max_response_chars: int = 600  # 最大回答字符数，避免过长回答
    enable_response_length_limit: bool = False  # 是否启用回答长度限制，设为False以测试回答质量
    
    # 性能优化配置
    enable_fast_mode: bool = True  # 启用快速模式，减少生成参数
    enable_performance_monitoring: bool = True  # 启用性能监控

    # 超时和性能配置
    max_generation_time: int = 30  # 生成超时时间（秒）

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
            self.encoder.cache_dir = EMBEDDING_CACHE_DIR  # 编码器使用embedding缓存目录
        if hasattr(self.reranker, 'cache_dir'):
            self.reranker.cache_dir = RERANKER_CACHE_DIR  # 重排序器使用DEFAULT_CACHE_DIR
        if hasattr(self.generator, 'cache_dir'):
            self.generator.cache_dir = GENERATOR_CACHE_DIR  # 生成器使用DEFAULT_CACHE_DIR

    @classmethod
    def load_environment_config(cls) -> 'Config':
        """Load configuration based on environment"""
        # Example of environment-based config loading
        if os.getenv("PRODUCTION") == "1":
            return cls(
                encoder=EncoderConfig(
                    # chinese_model_path="models/finetuned_alphafin_zh",
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