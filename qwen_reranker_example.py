#!/usr/bin/env python3
"""
改进的Qwen3-Reranker使用示例
包含错误处理、性能优化和设备管理
"""

import torch
import warnings
from transformers import AutoTokenizer, AutoModelForCausalLM
from typing import List, Tuple, Optional
import logging

# 设置日志
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# 忽略权重初始化警告
warnings.filterwarnings("ignore", message="Some weights of.*were not initialized.*")

class QwenReranker:
    def __init__(self, 
                 model_name: str = "Qwen/Qwen3-Reranker-0.6B",
                 device: Optional[str] = None,
                 max_length: int = 8192,
                 use_flash_attention: bool = True):
        """
        初始化Qwen3-Reranker
        
        Args:
            model_name: 模型名称
            device: 设备 (cuda/cpu/auto)
            max_length: 最大序列长度
            use_flash_attention: 是否使用flash attention
        """
        self.model_name = model_name
        self.max_length = max_length
        
        # 设备选择
        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"
        self.device = device
        
        logger.info(f"正在加载Qwen3-Reranker: {model_name}")
        logger.info(f"设备: {self.device}")
        
        # 加载tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(
            model_name, 
            padding_side='left',
            trust_remote_code=True
        )
        
        # 加载模型
        model_kwargs = {
            "trust_remote_code": True,
            "torch_dtype": torch.float16 if device.startswith('cuda') else torch.float32
        }
        
        if use_flash_attention and device.startswith('cuda'):
            try:
                model_kwargs["attn_implementation"] = "flash_attention_2"
                logger.info("启用Flash Attention 2")
            except Exception as e:
                logger.warning(f"Flash Attention 2不可用: {e}")
        
        self.model = AutoModelForCausalLM.from_pretrained(
            model_name, 
            **model_kwargs
        ).eval()
        
        # 移动到设备
        self.model = self.model.to(self.device)
        
        # 设置特殊token
        self.token_false_id = self.tokenizer.convert_tokens_to_ids("no")
        self.token_true_id = self.tokenizer.convert_tokens_to_ids("yes")
        
        # 设置prompt模板
        self.prefix = "<|im_start|>system\nJudge whether the Document meets the requirements based on the Query and the Instruct provided. Note that the answer can only be \"yes\" or \"no\".<|im_end|>\n<|im_start|>user\n"
        self.suffix = "<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n"
        self.prefix_tokens = self.tokenizer.encode(self.prefix, add_special_tokens=False)
        self.suffix_tokens = self.tokenizer.encode(self.suffix, add_special_tokens=False)
        
        logger.info("Qwen3-Reranker加载完成")
    
    def format_instruction(self, instruction: str, query: str, doc: str) -> str:
        """格式化指令"""
        if instruction is None:
            instruction = 'Given a web search query, retrieve relevant passages that answer the query'
        return f"<Instruct>: {instruction}\n<Query>: {query}\n<Document>: {doc}"
    
    def process_inputs(self, pairs: List[str]) -> dict:
        """处理输入文本"""
        try:
            inputs = self.tokenizer(
                pairs, 
                padding=False, 
                truncation='longest_first',
                return_attention_mask=False, 
                max_length=self.max_length - len(self.prefix_tokens) - len(self.suffix_tokens)
            )
            
            # 添加prefix和suffix
            for i, ele in enumerate(inputs['input_ids']):
                inputs['input_ids'][i] = self.prefix_tokens + ele + self.suffix_tokens
            
            # 填充
            inputs = self.tokenizer.pad(
                inputs, 
                padding=True, 
                return_tensors="pt", 
                max_length=self.max_length
            )
            
            # 移动到设备
            for key in inputs:
                inputs[key] = inputs[key].to(self.device)
            
            return inputs
            
        except Exception as e:
            logger.error(f"处理输入时出错: {e}")
            raise
    
    @torch.no_grad()
    def compute_scores(self, inputs: dict) -> List[float]:
        """计算相关性分数"""
        try:
            batch_scores = self.model(**inputs).logits[:, -1, :]
            true_vector = batch_scores[:, self.token_true_id]
            false_vector = batch_scores[:, self.token_false_id]
            batch_scores = torch.stack([false_vector, true_vector], dim=1)
            batch_scores = torch.nn.functional.log_softmax(batch_scores, dim=1)
            scores = batch_scores[:, 1].exp().tolist()
            return scores
            
        except Exception as e:
            logger.error(f"计算分数时出错: {e}")
            raise
    
    def rerank(self, 
               queries: List[str], 
               documents: List[str], 
               instruction: Optional[str] = None) -> List[float]:
        """
        重排序文档
        
        Args:
            queries: 查询列表
            documents: 文档列表
            instruction: 指令（可选）
            
        Returns:
            相关性分数列表
        """
        if len(queries) != len(documents):
            raise ValueError("查询和文档数量必须相同")
        
        # 格式化输入
        pairs = [self.format_instruction(instruction, query, doc) 
                for query, doc in zip(queries, documents)]
        
        # 处理输入
        inputs = self.process_inputs(pairs)
        
        # 计算分数
        scores = self.compute_scores(inputs)
        
        return scores
    
    def rerank_batch(self, 
                    query: str, 
                    documents: List[str], 
                    instruction: Optional[str] = None,
                    batch_size: int = 8) -> List[float]:
        """
        批量重排序文档
        
        Args:
            query: 单个查询
            documents: 文档列表
            instruction: 指令（可选）
            batch_size: 批次大小
            
        Returns:
            相关性分数列表
        """
        scores = []
        
        for i in range(0, len(documents), batch_size):
            batch_docs = documents[i:i + batch_size]
            batch_queries = [query] * len(batch_docs)
            
            batch_scores = self.rerank(batch_queries, batch_docs, instruction)
            scores.extend(batch_scores)
        
        return scores

def main():
    """示例用法"""
    # 初始化重排序器
    reranker = QwenReranker(
        device="cuda" if torch.cuda.is_available() else "cpu",
        use_flash_attention=True
    )
    
    # 示例数据
    task = 'Given a web search query, retrieve relevant passages that answer the query'
    
    queries = [
        "What is the capital of China?",
        "Explain gravity",
        "How does photosynthesis work?",
    ]
    
    documents = [
        "The capital of China is Beijing.",
        "Gravity is a force that attracts two bodies towards each other. It gives weight to physical objects and is responsible for the movement of planets around the sun.",
        "Photosynthesis is the process by which plants convert sunlight into energy.",
    ]
    
    # 重排序
    try:
        scores = reranker.rerank(queries, documents, task)
        
        print("重排序结果:")
        for i, (query, doc, score) in enumerate(zip(queries, documents, scores)):
            print(f"{i+1}. 查询: {query}")
            print(f"   文档: {doc[:100]}...")
            print(f"   分数: {score:.4f}")
            print()
            
    except Exception as e:
        logger.error(f"重排序失败: {e}")

if __name__ == "__main__":
    main() 