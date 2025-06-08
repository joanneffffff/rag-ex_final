import os
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
import psutil
from typing import Optional
import warnings

warnings.filterwarnings("ignore")

# 设置 Hugging Face 缓存目录到 D 盘
os.environ['HF_HOME'] = 'D:/AI/huggingface'
os.environ['TRANSFORMERS_CACHE'] = 'D:/AI/huggingface/transformers'

class LocalLLM:
    def __init__(
        self,
        model_name: str = "facebook/opt-125m",
        device: Optional[str] = None,
        temperature: float = 0.7,
        max_new_tokens: int = 50,
        top_p: float = 0.9,
        cache_dir: str = "D:/AI/huggingface"  # 自定义缓存目录
    ):
        """
        初始化本地LLM
        Args:
            model_name: 模型名称
            device: 设备 (cpu/cuda)
            temperature: 温度参数
            max_new_tokens: 最大生成token数
            top_p: top-p采样参数
            cache_dir: 模型缓存目录
        """
        self.model_name = model_name
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.temperature = temperature
        self.max_new_tokens = max_new_tokens
        self.top_p = top_p
        self.cache_dir = cache_dir
        
        # 创建缓存目录
        os.makedirs(self.cache_dir, exist_ok=True)
        
        # 打印系统信息
        self._print_system_info()
        
        # 加载模型和tokenizer
        self._load_model_and_tokenizer()
        
    def _print_system_info(self):
        """打印系统信息"""
        print(f"系统信息:")
        print(f"- PyTorch版本: {torch.__version__}")
        print(f"- 使用设备: {self.device}")
        print(f"- CUDA是否可用: {torch.cuda.is_available()}")
        print(f"- 模型缓存目录: {self.cache_dir}")
        self._print_memory_usage()
        
    def _print_memory_usage(self):
        """打印内存使用情况"""
        process = psutil.Process(os.getpid())
        print(f"当前内存使用: {process.memory_info().rss / 1024 / 1024:.2f} MB")
        
    def _load_model_and_tokenizer(self):
        """加载模型和tokenizer"""
        print(f"\n1. 加载tokenizer... 模型: {self.model_name}")
        self.tokenizer = AutoTokenizer.from_pretrained(
            self.model_name,
            cache_dir=self.cache_dir,
            use_fast=True
        )
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        self._print_memory_usage()
        
        print("\n2. 加载模型...")
        self.model = AutoModelForCausalLM.from_pretrained(
            self.model_name,
            cache_dir=self.cache_dir,
            torch_dtype=torch.float32,
            device_map="auto",
            low_cpu_mem_usage=True,
            use_cache=True
        )
        self._print_memory_usage()
        
    def generate(self, prompt: str) -> str:
        """
        生成回答
        Args:
            prompt: 输入提示
        Returns:
            生成的回答
        """
        # 准备输入
        inputs = self.tokenizer(
            prompt, 
            return_tensors="pt",
            truncation=True,
            max_length=512
        )
        
        # 生成回答
        print("\n生成回答中...")
        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=self.max_new_tokens,
                temperature=self.temperature,
                top_p=self.top_p,
                num_beams=1,
                pad_token_id=self.tokenizer.pad_token_id,
                eos_token_id=self.tokenizer.eos_token_id,
                use_cache=True,
                do_sample=True
            )
        
        # 解码输出
        response = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        return response

def test_simple():
    """简单测试"""
    print("\n模型将下载到 D:/AI/huggingface 目录")
    
    llm = LocalLLM()
    
    # 简单测试用例
    test_text = "What is Python programming language? Answer in one sentence:"
    
    print(f"\n输入: {test_text}")
    response = llm.generate(test_text)
    print(f"输出: {response}")
    llm._print_memory_usage()

if __name__ == "__main__":
    test_simple() 