from typing import List
import requests

from xlm.components.generator.generator import Generator
from xlm.registry import DEFAULT_LMS_ENDPOINT


class LLMGenerator(Generator):
    def __init__(
        self,
        model_name: str,
        endpoint: str = DEFAULT_LMS_ENDPOINT,
        split_lines: bool = False,
        max_new_tokens: int = 100,
        temperature: float = 0.7,
    ):
        """
        初始化LLM生成器
        Args:
            model_name: 模型名称
            endpoint: LMS服务端点
            split_lines: 是否按行分割输出
            max_new_tokens: 最大生成token数
            temperature: 温度参数
        """
        super().__init__(model_name=model_name)
        self.endpoint = endpoint
        self.split_lines = split_lines
        self.max_new_tokens = max_new_tokens
        self.temperature = temperature

    def generate(self, texts: List[str]) -> List[str]:
        """
        使用远程LLM服务生成文本
        Args:
            texts: 输入文本列表
        Returns:
            生成的文本列表
        """
        url = f"{self.endpoint}/generate"
        
        payload = {
            "texts": texts,
            "model_name": self.model_name,
            "split_lines": self.split_lines,
            "max_new_tokens": self.max_new_tokens,
            "temperature": self.temperature
        }
        
        try:
            response = requests.post(url, json=payload)
            response.raise_for_status()
            result = response.json()
            return result["generated_texts"] if "generated_texts" in result else result
        except requests.exceptions.RequestException as e:
            print(f"Error calling LMS service: {e}")
            return ["Error: Failed to generate response"] * len(texts)
