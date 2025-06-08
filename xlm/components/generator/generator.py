from abc import ABC, abstractmethod
from typing import List


class Generator(ABC):
    def __init__(self, model_name: str):
        self.model_name = model_name

    @abstractmethod
    def generate(self, texts: List[str]) -> List[str]:
        """
        生成文本
        Args:
            texts: 输入文本列表
        Returns:
            生成的文本列表
        """
        pass
