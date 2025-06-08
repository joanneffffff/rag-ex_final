from typing import List
from xlm.components.generator.llm_generator import LLMGenerator
from xlm.modules.perturber.perturber import Perturber


class LLMBasedPerturber(Perturber):
    def __init__(
        self,
        generator: LLMGenerator,
        prompt_template: str,
    ):
        """
        基于LLM的扰动器
        Args:
            generator: LLM生成器实例
            prompt_template: 提示模板，使用{text}作为占位符
        """
        self.generator = generator
        self.prompt_template = prompt_template

    def perturb(self, text: str, features: List[str]) -> List[str]:
        """
        对文本进行扰动
        Args:
            text: 原始文本
            features: 要扰动的特征列表
        Returns:
            扰动后的文本列表
        """
        perturbations = []
        prompts = []
        
        # 为每个特征生成提示
        for feature in features:
            prompt = self.prompt_template.format(text=feature)
            prompts.append(prompt)

        # 使用LLM生成替换文本
        responses = self.generator.generate(texts=prompts)

        # 替换原文中的特征
        for response, feature in zip(responses, features):
            perturbations.append(text.replace(feature, response.strip()).strip())
            
        return perturbations
