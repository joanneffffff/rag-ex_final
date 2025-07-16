import re
from typing import Dict, Any, List, Optional
from .base_perturber import BasePerturber # 确保正确导入BasePerturber

class TrendPerturber(BasePerturber):
    target = "both"
    def perturb_context(self, text):
        return self.perturb(text)
    def perturb_prompt(self, text):
        return self.perturb(text)
    """
    Perturbs financial trend terms in Chinese text by replacing them with antonyms.
    For each identifiable trend term instance, it generates a separate perturbed text.
    """
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        super().__init__(config)
        # Define a mapping of Chinese trend terms to their antonyms
        self.trend_map = {
            "上升": "下降", "上涨": "下跌", "增长": "减少", "提升": "降低", "增加": "减少",
            "下降": "上升", "下跌": "上涨", "减少": "增长", "降低": "提升",
            "好转": "恶化", "改善": "恶化", "积极": "消极", "盈利": "亏损",
            "扩张": "收缩", "持续增长": "持续下滑", "稳步增长": "显著下降",
            "强劲": "疲软", "高于": "低于", "优于": "劣于", "领先": "落后",
            "增加率": "减少率", "上升趋势": "下降趋势", "增长趋势": "减少趋势"
        }
        
        # 预编译中文趋势词的正则表达式模式
        # 中文使用字符匹配，不使用\b词边界
        self.compiled_patterns = {
            re.compile(r'(' + re.escape(k) + r')'): v 
            for k, v in self.trend_map.items()
        }

    @property
    def perturber_name(self) -> str:
        return "trend"

    def perturb(self, original_text: str, features: Optional[List[str]] = None) -> List[Dict[str, Any]]:
        """
        Perturbs financial trend terms in the original Chinese text.
        It finds all predefined trend terms and replaces each unique instance with its antonym.
        Generates one perturbed text for each unique perturbation.
        """
        perturbations = []
        seen_perturbed_texts = set()

        # Iterate through the trend patterns to find and replace terms
        for original_term_pattern_regex, antonym_term in self.compiled_patterns.items():
            for match in original_term_pattern_regex.finditer(original_text):
                original_matched_phrase = match.group(0) # The exact matched phrase (e.g., "增长")
                
                perturbed_text_candidate = (
                    original_text[:match.start()] + 
                    antonym_term + 
                    original_text[match.end():]
                )
                
                if perturbed_text_candidate not in seen_perturbed_texts:
                    perturbations.append({
                        'perturbed_text': perturbed_text_candidate,
                        'perturbation_detail': f"Changed trend term '{original_matched_phrase}' to '{antonym_term}'",
                        'original_feature': original_matched_phrase
                    })
                    seen_perturbed_texts.add(perturbed_text_candidate)
        
        # If no trend terms were found or perturbed, return original text as a 'no change' perturbation
        if not perturbations:
            perturbations.append({
                'perturbed_text': original_text,
                'perturbation_detail': "No identifiable trend term to perturb",
                'original_feature': None
            })

        return perturbations