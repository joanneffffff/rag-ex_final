import re
from typing import Dict, Any, List, Optional
from .base_perturber import BasePerturber # 确保正确导入BasePerturber

def is_chinese_text(text: str) -> bool:
    """Detects if the text contains Chinese characters."""
    return bool(re.search(r'[\u4e00-\u9fff]', text))

class TrendPerturber(BasePerturber):
    """
    Perturbs financial trend terms in the text by replacing them with antonyms.
    Supports both English and Chinese trend terms.
    For each identifiable trend term instance, it generates a separate perturbed text.
    """
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        super().__init__(config)
        # Define a mapping of trend terms to their antonyms for both languages
        self.trend_map = {
            "zh": {
                "上升": "下降", "上涨": "下跌", "增长": "减少", "提升": "降低", "增加": "减少",
                "下降": "上升", "下跌": "上涨", "减少": "增长", "降低": "提升",
                "好转": "恶化", "改善": "恶化", "积极": "消极", "盈利": "亏损",
                "扩张": "收缩", "持续增长": "持续下滑", "稳步增长": "显著下降",
                "强劲": "疲软", "高于": "低于", "优于": "劣于", "领先": "落后",
                "增加率": "减少率", "上升趋势": "下降趋势", "增长趋势": "减少趋势"
            },
            "en": {
                "increase": "decrease", "increased": "decreased", "increases": "decreases",
                "rise": "fall", "rose": "fell", "rising": "falling", "growth": "decline",
                "grow": "shrink", "grew": "shrank", "growing": "shrinking",
                "up": "down", "higher": "lower", "gain": "loss", "gains": "losses",
                "positive": "negative", "expansion": "contraction", "expanding": "contracting",
                "improve": "worsen", "improved": "worsened", "improves": "worsens",
                "strong": "weak", "stronger": "weaker", "above": "below", "exceed": "fall short",
                "exceeded": "fell short", "outperform": "underperform",
                "increase in": "decrease in", "decrease in": "increase in",
                "higher than": "lower than", "lower than": "higher than"
            }
        }
        # Pre-compile regex patterns for faster matching and whole word boundaries
        self.compiled_patterns = {}
        for lang, terms in self.trend_map.items():
            self.compiled_patterns[lang] = {
                re.compile(r'\b' + re.escape(k) + r'\b', re.IGNORECASE): v 
                for k, v in terms.items()
            }

    @property
    def perturber_name(self) -> str:
        return "trend" # Override default name if needed, or let BasePerturber handle

    def perturb(self, original_text: str, features: Optional[List[str]] = None) -> List[Dict[str, Any]]:
        """
        Perturbs financial trend terms in the original text.
        It finds all predefined trend terms and replaces each unique instance with its antonym.
        Generates one perturbed text for each unique perturbation.
        """
        perturbations = []
        seen_perturbed_texts = set()

        lang = "zh" if is_chinese_text(original_text) else "en"
        trend_patterns = self.compiled_patterns.get(lang, {})

        # Iterate through the trend patterns to find and replace terms
        for original_term_pattern_regex, antonym_term in trend_patterns.items():
            for match in original_term_pattern_regex.finditer(original_text):
                original_matched_phrase = match.group(0) # The exact matched phrase (e.g., "Growth" or "growth")
                
                # Check for direct self-antonymy to avoid infinite loops or illogical changes (e.g., 'increase' -> 'decrease' and 'decrease' -> 'increase')
                # If original_matched_phrase is 'decrease' and its antonym is 'increase', this is okay.
                # If we map 'increase' -> 'decrease', and then try to map 'decrease' -> 'increase',
                # we want to ensure we don't end up swapping back unintentionally in a single perturbation.
                # However, for this perturber, we generate *distinct* perturbations for each found term.
                # The crucial part is to ensure original_text[:match.start()] + antonym_term + original_text[match.end():] is truly unique.
                
                perturbed_text_candidate = (
                    original_text[:match.start()] + 
                    antonym_term + 
                    original_text[match.end():]
                )
                
                if perturbed_text_candidate not in seen_perturbed_texts:
                    perturbations.append({
                        'perturbed_text': perturbed_text_candidate,
                        'perturbation_detail': f"Changed trend term '{original_matched_phrase}' to '{antonym_term}'",
                        'original_feature': original_matched_phrase # The specific matched word/phrase
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