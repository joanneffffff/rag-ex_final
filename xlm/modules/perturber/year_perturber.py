import re
from typing import Dict, Any, List, Optional
from .base_perturber import BasePerturber # 确保正确导入BasePerturber

def _is_chinese_text(text: str) -> bool:
    """Detects if the text contains Chinese characters."""
    return bool(re.search(r'[\u4e00-\u9fff]', text))

class YearPerturber(BasePerturber):
    """
    Perturbs year mentions in the text.
    It finds 4-digit year mentions and replaces them using a predefined map or by incrementing.
    Generates one perturbed text for each unique instance of a perturbable year found.
    """
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        super().__init__(config)
        self.increment = self.config.get('increment', 1) # Default increment is +1 year
        self.min_year = self.config.get('min_year', 1900) # Minimum year to consider for perturbation
        self.max_year = self.config.get('max_year', 2050) # Maximum year to consider as a valid year in original text

        # 添加特定的年份映射 - 2018-2023年范围
        # 这个映射将优先于简单的增量扰动。
        # 键是原始年份字符串，值是目标年份字符串。
        self.year_mapping = self.config.get('year_mapping', {
            "2023": "2018", # 将2023改为2018
            "2022": "2019", # 将2022改为2019
            "2021": "2020", # 将2021改为2020
            "2020": "2019", # 将2020改为2019
            "2019": "2018", # 将2019改为2018
            "2018": "2017", # 将2018改为2017
        })

        # Define comprehensive year patterns for both languages
        # Note: The group ((\d{4})) ensures the year digits are in group 1.
        self.year_patterns = {
            "zh": [
                re.compile(r'(\d{4})\s*年'),      # "2023年"
                re.compile(r'(\d{4})\s*年度'),    # "2023年度"
                re.compile(r'(\d{4})年(\d{1,2})月'), # "2023年1月"
            ],
            "en": [
                re.compile(r'\b(FY\s*)?(\d{4})\b', re.IGNORECASE), # "FY 2023", "2023"
                re.compile(r'\bQ\d\s*(\d{4})\b', re.IGNORECASE), # "Q1 2023"
                re.compile(r'\b(\d{4})s\b', re.IGNORECASE), # "1990s", "2000s"
                re.compile(r'\b(\d{4})\b') # standalone 4-digit year (most general, put last)
            ]
        }

    @property
    def perturber_name(self) -> str:
        return "year"

    def perturb(self, original_text: str, features: Optional[List[str]] = None) -> List[Dict[str, Any]]:
        """
        Applies year perturbation to the original text.
        It finds all 4-digit year mentions and replaces them using year_mapping or by incrementing.
        Generates one perturbed text for each unique instance of a perturbable year found.
        """
        perturbations = []
        seen_perturbed_texts = set()

        lang = "zh" if _is_chinese_text(original_text) else "en"
        patterns_for_lang = self.year_patterns.get(lang, [])

        # Iterate through patterns specific to the detected language
        for pattern_regex in patterns_for_lang:
            for match in pattern_regex.finditer(original_text):
                # Extract the 4-digit year string. It's usually the last numeric group.
                original_year_str_matched = None
                for g in reversed(match.groups()):
                    if g and g.isdigit() and len(g) == 4:
                        original_year_str_matched = g
                        break
                
                if original_year_str_matched is None:
                    continue # Skip if no clear 4-digit year number found by current pattern's groups
                
                original_year_int = int(original_year_str_matched)
                
                # Check if the year is within a reasonable range for perturbation
                if self.min_year <= original_year_int <= self.max_year:
                    perturbed_year_str = None
                    
                    # 1. 优先使用 year_mapping
                    if original_year_str_matched in self.year_mapping:
                        perturbed_year_str = self.year_mapping[original_year_str_matched]
                    # 2. 否则，使用增量
                    else:
                        perturbed_year_int_inc = original_year_int + self.increment
                        # 可选：如果增量后的年份超出合理范围，可以跳过或设置为特定值
                        # if perturbed_year_int_inc > 2100: continue
                        perturbed_year_str = str(perturbed_year_int_inc)
                    
                    if perturbed_year_str is None: continue # 如果没有得到替换年份，跳过

                    # Construct the perturbed text by replacing only this specific instance of the matched year string
                    # We replace the original 4-digit year WITHIN the matched full phrase (e.g., "2023年")
                    # This ensures "2023年" becomes "2018年", not "2018"
                    full_matched_span_text = match.group(0) # e.g., "2023年" or "FY 2023"
                    perturbed_full_span_text = full_matched_span_text.replace(original_year_str_matched, perturbed_year_str)

                    perturbed_text_candidate = (
                        original_text[:match.start(0)] + 
                        perturbed_full_span_text + 
                        original_text[match.end(0):]
                    )
                    
                    if perturbed_text_candidate not in seen_perturbed_texts:
                        perturbations.append({
                            'perturbed_text': perturbed_text_candidate,
                            'perturbation_detail': f"Changed year '{full_matched_span_text}' to '{perturbed_full_span_text}'",
                            'original_feature': full_matched_span_text # 存储完整的原始特征表达
                        })
                        seen_perturbed_texts.add(perturbed_text_candidate)
                
        # If no years were found or perturbed, return original text as a 'no change' perturbation
        if not perturbations:
            perturbations.append({
                'perturbed_text': original_text,
                'perturbation_detail': "No identifiable year to perturb",
                'original_feature': None
            })

        return perturbations