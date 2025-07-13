import re
from typing import Dict, Any, List, Optional
from .base_perturber import BasePerturber # 确保正确导入BasePerturber

def _is_chinese_text(text: str) -> bool:
    """Detects if the text contains Chinese characters."""
    return bool(re.search(r'[\u4e00-\u9fff]', text))

class YearPerturber(BasePerturber):
    """
    Perturbs year mentions in the text by incrementing them by a specified amount (default: +1).
    Supports both English and Chinese numerical year formats.
    Generates one perturbed text for each unique instance of a perturbable year found.
    """
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        super().__init__(config)
        self.increment = self.config.get('increment', 1) # Default increment is +1 year
        self.min_year = self.config.get('min_year', 1900) # Minimum year to consider for perturbation
        self.max_year = self.config.get('max_year', 2050) # Maximum year to consider as a valid year

        # Define comprehensive year patterns for both languages
        self.year_patterns = {
            "zh": [
                r'(\d{4})\s*年',      # "2023年"
                r'(\d{4})\s*年度',    # "2023年度"
                r'(\d{4})年(\d{1,2})月', # "2023年1月"
            ],
            "en": [
                r'\b(FY\s*)?(\d{4})\b', # "FY 2023", "2023"
                r'\bQ\d\s*(\d{4})\b', # "Q1 2023"
                r'\b(\d{4})s\b', # "1990s", "2000s"
                r'\b(\d{4})\b' # standalone 4-digit year (most general, put last)
            ]
        }
        # Pre-compile regex patterns for faster matching
        self.compiled_patterns = {
            "zh": [re.compile(p) for p in self.year_patterns["zh"]],
            "en": [re.compile(p) for p in self.year_patterns["en"]]
        }

    @property
    def perturber_name(self) -> str:
        return "year" # Override default name if needed

    def perturb(self, original_text: str, features: Optional[List[str]] = None) -> List[Dict[str, Any]]:
        """
        Applies year perturbation to the original text.
        It finds all 4-digit year mentions (e.g., "2023", "1999") and increments them.
        Generates one perturbed text for each unique instance of a perturbable year found.
        """
        perturbations = []
        seen_perturbed_texts = set()

        lang = "zh" if _is_chinese_text(original_text) else "en"
        patterns_for_lang = self.compiled_patterns.get(lang, [])

        # Iterate through patterns specific to the detected language
        for pattern_regex in patterns_for_lang:
            for match in pattern_regex.finditer(original_text):
                # The actual year string is typically in the last captured group, or group 1 if no other groups
                original_year_str = None
                for g in reversed(match.groups()): # Try to find the year string from last group
                    if g and g.isdigit():
                        original_year_str = g
                        break
                
                if original_year_str is None:
                    continue # Skip if no clear year number found by current pattern's groups
                
                original_year_int = int(original_year_str)
                
                # Check if the year is within a reasonable range for perturbation
                if self.min_year <= original_year_int <= self.max_year:
                    perturbed_year_int = original_year_int + self.increment
                    
                    # Construct the perturbed text by replacing only this specific instance of the year
                    # Use match.start() and match.end() to get the exact span of the matched pattern
                    perturbed_text_candidate = (
                        original_text[:match.start(0)] + 
                        match.group(0).replace(original_year_str, str(perturbed_year_int)) + # Replace within the matched group
                        original_text[match.end(0):]
                    )
                    
                    if perturbed_text_candidate not in seen_perturbed_texts:
                        perturbations.append({
                            'perturbed_text': perturbed_text_candidate,
                            'perturbation_detail': f"Changed year '{original_year_str}' to '{perturbed_year_int}'",
                            'original_feature': original_year_str
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