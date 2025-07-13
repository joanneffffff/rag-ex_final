import re
import random # 依然需要，因为super().__init__可能用到config
from typing import Dict, Any, List, Optional
from .base_perturber import BasePerturber # 确保正确导入BasePerturber

class TermPerturber(BasePerturber):
    """
    Perturbs financial terms in the text by replacing them with other specified terms.
    Requires a predefined list of financial terms and their exact replacements.
    """
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        super().__init__(config)
        
        # 定义一个金融术语到其替换词的精确映射
        # 请根据你的实际需求，补充更全面的中英文术语映射
        self.term_replacement_map = self.config.get('term_map', {
            # 中文术语精确替换 (建议你补充和精炼这个列表)
            "市盈率": "净利润",
            "净利润": "市盈率",
            "市净率": "市销率",
            "营收": "收入",
            "营业收入": "营业利润",
            "营业利润": "营业收入",
            "总资产": "净资产",
            "净资产": "总资产",
            "负债": "资产",
            "资产": "负债",
            # 英文术语精确替换 (示例，你可根据TatQA数据补充)
            "revenue": "profit",
            "net income": "gross profit",
            "assets": "liabilities",
            "liabilities": "assets",
            "equity": "debt",
            "pe ratio": "pb ratio",
            "pb ratio": "pe ratio",
        })

        self.random_seed = self.config.get('random_seed', 42)
        random.seed(self.random_seed) # 保持随机种子，如果未来需要随机选择替换词

    def perturb(self, original_text: str, features: Optional[List[str]] = None) -> List[Dict[str, Any]]:
        """
        Perturbs financial terms in the original text by replacing them with their specified replacements.
        Generates one perturbed text for each unique instance of a replaceable term found.
        """
        perturbations = []
        seen_perturbed_texts = set()

        # Iterate through the defined term map
        for original_term, replacement_term_candidate in self.term_replacement_map.items():
            # Use whole word boundary to avoid partial matches
            # re.escape ensures special characters in original_term are treated as literals
            pattern = re.compile(r'\b' + re.escape(original_term) + r'\b', re.IGNORECASE)
            
            # Find all occurrences of the original_term in the text
            for match in pattern.finditer(original_text):
                matched_original_term_exact_case = match.group(0) # e.g., "Revenue" vs "revenue"
                
                # Determine the exact replacement term. If replacement_term_candidate is a list, choose one.
                # For your current term_map, it's a direct string, so no random.choice needed here.
                actual_replacement_term = replacement_term_candidate

                # Perform the replacement on a copy of the original text
                # Replace only the specific matched instance
                perturbed_text_candidate = (
                    original_text[:match.start()] + 
                    actual_replacement_term + # Use the chosen replacement term
                    original_text[match.end():]
                )
                
                if perturbed_text_candidate not in seen_perturbed_texts:
                    perturbations.append({
                        'perturbed_text': perturbed_text_candidate,
                        'perturbation_detail': f"Changed financial term '{matched_original_term_exact_case}' to '{actual_replacement_term}'",
                        'original_feature': matched_original_term_exact_case
                    })
                    seen_perturbed_texts.add(perturbed_text_candidate)
        
        # If no financial terms were found or perturbed, return original text as a 'no change' perturbation
        if not perturbations:
            perturbations.append({
                'perturbed_text': original_text,
                'perturbation_detail': "No identifiable financial term to perturb",
                'original_feature': None
            })

        return perturbations