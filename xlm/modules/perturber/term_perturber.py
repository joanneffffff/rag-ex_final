from xlm.modules.perturber.perturber import Perturber
from typing import List

class TermPerturber(Perturber):
    """金融术语替换扰动"""
    def perturb(self, text: str, features: List[str]) -> List[str]:
        term_map = {
            "市盈率": "净利润",
            "净利润": "市盈率",
            "市净率": "市销率",
            "营收": "收入",
        }
        perturbations = []
        for feature in features:
            for k, v in term_map.items():
                if k in feature:
                    perturbed = text.replace(k, v)
                    perturbations.append(perturbed)
        return perturbations 