from xlm.modules.perturber.perturber import Perturber
from typing import List
import re

def is_chinese(text: str) -> bool:
    return bool(re.search(r'[\u4e00-\u9fff]', text))

class TrendPerturber(Perturber):
    """
    趋势上升↔下降扰动，支持中英文
    """
    def __init__(self):
        self.trend_map = {
            "zh": {
                "上升": "下降", "上涨": "下跌", "增长": "减少", "提升": "降低", "增加": "减少",
                "下降": "上升", "下跌": "上涨", "减少": "增长", "降低": "提升"
            },
            "en": {
                "increase": "decrease", "rise": "fall", "growth": "decline", "up": "down", "gain": "loss",
                "decrease": "increase", "fall": "rise", "decline": "growth", "down": "up", "loss": "gain"
            }
        }

    def perturb(self, text: str, features: List[str]) -> List[str]:
        lang = "zh" if is_chinese(text) else "en"
        trend_dict = self.trend_map.get(lang, {})
        perturbations = []
        for feature in features:
            for k, v in trend_dict.items():
                if k in feature:
                    perturbed = text.replace(k, v)
                    perturbations.append(perturbed)
        return perturbations 