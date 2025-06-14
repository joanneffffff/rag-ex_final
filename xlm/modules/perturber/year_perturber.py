from xlm.modules.perturber.perturber import Perturber
from typing import List
import re

def is_chinese(text: str) -> bool:
    return bool(re.search(r'[\u4e00-\u9fff]', text))

class YearPerturber(Perturber):
    """
    时间篡改扰动，支持中英文
    """
    def perturb(self, text: str, features: List[str]) -> List[str]:
        perturbations = []
        if is_chinese(text):
            # 匹配中文年份
            years = re.findall(r"(20\\d{2})年", text)
            for year in years:
                new_year = str(int(year) + 1)
                perturbed = text.replace(f"{year}年", f"{new_year}年")
                perturbations.append(perturbed)
        else:
            # 匹配英文年份
            years = re.findall(r"(20\\d{2})", text)
            for year in years:
                new_year = str(int(year) + 1)
                perturbed = text.replace(year, new_year)
                perturbations.append(perturbed)
        return perturbations 