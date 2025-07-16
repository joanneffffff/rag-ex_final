#!/usr/bin/env python3
"""
测试year扰动器是否能正确识别样本中的年份
"""

import sys
import os
import re
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from xlm.modules.perturber.year_perturber import YearPerturber

def test_year_perturber():
    """测试year扰动器"""
    print("🔧 测试YearPerturber...")
    
    # 初始化扰动器
    perturber = YearPerturber()
    
    # 测试样本文本
    test_texts = [
        "阳光电源在2023年4月24日的市销率是多少？",
        "神火股份（000933）公司三季报显示业绩大幅改善，云南神火并表驱动增长，煤铝主业经营改善，新疆神火和泉店煤矿增利，同时原材料价格下降降低成本。集团计划未来六个月增持股份，显示对业绩增长的信心。基于近期市场数据，该股票下个月的最终收益预测为'涨'，上涨概率为'极大'。请问这一预测是如何得出的？",
        "一汽解放于2006年9月22日的股票分析数据显示，其股息率为6.4309%。",
        "崇达技术（002815）在最近的研究报告中，其业务结构和投资扩产状况如何？"
    ]
    
    for i, text in enumerate(test_texts):
        print(f"\n--- 测试文本 {i+1} ---")
        print(f"原文: {text}")
        
        # 测试正则表达式匹配
        print("🔍 正则表达式匹配测试:")
        lang = "zh"
        patterns = perturber.year_patterns.get(lang, [])
        
        for j, pattern in enumerate(patterns):
            matches = list(pattern.finditer(text))
            if matches:
                print(f"  模式{j+1}: 找到 {len(matches)} 个匹配")
                for match in matches:
                    print(f"    匹配: '{match.group(0)}' (位置: {match.start()}-{match.end()})")
                    print(f"    组: {match.groups()}")
            else:
                print(f"  模式{j+1}: 无匹配")
        
        # 测试扰动器
        print("🔧 扰动器测试:")
        perturbations = perturber.perturb(text)
        
        for j, perturbation in enumerate(perturbations):
            print(f"  扰动{j+1}:")
            print(f"    原始文本: {perturbation.get('original_feature', 'None')}")
            print(f"    扰动后文本: {perturbation.get('perturbed_text', 'None')}")
            print(f"    扰动详情: {perturbation.get('perturbation_detail', 'None')}")
            
            # 检查是否有实际变化
            original_text = text
            perturbed_text = perturbation.get('perturbed_text', text)
            if original_text != perturbed_text:
                print(f"    ✅ 有变化")
            else:
                print(f"    ❌ 无变化")

if __name__ == "__main__":
    test_year_perturber() 