#!/usr/bin/env python3
"""
测试年份扰动器的修改
"""

import sys
from pathlib import Path

# 添加项目根目录到路径
sys.path.append(str(Path(__file__).parent))

from xlm.modules.perturber.year_perturber import YearPerturber

def test_year_perturber():
    """测试年份扰动器"""
    print("🔬 测试年份扰动器...")
    
    # 创建扰动器
    perturber = YearPerturber()
    
    # 测试用例
    test_cases = [
        "2024年营收增长10%",
        "2025年度报告显示利润下降",
        "2023年第一季度业绩",
        "2022年财务数据",
        "2021年公司表现良好",
        "2020年疫情影响较大"
    ]
    
    for i, test_text in enumerate(test_cases):
        print(f"\n📊 测试用例 {i+1}: {test_text}")
        
        # 应用扰动
        perturbations = perturber.perturb(test_text)
        
        for j, perturbation in enumerate(perturbations):
            if isinstance(perturbation, dict):
                perturbed_text = perturbation.get('perturbed_text', test_text)
                detail = perturbation.get('perturbation_detail', '')
            else:
                perturbed_text = perturbation
                detail = '直接扰动'
            
            print(f"  扰动结果 {j+1}: {perturbed_text}")
            print(f"  扰动详情: {detail}")
            
            # 检查是否有实际变化
            if perturbed_text != test_text:
                print(f"  ✅ 成功扰动")
            else:
                print(f"  ⚠️ 无变化")

if __name__ == "__main__":
    test_year_perturber() 