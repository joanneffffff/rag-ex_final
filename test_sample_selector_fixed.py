#!/usr/bin/env python3
"""
测试修复后的样本选择器
"""

import sys
import os
import json
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from rag_perturbation_experiment import PerturbationSampleSelector

def test_sample_selector():
    """测试样本选择器"""
    print("🔧 测试修复后的PerturbationSampleSelector...")
    
    # 初始化选择器
    selector = PerturbationSampleSelector()
    
    # 测试文本
    test_texts = [
        "神火股份（000933）公司三季报显示业绩大幅改善，云南神火并表驱动增长，煤铝主业经营改善，新疆神火和泉店煤矿增利，同时原材料价格下降降低成本。集团计划未来六个月增持股份，显示对业绩增长的信心。基于近期市场数据，该股票下个月的最终收益预测为'涨'，上涨概率为'极大'。请问这一预测是如何得出的？",
        "一汽解放于2006年9月22日的股票分析数据显示，其股息率为6.4309%。",
        "阳光电源在2023年4月24日的市销率是多少？",
        "崇达技术（002815）在最近的研究报告中，其业务结构和投资扩产状况如何？"
    ]
    
    for i, text in enumerate(test_texts):
        print(f"\n--- 测试文本 {i+1} ---")
        print(f"原文: {text[:100]}...")
        
        # 测试年份检测
        year_terms = selector._detect_year_terms(text)
        print(f"检测到的年份: {year_terms}")
        
        # 测试趋势词检测
        trend_terms = selector._detect_trend_terms(text)
        print(f"检测到的趋势词: {trend_terms}")
        
        # 测试术语检测
        term_terms = selector._detect_term_terms(text)
        print(f"检测到的术语: {term_terms}")
        
        # 计算分数
        year_score = len(year_terms)
        trend_score = len(trend_terms)
        term_score = len(term_terms)
        total_score = year_score + trend_score + term_score
        
        print(f"分数: 年份={year_score}, 趋势={trend_score}, 术语={term_score}, 总分={total_score}")

def test_with_real_samples():
    """使用真实样本测试"""
    print("\n🔧 使用真实样本测试...")
    
    # 加载样本数据
    with open("selected_perturbation_samples.json", "r", encoding="utf-8") as f:
        data = json.load(f)
    
    # 获取所有样本
    all_samples = []
    for category, samples in data.get("categorized_samples", {}).items():
        all_samples.extend(samples)
    
    print(f"📊 总样本数: {len(all_samples)}")
    
    # 初始化选择器
    selector = PerturbationSampleSelector()
    
    # 分析前5个样本
    for i, sample in enumerate(all_samples[:5]):
        print(f"\n--- 样本 {i+1}: {sample.get('sample_id', 'unknown')} ---")
        
        # 分析样本
        analyzed = selector.analyze_sample(sample)
        
        print(f"年份关键词: {analyzed.get('year_keywords', [])}")
        print(f"趋势关键词: {analyzed.get('trend_keywords', [])}")
        print(f"术语关键词: {analyzed.get('term_keywords', [])}")
        print(f"年份分数: {analyzed.get('year_score', 0)}")
        print(f"趋势分数: {analyzed.get('trend_score', 0)}")
        print(f"术语分数: {analyzed.get('term_score', 0)}")
        print(f"总分: {analyzed.get('total_score', 0)}")

if __name__ == "__main__":
    test_sample_selector()
    test_with_real_samples() 