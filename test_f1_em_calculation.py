#!/usr/bin/env python3
"""
测试F1和EM计算是否正确
"""

import sys
import os
from pathlib import Path

# 添加项目根目录到Python路径
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

from llm_comparison.chinese_llm_evaluation import (
    calculate_f1_score, 
    calculate_exact_match, 
    normalize_answer_chinese,
    get_tokens_chinese
)
import logging

# 设置日志
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def test_f1_em_calculation():
    """测试F1和EM计算"""
    
    logger.info("🧪 开始测试F1和EM计算")
    
    # 测试案例
    test_cases = [
        {
            "name": "完全匹配",
            "prediction": "该公司在2012年第四季度的财务费用是3650812.88元",
            "ground_truth": "该公司在2012年第四季度的财务费用是3650812.88元",
            "expected_f1": 1.0,
            "expected_em": 1.0
        },
        {
            "name": "部分匹配",
            "prediction": "该公司在2012年第四季度的财务费用是3650812.88元",
            "ground_truth": "该公司在2012年第四季度的财务费用是3650812.88",
            "expected_f1": 0.9,  # 应该很高
            "expected_em": 0.0    # 不完全匹配
        },
        {
            "name": "无匹配",
            "prediction": "根据现有信息，无法提供此项信息",
            "ground_truth": "该公司在2012年第四季度的财务费用是3650812.88元",
            "expected_f1": 0.0,
            "expected_em": 0.0
        },
        {
            "name": "实际案例1",
            "prediction": "根据提供的资料，完整公司财务报告片段中并未包含2008年的数据，只有2019年至2022Q1的相关信息。 因此，无法从现有信息中获取片仔癀公司在2008年第一季度的净利润。 答案为：根据现有信息，无法提供此项信息。",
            "ground_truth": "该公司在2008年第一季度的净利润是29694885.63。",
            "expected_f1": 0.0,  # 应该很低
            "expected_em": 0.0
        },
        {
            "name": "实际案例2",
            "prediction": "瀚蓝环境2020年度财报的关键业绩指标包括：净利润同比增长15.9%至10.57亿元； 营业收入达74.81亿元，同比增长21.45%； 经营活动现金流增长47.68%至19.56亿元；。",
            "ground_truth": "这个股票的下月最终收益结果是：跌，下跌概率：较大",
            "expected_f1": 0.0,  # 应该很低
            "expected_em": 0.0
        }
    ]
    
    for i, test_case in enumerate(test_cases, 1):
        logger.info(f"\n📊 测试案例 {i}: {test_case['name']}")
        
        # 计算F1和EM
        f1_score = calculate_f1_score(test_case['prediction'], test_case['ground_truth'])
        em_score = calculate_exact_match(test_case['prediction'], test_case['ground_truth'])
        
        # 显示归一化结果
        normalized_pred = normalize_answer_chinese(test_case['prediction'])
        normalized_truth = normalize_answer_chinese(test_case['ground_truth'])
        
        logger.info(f"   预测答案: {test_case['prediction'][:50]}...")
        logger.info(f"   期望答案: {test_case['ground_truth'][:50]}...")
        logger.info(f"   归一化预测: {normalized_pred[:50]}...")
        logger.info(f"   归一化期望: {normalized_truth[:50]}...")
        logger.info(f"   F1分数: {f1_score:.4f} (期望: {test_case['expected_f1']:.4f})")
        logger.info(f"   EM分数: {em_score:.4f} (期望: {test_case['expected_em']:.4f})")
        
        # 检查分词结果
        pred_tokens = get_tokens_chinese(test_case['prediction'])
        truth_tokens = get_tokens_chinese(test_case['ground_truth'])
        
        logger.info(f"   预测分词: {pred_tokens[:10]}...")
        logger.info(f"   期望分词: {truth_tokens[:10]}...")
        
        # 验证结果是否合理
        if f1_score >= 0 and f1_score <= 1:
            logger.info(f"   ✅ F1分数在合理范围内")
        else:
            logger.error(f"   ❌ F1分数超出范围: {f1_score}")
            
        if em_score >= 0 and em_score <= 1:
            logger.info(f"   ✅ EM分数在合理范围内")
        else:
            logger.error(f"   ❌ EM分数超出范围: {em_score}")
    
    logger.info("\n🎉 F1和EM计算测试完成！")

def test_normalization():
    """测试归一化函数"""
    
    logger.info("\n🧪 测试归一化函数")
    
    test_texts = [
        "该公司在2012年第四季度的财务费用是3,650,812.88。",
        "该公司在2012年第四季度的财务费用是3650812.88元",
        "根据现有信息，无法提供此项信息。",
        "瀚蓝环境2020年度财报的关键业绩指标包括：净利润同比增长15.9%至10.57亿元；"
    ]
    
    for i, text in enumerate(test_texts, 1):
        normalized = normalize_answer_chinese(text)
        tokens = get_tokens_chinese(text)
        
        logger.info(f"   原文 {i}: {text}")
        logger.info(f"   归一化: {normalized}")
        logger.info(f"   分词: {tokens}")
        logger.info("")

if __name__ == "__main__":
    test_f1_em_calculation()
    test_normalization() 