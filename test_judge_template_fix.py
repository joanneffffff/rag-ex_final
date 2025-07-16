#!/usr/bin/env python3
"""
测试Judge模板修复的脚本
"""

import sys
import os
from pathlib import Path

# 添加项目根目录到Python路径
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

from llm_comparison.chinese_llm_judge import SingletonLLMJudge
import logging

# 设置日志
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def test_judge_template():
    """测试Judge模板修复"""
    
    logger.info("🧪 开始测试Judge模板修复")
    
    # 创建测试数据
    test_cases = [
        {
            "query": "片仔癀公司在2008年第一季度的净利润是多少？",
            "expected_answer": "根据提供的资料，片仔癀公司在2008年第一季度的净利润是29694885.63元。",
            "model_final_answer": "根据现有信息，无法提供此项信息。原报告片段主要涉及2020年度、2021年一季度、2019年业绩快报及2022Q1的表现，并未提及2008年的数据。因此无法确定片仔癀公司在2008年第一季度的具体净利润情况。"
        },
        {
            "query": "瀚蓝环境（600323）在2020年度财报中的关键业绩指标有哪些？",
            "expected_answer": "瀚蓝环境2020年度财报的关键业绩指标包括：净利润同比增长15.9%至10.57亿元；营业收入达74.81亿元，同比增长21.45%；经营活动现金流增长47.68%至19.56亿元。",
            "model_final_answer": "瀚蓝环境2020年度财报中的关键业绩指标包括：市盈率同比增长15.9%至10.57亿元；营业收入74.81亿元，同比增长21.45%；固废业务收入占比提升至53.98%。"
        }
    ]
    
    # 创建LLM Judge实例
    judge = SingletonLLMJudge()
    
    try:
        # 初始化LLM Judge
        logger.info("🤖 初始化LLM Judge...")
        judge.initialize("Qwen3-8B", "cuda:1")
        
        # 测试每个案例
        for i, test_case in enumerate(test_cases, 1):
            logger.info(f"📊 测试案例 {i}: {test_case['query'][:50]}...")
            
            try:
                result = judge.evaluate(
                    query=test_case['query'],
                    expected_answer=test_case['expected_answer'],
                    model_final_answer=test_case['model_final_answer']
                )
                
                logger.info(f"✅ 测试案例 {i} 评估完成:")
                logger.info(f"   准确性: {result.get('accuracy', 0)}")
                logger.info(f"   简洁性: {result.get('conciseness', 0)}")
                logger.info(f"   专业性: {result.get('professionalism', 0)}")
                logger.info(f"   综合评分: {result.get('overall_score', 0)}")
                logger.info(f"   推理过程: {result.get('reasoning', '')[:100]}...")
                
                # 检查是否有JSON解析警告
                if "Judge输出无JSON格式" in result.get('reasoning', ''):
                    logger.warning(f"⚠️ 测试案例 {i} 仍然出现JSON解析问题")
                else:
                    logger.info(f"✅ 测试案例 {i} JSON解析成功")
                
            except Exception as e:
                logger.error(f"❌ 测试案例 {i} 失败: {e}")
        
        logger.info("🎉 Judge模板修复测试完成！")
        
    except Exception as e:
        logger.error(f"❌ LLM Judge初始化失败: {e}")
        logger.info("💡 提示: 如果GPU内存不足，可以尝试使用CPU或减少模型大小")
    
    finally:
        # 清理资源
        try:
            judge.cleanup()
            logger.info("✅ 资源清理完成")
        except:
            pass

if __name__ == "__main__":
    test_judge_template() 