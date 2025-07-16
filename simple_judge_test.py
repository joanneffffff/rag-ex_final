#!/usr/bin/env python3
"""
简单的Judge测试脚本
"""

import sys
import os
from pathlib import Path

# 添加项目根目录到Python路径
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

from llm_comparison.chinese_llm_judge import SingletonLLMJudge, _get_builtin_judge_template
import logging

# 设置日志
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def test_template():
    """测试模板功能"""
    
    logger.info("🧪 开始测试Judge模板")
    
    # 测试内置模板
    template = _get_builtin_judge_template()
    logger.info(f"✅ 内置模板长度: {len(template)} 字符")
    logger.info(f"✅ 模板包含JSON格式要求: {'JSON' in template}")
    logger.info(f"✅ 模板包含评分标准: {'准确性' in template and '简洁性' in template and '专业性' in template}")
    
    # 测试模板替换
    test_query = "测试问题"
    test_expected = "期望答案"
    test_model_answer = "模型答案"
    
    template_with_vars = template.replace('{query}', test_query)
    template_with_vars = template_with_vars.replace('{expected_answer}', test_expected)
    template_with_vars = template_with_vars.replace('{model_final_answer}', test_model_answer)
    
    logger.info(f"✅ 模板变量替换成功: {test_query in template_with_vars}")
    
    logger.info("🎉 模板测试完成！")

def test_judge_initialization():
    """测试Judge初始化"""
    
    logger.info("🤖 测试Judge初始化...")
    
    try:
        judge = SingletonLLMJudge()
        logger.info("✅ SingletonLLMJudge创建成功")
        
        # 测试初始化（不实际加载模型）
        logger.info("✅ Judge类测试完成")
        
    except Exception as e:
        logger.error(f"❌ Judge初始化失败: {e}")

if __name__ == "__main__":
    test_template()
    test_judge_initialization()
    logger.info("🎉 所有测试完成！") 