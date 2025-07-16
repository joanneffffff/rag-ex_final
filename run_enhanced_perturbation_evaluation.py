#!/usr/bin/env python3
"""
运行增强扰动评估的示例脚本
"""

import sys
import os
from pathlib import Path

# 添加项目根目录到Python路径
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

from llm_comparison.enhanced_perturbation_evaluation import EnhancedPerturbationEvaluator
import logging

# 设置日志
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def main():
    """主函数 - 运行增强扰动评估"""
    
    # 文件路径配置
    perturbation_file = "perturbation_results_incremental.json"
    alphafin_data_file = "data/alphafin/alphafin_eval_samples_updated.jsonl"
    output_file = "enhanced_perturbation_evaluation_results.json"
    
    # 检查文件是否存在
    if not os.path.exists(perturbation_file):
        logger.error(f"❌ 扰动结果文件不存在: {perturbation_file}")
        return
    
    if not os.path.exists(alphafin_data_file):
        logger.error(f"❌ AlphaFin数据文件不存在: {alphafin_data_file}")
        return
    
    logger.info("🚀 开始增强扰动评估")
    logger.info(f"📁 扰动结果文件: {perturbation_file}")
    logger.info(f"📁 AlphaFin数据文件: {alphafin_data_file}")
    logger.info(f"📁 输出文件: {output_file}")
    
    # 创建评估器
    evaluator = EnhancedPerturbationEvaluator()
    
    try:
        # 1. 加载期望答案
        logger.info("📊 步骤1: 加载期望答案...")
        evaluator.load_expected_answers(alphafin_data_file)
        
        # 2. 初始化LLM Judge (可选，如果不需要LLM Judge评估可以跳过)
        logger.info("🤖 步骤2: 初始化LLM Judge...")
        try:
            evaluator.initialize_llm_judge("Qwen3-8B", "cuda:1")
        except Exception as e:
            logger.warning(f"⚠️ LLM Judge初始化失败，将跳过LLM Judge评估: {e}")
            # 继续执行，只是没有LLM Judge评估
        
        # 3. 执行评估
        logger.info("🔍 步骤3: 执行增强评估...")
        evaluator.evaluate_perturbation_results(perturbation_file, output_file)
        
        logger.info("🎉 增强扰动评估完成！")
        logger.info(f"📄 结果已保存到: {output_file}")
        
    except Exception as e:
        logger.error(f"❌ 评估过程中发生错误: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main() 