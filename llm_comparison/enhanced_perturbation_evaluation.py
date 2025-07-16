#!/usr/bin/env python3
"""
增强的扰动评估脚本
使用llm-judge评估扰动结果，并使用F1和EM逻辑计算扰动答案与期望答案、扰动答案与原始答案的对比
"""

import sys
import os
import json
import time
import argparse
from pathlib import Path
from typing import List, Dict, Any, Optional
import logging

# 添加项目根目录到Python路径
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

# 导入F1和EM计算逻辑
from llm_comparison.chinese_llm_evaluation import (
    calculate_f1_score, 
    calculate_exact_match, 
    normalize_answer_chinese
)

# 导入LLM Judge
from llm_comparison.chinese_llm_judge import SingletonLLMJudge

# 设置日志
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class EnhancedPerturbationEvaluator:
    """增强的扰动评估器"""
    
    def __init__(self):
        """初始化评估器"""
        self.llm_judge = None
        self.expected_answers = {}
        
    def load_expected_answers(self, alphafin_data_path: str):
        """从alphafin数据文件加载期望答案"""
        logger.info(f"📊 加载期望答案从: {alphafin_data_path}")
        
        try:
            with open(alphafin_data_path, 'r', encoding='utf-8') as f:
                for line_num, line in enumerate(f, 1):
                    if line.strip():
                        try:
                            data = json.loads(line.strip())
                            # 使用问题作为key，期望答案作为value
                            question = data.get('generated_question', '')
                            expected_answer = data.get('answer', '')
                            if question and expected_answer:
                                self.expected_answers[question] = expected_answer
                        except json.JSONDecodeError as e:
                            logger.warning(f"⚠️ 第{line_num}行JSON解析失败: {e}")
                            
            logger.info(f"✅ 成功加载 {len(self.expected_answers)} 个期望答案")
            
        except Exception as e:
            logger.error(f"❌ 加载期望答案失败: {e}")
            raise
    
    def initialize_llm_judge(self, model_name: str = "Qwen3-8B", device: str = "cuda:0"):
        """初始化LLM Judge"""
        logger.info(f"🤖 初始化LLM Judge: {model_name} on {device}")
        
        try:
            self.llm_judge = SingletonLLMJudge()
            self.llm_judge.initialize(model_name=model_name, device=device)
            logger.info("✅ LLM Judge初始化完成")
        except Exception as e:
            logger.error(f"❌ LLM Judge初始化失败: {e}")
            raise
    
    def calculate_enhanced_metrics(self, original_answer: str, perturbed_answer: str, 
                                 expected_answer: str, query: str) -> Dict[str, Any]:
        """计算增强的评估指标"""
        
        # 1. 使用F1和EM逻辑计算各种对比
        metrics = {}
        
        # 原始答案 vs 期望答案
        metrics['f1_original_vs_expected'] = calculate_f1_score(original_answer, expected_answer)
        metrics['em_original_vs_expected'] = calculate_exact_match(original_answer, expected_answer)
        
        # 扰动答案 vs 期望答案
        metrics['f1_perturbed_vs_expected'] = calculate_f1_score(perturbed_answer, expected_answer)
        metrics['em_perturbed_vs_expected'] = calculate_exact_match(perturbed_answer, expected_answer)
        
        # 扰动答案 vs 原始答案
        metrics['f1_perturbed_vs_original'] = calculate_f1_score(perturbed_answer, original_answer)
        metrics['em_perturbed_vs_original'] = calculate_exact_match(perturbed_answer, original_answer)
        
        # 2. 计算F1改进
        metrics['f1_improvement'] = metrics['f1_perturbed_vs_expected'] - metrics['f1_original_vs_expected']
        
        # 3. 使用LLM Judge评估
        if self.llm_judge:
            try:
                judge_result = self.llm_judge.evaluate(
                    query=query,
                    expected_answer=expected_answer,
                    model_final_answer=perturbed_answer
                )
                metrics['llm_judge_scores'] = judge_result
                logger.debug(f"✅ LLM Judge评估完成: 准确性={judge_result.get('accuracy', 0)}, 简洁性={judge_result.get('conciseness', 0)}, 专业性={judge_result.get('professionalism', 0)}")
            except Exception as e:
                logger.warning(f"⚠️ LLM Judge评估失败: {e}")
                metrics['llm_judge_scores'] = {
                    'accuracy': 5,  # 使用默认中等分数
                    'conciseness': 5,
                    'professionalism': 5,
                    'overall_score': 5,
                    'reasoning': f"评估失败，使用默认分数: {str(e)}",
                    'raw_output': ""
                }
        else:
            metrics['llm_judge_scores'] = {
                'accuracy': 0,
                'conciseness': 0,
                'professionalism': 0,
                'overall_score': 0,
                'reasoning': "LLM Judge未初始化",
                'raw_output': ""
            }
        
        return metrics
    
    def evaluate_perturbation_results(self, perturbation_file: str, output_file: str):
        """评估扰动结果文件"""
        logger.info(f"🔍 开始评估扰动结果: {perturbation_file}")
        
        # 加载扰动结果
        try:
            with open(perturbation_file, 'r', encoding='utf-8') as f:
                perturbation_results = json.load(f)
            logger.info(f"✅ 加载了 {len(perturbation_results)} 个扰动结果")
        except Exception as e:
            logger.error(f"❌ 加载扰动结果失败: {e}")
            return
        
        # 评估每个结果
        enhanced_results = []
        
        for i, result in enumerate(perturbation_results):
            logger.info(f"📊 评估第 {i+1}/{len(perturbation_results)} 个结果")
            
            query = result.get('question', '')
            original_answer = result.get('original_answer', '')
            perturbed_answer = result.get('perturbed_answer', '')
            
            # 查找期望答案
            expected_answer = self.expected_answers.get(query, '')
            if not expected_answer:
                logger.warning(f"⚠️ 未找到问题 '{query[:50]}...' 的期望答案")
                expected_answer = ""
            
            # 计算增强指标
            enhanced_metrics = self.calculate_enhanced_metrics(
                original_answer=original_answer,
                perturbed_answer=perturbed_answer,
                expected_answer=expected_answer,
                query=query
            )
            
            # 合并结果
            enhanced_result = {
                **result,  # 保留原始数据
                **enhanced_metrics,  # 添加增强指标
                'expected_answer': expected_answer,
                'evaluation_timestamp': time.strftime('%Y-%m-%d %H:%M:%S')
            }
            
            enhanced_results.append(enhanced_result)
            
            # 打印进度
            if (i + 1) % 10 == 0:
                logger.info(f"✅ 已完成 {i+1}/{len(perturbation_results)} 个评估")
        
        # 保存增强结果
        try:
            with open(output_file, 'w', encoding='utf-8') as f:
                json.dump(enhanced_results, f, ensure_ascii=False, indent=2)
            logger.info(f"🎉 增强评估结果已保存到: {output_file}")
        except Exception as e:
            logger.error(f"❌ 保存结果失败: {e}")
        
        # 打印统计摘要
        self.print_evaluation_summary(enhanced_results)
    
    def print_evaluation_summary(self, results: List[Dict[str, Any]]):
        """打印评估摘要"""
        logger.info("\n" + "="*60)
        logger.info("📊 增强扰动评估摘要")
        logger.info("="*60)
        
        if not results:
            logger.info("❌ 没有评估结果")
            return
        
        # 计算统计信息
        total_results = len(results)
        
        # F1统计
        f1_original_vs_expected = [r.get('f1_original_vs_expected', 0) for r in results]
        f1_perturbed_vs_expected = [r.get('f1_perturbed_vs_expected', 0) for r in results]
        f1_perturbed_vs_original = [r.get('f1_perturbed_vs_original', 0) for r in results]
        
        # EM统计
        em_original_vs_expected = [r.get('em_original_vs_expected', 0) for r in results]
        em_perturbed_vs_expected = [r.get('em_perturbed_vs_expected', 0) for r in results]
        em_perturbed_vs_original = [r.get('em_perturbed_vs_original', 0) for r in results]
        
        # LLM Judge统计
        judge_scores = [r.get('llm_judge_scores', {}).get('overall_score', 0) for r in results]
        
        logger.info(f"📈 评估样本总数: {total_results}")
        logger.info(f"📊 F1分数统计:")
        logger.info(f"   原始答案 vs 期望答案: 平均 {sum(f1_original_vs_expected)/len(f1_original_vs_expected):.4f}")
        logger.info(f"   扰动答案 vs 期望答案: 平均 {sum(f1_perturbed_vs_expected)/len(f1_perturbed_vs_expected):.4f}")
        logger.info(f"   扰动答案 vs 原始答案: 平均 {sum(f1_perturbed_vs_original)/len(f1_perturbed_vs_original):.4f}")
        
        logger.info(f"📊 EM分数统计:")
        logger.info(f"   原始答案 vs 期望答案: 平均 {sum(em_original_vs_expected)/len(em_original_vs_expected):.4f}")
        logger.info(f"   扰动答案 vs 期望答案: 平均 {sum(em_perturbed_vs_expected)/len(em_perturbed_vs_expected):.4f}")
        logger.info(f"   扰动答案 vs 原始答案: 平均 {sum(em_perturbed_vs_original)/len(em_perturbed_vs_original):.4f}")
        
        logger.info(f"🤖 LLM Judge评分: 平均 {sum(judge_scores)/len(judge_scores):.2f}")
        
        # 扰动器统计
        perturber_stats = {}
        for result in results:
            perturber = result.get('perturber_name', 'unknown')
            if perturber not in perturber_stats:
                perturber_stats[perturber] = {
                    'count': 0,
                    'avg_f1_improvement': 0,
                    'avg_judge_score': 0
                }
            
            perturber_stats[perturber]['count'] += 1
            perturber_stats[perturber]['avg_f1_improvement'] += result.get('f1_improvement', 0)
            perturber_stats[perturber]['avg_judge_score'] += result.get('llm_judge_scores', {}).get('overall_score', 0)
        
        logger.info(f"🔄 扰动器统计:")
        for perturber, stats in perturber_stats.items():
            count = stats['count']
            avg_f1_improvement = stats['avg_f1_improvement'] / count if count > 0 else 0
            avg_judge_score = stats['avg_judge_score'] / count if count > 0 else 0
            logger.info(f"   {perturber}: {count}个样本, F1改进: {avg_f1_improvement:.4f}, Judge评分: {avg_judge_score:.2f}")
        
        logger.info("="*60)

def main():
    """主函数"""
    parser = argparse.ArgumentParser(description="增强的扰动评估脚本")
    parser.add_argument("--perturbation_file", type=str, required=True,
                       help="扰动结果文件路径 (例如: perturbation_results_incremental.json)")
    parser.add_argument("--alphafin_data", type=str, required=True,
                       help="AlphaFin数据文件路径 (例如: data/alphafin/alphafin_eval_samples_updated.jsonl)")
    parser.add_argument("--output_file", type=str, default=None,
                       help="输出文件路径 (默认: enhanced_perturbation_evaluation_results.json)")
    parser.add_argument("--judge_model", type=str, default="Qwen3-8B",
                       help="LLM Judge模型名称")
    parser.add_argument("--judge_device", type=str, default="cuda:1",
                       help="LLM Judge设备")
    
    args = parser.parse_args()
    
    # 设置默认输出文件
    if args.output_file is None:
        timestamp = time.strftime('%Y%m%d_%H%M%S')
        args.output_file = f"enhanced_perturbation_evaluation_results_{timestamp}.json"
    
    # 创建评估器
    evaluator = EnhancedPerturbationEvaluator()
    
    try:
        # 加载期望答案
        evaluator.load_expected_answers(args.alphafin_data)
        
        # 初始化LLM Judge
        evaluator.initialize_llm_judge(args.judge_model, args.judge_device)
        
        # 执行评估
        evaluator.evaluate_perturbation_results(args.perturbation_file, args.output_file)
        
        logger.info("🎉 增强扰动评估完成！")
        
    except Exception as e:
        logger.error(f"❌ 评估过程中发生错误: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main() 