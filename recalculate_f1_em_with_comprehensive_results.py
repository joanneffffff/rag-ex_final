#!/usr/bin/env python3
"""
使用期望答案（ground truth）和扰动答案进行比较，计算F1、EM和LLM-Judge评分
"""

import sys
import os
import json
import time
from pathlib import Path
from typing import List, Dict, Any, Optional
import logging

# 添加项目根目录到Python路径
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

# 导入F1和EM计算逻辑
from llm_comparison.chinese_llm_evaluation import (
    calculate_f1_score, 
    calculate_exact_match, 
    normalize_answer_chinese
)

# 导入LLM-Judge评估逻辑
from llm_comparison.chinese_llm_judge import SingletonLLMJudge

# 设置日志
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class GroundTruthPerturbationEvaluator:
    """比较期望答案（ground truth）和扰动答案的评估器"""
    
    def __init__(self):
        """初始化评估器"""
        self.expected_answers = {}
        self.original_answers = {}  # 添加原始答案存储
        self.perturbation_results = {}
        self.judge = SingletonLLMJudge()
        self.judge.initialize(device="cuda:1")
        
    def load_expected_answers(self, file_path: str):
        """加载期望答案（ground truth）"""
        logger.info(f"加载期望答案: {file_path}")
        with open(file_path, 'r', encoding='utf-8') as f:
            for line in f:
                data = json.loads(line.strip())
                question = data.get('generated_question', '')
                answer = data.get('answer', '')
                if question and answer:
                    self.expected_answers[question] = answer
        
        logger.info(f"加载了 {len(self.expected_answers)} 个期望答案")
        
    def load_perturbation_results(self, file_path: str):
        """加载扰动结果并重新组织数据结构"""
        logger.info(f"加载扰动结果: {file_path}")
        with open(file_path, 'r', encoding='utf-8') as f:
            raw_results = json.load(f)
        
        # 重新组织数据结构：按问题分组，每个问题包含多个扰动器
        reorganized_results = {}
        
        for sample in raw_results:
            question = sample.get('question', '').strip()
            perturber_name = sample.get('perturber_name', '')
            perturbed_answer = sample.get('perturbed_answer', '')
            
            if not question or not perturber_name or not perturbed_answer:
                continue
                
            # 如果问题不存在，创建新的问题条目
            if question not in reorganized_results:
                reorganized_results[question] = {
                    'generated_question': question,
                    'perturbations': {}
                }
            
            # 添加扰动器结果
            reorganized_results[question]['perturbations'][perturber_name] = {
                'perturbed_answer': perturbed_answer
            }
        
        # 转换为列表格式
        self.perturbation_results = list(reorganized_results.values())
        
        logger.info(f"重新组织了扰动结果，包含 {len(self.perturbation_results)} 个样本")
        
        # 打印一些统计信息
        total_perturbations = sum(len(sample.get('perturbations', {})) for sample in self.perturbation_results)
        logger.info(f"总扰动器数量: {total_perturbations}")
        
        # 统计各扰动器数量
        perturber_counts = {}
        for sample in self.perturbation_results:
            for perturber_name in sample.get('perturbations', {}):
                perturber_counts[perturber_name] = perturber_counts.get(perturber_name, 0) + 1
        
        for perturber, count in perturber_counts.items():
            logger.info(f"  {perturber}: {count} 个样本")
        
    def load_original_answers(self, comprehensive_file: str):
        """从comprehensive_evaluation_results文件加载原始答案"""
        logger.info(f"加载原始答案: {comprehensive_file}")
        try:
            with open(comprehensive_file, 'r', encoding='utf-8') as f:
                data = json.load(f)
                
            # 提取data字段中的答案，使用query作为键
            for item in data.get('data', []):
                query = item.get('query', '').strip()
                answer = item.get('answer', '')
                if query and answer:
                    self.original_answers[query] = answer
                    
            logger.info(f"加载了 {len(self.original_answers)} 个原始答案")
            
        except Exception as e:
            logger.error(f"加载原始答案失败: {e}")
            raise
        
    def evaluate_sample(self, sample: Dict[str, Any]) -> Dict[str, Any]:
        """评估单个样本"""
        # 兼容多种主问题字段名，并自动strip
        generated_question = (
            sample.get('generated_question') or
            sample.get('question') or
            sample.get('query') or
            ''
        ).strip()
        
        # 查找期望答案和原始答案
        expected_answer = self.expected_answers.get(generated_question, '')
        original_answer = self.original_answers.get(generated_question, '')
        
        if not expected_answer and generated_question:
            # 尝试用strip后的key再查一次
            for k, v in self.expected_answers.items():
                if k.strip() == generated_question:
                    expected_answer = v
                    break
                    
        if not original_answer and generated_question:
            # 尝试用strip后的key再查一次原始答案
            for k, v in self.original_answers.items():
                if k.strip() == generated_question:
                    original_answer = v
                    break
        
        if not expected_answer:
            logger.warning(f"未找到期望答案: {generated_question}")
            return {}
            
        results = {
            'generated_question': generated_question,
            'expected_answer': expected_answer,
            'original_answer': original_answer,
            'perturbations': {}
        }
        
        # 评估每个扰动器
        for perturbator_name, perturbator_results in sample.get('perturbations', {}).items():
            perturb_answer = perturbator_results.get('perturbed_answer', '')
            
            if not perturb_answer:
                continue
                
            # 计算各种对比的F1和EM
            metrics = {}
            
            # 1. 期望答案 vs 扰动答案
            metrics['f1_expected_vs_perturbed'] = calculate_f1_score(expected_answer, perturb_answer)
            metrics['em_expected_vs_perturbed'] = calculate_exact_match(expected_answer, perturb_answer)
            
            # 2. 原始答案 vs 扰动答案
            if original_answer:
                metrics['f1_original_vs_perturbed'] = calculate_f1_score(original_answer, perturb_answer)
                metrics['em_original_vs_perturbed'] = calculate_exact_match(original_answer, perturb_answer)
            else:
                metrics['f1_original_vs_perturbed'] = 0.0
                metrics['em_original_vs_perturbed'] = 0.0
            
            # 3. 期望答案 vs 原始答案
            if original_answer:
                metrics['f1_expected_vs_original'] = calculate_f1_score(expected_answer, original_answer)
                metrics['em_expected_vs_original'] = calculate_exact_match(expected_answer, original_answer)
            else:
                metrics['f1_expected_vs_original'] = 0.0
                metrics['em_expected_vs_original'] = 0.0
            
            # LLM-Judge评估
            try:
                judge_result = self.judge.evaluate(
                    query=generated_question,
                    expected_answer=expected_answer,
                    model_final_answer=perturb_answer,
                    template_file_name="qwen_judge_template.txt"  # 使用指定的模板文件
                )
                # 分别获取三个角度的分数
                judge_accuracy = judge_result.get('accuracy', 0.0)
                judge_conciseness = judge_result.get('conciseness', 0.0)
                judge_professionalism = judge_result.get('professionalism', 0.0)
                judge_overall = judge_result.get('overall_score', 0.0)
            except Exception as e:
                logger.error(f"LLM-Judge评估失败: {e}")
                judge_accuracy = judge_conciseness = judge_professionalism = judge_overall = 0.0
            
            results['perturbations'][perturbator_name] = {
                'perturbed_answer': perturb_answer,
                'f1_expected_vs_perturbed': metrics['f1_expected_vs_perturbed'],
                'em_expected_vs_perturbed': metrics['em_expected_vs_perturbed'],
                'f1_original_vs_perturbed': metrics['f1_original_vs_perturbed'],
                'em_original_vs_perturbed': metrics['em_original_vs_perturbed'],
                'f1_expected_vs_original': metrics['f1_expected_vs_original'],
                'em_expected_vs_original': metrics['em_expected_vs_original'],
                'judge_accuracy': judge_accuracy,
                'judge_conciseness': judge_conciseness,
                'judge_professionalism': judge_professionalism,
                'judge_overall': judge_overall
            }
            
        return results
        
    def evaluate_all(self) -> Dict[str, Any]:
        """评估所有样本"""
        logger.info("开始评估所有样本...")
        
        all_results = []
        total_samples = len(self.perturbation_results)
        
        for i, sample in enumerate(self.perturbation_results):
            logger.info(f"评估样本 {i+1}/{total_samples}")
            result = self.evaluate_sample(sample)
            if result:
                all_results.append(result)
                
        # 计算汇总统计
        summary = self.calculate_summary(all_results)
        
        return {
            'results': all_results,
            'summary': summary
        }
        
    def calculate_summary(self, results: List[Dict[str, Any]]) -> Dict[str, Any]:
        """计算汇总统计"""
        summary = {
            'total_samples': len(results),
            'perturbators': {}
        }
        
        # 收集所有扰动器的分数
        perturbator_scores = {}
        
        for result in results:
            for perturbator_name, perturbator_data in result.get('perturbations', {}).items():
                if perturbator_name not in perturbator_scores:
                    perturbator_scores[perturbator_name] = {
                        'f1_expected_vs_perturbed': [],
                        'em_expected_vs_perturbed': [],
                        'f1_original_vs_perturbed': [],
                        'em_original_vs_perturbed': [],
                        'f1_expected_vs_original': [],
                        'em_expected_vs_original': [],
                        'judge_accuracy_scores': [],
                        'judge_conciseness_scores': [],
                        'judge_professionalism_scores': [],
                        'judge_overall_scores': []
                    }
                
                perturbator_scores[perturbator_name]['f1_expected_vs_perturbed'].append(
                    perturbator_data['f1_expected_vs_perturbed']
                )
                perturbator_scores[perturbator_name]['em_expected_vs_perturbed'].append(
                    perturbator_data['em_expected_vs_perturbed']
                )
                perturbator_scores[perturbator_name]['f1_original_vs_perturbed'].append(
                    perturbator_data['f1_original_vs_perturbed']
                )
                perturbator_scores[perturbator_name]['em_original_vs_perturbed'].append(
                    perturbator_data['em_original_vs_perturbed']
                )
                perturbator_scores[perturbator_name]['f1_expected_vs_original'].append(
                    perturbator_data['f1_expected_vs_original']
                )
                perturbator_scores[perturbator_name]['em_expected_vs_original'].append(
                    perturbator_data['em_expected_vs_original']
                )
                perturbator_scores[perturbator_name]['judge_accuracy_scores'].append(
                    perturbator_data['judge_accuracy']
                )
                perturbator_scores[perturbator_name]['judge_conciseness_scores'].append(
                    perturbator_data['judge_conciseness']
                )
                perturbator_scores[perturbator_name]['judge_professionalism_scores'].append(
                    perturbator_data['judge_professionalism']
                )
                perturbator_scores[perturbator_name]['judge_overall_scores'].append(
                    perturbator_data['judge_overall']
                )
        
        # 计算每个扰动器的平均分数
        for perturbator_name, scores in perturbator_scores.items():
            summary['perturbators'][perturbator_name] = {
                'avg_f1_expected_vs_perturbed': sum(scores['f1_expected_vs_perturbed']) / len(scores['f1_expected_vs_perturbed']),
                'avg_em_expected_vs_perturbed': sum(scores['em_expected_vs_perturbed']) / len(scores['em_expected_vs_perturbed']),
                'avg_f1_original_vs_perturbed': sum(scores['f1_original_vs_perturbed']) / len(scores['f1_original_vs_perturbed']),
                'avg_em_original_vs_perturbed': sum(scores['em_original_vs_perturbed']) / len(scores['em_original_vs_perturbed']),
                'avg_f1_expected_vs_original': sum(scores['f1_expected_vs_original']) / len(scores['f1_expected_vs_original']),
                'avg_em_expected_vs_original': sum(scores['em_expected_vs_original']) / len(scores['em_expected_vs_original']),
                'avg_judge_accuracy': sum(scores['judge_accuracy_scores']) / len(scores['judge_accuracy_scores']),
                'avg_judge_conciseness': sum(scores['judge_conciseness_scores']) / len(scores['judge_conciseness_scores']),
                'avg_judge_professionalism': sum(scores['judge_professionalism_scores']) / len(scores['judge_professionalism_scores']),
                'avg_judge_overall': sum(scores['judge_overall_scores']) / len(scores['judge_overall_scores']),
                'sample_count': len(scores['f1_expected_vs_perturbed'])
            }
        
        return summary
        
    def save_results(self, results: Dict[str, Any], output_file: str):
        """保存评估结果"""
        logger.info(f"保存结果到: {output_file}")
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(results, f, ensure_ascii=False, indent=2)
            
        # 打印汇总统计
        self.print_summary(results['summary'])
        
    def print_summary(self, summary: Dict[str, Any]):
        """打印汇总统计"""
        print("\n" + "="*80)
        print("评估结果汇总")
        print("="*80)
        print(f"总样本数: {summary['total_samples']}")
        print("\n各扰动器表现:")
        print("-" * 80)
        
        for perturbator_name, stats in summary['perturbators'].items():
            print(f"{perturbator_name}:")
            print(f"  期望答案 vs 扰动答案:")
            print(f"    平均F1分数: {stats['avg_f1_expected_vs_perturbed']:.4f}")
            print(f"    平均EM分数: {stats['avg_em_expected_vs_perturbed']:.4f}")
            print(f"  原始答案 vs 扰动答案:")
            print(f"    平均F1分数: {stats['avg_f1_original_vs_perturbed']:.4f}")
            print(f"    平均EM分数: {stats['avg_em_original_vs_perturbed']:.4f}")
            print(f"  期望答案 vs 原始答案:")
            print(f"    平均F1分数: {stats['avg_f1_expected_vs_original']:.4f}")
            print(f"    平均EM分数: {stats['avg_em_expected_vs_original']:.4f}")
            print(f"  Judge评估分数:")
            print(f"    准确性: {stats['avg_judge_accuracy']:.4f}")
            print(f"    简洁性: {stats['avg_judge_conciseness']:.4f}")
            print(f"    专业性: {stats['avg_judge_professionalism']:.4f}")
            print(f"    综合分数: {stats['avg_judge_overall']:.4f}")
            print(f"  样本数: {stats['sample_count']}")
            print()

def main():
    """主函数"""
    # 文件路径
    expected_answers_file = "data/alphafin/alphafin_eval_samples_updated.jsonl"
    original_answers_file = "comprehensive_evaluation_results/chinese_e2e_combined_results_recalculated_2025-07-14_00-00-00.json"
    perturbation_results_file = "perturbation_results_incremental.json"
    output_file = "ground_truth_perturbation_evaluation_results.json"
    
    # 检查文件是否存在
    if not os.path.exists(expected_answers_file):
        logger.error(f"期望答案文件不存在: {expected_answers_file}")
        return
        
    if not os.path.exists(original_answers_file):
        logger.error(f"原始答案文件不存在: {original_answers_file}")
        return
        
    if not os.path.exists(perturbation_results_file):
        logger.error(f"扰动结果文件不存在: {perturbation_results_file}")
        return
    
    # 创建评估器
    evaluator = GroundTruthPerturbationEvaluator()
    
    # 加载数据
    evaluator.load_expected_answers(expected_answers_file)
    evaluator.load_original_answers(original_answers_file)
    evaluator.load_perturbation_results(perturbation_results_file)
    
    # 执行评估
    results = evaluator.evaluate_all()
    
    # 保存结果
    evaluator.save_results(results, output_file)
    
    logger.info("评估完成！")

if __name__ == "__main__":
    main() 