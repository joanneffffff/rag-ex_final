#!/usr/bin/env python3
"""
全面评估脚本 - 在更大数据集上进行MRR和生成质量评估
验证英文Prompt流程的泛化性
"""

# 临时关闭warnings，避免transformers参数警告
import warnings
warnings.filterwarnings("ignore")

# 更精确地过滤transformers生成参数警告
import logging
logging.getLogger("transformers.generation.utils").setLevel(logging.ERROR)

# 减少其他模块的日志输出
logging.getLogger("transformers").setLevel(logging.WARNING)
logging.getLogger("torch").setLevel(logging.WARNING)
logging.getLogger("xlm").setLevel(logging.WARNING)

# 设置环境变量减少transformers的详细输出
import os
os.environ['TRANSFORMERS_VERBOSITY'] = 'error'

import json
import time
import torch
import numpy as np
from typing import List, Dict, Any, Optional
from pathlib import Path
import sys
from tqdm import tqdm

# 添加项目根目录到路径
sys.path.append(str(Path(__file__).parent))

# 导入RAG系统的LocalLLMGenerator
try:
    from xlm.components.generator.local_llm_generator import LocalLLMGenerator
    USE_RAG_GENERATOR = True
    print("✅ 使用RAG系统的LocalLLMGenerator")
except ImportError:
    USE_RAG_GENERATOR = False
    print("⚠️ 无法导入RAG系统的LocalLLMGenerator，使用备用方案")
    from transformers import AutoTokenizer, AutoModelForCausalLM
    from transformers.utils.quantization_config import BitsAndBytesConfig

from test_english_template import LLMTemplateTester, get_final_optimized_english_prompt_messages

class ComprehensiveEvaluator:
    """全面评估器"""
    
    def __init__(self, model_name: str = "SUFE-AIFLM-Lab/Fin-R1", device: str = "auto"):
        self.tester = LLMTemplateTester(model_name, device)
        self.tester.load_model()
        
    def load_evaluation_data(self, sample_size: int = 500) -> List[Dict[str, Any]]:
        """加载评估数据"""
        # 加载增强版评估数据
        eval_data = []
        with open('evaluate_mrr/tatqa_eval_enhanced.jsonl', 'r') as f:
            for line in tqdm(f, desc="📖 读取数据文件", unit="行"):
                eval_data.append(json.loads(line))
        
        # 随机采样指定数量的样本
        if len(eval_data) > sample_size:
            np.random.seed(42)  # 确保可重现性
            eval_data = np.random.choice(eval_data, sample_size, replace=False).tolist()
        
        return eval_data
    
    def evaluate_single_sample(self, sample: Dict[str, Any]) -> Dict[str, Any]:
        """评估单个样本"""
        try:
            # 构建Prompt
            messages = get_final_optimized_english_prompt_messages(
                context_content=sample["context"],
                question_text=sample["query"],
                summary_content=sample["context"]
            )
            
            # 生成回答
            generation_result = self.tester.generate_response(messages)
            
            # 评估质量
            evaluation = self.tester.evaluate_answer_quality(
                generated_answer=generation_result["cleaned_answer"],
                expected_answer=sample["answer"],
                context=sample["context"],
                question=sample["query"],
                raw_answer=generation_result["generated_answer"]  # 传递原始未清理的答案
            )
            
            return {
                "sample_id": sample.get("id", "unknown"),
                "query": sample["query"],
                "expected_answer": sample["answer"],
                "context_type": "table" if "Table ID:" in sample["context"] else "paragraph",
                "generation": generation_result,
                "evaluation": evaluation,
                "success": evaluation["exact_match"] or evaluation["contains_expected"]
            }
            
        except Exception as e:
            # 静默处理错误，不打印详细日志
            return {
                "sample_id": sample.get("id", "unknown"),
                "query": sample["query"],
                "expected_answer": sample["answer"],
                "error": str(e),
                "success": False
            }
    
    def run_comprehensive_evaluation(self, sample_size: int = 100) -> Dict[str, Any]:
        """运行全面评估"""
        # 加载数据
        eval_data = self.load_evaluation_data(sample_size)
        
        # 评估所有样本
        results = []
        start_time = time.time()
        
        # 使用tqdm进度条
        for sample in tqdm(eval_data, desc="🔍 评估样本", unit="样本", 
                          bar_format="{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}, {rate_fmt}]"):
            result = self.evaluate_single_sample(sample)
            results.append(result)
        
        total_time = time.time() - start_time
        
        # 分析结果
        analysis = self.analyze_results(results, total_time)
        
        return {
            "results": results,
            "analysis": analysis,
            "timestamp": time.time()
        }
    
    def analyze_results(self, results: List[Dict[str, Any]], total_time: float) -> Dict[str, Any]:
        """分析评估结果"""
        # 基础统计
        total_samples = len(results)
        successful_samples = sum(1 for r in results if r.get("success", False))
        failed_samples = total_samples - successful_samples
        
        # 质量指标
        quality_scores = [r.get("evaluation", {}).get("quality_score", 0) for r in results]
        exact_matches = sum(1 for r in results if r.get("evaluation", {}).get("exact_match", False))
        contains_expected = sum(1 for r in results if r.get("evaluation", {}).get("contains_expected", False))
        semantic_similarities = [r.get("evaluation", {}).get("semantic_similarity", 0) for r in results]
        
        # 生成时间统计
        generation_times = [r.get("generation", {}).get("generation_time", 0) for r in results]
        
        # 按上下文类型分组
        table_results = [r for r in results if r.get("context_type") == "table"]
        paragraph_results = [r for r in results if r.get("context_type") == "paragraph"]
        
        # 计算MRR (Mean Reciprocal Rank) - 这里简化为精确匹配率
        mrr = exact_matches / total_samples if total_samples > 0 else 0
        
        analysis = {
            "overall_metrics": {
                "total_samples": total_samples,
                "successful_samples": successful_samples,
                "failed_samples": failed_samples,
                "success_rate": successful_samples / total_samples if total_samples > 0 else 0,
                "exact_match_rate": exact_matches / total_samples if total_samples > 0 else 0,
                "contains_expected_rate": contains_expected / total_samples if total_samples > 0 else 0,
                "mrr": mrr,
                "avg_quality_score": np.mean(quality_scores) if quality_scores else 0,
                "avg_semantic_similarity": np.mean(semantic_similarities) if semantic_similarities else 0,
                "avg_generation_time": np.mean(generation_times) if generation_times else 0,
                "total_evaluation_time": total_time
            },
            "context_type_analysis": {
                "table_samples": {
                    "count": len(table_results),
                    "success_rate": sum(1 for r in table_results if r.get("success", False)) / len(table_results) if table_results else 0,
                    "exact_match_rate": sum(1 for r in table_results if r.get("evaluation", {}).get("exact_match", False)) / len(table_results) if table_results else 0,
                    "avg_quality_score": np.mean([r.get("evaluation", {}).get("quality_score", 0) for r in table_results]) if table_results else 0
                },
                "paragraph_samples": {
                    "count": len(paragraph_results),
                    "success_rate": sum(1 for r in paragraph_results if r.get("success", False)) / len(paragraph_results) if paragraph_results else 0,
                    "exact_match_rate": sum(1 for r in paragraph_results if r.get("evaluation", {}).get("exact_match", False)) / len(paragraph_results) if paragraph_results else 0,
                    "avg_quality_score": np.mean([r.get("evaluation", {}).get("quality_score", 0) for r in paragraph_results]) if paragraph_results else 0
                }
            },
            "quality_distribution": {
                "excellent_quality": sum(1 for score in quality_scores if score >= 0.8),
                "good_quality": sum(1 for score in quality_scores if 0.6 <= score < 0.8),
                "fair_quality": sum(1 for score in quality_scores if 0.4 <= score < 0.6),
                "poor_quality": sum(1 for score in quality_scores if score < 0.4)
            },
            "performance_insights": []
        }
        
        # 生成性能洞察
        if analysis["overall_metrics"]["success_rate"] >= 0.8:
            analysis["performance_insights"].append("🎉 整体表现优秀，成功率达到80%以上")
        elif analysis["overall_metrics"]["success_rate"] >= 0.6:
            analysis["performance_insights"].append("✅ 整体表现良好，成功率达到60%以上")
        else:
            analysis["performance_insights"].append("⚠️ 整体表现需要改进")
        
        if analysis["overall_metrics"]["exact_match_rate"] >= 0.7:
            analysis["performance_insights"].append("🎯 精确匹配率很高，模型输出质量优秀")
        
        if analysis["context_type_analysis"]["table_samples"]["success_rate"] > analysis["context_type_analysis"]["paragraph_samples"]["success_rate"]:
            analysis["performance_insights"].append("📊 表格数据表现优于段落数据")
        else:
            analysis["performance_insights"].append("📝 段落数据表现优于表格数据")
        
        return analysis
    
    def print_analysis_summary(self, analysis: Dict[str, Any]):
        """打印分析摘要"""
        print("\n" + "="*80)
        print("📊 全面评估结果摘要")
        print("="*80)
        
        metrics = analysis["overall_metrics"]
        print(f"📈 整体指标:")
        print(f"   总样本数: {metrics['total_samples']}")
        print(f"   成功样本数: {metrics['successful_samples']}")
        print(f"   成功率: {metrics['success_rate']:.3f} ({metrics['success_rate']*100:.1f}%)")
        print(f"   精确匹配率: {metrics['exact_match_rate']:.3f} ({metrics['exact_match_rate']*100:.1f}%)")
        print(f"   包含期望答案率: {metrics['contains_expected_rate']:.3f} ({metrics['contains_expected_rate']*100:.1f}%)")
        print(f"   MRR: {metrics['mrr']:.3f}")
        print(f"   平均质量分数: {metrics['avg_quality_score']:.3f}")
        print(f"   平均语义相似度: {metrics['avg_semantic_similarity']:.3f}")
        print(f"   平均生成时间: {metrics['avg_generation_time']:.2f}s")
        print(f"   总评估时间: {metrics['total_evaluation_time']:.2f}s")
        
        print(f"\n📊 上下文类型分析:")
        table_analysis = analysis["context_type_analysis"]["table_samples"]
        paragraph_analysis = analysis["context_type_analysis"]["paragraph_samples"]
        print(f"   表格数据 ({table_analysis['count']} 样本):")
        print(f"     成功率: {table_analysis['success_rate']:.3f} ({table_analysis['success_rate']*100:.1f}%)")
        print(f"     精确匹配率: {table_analysis['exact_match_rate']:.3f} ({table_analysis['exact_match_rate']*100:.1f}%)")
        print(f"     平均质量分数: {table_analysis['avg_quality_score']:.3f}")
        print(f"   段落数据 ({paragraph_analysis['count']} 样本):")
        print(f"     成功率: {paragraph_analysis['success_rate']:.3f} ({paragraph_analysis['success_rate']*100:.1f}%)")
        print(f"     精确匹配率: {paragraph_analysis['exact_match_rate']:.3f} ({paragraph_analysis['exact_match_rate']*100:.1f}%)")
        print(f"     平均质量分数: {paragraph_analysis['avg_quality_score']:.3f}")
        
        print(f"\n📈 质量分布:")
        quality_dist = analysis["quality_distribution"]
        total = sum(quality_dist.values())
        print(f"   优秀质量 (≥0.8): {quality_dist['excellent_quality']} ({quality_dist['excellent_quality']/total*100:.1f}%)")
        print(f"   良好质量 (0.6-0.8): {quality_dist['good_quality']} ({quality_dist['good_quality']/total*100:.1f}%)")
        print(f"   一般质量 (0.4-0.6): {quality_dist['fair_quality']} ({quality_dist['fair_quality']/total*100:.1f}%)")
        print(f"   较差质量 (<0.4): {quality_dist['poor_quality']} ({quality_dist['poor_quality']/total*100:.1f}%)")
        
        print(f"\n💡 性能洞察:")
        for insight in analysis["performance_insights"]:
            print(f"   {insight}")
        
        print("="*80)

def main():
    """主函数"""
    # 创建评估器
    evaluator = ComprehensiveEvaluator()
    
    # 运行评估 (可以根据需要调整样本数)
    sample_sizes = [100, 500]  # 先测试100个，再测试500个
    
    for sample_size in tqdm(sample_sizes, desc="📊 评估不同样本数", unit="样本数"):
        print(f"\n{'='*60}")
        print(f"📊 评估样本数: {sample_size}")
        print(f"{'='*60}")
        
        # 运行评估
        evaluation_results = evaluator.run_comprehensive_evaluation(sample_size)
        
        # 打印摘要
        evaluator.print_analysis_summary(evaluation_results["analysis"])
        
        # 保存结果
        output_file = f"comprehensive_evaluation_{sample_size}_samples.json"
        with open(output_file, "w", encoding="utf-8") as f:
            json.dump(evaluation_results, f, indent=2, ensure_ascii=False)
        print(f"\n✅ 详细结果已保存到: {output_file}")
    
    print("\n🎉 全面评估完成！")

if __name__ == "__main__":
    main() 