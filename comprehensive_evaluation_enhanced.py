#!/usr/bin/env python3
"""
增强版全面评估脚本 - 确保tqdm进度条正常显示
使用Minimal模板进行100样本评估
"""

# 临时关闭warnings，避免transformers参数警告
import warnings
warnings.filterwarnings("ignore")

# 更精确地过滤transformers生成参数警告
import logging
logging.getLogger("transformers.generation.utils").setLevel(logging.ERROR)
logging.getLogger("transformers").setLevel(logging.WARNING)
logging.getLogger("torch").setLevel(logging.WARNING)
logging.getLogger("xlm").setLevel(logging.WARNING)

# 设置环境变量减少transformers的详细输出
import os
os.environ['TRANSFORMERS_VERBOSITY'] = 'error'

import json
import re
import torch
import numpy as np
from pathlib import Path
from typing import List, Dict, Any, Optional
import random
import time
from difflib import SequenceMatcher
import sys
import argparse
from collections import Counter

# 确保tqdm正确导入和配置
try:
    from tqdm import tqdm
    # 强制启用tqdm进度条
    tqdm.monitor_interval = 0
except ImportError:
    print("❌ tqdm未安装，请运行: pip install tqdm")
    sys.exit(1)

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


import re # 确保你的脚本顶部有 import re

def extract_final_answer(raw_output: str) -> str:
    """从模型的原始输出中提取<answer>标签内的内容"""
    match = re.search(r'<answer>(.*?)</answer>', raw_output, re.DOTALL)
    if match:
        # 如果找到标签，返回标签内的干净内容
        return match.group(1).strip()
    
    # 如果没找到，作为备用方案，返回整个输出的最后一行，这可能包含答案
    lines = raw_output.strip().split('\n')
    if lines:
        return lines[-1].strip()
    return "" # 如果连最后一行都没有，返回空字符串


def calculate_f1_score(prediction: str, ground_truth: str) -> float:
    """
    计算两个字符串之间基于词语重叠的F1分数。
    """
    # 文本规范化：转小写，移除标点，按空格分词
    def normalize(text):
        return re.sub(r'[^\w\s]', '', text.lower()).split()

    prediction_tokens = normalize(prediction)
    ground_truth_tokens = normalize(ground_truth)

    if not ground_truth_tokens:
        return 1.0 if not prediction_tokens else 0.0
    if not prediction_tokens:
        return 0.0

    # 使用Counter来处理词频
    common = Counter(prediction_tokens) & Counter(ground_truth_tokens)
    num_same = sum(common.values())

    if num_same == 0:
        return 0.0

    # 计算精确率、召回率和F1
    precision = 1.0 * num_same / len(prediction_tokens)
    recall = 1.0 * num_same / len(ground_truth_tokens)
    f1 = (2 * precision * recall) / (precision + recall)
    
    return f1


class LLMTemplateTester:
    """LLM模板测试器"""
    
    def __init__(self, model_name: str = "SUFE-AIFLM-Lab/Fin-R1", device: str = "auto"):
        self.model_name = model_name
        self.device = self._setup_device(device)
        self.llm_generator = None  # 使用RAG的LocalLLMGenerator
        self.max_length = 2048 # Increased context window for complex prompts
        self.max_new_tokens = 150  # 默认token限制
        
    def _setup_device(self, device: str) -> str:
        """设置设备"""
        if device == "auto":
            if torch.cuda.is_available():
                return "cuda"
            else:
                return "cpu"
        return device
    
    def load_model(self):
        """加载模型"""
        if USE_RAG_GENERATOR:
            # 使用RAG系统的LocalLLMGenerator
            self.llm_generator = LocalLLMGenerator(
                model_name=self.model_name,
                device=self.device
            )
        else:
            # 备用方案：直接使用transformers
            print("⚠️ 使用备用transformers方案")
            self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
            self.model = AutoModelForCausalLM.from_pretrained(
                self.model_name,
                torch_dtype=torch.float16,
                device_map=self.device
            )
    
    def _convert_messages_to_text(self, messages: List[Dict[str, str]]) -> str:
        """将messages转换为文本格式"""
        text = ""
        for message in messages:
            role = message["role"]
            content = message["content"]
            if role == "system":
                text += f"System: {content}\n\n"
            elif role == "user":
                text += f"User: {content}\n\n"
            elif role == "assistant":
                text += f"Assistant: {content}\n\n"
        return text.strip()
    
    def generate_response(self, messages: List[Dict[str, str]]) -> Dict[str, Any]:
        """生成回答"""
        start_time = time.time()
        
        if USE_RAG_GENERATOR and self.llm_generator:
            # 使用RAG系统的LocalLLMGenerator
            try:
                # 将messages转换为文本格式
                prompt_text = self._convert_messages_to_text(messages)
                
                # 设置生成参数
                generation_params = {
                    "max_new_tokens": self.max_new_tokens,
                    "do_sample": False,  # Fin-R1使用确定性生成
                    "repetition_penalty": 1.1
                }
                
                # 生成回答
                generated_text = self.llm_generator.generate([prompt_text])[0]
                
                generation_time = time.time() - start_time
                
                # 清理回答
                cleaned_answer = self._clean_answer(generated_text)
                
                return {
                    "generated_answer": generated_text,
                    "cleaned_answer": cleaned_answer,
                    "generation_time": generation_time
                }
                
            except Exception as e:
                print(f"⚠️ RAG生成器错误: {e}")
                return {
                    "generated_answer": f"Error: {e}",
                    "cleaned_answer": f"Error: {e}",
                    "generation_time": time.time() - start_time
                }
        else:
            # 备用方案
            return {
                "generated_answer": "RAG generator not available",
                "cleaned_answer": "RAG generator not available",
                "generation_time": time.time() - start_time
            }
    
    def _clean_answer(self, answer: str) -> str:
        """清理回答"""
        # 移除多余的空白字符
        cleaned = answer.strip()
        
        # 如果回答太长，截断到合理长度
        if len(cleaned) > 1000:
            cleaned = cleaned[:1000] + "..."
        
        return cleaned
    
    def evaluate_answer_quality(self, generated_answer: str, expected_answer: str, 
                              context: str, question: str, raw_answer: str = "") -> Dict[str, Any]:
        """评估答案质量"""
        # 基础评估
        exact_match = generated_answer.strip().lower() == expected_answer.strip().lower()
        
        # 包含检查
        contains_expected = expected_answer.strip().lower() in generated_answer.strip().lower()
        
        # 语义相似度（简化版本）
        similarity = SequenceMatcher(None, generated_answer.lower(), expected_answer.lower()).ratio()
        
        # 质量分数计算
        quality_score = 0.0
        
        if exact_match:
            quality_score = 1.0
        elif contains_expected:
            quality_score = 0.8
        elif similarity > 0.7:
            quality_score = 0.6
        elif similarity > 0.5:
            quality_score = 0.4
        elif similarity > 0.3:
            quality_score = 0.2
        else:
            quality_score = 0.0
        
        # 格式违规检查
        format_violations = []
        if len(generated_answer) > 500:
            format_violations.append("回答过长")
        if not generated_answer.strip():
            format_violations.append("空回答")
        
        f1_score = calculate_f1_score(generated_answer, expected_answer)
        return {
            "quality_score": quality_score,
            "exact_match": exact_match,
            "contains_expected": contains_expected,
            "semantic_similarity": similarity,
            "format_violations": format_violations,
            "f1_score": f1_score
        }

def get_detailed_english_prompt_messages(context_content: str, question_text: str, summary_content: Optional[str] = None) -> List[Dict[str, str]]:
    """
    生成 LLM 期望的 messages 列表。
    使用详细的Chain-of-Thought模板，解析system和user标签。
    """
    
    # 读取模板文件
    try:
        with open('rag_english_template.txt', 'r', encoding='utf-8') as f:
            template_content = f.read().strip()
    except FileNotFoundError:
        print("⚠️ 模板文件未找到，使用默认模板")
        return [
            {"role": "system", "content": "You are a world-class quantitative financial analyst AI."},
            {"role": "user", "content": f"Context:\n{context_content}\n\nQuestion:\n{question_text}\n\nA:"}
        ]
    
    # 解析system和user标签
    import re
    
    # 提取system内容
    system_match = re.search(r'<system>(.*?)</system>', template_content, re.DOTALL)
    if system_match:
        system_content = system_match.group(1).strip()
    else:
        system_content = "You are a world-class quantitative financial analyst AI."
    
    # 提取user模板
    user_match = re.search(r'<user>(.*?)</user>', template_content, re.DOTALL)
    if user_match:
        user_template = user_match.group(1).strip()
        # 替换占位符
        user_content = user_template.replace('{context}', context_content).replace('{question}', question_text)
    else:
        user_content = f"Context:\n{context_content}\n\nQuestion:\n{question_text}\n\nA:"

    return [
        {"role": "system", "content": system_content},
        {"role": "user", "content": user_content}
    ]

class EnhancedComprehensiveEvaluator:
    """增强版全面评估器"""
    
    def __init__(self, model_name: str = "SUFE-AIFLM-Lab/Fin-R1", device: str = "auto"):
        self.tester = LLMTemplateTester(model_name, device)
        # 增加max_length以支持更长的模板
        self.tester.max_length = 4096
        # 增加max_new_tokens以支持完整的Chain-of-Thought推理
        self.tester.max_new_tokens = 1024
        print("🔄 加载模型...")
        self.tester.load_model()
        print("✅ 模型加载完成")
        
    def load_evaluation_data(self, sample_size: int = 100) -> List[Dict[str, Any]]:
        """加载评估数据"""
        print(f"📖 加载评估数据，目标样本数: {sample_size}")
        
        # 加载增强版评估数据
        eval_data = []
        data_file = 'evaluate_mrr/tatqa_eval_enhanced.jsonl'
        
        if not os.path.exists(data_file):
            print(f"❌ 数据文件不存在: {data_file}")
            return []
        
        # 使用tqdm显示文件读取进度
        with open(data_file, 'r') as f:
            lines = f.readlines()
            for line in tqdm(lines, desc="📖 读取数据文件", unit="行", 
                           bar_format="{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}, {rate_fmt}]"):
                eval_data.append(json.loads(line))
        
        print(f"✅ 读取了 {len(eval_data)} 行数据")
        
        # 随机采样指定数量的样本
        if len(eval_data) > sample_size:
            np.random.seed(42)  # 确保可重现性
            eval_data = np.random.choice(eval_data, sample_size, replace=False).tolist()
            print(f"✅ 随机采样了 {len(eval_data)} 个样本")
        
        return eval_data
    
    # 在 EnhancedComprehensiveEvaluator 类中
    def evaluate_single_sample(self, sample: Dict[str, Any]) -> Dict[str, Any]:
        """
        评估单个样本，包含提取最终答案的逻辑。
        """
        try:
            # 1. 构建Prompt (这部分逻辑不变)
            messages = get_detailed_english_prompt_messages(
                context_content=sample["context"],
                question_text=sample["query"],
                summary_content=sample["context"]
            )
            
            # 2. 生成完整回答 (模型会输出思考过程和<answer>标签) (这部分逻辑不变)
            generation_result = self.tester.generate_response(messages)
            
            # 3. 从原始输出中提取<answer>标签内的最终答案
            final_answer_to_evaluate = extract_final_answer(generation_result["generated_answer"])
            
            # 4. 使用提取出的干净答案进行质量评估
            evaluation = self.tester.evaluate_answer_quality(
                generated_answer=final_answer_to_evaluate,
                expected_answer=sample["answer"],
                context=sample["context"],
                question=sample["query"],
                raw_answer=generation_result["generated_answer"]
            )
            
            # 5. 组装并返回结果
            return {
                "sample_id": sample.get("id", "unknown"),
                "query": sample["query"],
                "expected_answer": sample["answer"],
                "generation": generation_result,
                "evaluation": evaluation,
                "context_type": "table" if "Table ID:" in sample["context"] else "paragraph",
                "success": evaluation["exact_match"] or evaluation["contains_expected"]
            }
            
        except Exception as e:
            print(f"⚠️ 样本评估失败: {str(e)[:100]}...")
            return {
                "sample_id": sample.get("id", "unknown"),
                "query": sample["query"],
                "expected_answer": sample["answer"],
                "error": str(e),
                "success": False
            }
    
    def run_comprehensive_evaluation(self, sample_size: int = 100) -> Dict[str, Any]:
        """运行全面评估"""
        print(f"\n🚀 开始全面评估，样本数: {sample_size}")
        print("="*60)
        
        # 加载数据
        eval_data = self.load_evaluation_data(sample_size)
        
        if not eval_data:
            print("❌ 没有可用的评估数据")
            return {"results": [], "analysis": {}, "timestamp": time.time()}
        
        # 评估所有样本
        results = []
        start_time = time.time()
        
        print(f"🔍 开始评估 {len(eval_data)} 个样本...")
        
        # 使用tqdm进度条，确保显示
        pbar = tqdm(eval_data, desc="🔍 评估样本", unit="样本", 
                   bar_format="{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}, {rate_fmt}]",
                   ncols=100, leave=True)
        
        for i, sample in enumerate(pbar):
            # 更新进度条描述
            pbar.set_description(f"🔍 评估样本 {i+1}/{len(eval_data)}")
            
            result = self.evaluate_single_sample(sample)
            results.append(result)
            
            # 每10个样本显示一次进度
            if (i + 1) % 10 == 0:
                success_count = sum(1 for r in results if r.get("success", False))
                pbar.set_postfix({
                    "成功": f"{success_count}/{i+1}",
                    "成功率": f"{success_count/(i+1)*100:.1f}%"
                })
        
        pbar.close()
        
        total_time = time.time() - start_time
        print(f"✅ 评估完成，耗时: {total_time:.2f}秒")
        
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
        
        # 新增：收集所有的f1_score
        f1_scores = [r.get("evaluation", {}).get("f1_score", 0) for r in results]

       

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
        
        # ... 你现有的 analysis 字典 ...
        
        # 在 analysis["overall_metrics"] 中添加 avg_f1_score
        analysis["overall_metrics"]["avg_f1_score"] = np.mean(f1_scores) if f1_scores else 0

        # 你也可以为表格和段落数据分别计算平均F1
        table_f1_scores = [r.get("evaluation", {}).get("f1_score", 0) for r in table_results]
        paragraph_f1_scores = [r.get("evaluation", {}).get("f1_score", 0) for r in paragraph_results]
        analysis["context_type_analysis"]["table_samples"]["avg_f1_score"] = np.mean(table_f1_scores) if table_f1_scores else 0
        analysis["context_type_analysis"]["paragraph_samples"]["avg_f1_score"] = np.mean(paragraph_f1_scores) if paragraph_f1_scores else 0


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
        print(f"   F1 Score (词语重叠): {metrics['avg_f1_score']:.3f}")
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
        print(f"     平均F1 Score: {table_analysis['avg_f1_score']:.3f}")
        print(f"     平均质量分数: {table_analysis['avg_quality_score']:.3f}")
        print(f"   段落数据 ({paragraph_analysis['count']} 样本):")
        print(f"     成功率: {paragraph_analysis['success_rate']:.3f} ({paragraph_analysis['success_rate']*100:.1f}%)")
        print(f"     精确匹配率: {paragraph_analysis['exact_match_rate']:.3f} ({paragraph_analysis['exact_match_rate']*100:.1f}%)")
        print(f"     平均F1 Score: {paragraph_analysis['avg_f1_score']:.3f}")
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
    # 解析命令行参数
    parser = argparse.ArgumentParser(description='增强版全面评估')
    parser.add_argument('--n', type=int, default=100, help='评估样本数量 (默认: 100)')
    args = parser.parse_args()
    
    print("🚀 增强版全面评估开始")
    print(f"使用详细Chain-of-Thought模板进行{args.n}样本评估")
    print("="*60)
    
    # 创建评估器
    evaluator = EnhancedComprehensiveEvaluator()
    
    # 运行指定样本数量的评估
    sample_size = args.n
    print(f"\n📊 评估样本数: {sample_size}")
    
    # 运行评估
    evaluation_results = evaluator.run_comprehensive_evaluation(sample_size)
    
    # 打印摘要
    evaluator.print_analysis_summary(evaluation_results["analysis"])
    
    # 保存结果
    output_file = f"comprehensive_evaluation_{sample_size}_samples_enhanced.json"
    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(evaluation_results, f, indent=2, ensure_ascii=False)
    print(f"\n✅ 详细结果已保存到: {output_file}")
    
    print("\n🎉 增强版全面评估完成！")

if __name__ == "__main__":
    main() 