#!/usr/bin/env python3
"""
优化版全面评估脚本 - 解决生成超时问题
"""

import json
import os
import time
import re
import numpy as np
from typing import List, Dict, Any, Optional
from collections import Counter
from difflib import SequenceMatcher
from tqdm import tqdm
import torch
import signal
from contextlib import contextmanager

# 导入RAG系统的LocalLLMGenerator
try:
    from xlm.components.generator.local_llm_generator import LocalLLMGenerator
    USE_RAG_GENERATOR = True
except ImportError:
    print("⚠️ 无法导入LocalLLMGenerator，使用备用方案")
    USE_RAG_GENERATOR = False

class TimeoutException(Exception):
    pass

@contextmanager
def timeout(seconds):
    """超时控制上下文管理器"""
    def signal_handler(signum, frame):
        raise TimeoutException(f"操作超时 ({seconds}秒)")
    
    # 设置信号处理器
    old_handler = signal.signal(signal.SIGALRM, signal_handler)
    signal.alarm(seconds)
    
    try:
        yield
    finally:
        signal.alarm(0)
        signal.signal(signal.SIGALRM, old_handler)

# 你脚本中的这个函数现在是完美的，因为它做的就是这件事
def extract_final_answer(raw_output: str) -> str:
    """从模型的原始输出中提取<answer>标签内的内容"""
    match = re.search(r'<answer>(.*?)</answer>', raw_output, re.DOTALL)
    if match:
        # 完整地返回标签内的所有内容
        return match.group(1).strip()
    # 如果没找到标签，返回空字符串或整个输出作为备用
    return ""

def calculate_f1_score(prediction: str, ground_truth: str) -> float:
    """计算两个字符串之间基于词语重叠的F1分数"""
    def normalize(text):
        return re.sub(r'[^\w\s]', '', text.lower()).split()

    prediction_tokens = normalize(prediction)
    ground_truth_tokens = normalize(ground_truth)

    if not ground_truth_tokens:
        return 1.0 if not prediction_tokens else 0.0
    if not prediction_tokens:
        return 0.0

    common = Counter(prediction_tokens) & Counter(ground_truth_tokens)
    num_same = sum(common.values())

    if num_same == 0:
        return 0.0

    precision = 1.0 * num_same / len(prediction_tokens)
    recall = 1.0 * num_same / len(ground_truth_tokens)
    f1 = (2 * precision * recall) / (precision + recall)
    
    return f1

class OptimizedLLMTemplateTester:
    """优化版LLM模板测试器 - 解决超时问题"""
    
    def __init__(self, model_name: str = "SUFE-AIFLM-Lab/Fin-R1", device: str = "auto"):
        self.model_name = model_name
        self.device = self._setup_device(device)
        self.llm_generator = None
        self.max_length = 8192  # 增加最大长度
        self.max_new_tokens = 2048  # 增加token数量以生成完整答案
        self.timeout_seconds = 120   # 增加超时时间到2分钟
        
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
        global USE_RAG_GENERATOR
        if USE_RAG_GENERATOR:
            try:
                self.llm_generator = LocalLLMGenerator(
                    model_name=self.model_name,
                    device=self.device
                )
                print("✅ RAG生成器加载成功")
            except Exception as e:
                print(f"❌ RAG生成器加载失败: {e}")
                USE_RAG_GENERATOR = False
        else:
            print("⚠️ 使用备用transformers方案")
    
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
        """生成回答 - 带超时控制"""
        start_time = time.time()
        
        if USE_RAG_GENERATOR and self.llm_generator:
            try:
                # 将messages转换为文本格式
                prompt_text = self._convert_messages_to_text(messages)
                
                # 优化的生成参数
                generation_params = {
                    "max_new_tokens": self.max_new_tokens,
                    "do_sample": True,  # 启用采样以获得更好的生成
                    "repetition_penalty": 1.1,
                    "temperature": 0.3,  # 适中的温度以获得平衡的生成
                    "top_p": 0.9,
                    "top_k": 50,  # 添加top_k参数
                    "pad_token_id": 0,
                    "eos_token_id": 2,
                    "length_penalty": 1.0,  # 不惩罚长答案
                    "no_repeat_ngram_size": 3  # 避免重复
                }
                
                # 带超时控制的生成
                try:
                    with timeout(self.timeout_seconds):
                        generated_text = self.llm_generator.generate([prompt_text])[0]
                except TimeoutException:
                    print(f"⚠️ 生成超时 ({self.timeout_seconds}秒)，返回部分结果...")
                    # 返回一个提示信息
                    generated_text = "<think>Generation timeout occurred. Please try with a simpler prompt or increase timeout.</think>"
                
                generation_time = time.time() - start_time
                
                # 清理回答
                cleaned_answer = self._clean_answer(generated_text)
                
                return {
                    "generated_answer": generated_text,
                    "cleaned_answer": cleaned_answer,
                    "generation_time": generation_time,
                    "timeout_occurred": "timeout" in generated_text.lower()
                }
                
            except Exception as e:
                print(f"⚠️ RAG生成器错误: {e}")
                return {
                    "generated_answer": f"Error: {e}",
                    "cleaned_answer": f"Error: {e}",
                    "generation_time": time.time() - start_time,
                    "timeout_occurred": False
                }
        else:
            return {
                "generated_answer": "RAG generator not available",
                "cleaned_answer": "RAG generator not available",
                "generation_time": time.time() - start_time,
                "timeout_occurred": False
            }
    
    def _clean_answer(self, answer: str) -> str:
        """清理回答"""
        return answer.strip()
    
    def evaluate_answer_quality(self, generated_answer: str, expected_answer: str, 
                              context: str, question: str, raw_answer: str = "") -> Dict[str, Any]:
        """评估答案质量"""
        # 检查是否超时
        if "timeout" in generated_answer.lower():
            return {
                "quality_score": 0.0,
                "exact_match": False,
                "contains_expected": False,
                "semantic_similarity": 0.0,
                "format_violations": ["生成超时"],
                "f1_score": 0.0,
                "timeout_occurred": True
            }
        
        # 基础评估
        exact_match = generated_answer.strip().lower() == expected_answer.strip().lower()
        contains_expected = expected_answer.strip().lower() in generated_answer.strip().lower()
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
        
        format_violations = []
        if not generated_answer.strip():
            format_violations.append("空回答")
        
        f1_score = calculate_f1_score(generated_answer, expected_answer)
        
        return {
            "quality_score": quality_score,
            "exact_match": exact_match,
            "contains_expected": contains_expected,
            "semantic_similarity": similarity,
            "format_violations": format_violations,
            "f1_score": f1_score,
            "timeout_occurred": False
        }

def get_detailed_english_prompt_messages(context_content: str, question_text: str, summary_content: Optional[str] = None) -> List[Dict[str, str]]:
    """生成LLM期望的messages列表"""
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
    system_match = re.search(r'<system>(.*?)</system>', template_content, re.DOTALL)
    if system_match:
        system_content = system_match.group(1).strip()
    else:
        system_content = "You are a world-class quantitative financial analyst AI."
    
    user_match = re.search(r'<user>(.*?)</user>', template_content, re.DOTALL)
    if user_match:
        user_template = user_match.group(1).strip()
        user_content = user_template.replace('{context}', context_content).replace('{question}', question_text)
    else:
        user_content = f"Context:\n{context_content}\n\nQuestion:\n{question_text}\n\nA:"

    return [
        {"role": "system", "content": system_content},
        {"role": "user", "content": user_content}
    ]

class OptimizedComprehensiveEvaluator:
    """优化版全面评估器 - 解决超时问题"""
    
    def __init__(self, model_name: str = "SUFE-AIFLM-Lab/Fin-R1", device: str = "auto"):
        self.tester = OptimizedLLMTemplateTester(model_name, device)
        print("🔄 加载模型...")
        self.tester.load_model()
        print("✅ 模型加载完成")
        
    def load_evaluation_data(self, sample_size: int = 20) -> List[Dict[str, Any]]:
        """加载评估数据"""
        print(f"📖 加载评估数据，目标样本数: {sample_size}")
        
        eval_data = []
        data_file = 'evaluate_mrr/tatqa_eval_enhanced.jsonl'
        
        if not os.path.exists(data_file):
            print(f"❌ 数据文件不存在: {data_file}")
            return []
        
        with open(data_file, 'r') as f:
            lines = f.readlines()
            for line in tqdm(lines, desc="📖 读取数据文件", unit="行"):
                eval_data.append(json.loads(line))
        
        print(f"✅ 读取了 {len(eval_data)} 行数据")
        
        # 随机采样指定数量的样本
        if len(eval_data) > sample_size:
            np.random.seed(42)
            eval_data = np.random.choice(eval_data, sample_size, replace=False).tolist()
            print(f"✅ 随机采样了 {len(eval_data)} 个样本")
        
        return eval_data
    
    def evaluate_single_sample(self, sample: Dict[str, Any]) -> Dict[str, Any]:
        """评估单个样本"""
        try:
            # 构建Prompt
            messages = get_detailed_english_prompt_messages(
                context_content=sample["context"],
                question_text=sample["query"],
                summary_content=sample["context"]
            )
            
            # 生成回答
            generation_result = self.tester.generate_response(messages)
            
            # 提取最终答案
            final_answer_to_evaluate = extract_final_answer(generation_result["generated_answer"])
            
            # 质量评估
            evaluation = self.tester.evaluate_answer_quality(
                generated_answer=final_answer_to_evaluate,
                expected_answer=sample["answer"],
                context=sample["context"],
                question=sample["query"],
                raw_answer=generation_result["generated_answer"]
            )
            
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
    
    def run_comprehensive_evaluation(self, sample_size: int = 20) -> Dict[str, Any]:
        """运行全面评估"""
        print(f"\n🚀 开始优化版全面评估，样本数: {sample_size}")
        print("="*60)
        
        # 加载数据
        eval_data = self.load_evaluation_data(sample_size)
        
        if not eval_data:
            print("❌ 没有可用的评估数据")
            return {"results": [], "analysis": {}, "timestamp": time.time()}
        
        # 评估所有样本
        results = []
        start_time = time.time()
        timeout_count = 0
        
        print(f"🔍 开始评估 {len(eval_data)} 个样本...")
        
        pbar = tqdm(eval_data, desc="🔍 评估样本", unit="样本")
        
        for i, sample in enumerate(pbar):
            pbar.set_description(f"🔍 评估样本 {i+1}/{len(eval_data)}")
            
            result = self.evaluate_single_sample(sample)
            results.append(result)
            
            # 统计超时情况
            if result.get("evaluation", {}).get("timeout_occurred", False):
                timeout_count += 1
            
            # 每5个样本显示一次进度
            if (i + 1) % 5 == 0:
                success_count = sum(1 for r in results if r.get("success", False))
                pbar.set_postfix({
                    "成功": f"{success_count}/{i+1}",
                    "超时": f"{timeout_count}/{i+1}",
                    "成功率": f"{success_count/(i+1)*100:.1f}%"
                })
        
        pbar.close()
        
        total_time = time.time() - start_time
        print(f"✅ 评估完成，耗时: {total_time:.2f}秒")
        print(f"⚠️ 超时次数: {timeout_count}/{len(eval_data)}")
        
        # 分析结果
        analysis = self.analyze_results(results, total_time)
        
        return {
            "results": results,
            "analysis": analysis,
            "timestamp": time.time()
        }
    
    def analyze_results(self, results: List[Dict[str, Any]], total_time: float) -> Dict[str, Any]:
        """分析评估结果"""
        total_samples = len(results)
        successful_samples = sum(1 for r in results if r.get("success", False))
        timeout_samples = sum(1 for r in results if r.get("evaluation", {}).get("timeout_occurred", False))
        
        quality_scores = [r.get("evaluation", {}).get("quality_score", 0) for r in results]
        exact_matches = sum(1 for r in results if r.get("evaluation", {}).get("exact_match", False))
        contains_expected = sum(1 for r in results if r.get("evaluation", {}).get("contains_expected", False))
        semantic_similarities = [r.get("evaluation", {}).get("semantic_similarity", 0) for r in results]
        f1_scores = [r.get("evaluation", {}).get("f1_score", 0) for r in results]
        generation_times = [r.get("generation", {}).get("generation_time", 0) for r in results]
        
        analysis = {
            "overall_metrics": {
                "total_samples": total_samples,
                "successful_samples": successful_samples,
                "timeout_samples": timeout_samples,
                "success_rate": successful_samples / total_samples if total_samples > 0 else 0,
                "timeout_rate": timeout_samples / total_samples if total_samples > 0 else 0,
                "exact_match_rate": exact_matches / total_samples if total_samples > 0 else 0,
                "contains_expected_rate": contains_expected / total_samples if total_samples > 0 else 0,
                "avg_quality_score": np.mean(quality_scores) if quality_scores else 0,
                "avg_semantic_similarity": np.mean(semantic_similarities) if semantic_similarities else 0,
                "avg_f1_score": np.mean(f1_scores) if f1_scores else 0,
                "avg_generation_time": np.mean(generation_times) if generation_times else 0,
                "total_evaluation_time": total_time
            }
        }
        
        return analysis
    
    def print_analysis_summary(self, analysis: Dict[str, Any]):
        """打印分析摘要"""
        print("\n" + "="*80)
        print("📊 优化版全面评估结果摘要")
        print("="*80)
        
        metrics = analysis["overall_metrics"]
        
        print(f"📈 总体指标:")
        print(f"   • 总样本数: {metrics['total_samples']}")
        print(f"   • 成功样本: {metrics['successful_samples']}")
        print(f"   • 超时样本: {metrics['timeout_samples']}")
        print(f"   • 成功率: {metrics['success_rate']:.2%}")
        print(f"   • 超时率: {metrics['timeout_rate']:.2%}")
        print(f"   • 精确匹配率: {metrics['exact_match_rate']:.2%}")
        print(f"   • 包含期望答案率: {metrics['contains_expected_rate']:.2%}")
        print(f"   • 平均质量分数: {metrics['avg_quality_score']:.3f}")
        print(f"   • 平均语义相似度: {metrics['avg_semantic_similarity']:.3f}")
        print(f"   • 平均F1分数: {metrics['avg_f1_score']:.3f}")
        print(f"   • 平均生成时间: {metrics['avg_generation_time']:.2f}秒")
        print(f"   • 总评估时间: {metrics['total_evaluation_time']:.2f}秒")
        
        print("\n" + "="*80)

def main():
    """主函数"""
    print("🚀 启动优化版全面评估系统")
    print("="*60)
    
    # 创建评估器
    evaluator = OptimizedComprehensiveEvaluator(
        model_name="SUFE-AIFLM-Lab/Fin-R1",
        device="auto"
    )
    
    # 运行评估
    sample_size = 20  # 减少样本数量以避免长时间运行
    results = evaluator.run_comprehensive_evaluation(sample_size)
    
    # 打印结果
    if results["analysis"]:
        evaluator.print_analysis_summary(results["analysis"])
    
    # 保存结果
    output_file = f"comprehensive_evaluation_optimized_{sample_size}_samples.json"
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    
    print(f"\n💾 结果已保存到: {output_file}")
    print("✅ 评估完成！")

if __name__ == "__main__":
    main() 