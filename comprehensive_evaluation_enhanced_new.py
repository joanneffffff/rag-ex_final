#!/usr/bin/env python3
"""
最终版全面评估脚本 - 修复版本
使用与comprehensive_evaluation_enhanced.py相同的逻辑，但只使用一个统一模板，包含context分离功能
"""

import warnings
import logging
import os
import json
import re
import torch
import numpy as np
from pathlib import Path
from typing import List, Dict, Any, Optional, Union
import time
import argparse
from collections import Counter
from difflib import SequenceMatcher
import sys
import gc
import signal
import atexit

# 环境设置
warnings.filterwarnings("ignore")
logging.getLogger("transformers").setLevel(logging.ERROR)
os.environ['TRANSFORMERS_VERBOSITY'] = 'error'

try:
    from tqdm import tqdm
except ImportError:
    print("❌ tqdm未安装，请运行: pip install tqdm")
    sys.exit(1)

sys.path.append(str(Path(__file__).parent))

try:
    from xlm.components.generator.local_llm_generator import LocalLLMGenerator
    USE_RAG_GENERATOR = True
    print("✅ 使用RAG系统的LocalLLMGenerator")
except ImportError:
    USE_RAG_GENERATOR = False
    print("⚠️ 无法导入RAG系统的LocalLLMGenerator，脚本将无法运行。")
    sys.exit(1)

try:
    from xlm.utils.context_separator import context_separator
    USE_CONTEXT_SEPARATOR = True
    print("✅ 使用上下文分离功能")
except ImportError:
    USE_CONTEXT_SEPARATOR = False
    print("⚠️ 无法导入上下文分离器，将使用原始上下文处理方式")

# ===================================================================
# 资源清理机制
# ===================================================================

class ResourceManager:
    """资源管理器，确保程序结束时正确清理资源"""
    
    def __init__(self):
        self.generator = None
        self.cleanup_registered = False
        self._register_cleanup()
    
    def _register_cleanup(self):
        """注册清理函数"""
        if not self.cleanup_registered:
            atexit.register(self.cleanup_resources)
            signal.signal(signal.SIGINT, self._signal_handler)
            signal.signal(signal.SIGTERM, self._signal_handler)
            self.cleanup_registered = True
    
    def _signal_handler(self, signum, frame):
        """信号处理器"""
        print(f"\n🛑 收到信号 {signum}，开始清理资源...")
        self.cleanup_resources()
        sys.exit(0)
    
    def set_generator(self, generator):
        """设置生成器引用"""
        self.generator = generator
    
    def cleanup_resources(self):
        """清理所有资源"""
        print("🧹 开始清理资源...")
        
        try:
            # 1. 清理生成器
            if self.generator:
                print("🗑️ 清理生成器...")
                if hasattr(self.generator, 'model'):
                    del self.generator.model
                if hasattr(self.generator, 'tokenizer'):
                    del self.generator.tokenizer
                self.generator = None
            
            # 2. 清理GPU内存
            if torch.cuda.is_available():
                print("🗑️ 清理GPU内存...")
                torch.cuda.empty_cache()
                torch.cuda.synchronize()
                
                # 显示清理后的内存状态
                for i in range(torch.cuda.device_count()):
                    allocated = torch.cuda.memory_allocated(i) / 1024**3
                    cached = torch.cuda.memory_reserved(i) / 1024**3
                    print(f"   GPU {i}: 已分配 {allocated:.2f}GB, 缓存 {cached:.2f}GB")
            
            # 3. 强制垃圾回收
            print("🗑️ 强制垃圾回收...")
            gc.collect()
            
            print("✅ 资源清理完成")
            
        except Exception as e:
            print(f"⚠️ 资源清理过程中出现错误: {e}")

# 全局资源管理器
resource_manager = ResourceManager()

# ===================================================================
# 核心辅助函数
# ===================================================================

def extract_final_answer_with_rescue(raw_output: str) -> str:
    """
    从模型的原始输出中智能提取最终答案。
    它首先尝试寻找<answer>标签，如果失败或为空，则启动救援逻辑从<think>标签中提取。
    """
    def _clean_extracted_text(text: str) -> str:
        """对提取出的文本进行通用清理，以匹配期望的答案格式"""
        text = text.strip()
        # 移除数字中的逗号 (如果你的 expected_answer 不包含逗号)
        text = text.replace(',', '')
        # 移除负数括号 (例如 "(33)" -> "-33")
        if text.startswith('(') and text.endswith(')'):
            text = '-' + text[1:-1]
        
        # 标准化百分号，确保 "15.2%" 和 "15.2 %" 匹配
        text = text.replace('%', ' %').strip()
        text = text.replace(' %', '%')

        # 移除常见的引导词句 (应与 Prompt 优化后减少出现)
        text = re.sub(r'^(the\s*answer\s*is|it\s*was|the\s*value\s*is|resulting\s*in|this\s*represents|the\s*effect\s*is|therefore|so|thus|in\s*conclusion|final\s*answer\s*is|final\s*number\s*is)\s*', '', text, flags=re.IGNORECASE).strip()
        
        # 移除末尾可能的多余标点符号，如句号、逗号、分号 (但保留百分号)
        text = re.sub(r'[\.。;,]$', '', text).strip()
        
        # 移除常见的货币符号和单位词 (如果你的 expected_answer 不包含这些)
        text = re.sub(r'(\$|million|billion|usd|eur|pounds|£)', '', text, flags=re.IGNORECASE).strip()

        return text

    # 1. 尝试从 <answer> 标签中提取 (这是首要且最期望的)
    answer_match = re.search(r'<answer>(.*?)</answer>', raw_output, re.DOTALL)
    if answer_match:
        content = answer_match.group(1).strip()
        if content:
            return _clean_extracted_text(content)
        # 如果 <answer> 标签存在但内容为空，则继续救援

    # 2. 救援逻辑：如果 <answer> 标签不存在或为空，尝试从 <think> 标签中提取
    # 查找 <think> 标签的内容
    think_match = re.search(r'<think>(.*?)(?:</think>|$)', raw_output, re.DOTALL)
    if not think_match:
        # 如果连 <think> 标签都没有，回退到原始输出的最后一行
        lines = raw_output.strip().split('\n')
        return _clean_extracted_text(lines[-1]) if lines else ""

    think_content = think_match.group(1)
    
    # --- 2.1. 尝试寻找结论性短语 ---
    conclusion_phrases = [
        r'final\s*answer\s*is[:\s]*', r'the\s*answer\s*is[:\s]*', 
        r'therefore,\s*the\s*answer\s*is[:\s]*', r'the\s*result\s*is[:\s]*', 
        r'equals\s*to[:\s]*', r'is\s*equal\s*to[:\s]*', 
        r'the\s*value\s*is[:\s]*', r'the\s*change\s*is[:\s]*', 
        r'the\s*amount\s*is[:\s]*', r'conclusion[:\s]*', 
        r'final\s*extracted\s*value/calculated\s*result[:\s]*', r'final\s*number[:\s]*',
        r'adjusted\s*net\s*income\s*is[:\s]*', r'percentage\s*change\s*is[:\s]*', 
        r'decreased\s*by[:\s]*', r'increased\s*by[:\s]*',
        r'net\s*change\s*is[:\s]*', r'total\s*is[:\s]*',
        r'resulting\s*in[:\s]*', r'is[:\s]*([-+]?[\d,\.]+%?)' # 捕获"is:"后面直接跟的数字或百分比
    ]
    
    for phrase_pattern in conclusion_phrases:
        # 捕获短语后到下一个标签、双换行符或字符串结束的内容 (非贪婪)
        conclusion_match = re.search(
            f'{phrase_pattern}(.*?)(?:$|<answer>|<think>|\\n\\n|\\Z)', 
            think_content, 
            re.IGNORECASE | re.DOTALL 
        )
        if conclusion_match:
            conclusion = conclusion_match.group(1).strip()
            # 确保提取的内容不包含思考过程中的步骤编号
            if conclusion and re.fullmatch(r'\d+\.', conclusion.split('\n')[0].strip()):
                continue # 如果第一行是步骤编号，跳过
            
            return _clean_extracted_text(conclusion)
    
    # --- 2.2. 如果结论性短语不匹配，尝试寻找最后一个符合数值/百分比/常见格式的字符串 ---
    potential_answers_raw = re.findall(r'([-+]?\s*\(?[\d,\.]+\)?%?)\s*$', think_content, re.MULTILINE) # 捕获组
    if not potential_answers_raw:
        potential_answers_raw = re.findall(r'([-+]?\s*\(?[\d,\.]+\)?%?)', think_content) # 捕获组
    
    if potential_answers_raw:
        for item_raw in reversed(potential_answers_raw):
            item = item_raw.strip()
            if not item: continue
            
            # 排除明显的步骤编号或短语 (如"1.", "2.", "Step 1:")
            if re.fullmatch(r'(\d+\.|\bstep\s*\d+\b)[:\s]*', item, re.IGNORECASE):
                continue

            cleaned_item = _clean_extracted_text(item)
            
            # 简单的验证，确保不是空的或纯粹的标点
            if cleaned_item and len(cleaned_item) > 0 and not re.fullmatch(r'[^\w\s\d%.-]*', cleaned_item):
                return cleaned_item
                
    # --- 2.3. 最后回退：如果以上都失败，取 <think> 内容的最后一行 ---
    lines = [line for line in think_content.strip().split('\n') if line.strip()]
    if lines:
        return _clean_extracted_text(lines[-1])
    return "" # 如果 think 也是空的，返回空字符串

def calculate_f1_score(prediction: str, ground_truth: str) -> float:
    """计算F1分数，包含更鲁棒的归一化，与答案提取逻辑保持高度一致"""
    def normalize_for_f1(text):
        text = text.strip()
        
        # 移除数字中的逗号
        text = text.replace(',', '')
        # 移除负数括号 (例如 "(33)" -> "-33")
        if text.startswith('(') and text.endswith(')'):
            text = '-' + text[1:-1]
        
        # 标准化百分号，确保 "15.2%" 和 "15.2%" 匹配
        text = text.replace('%', ' %').strip()
        text = text.replace(' %', '%')

        # 移除常见的引导词句 (应与 Prompt 优化后减少出现)
        text = re.sub(r'^(the\s*answer\s*is|it\s*was|the\s*value\s*is|resulting\s*in|this\s*represents|the\s*effect\s*is|therefore|so|thus|in\s*conclusion|final\s*answer\s*is|final\s*number\s*is)\s*', '', text, flags=re.IGNORECASE).strip()
        
        # 移除末尾可能的多余标点 (例如句号)
        text = text.rstrip('.')
        
        # 最终全部小写并分割
        return text.lower().split()

    prediction_tokens = normalize_for_f1(prediction)
    ground_truth_tokens = normalize_for_f1(ground_truth)

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

def _parse_template_string_to_messages(template_full_string: str, query: str, context: str = "", table_context: str = "", text_context: str = "") -> List[Dict[str, str]]:
    """
    解析模板字符串为消息列表，支持分离的上下文
    """
    # 替换占位符
    formatted_template = template_full_string.replace("{query}", query)
    
    if table_context and text_context:
        # 使用分离的上下文
        formatted_template = formatted_template.replace("{table_context}", table_context)
        formatted_template = formatted_template.replace("{text_context}", text_context)
    else:
        # 使用统一上下文
        formatted_template = formatted_template.replace("{context_content}", context)
    
    # 解析 ===SYSTEM=== 和 ===USER=== 标签
    system_match = re.search(r'===SYSTEM===(.*?)(?:===USER===|$)', formatted_template, re.DOTALL)
    user_match = re.search(r'===USER===(.*?)(?:===|$)', formatted_template, re.DOTALL)
    
    messages = []
    
    if system_match:
        system_content = system_match.group(1).strip()
        if system_content:
            messages.append({"role": "system", "content": system_content})
    
    if user_match:
        user_content = user_match.group(1).strip()
        if user_content:
            messages.append({"role": "user", "content": user_content})
    
    return messages

def load_and_format_template(template_name: str, context: str, query: str) -> List[Dict[str, str]]:
    """
    加载并格式化指定的prompt模板（统一上下文）
    """
    template_path = Path("data/prompt_templates") / template_name
    try:
        with open(template_path, 'r', encoding='utf-8') as f:
            template_full_string = f.read().strip()
    except FileNotFoundError:
        print(f"❌ 模板文件未找到: {template_path}")
        # 使用默认模板
        template_full_string = """===SYSTEM===
You are a helpful assistant that answers questions based on the provided context.

===USER===
Context: {context_content}

Question: {query}

Answer:"""
    
    return _parse_template_string_to_messages(template_full_string, query, context=context)

def load_and_format_template_with_separated_context(template_name: str, table_context: str, text_context: str, query: str) -> List[Dict[str, str]]:
    """
    加载并格式化指定的prompt模板（分离的上下文）
    """
    template_path = Path("data/prompt_templates") / template_name
    try:
        with open(template_path, 'r', encoding='utf-8') as f:
            template_full_string = f.read().strip()
    except FileNotFoundError:
        print(f"❌ 模板文件未找到: {template_path}")
        # 使用默认模板
        template_full_string = """===SYSTEM===
You are a helpful assistant that answers questions based on the provided context.

===USER===
Table Context: {table_context}

Text Context: {text_context}

Question: {query}

Answer:"""
    
    return _parse_template_string_to_messages(template_full_string, query, table_context=table_context, text_context=text_context)

def get_final_prompt(context: str, query: str) -> List[Dict[str, str]]:
    """使用统一的模板，包含context分离功能"""
    # 始终使用统一的模板
    template_file = 'unified_english_template.txt'
    
    # 使用上下文分离功能
    if USE_CONTEXT_SEPARATOR:
        try:
            # 分离上下文
            separated = context_separator.separate_context(context)
            
            # 格式化 prompt 参数
            prompt_params = context_separator.format_for_prompt(separated, query)
            
            # 使用分离后的上下文格式化模板
            return load_and_format_template_with_separated_context(
                template_file, 
                prompt_params["table_context"], 
                prompt_params["text_context"], 
                query
            )
        except Exception as e:
            print(f"⚠️ 上下文分离失败: {e}，回退到原始方式")
            return load_and_format_template(template_file, context, query)
    else:
        # 回退到原始方式
        return load_and_format_template(template_file, context, query)

# ===================================================================
# 核心评估类
# ===================================================================

class ComprehensiveEvaluator:
    def __init__(self, model_name: str, device: str):
        self.model_name = model_name
        self.device = device
        self.max_new_tokens = 4096
        print("🔄 加载模型...")
        self.generator = LocalLLMGenerator(model_name=self.model_name, device=self.device)
        
        # 注册生成器到资源管理器
        resource_manager.set_generator(self.generator)
        
        print("✅ 模型加载完成")

    def run_evaluation(self, eval_data: List[Dict[str, Any]]) -> Dict[str, Any]:
        results = []
        start_time = time.time()
        pbar = tqdm(eval_data, desc="🔍 评估样本", unit="个")

        try:
            for sample in pbar:
                result = self._evaluate_single_sample(sample)
                results.append(result)
                
                # 定期清理GPU内存
                if len(results) % 5 == 0:
                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()
        except KeyboardInterrupt:
            print("\n🛑 用户中断，开始清理资源...")
            resource_manager.cleanup_resources()
            raise
        except Exception as e:
            print(f"\n❌ 评估过程中出现错误: {e}")
            resource_manager.cleanup_resources()
            raise
        finally:
            total_time = time.time() - start_time
            print(f"\n✅ 评估完成，总耗时: {total_time:.2f}秒")
            print(f"📊 处理了 {len(results)} 个结果")
        
        analysis = self.analyze_results(results)
        analysis['performance'] = {'total_time': total_time, 'avg_time_per_sample': total_time / len(results) if results else 0}
        return {"results": results, "analysis": analysis}

    def _evaluate_single_sample(self, sample: Dict[str, Any]) -> Dict[str, Any]:
        try:
            # 使用统一的模板，包含context分离
            messages = get_final_prompt(sample["context"], sample["query"])
            
            # 转换为文本格式
            prompt_text = self._convert_messages_to_text(messages)

            gen_start_time = time.time()
            # generator.generate 期望 List[str]，所以用 [prompt_text] 包裹
            generation_result = self.generator.generate([prompt_text])[0]
            gen_time = time.time() - gen_start_time
            
            final_answer_to_evaluate = extract_final_answer_with_rescue(generation_result)
            evaluation = self._evaluate_quality(final_answer_to_evaluate, sample["answer"])
            
            return {
                "query": sample["query"],
                "expected_answer": sample["answer"],
                "generated_answer": generation_result,      # 原始模型输出
                "extracted_answer": final_answer_to_evaluate, # 经过 extract_final_answer_with_rescue 处理后的答案
                "evaluation": evaluation,
                "answer_from": sample.get("answer_from", "unknown"), 
                "predicted_answer_from": "unified",
                "decision_confidence": 1.0,
                "is_difficult_decision": False,
                "context_type": "unified",
                "content_ratio": {"table_ratio":0.0, "text_ratio":0.0, "mixed_ratio":0.0},
                "generation_time": gen_time
            }
        except Exception as e:
            # 详细打印错误信息，包括发生错误的样本ID或查询
            sample_id = sample.get("id", "N/A") # 如果你的样本有ID
            print(f"\n❌ 处理样本失败 (ID: {sample_id}, Query: '{sample.get('query', 'N/A')[:50]}...', Error: {e})", file=sys.stderr)
            return {
                "query": sample["query"], 
                "expected_answer": sample["answer"], 
                "error": str(e),
                "context": sample.get("context", "N/A"), # 包含上下文以供调试
                "evaluation": {"exact_match": False, "f1_score": 0.0},
                "generated_answer": "", # 确保有此字段，即使是错误样本
                "extracted_answer": ""  # 确保有此字段，即使是错误样本
            }

    def _evaluate_quality(self, generated: str, expected: str) -> Dict[str, Any]:
        exact_match = generated.strip().lower() == expected.strip().lower()
        f1 = calculate_f1_score(generated, expected)
        return {"exact_match": exact_match, "f1_score": f1}

    def _convert_messages_to_text(self, messages: List[Dict[str, str]]) -> str:
        """
        将 messages 列表转换为Fin-R1（Qwen2.5 based）期望的ChatML格式字符串。
        这是最终传递给 LocalLLMGenerator 的字符串。
        """
        if not messages:
            return ""
        
        # Qwen2.5 使用 ChatML 格式
        formatted_prompt = ""
        for message in messages:
            role = message.get("role", "")
            content = message.get("content", "")
            
            if role == "system":
                formatted_prompt += f"<|im_start|>system\n{content.strip()}<|im_end|>\n"
            elif role == "user":
                formatted_prompt += f"<|im_start|>user\n{content.strip()}<|im_end|>\n"
            elif role == "assistant":
                formatted_prompt += f"<|im_start|>assistant\n{content.strip()}<|im_end|>\n"
        
        # 提示模型开始生成
        formatted_prompt += "<|im_start|>assistant\n" 
        
        return formatted_prompt

    def analyze_results(self, results: List[Dict[str, Any]]) -> Dict[str, Any]:
        print(f"🔍 开始分析 {len(results)} 个结果...")
        
        if not results: 
            print("❌ 没有结果可分析")
            return {}
        
        # 检查结果结构
        valid_results = [r for r in results if 'evaluation' in r]
        error_results = [r for r in results if 'error' in r]
        print(f"✅ 有效结果: {len(valid_results)}, ❌ 错误结果: {len(error_results)}")
        
        all_f1 = [r['evaluation']['f1_score'] for r in valid_results]
        all_em = [r['evaluation']['exact_match'] for r in valid_results]

        analysis = {
            'total_samples': len(results),
            'valid_samples': len(valid_results),
            'error_samples': len(error_results),
            'exact_match_rate': np.mean(all_em) if all_em else 0.0,
            'average_f1_score': np.mean(all_f1) if all_f1 else 0.0,
            'average_generation_time': np.mean([r.get('generation_time', 0) for r in valid_results]) if valid_results else 0.0
        }
        
        return analysis

    def print_summary(self, analysis: Dict[str, Any]):
        """打印评估摘要"""
        print("\n" + "="*80)
        print("📊 全面评估结果摘要")
        print("="*80)
        print("📈 整体指标:")
        print(f"   - 总样本数: {analysis['total_samples']}")
        print(f"   - 有效样本数: {analysis['valid_samples']}")
        print(f"   - 错误样本数: {analysis['error_samples']}")
        print(f"   - 精确匹配率: {analysis['exact_match_rate']:.2%}")
        print(f"   - 平均F1分数: {analysis['average_f1_score']:.4f}")
        print(f"   - 平均生成时间: {analysis['average_generation_time']:.2f}秒")
        
        print("\n💡 性能洞察:")
        if analysis['average_f1_score'] < 0.3:
            print("   - ⚠️ 整体F1分数较低，需要显著改进。")
        elif analysis['average_f1_score'] < 0.6:
            print("   - 🔶 F1分数中等，有改进空间。")
        else:
            print("   - ✅ F1分数良好。")
        
        print("="*80)

def load_evaluation_data(data_path: str, sample_size: Optional[int] = None) -> List[Dict[str, Any]]:
    """加载评估数据"""
    try:
        from utils.data_loader import load_json_or_jsonl, sample_data
        eval_data = load_json_or_jsonl(data_path)
        
        if sample_size and sample_size < len(eval_data):
            eval_data = sample_data(eval_data, sample_size, 42)
            print(f"✅ 随机采样 {len(eval_data)} 个样本进行评估。")
        else:
            print(f"✅ 加载所有 {len(eval_data)} 个样本进行评估。")
        return eval_data
            
    except Exception as e:
        print(f"❌ 数据加载失败 {data_path}: {e}")
        raise

def main():
    parser = argparse.ArgumentParser(description="最终全面评估脚本")
    parser.add_argument("--model", type=str, default="SUFE-AIFLM-Lab/Fin-R1", help="要评估的LLM名称")
    parser.add_argument("--data_path", type=str, required=True, help="评估数据集文件路径 (jsonl 或 json)")
    parser.add_argument("--sample_size", type=int, default=None, help="要评估的随机样本数量 (None表示全部)")
    parser.add_argument("--device", type=str, default="cuda", help="设备 (cuda/cpu/auto)")
    args = parser.parse_args()

    # 设备设置
    device = args.device
    if device == "auto":
        device = "cuda:0" if torch.cuda.is_available() else "cpu"
    elif device == "cuda":
        if not torch.cuda.is_available():
            print("❌ CUDA不可用，回退到CPU")
            device = "cpu"
        else:
            device = "cuda:1"
            print(f"✅ 使用GPU: {torch.cuda.get_device_name(1)}")
            print(f"GPU内存: {torch.cuda.get_device_properties(1).total_memory / 1024**3:.1f} GB")
    
    try:
        # 1. 加载评估数据
        eval_data = load_evaluation_data(args.data_path, args.sample_size)
        
        # 2. 初始化和运行评估器
        evaluator = ComprehensiveEvaluator(model_name=args.model, device=device)
        analysis_results = evaluator.run_evaluation(eval_data)
        
        # 3. 打印和保存结果
        evaluator.print_summary(analysis_results['analysis'])
        output_filename = f"comprehensive_evaluation_results_{time.strftime('%Y%m%d_%H%M%S')}.json"
        with open(output_filename, 'w', encoding='utf-8') as f:
            json.dump(analysis_results, f, indent=2, ensure_ascii=False)
        print(f"\n💾 结果保存到: {output_filename}")
        
    except KeyboardInterrupt:
        print("\n🛑 用户中断程序")
        resource_manager.cleanup_resources()
    except Exception as e:
        print(f"\n❌ 程序执行失败: {e}")
        resource_manager.cleanup_resources()
        raise
    finally:
        # 确保程序结束时清理资源
        resource_manager.cleanup_resources()

if __name__ == "__main__":
    main()