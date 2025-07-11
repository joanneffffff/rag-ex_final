#!/usr/bin/env python3
"""
最终版全面评估脚本 - 修复版本2
修复Prompt模板与答案提取逻辑的不匹配问题，统一使用<answer>...</answer>标签格式
目标：使生成器(Fin-R1)的F1分数恢复到并稳定在0.4以上
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
# 核心辅助函数 - 修复版本
# ===================================================================

def _shared_text_standardizer(text: str) -> str:
    """
    Helper function to standardize text for both answer extraction and F1 score calculation.
    Ensures commas are removed, negative numbers in parentheses are handled,
    percentage signs are handled, common introductory phrases are removed,
    trailing punctuation is removed, and currency symbols/unit words are removed.
    """
    text = text.strip()
    # Remove commas from numbers
    text = text.replace(',', '')
    # Handle negative numbers in parentheses (e.g., "(33)" -> "-33")
    if text.startswith('(') and text.endswith(')'):
        text = '-' + text[1:-1]
    
    # Remove common introductory phrases (should be less frequent with optimized prompt)
    # This list should be aligned with phrases you *don't* want in the final answer.
    text = re.sub(r'^(the\s*answer\s*is|it\s*was|the\s*value\s*is|resulting\s*in|this\s*represents|the\s*effect\s*is|therefore|so|thus|in\s*conclusion|final\s*answer\s*is|final\s*number\s*is)\s*', '', text, flags=re.IGNORECASE).strip()
    
    # Remove trailing punctuation (e.g., periods, commas, semicolons, but ensure percentage sign is removed if numeric)
    # This regex is made more aggressive to ensure any trailing punctuation OR a standalone % is removed.
    text = re.sub(r'[\.。;,]$', '', text).strip() # General trailing punctuation
    
    # <<< NEW ADDITION / REVISION >>>: Explicitly remove percentage sign at the end of a numeric string
    # This helps when expected_answer is "0.2" but generated is "0.2%"
    if text.endswith('%'):
        # Check if the part before % is numeric (allows for negative, decimal numbers)
        numeric_part_match = re.fullmatch(r'([-+]?[\d.]+)', text[:-1].strip())
        if numeric_part_match:
            text = numeric_part_match.group(1) # Keep only the numeric part
        else:
            text = text[:-1].strip() # If not purely numeric, just strip the %
    
    # Remove common currency symbols and unit words
    text = re.sub(r'(\$|million|billion|usd|eur|pounds|£)', '', text, flags=re.IGNORECASE).strip()

    return text

def extract_final_answer_with_rescue(raw_output: str) -> str:
    """
    Extracts the final answer from the model's raw output.
    It exclusively looks for the <answer> tag. If not found or empty, it returns a specific phrase.
    This version implements the "I cannot find the answer" explicit fallback.
    """
    cleaned_output = raw_output.strip()
    # Define the specific phrase for "answer not found" in English
    NOT_FOUND_REPLY_ENGLISH = "I cannot find the answer in the provided context."

    # 1. 精确寻找 <answer>...</answer> 标签
    # Use non-greedy matching .*? to capture content inside the tag
    match = re.search(r'<answer>(.*?)</answer>', cleaned_output, re.DOTALL)
    
    if match:
        content = match.group(1).strip()
        # Ensure extracted content is not empty or an empty tag itself (e.g., <answer></answer>)
        if content and content.lower() not in ['<final></final>', '<answer></answer>', '<final-answer></final-answer>']:
            return _shared_text_standardizer(content)
    
    # If no valid <answer> structure is found or content is invalid,
    # return the specific "not found" phrase.
    return NOT_FOUND_REPLY_ENGLISH

def calculate_f1_score(prediction: str, ground_truth: str) -> float:
    # Define the specific phrase for "answer not found" (standardized lowercase form)
    NOT_FOUND_ANSWER_PHRASE = "i cannot find the answer in the provided context."

    # Standardize both prediction and ground truth texts
    normalized_prediction = _shared_text_standardizer(prediction).lower()
    normalized_ground_truth = _shared_text_standardizer(ground_truth).lower()

    # 1. Handle cases where the model explicitly states "I cannot find the answer..."
    if normalized_prediction == NOT_FOUND_ANSWER_PHRASE:
        # If the ground truth is also "I cannot find the answer...", it's a correct match
        if normalized_ground_truth == NOT_FOUND_ANSWER_PHRASE:
            return 1.0
        # Otherwise, the model said "I cannot find..." but the answer exists, so it's an error
        else:
            return 0.0
    
    # 2. Handle cases where the ground truth is "I cannot find the answer...", but the model gave a factual answer (which is an error)
    if normalized_ground_truth == NOT_FOUND_ANSWER_PHRASE:
        return 0.0

    # 3. Standard F1 score calculation for factual answers
    prediction_tokens = normalized_prediction.split()
    ground_truth_tokens = normalized_ground_truth.split()

    if not ground_truth_tokens: 
        return 1.0 if not prediction_tokens else 0.0 # If ground truth is empty, predict empty for 1.0 F1
    if not prediction_tokens: 
        return 0.0 # If prediction is empty, but ground truth is not, 0.0 F1

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
    解析模板字符串并格式化为消息列表。
    根据当前模板设计，处理分离的上下文，并精确解析 SYSTEM 和 USER 块。
    """
    # 替换模板中的占位符
    formatted_template = template_full_string.replace("{query}", query)
    formatted_template = formatted_template.replace("{table_context}", table_context)
    formatted_template = formatted_template.replace("{text_context}", text_context)
    
    # --- 关键的正则表达式调整 ---
    # SYSTEM 块：从 ===SYSTEM=== 后到下一个 ===USER=== 或字符串末尾
    # 使用 \s* 匹配可能存在的空格或换行符
    system_match = re.search(r'===SYSTEM===\s*\n(.*?)(?=\n===USER===|\Z)', formatted_template, re.DOTALL)
    # USER 块：从 ===USER=== 后到字符串末尾
    # 使用 \s* 匹配可能存在的空格或换行符
    user_match = re.search(r'===USER===\s*\n(.*?)\Z', formatted_template, re.DOTALL) 
    
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
    注意：此函数在新流程中可能不直接使用，但其默认模板已更新。
    """
    template_path = Path("data/prompt_templates") / template_name
    try:
        with open(template_path, 'r', encoding='utf-8') as f:
            template_full_string = f.read().strip()
    except FileNotFoundError:
        print(f"❌ 模板文件未找到: {template_path}")
        # 使用与新策略一致的默认模板
        template_full_string = """===SYSTEM===
You are a helpful assistant that answers questions based on the provided context.
Your ONLY output MUST be the final, direct, and concise answer enclosed STRICTLY within an <answer> tag. You MUST NOT include any thinking process, intermediate steps, or conversational filler outside this tag.

===USER===
Context: {context_content}

Question: {query}
<answer>""" # 确保这里是 <answer>
    
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
        # 使用与新策略一致的默认模板
        template_full_string = """===SYSTEM===
You are a helpful assistant that answers questions based on the provided context.
Your ONLY output MUST be the final, direct, and concise answer enclosed STRICTLY within an <answer> tag. You MUST NOT include any thinking process, intermediate steps, or conversational filler outside this tag.

===USER===
Table Context: {table_context}

Text Context: {text_context}

Question: {query}
<answer>""" # 确保这里是 <answer>
    
    return _parse_template_string_to_messages(template_full_string, query, table_context=table_context, text_context=text_context)

def get_final_prompt(context: str, query: str) -> List[Dict[str, str]]:
    """使用统一的模板，包含context分离功能"""
    # 使用我们最新确定的 Prompt 模板文件名
    template_file = 'unified_english_template_no_think.txt' # **请务必确保这个文件名与您保存的模板文件名一致**
    
    # 强制使用上下文分离功能
    if not USE_CONTEXT_SEPARATOR:
        print("❌ 未启用上下文分离功能，但当前Prompt模板要求分离上下文。脚本将无法继续。")
        raise NotImplementedError("当前Prompt模板要求上下文分离，但功能未启用。请检查 USE_CONTEXT_SEPARATOR 配置。")

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
        # 如果上下文分离失败，无法构造带有 table_context/text_context 的 prompt，
        # 并且新模板不兼容统一上下文，则直接抛出错误。
        print(f"❌ 上下文分离失败，且无法使用兼容的Prompt模板。错误: {e}", file=sys.stderr)
        raise # 强制抛出错误，因为无法正确构造 Prompt

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
            table_len_ratio, text_len_ratio = 0.0, 0.0 # 初始化为0

            # 确保在调用 get_final_prompt 之前获取这些比例，如果 context_separator 支持
            if USE_CONTEXT_SEPARATOR:
                try:
                    # 假设 context_separator.separate_context 能够返回原始的 table/text 长度
                    # 这需要 context_separator 模块的支持
                    separated_data = context_separator.separate_context(sample["context"])
                    # 假设 separated_data 包含 'table_content_length' 和 'text_content_length'
                    # 如果 context_separator 不提供，则此部分跳过或自行计算
                    total_context_length = len(sample["context"])
                    if total_context_length > 0:
                        # 这是一个示例，您需要根据 context_separator 的实际返回来获取长度
                        # 例如：table_len = len(separated_data.get('table_context_raw', ''))
                        # text_len = len(separated_data.get('text_context_raw', ''))
                        # table_len_ratio = table_len / total_context_length
                        # text_len_ratio = text_len / total_context_length
                        pass # 如果 context_separator 无法提供，则保持为0
                except Exception as e:
                    # 分离失败会在 get_final_prompt 中处理，这里捕获是为了避免重复错误信息
                    pass

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
                "predicted_answer_from": "separated_context_answer_only",
                "decision_confidence": 1.0,
                "is_difficult_decision": False,
                "context_type": "separated_context",
                "content_ratio": {"table_ratio": table_len_ratio, "text_ratio": text_len_ratio, "mixed_ratio": table_len_ratio + text_len_ratio}, # mixed_ratio 可以是两者之和
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
            elif role == "assistant": # 示例中可能会有 assistant 轮次
                formatted_prompt += f"<|im_start|>assistant\n{content.strip()}<|im_end|>\n"
        
        # <<< 关键修改 >>>
        # 移除或注释掉这一行，因为Prompt模板的末尾是用户消息的一部分（以 <think> 结尾），
        # 模型会根据ChatML的规则自动在用户消息后生成助手回应，无需额外添加 <|im_start|>assistant。
        # formatted_prompt += "<|im_start|>assistant\n" 
        
        return formatted_prompt

    def analyze_results(self, results: List[Dict[str, Any]]) -> Dict[str, Any]:
        """分析评估结果"""
        if not results:
            return {"error": "没有有效结果可分析"}
        
        # 过滤掉有错误的样本
        valid_results = [r for r in results if "error" not in r]
        error_count = len(results) - len(valid_results)
        
        if not valid_results:
            return {"error": "所有样本都有错误", "error_count": error_count}
        
        # 计算基本指标
        exact_matches = sum(1 for r in valid_results if r["evaluation"]["exact_match"])
        f1_scores = [r["evaluation"]["f1_score"] for r in valid_results]
        
        # 计算统计信息
        avg_f1 = np.mean(f1_scores)
        std_f1 = np.std(f1_scores)
        median_f1 = np.median(f1_scores)
        
        # 计算时间统计
        generation_times = [r.get("generation_time", 0) for r in valid_results]
        avg_gen_time = np.mean(generation_times) if generation_times else 0
        
        return {
            "total_samples": len(results),
            "valid_samples": len(valid_results),
            "error_samples": error_count,
            "exact_match_count": exact_matches,
            "exact_match_rate": exact_matches / len(valid_results),
            "avg_f1_score": avg_f1,
            "std_f1_score": std_f1,
            "median_f1_score": median_f1,
            "min_f1_score": min(f1_scores),
            "max_f1_score": max(f1_scores),
            "avg_generation_time": avg_gen_time,
            "performance": {
                "total_time": sum(generation_times),
                "avg_time_per_sample": avg_gen_time
            }
        }

    def print_summary(self, analysis: Dict[str, Any]):
        """打印评估摘要"""
        print("\n" + "="*60)
        print("📊 评估结果摘要")
        print("="*60)
        
        if "error" in analysis:
            print(f"❌ 分析失败: {analysis['error']}")
            return
        
        print(f"📈 样本统计:")
        print(f"   总样本数: {analysis['total_samples']}")
        print(f"   有效样本: {analysis['valid_samples']}")
        print(f"   错误样本: {analysis['error_samples']}")
        
        print(f"\n🎯 准确率指标:")
        print(f"   精确匹配数: {analysis['exact_match_count']}")
        print(f"   精确匹配率: {analysis['exact_match_rate']:.4f} ({analysis['exact_match_rate']*100:.2f}%)")
        
        print(f"\n📊 F1分数统计:")
        print(f"   平均F1: {analysis['avg_f1_score']:.4f}")
        print(f"   标准差: {analysis['std_f1_score']:.4f}")
        print(f"   中位数: {analysis['median_f1_score']:.4f}")
        print(f"   最小值: {analysis['min_f1_score']:.4f}")
        print(f"   最大值: {analysis['max_f1_score']:.4f}")
        
        print(f"\n⏱️ 性能统计:")
        print(f"   平均生成时间: {analysis['avg_generation_time']:.3f}秒")
        if 'performance' in analysis:
            print(f"   总处理时间: {analysis['performance']['total_time']:.2f}秒")
            print(f"   平均样本时间: {analysis['performance']['avg_time_per_sample']:.3f}秒")
        
        print("="*60)

def load_evaluation_data(data_path: str, sample_size: Optional[int] = None) -> List[Dict[str, Any]]:
    """加载评估数据"""
    print(f"📖 正在从 {data_path} 加载数据...")
    
    try:
        if data_path.endswith('.jsonl'):
            data = []
            with open(data_path, 'r', encoding='utf-8') as f:
                for line in f:
                    if line.strip(): # 避免空行
                        data.append(json.loads(line))
            print(f"✅ 成功加载为JSONL，样本数: {len(data)}")
        else: # 假设是 .json 文件
            with open(data_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
            # 确保数据是列表格式 (兼容 TATQA 常见的 JSON 格式)
            if isinstance(data, dict):
                if "data" in data: # TATQA数据集通常是 {"data": [...]}
                    data = data["data"]
                elif "samples" in data:
                    data = data["samples"]
                else: # 尝试找到包含列表的键
                    found_list = False
                    for key, value in data.items():
                        if isinstance(value, list):
                            data = value
                            found_list = True
                            break
                    if not found_list:
                        raise ValueError("无法在JSON文件中找到样本数据列表")
            if not isinstance(data, list):
                raise ValueError("数据必须是样本列表格式")
            print(f"✅ 成功加载为JSON数组，样本数: {len(data)}")
        
        # 限制样本数量
        if sample_size and sample_size < len(data):
            eval_data = data[:sample_size] # 使用切片
            print(f"✅ 限制为前 {len(eval_data)} 个样本进行评估。")
        else:
            eval_data = data # 复制一份，避免原数据被修改的风险
            print(f"✅ 加载所有 {len(eval_data)} 个样本进行评估。")
        
        return eval_data
        
    except FileNotFoundError:
        print(f"❌ 文件未找到: {data_path}")
        sys.exit(1)
    except json.JSONDecodeError as e:
        print(f"❌ JSON解析错误: {e}")
        sys.exit(1)
    except Exception as e:
        print(f"❌ 数据加载错误: {e}")
        sys.exit(1)

def main():
    parser = argparse.ArgumentParser(description="全面评估脚本 - 修复版本2")
    parser.add_argument("--model", type=str, default=None, help="要评估的LLM名称（默认使用config/parameters.py中的设置）")
    parser.add_argument("--data_path", type=str, required=True, help="评估数据集文件路径 (jsonl 或 json)")
    parser.add_argument("--sample_size", type=int, default=None, help="要评估的随机样本数量 (None表示全部)")
    parser.add_argument("--device", type=str, default=None, help="设备 (cuda:0/cuda:1/cpu/auto，默认使用config/parameters.py中的设置）")
    
    args = parser.parse_args()
    
    # 使用配置文件中的设置
    try:
        from config.parameters import config
        print("📖 加载配置文件设置...")
        
        # 设置模型名称
        if args.model is None:
            args.model = config.generator.model_name
            print(f"📖 使用配置文件中的模型: {args.model}")
        else:
            print(f"📖 使用命令行指定的模型: {args.model}")
        
        # 设置设备
        if args.device is None:
            args.device = config.generator.device
            print(f"📖 使用配置文件中的设备: {args.device}")
        else:
            print(f"📖 使用命令行指定的设备: {args.device}")
            
    except ImportError:
        print("⚠️ 无法导入config/parameters.py，使用默认设置")
        if args.model is None:
            args.model = "SUFE-AIFLM-Lab/Fin-R1"
        if args.device is None:
            args.device = "cuda:0"
    
    # 设备设置
    device = args.device or "cuda:0"  # 确保device不为None
    if device == "auto":
        device = "cuda:0" if torch.cuda.is_available() else "cpu"
    elif device and device.startswith("cuda"): # 检查是否以cuda开头，允许 cuda:0, cuda:1 等
        try:
            if not torch.cuda.is_available():
                print("❌ CUDA不可用，回退到CPU")
                device = "cpu"
            else:
                # 尝试解析具体的GPU ID
                device_id = int(device.split(':')[1]) if ':' in device else 0
                if device_id >= torch.cuda.device_count():
                    print(f"❌ GPU ID {device_id} 不可用，最大ID为 {torch.cuda.device_count() - 1}。回退到cuda:0")
                    device = "cuda:0"
                else:
                    device = f"cuda:{device_id}"
                    print(f"✅ 使用GPU: {torch.cuda.get_device_name(device_id)}")
                    print(f"GPU内存: {torch.cuda.get_device_properties(device_id).total_memory / 1024**3:.1f} GB")
        except (ValueError, IndexError): # 处理 cuda:bad_id 的情况
            print(f"❌ 无效的CUDA设备参数 '{args.device}'。回退到cuda:0")
            device = "cuda:0"
    else:
        print(f"⚠️ 使用设备: {device}")
    
    # 加载数据
    eval_data = load_evaluation_data(args.data_path, args.sample_size)
    
    # 确保模型名称不为None
    model_name = args.model or "SUFE-AIFLM-Lab/Fin-R1"
    
    # 创建评估器
    evaluator = ComprehensiveEvaluator(model_name, device)
    
    try:
        # 运行评估
        results = evaluator.run_evaluation(eval_data)
        
        # 打印摘要
        evaluator.print_summary(results["analysis"])
        
        # 保存结果
        timestamp = time.strftime("%Y%m%d_%H%M%S")
        output_file = f"comprehensive_evaluation_results_fixed_v2_{timestamp}.json"
        
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(results, f, ensure_ascii=False, indent=2)
        
        print(f"\n💾 结果已保存到: {output_file}")
        
    except KeyboardInterrupt:
        print("\n🛑 用户中断评估")
    except Exception as e:
        print(f"\n❌ 评估过程中出现错误: {e}")
        raise
    finally:
        # 确保资源清理
        resource_manager.cleanup_resources()

if __name__ == "__main__":
    main()