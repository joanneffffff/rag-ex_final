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

def _shared_text_standardizer(text: str) -> str:
    """
    辅助函数，用于标准化文本，供答案提取和F1分数计算使用。
    确保移除逗号、处理负数括号、标准化百分号、移除引导词句、移除末尾标点和货币单位。
    """
    text = text.strip()
    # 移除数字中的逗号
    text = text.replace(',', '')
    # 移除负数括号 (例如 "(33)" -> "-33")
    if text.startswith('(') and text.endswith(')'):
        text = '-' + text[1:-1]
    
    # 标准化百分号，确保 "15.2%" 和 "15.2 %" 匹配
    text = text.replace('%', ' %').strip()
    text = text.replace(' %', '%')

    # 移除常见的引导词句 (应与 Prompt 优化后减少出现)
    text = re.sub(r'^(the\s*answer\s*is|it\s*was|the\s*value\s*is|resulting\s*in|this\s*represents|the\s*effect\s*is|therefore|so|thus|in\s*conclusion|final\s*answer\s*is|final\s*number\s*is)\s*', '', text, flags=re.IGNORECASE).strip()
    
    # 移除末尾可能的多余标点 (例如句号、逗号、分号，但保留百分号)
    text = re.sub(r'[\.。;,]$', '', text).strip()
    
    # 移除常见的货币符号和单位词
    text = re.sub(r'(\$|million|billion|usd|eur|pounds|£)', '', text, flags=re.IGNORECASE).strip()

    return text

def extract_final_answer_with_rescue(raw_output: str) -> str:
    """
    从模型的原始输出中提取最终答案。
    优先寻找 <final-answer> 标签。其次尝试寻找 "FINAL ANSWER: " 前缀。
    如果失败或未找到，则返回空字符串。
    """
    cleaned_output = raw_output.strip()

    # --- 1. 首要尝试：寻找 <final-answer> 标签 (模型实际输出的标签) ---
    # 使用非贪婪匹配 .*? 来捕获 <final-answer> 标签内部的内容
    final_answer_tag_match = re.search(r'<final-answer>(.*?)</final-answer>', cleaned_output, re.DOTALL)
    if final_answer_tag_match:
        content = final_answer_tag_match.group(1).strip()
        # 确认提取的内容非空，且不只是另一个空标签
        if content and content.lower() not in ['<final></final>', '<answer></answer>']:
            return _shared_text_standardizer(content)

    # --- 2. 其次尝试：寻找 "FINAL ANSWER: " 前缀 (Prompt 中指定的格式) ---
    # 这通常会在 <think> 标签内部或其附近，所以我们可以在整个 cleaned_output 中寻找
    final_answer_prefix_match = re.search(r'FINAL ANSWER:\s*(.*?)(?:\n|$)', cleaned_output, re.IGNORECASE | re.DOTALL)
    if final_answer_prefix_match:
        content = final_answer_prefix_match.group(1).strip()
        # 排除模型偶尔会在 FINAL ANSWER: 后跟一个空的标签对
        if content and content.lower() not in ['<final-answer></final-answer>', '<answer></answer>', '<final></final>']:
            return _shared_text_standardizer(content)

    # --- 3. 最后救援：从 <think> 标签内部的最后部分提取 ---
    # 如果以上都没有找到，但模型输出了 <think>，则尝试从其内部内容中提取答案。
    # 这捕获了模型仍然在 <think> 里"思考"并给出答案但没加明确前缀或标签的情况。
    think_match = re.search(r'<think>(.*?)</think>', cleaned_output, re.DOTALL)
    if think_match:
        think_content = think_match.group(1).strip()
        lines = [line.strip() for line in think_content.split('\n') if line.strip()]
        if lines:
            last_line_content = lines[-1]
            # 简单判断最后一行是否可能是答案（例如，包含数字或字母，且不太长）
            if re.search(r'[-+]?\s*\(?[\d,\.]+\)?%?|[a-zA-Z]', last_line_content) and \
               not re.search(r'^(okay|let\'s|wait|but|hmm|alternatively|given|so|thus|first|to|from)', last_line_content, re.IGNORECASE) and \
               len(last_line_content.split()) < 25: # 增加长度限制，排除过长的思考片段
                return _shared_text_standardizer(last_line_content)
    
    # 如果以上所有尝试都失败，返回空字符串
    return ""

def calculate_f1_score(prediction: str, ground_truth: str) -> float:
    """计算F1分数，包含更鲁棒的归一化，与答案提取逻辑保持高度一致"""
    def normalize_for_f1(text):
        return _shared_text_standardizer(text).lower().split() # 调用共享函数
    
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
    解析模板字符串并格式化为消息列表。
    根据当前模板设计，处理分离的上下文，并精确解析 SYSTEM 和 USER 块。
    """
    # 替换模板中的占位符
    formatted_template = template_full_string.replace("{query}", query)
    formatted_template = formatted_template.replace("{table_context}", table_context)
    formatted_template = formatted_template.replace("{text_context}", text_context)
    
    # 解析 ===SYSTEM=== 和 ===USER=== 标签
    # 这里的正则表达式需要精确匹配 SYSTEM/USER 块
    # 使用非贪婪匹配 .*? 和明确的结束边界 (?=...)
    system_match = re.search(r'===SYSTEM===\n(.*?)(?=\n===USER===|$)', formatted_template, re.DOTALL)
    user_match = re.search(r'===USER===\n(.*?)$', formatted_template, re.DOTALL) # 匹配到字符串末尾
    
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
You are a helpful assistant that provides well-reasoned and detailed responses.
You MUST respond with a <think> tag containing your detailed, step-by-step reasoning process. Within the <think> tag, provide the final, direct, and concise answer on a new line prefixed with "FINAL ANSWER: ".

===USER===
Context: {context_content}

Question: {query}
<think>""" # 更新为 <think> 标签开始
    
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
You are a helpful assistant that provides well-reasoned and detailed responses.
You MUST respond with a <think> tag containing your detailed, step-by-step reasoning process. Within the <think> tag, provide the final, direct, and concise answer on a new line prefixed with "FINAL ANSWER: ".

===USER===
Table Context: {table_context}

Text Context: {text_context}

Question: {query}
<think>""" # 更新为 <think> 标签开始
    
    return _parse_template_string_to_messages(template_full_string, query, table_context=table_context, text_context=text_context)

def get_final_prompt(context: str, query: str) -> List[Dict[str, str]]:
    """使用统一的模板，包含context分离功能"""
    # 使用包含受控思考过程的新模板
    template_file = 'unified_english_template_with_controlled_think.txt' # **修改这里**
    
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

            if USE_CONTEXT_SEPARATOR:
                try:
                    # 假设 context_separator.separate_context 能够返回原始的 table/text 长度
                    # 你需要根据 context_separator 的实际返回来获取长度
                    # 示例：
                    # separated_data_raw = context_separator.separate_context_raw(sample["context"]) # 假设有返回原始长度的函数
                    # total_context_length = len(sample["context"])
                    # if total_context_length > 0:
                    #     table_len_ratio = len(separated_data_raw.get('table_content_raw', '')) / total_context_length
                    #     text_len_ratio = len(separated_data_raw.get('text_content_raw', '')) / total_context_length
                    pass # 如果 context_separator 无法提供，则保持为0
                except Exception:
                    pass # 这里的错误已在 get_final_prompt 中处理，此处忽略

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
                "predicted_answer_from": "separated_context_controlled_think", # 更新描述
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
            elif role == "assistant": 
                formatted_prompt += f"<|im_start|>assistant\n{content.strip()}<|im_end|>\n"
        
        # 移除或注释掉这一行，因为Prompt模板的末尾已经有 <think>，
        # 且 <think> 会在 user 消息中被解析，模型会理解这是用户请求后开始生成。
        # formatted_prompt += "<|im_start|>assistant\n" # <-- 移除或注释掉此行
        
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
    parser = argparse.ArgumentParser(description="全面评估脚本")
    parser.add_argument("--model", type=str, default="SUFE-AIFLM-Lab/Fin-R1", help="要评估的LLM名称")
    parser.add_argument("--data_path", type=str, required=True, help="评估数据集文件路径 (jsonl 或 json)")
    parser.add_argument("--sample_size", type=int, default=None, help="要评估的随机样本数量 (None表示全部)")
    parser.add_argument("--device", type=str, default="cuda:0", help="设备 (cuda:0/cuda:1/cpu/auto)")
    
    args = parser.parse_args()
    
    # 设备设置
    device = args.device
    if device == "auto":
        device = "cuda:0" if torch.cuda.is_available() else "cpu"
    elif device.startswith("cuda"): # 检查是否以cuda开头，允许 cuda:0, cuda:1 等
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
    
    # 创建评估器
    evaluator = ComprehensiveEvaluator(args.model, device)
    
    try:
        # 运行评估
        results = evaluator.run_evaluation(eval_data)
        
        # 打印摘要
        evaluator.print_summary(results["analysis"])
        
        # 保存结果
        timestamp = time.strftime("%Y%m%d_%H%M%S")
        output_file = f"comprehensive_evaluation_results_{timestamp}.json"
        
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