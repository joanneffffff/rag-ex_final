#!/usr/bin/env python3
"""
TatQA英文LLM模型对比评估脚本 - 基于comprehensive_evaluation_enhanced.py的逻辑
支持Fin-R1和Qwen3-8B在TatQA英文数据集上的表现对比
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

# 环境设置
warnings.filterwarnings("ignore")
logging.getLogger("transformers").setLevel(logging.ERROR)
os.environ['TRANSFORMERS_VERBOSITY'] = 'error'

try:
    from tqdm import tqdm
except ImportError:
    print("❌ tqdm未安装，请运行: pip install tqdm")
    sys.exit(1)

from transformers import AutoTokenizer, AutoModelForCausalLM

# ===================================================================
# 核心辅助函数 (基于comprehensive_evaluation_enhanced.py)
# ===================================================================

def extract_final_answer_with_rescue(raw_output: str) -> str:
    """
    从模型的原始输出中智能提取最终答案 (支持中英文)
    """
    def _clean_extracted_text(text: str) -> str:
        """对提取出的文本进行通用清理"""
        text = text.strip()
        # 移除数字中的逗号
        text = text.replace(',', '')
        # 移除负数括号 (例如 "(33)" -> "-33")
        if text.startswith('(') and text.endswith(')'):
            text = '-' + text[1:-1]
        
        # 标准化百分号
        text = text.replace('%', ' %').strip()
        text = text.replace(' %', '%')

        # 移除常见的引导词句
        text = re.sub(r'^(the\s*answer\s*is|it\s*was|the\s*value\s*is|resulting\s*in|this\s*represents|the\s*effect\s*is|therefore|so|thus|in\s*conclusion|final\s*answer\s*is|final\s*number\s*is)\s*', '', text, flags=re.IGNORECASE).strip()
        
        # 移除末尾可能的多余标点符号
        text = re.sub(r'[\.;,]$', '', text).strip()
        
        # 移除常见的货币符号和单位词
        text = re.sub(r'(\$|million|billion|usd|eur|pounds|£)', '', text, flags=re.IGNORECASE).strip()

        return text

    # 1. 尝试从 <answer> 标签中提取
    answer_match = re.search(r'<answer>(.*?)</answer>', raw_output, re.DOTALL)
    if answer_match:
        content = answer_match.group(1).strip()
        if content:
            return _clean_extracted_text(content)

    # 2. 救援逻辑：从 <think> 标签中提取
    think_match = re.search(r'<think>(.*?)(?:</think>|$)', raw_output, re.DOTALL)
    if not think_match:
        # 如果连 <think> 标签都没有，回退到原始输出的最后一行
        lines = raw_output.strip().split('\n')
        return _clean_extracted_text(lines[-1]) if lines else ""

    think_content = think_match.group(1)
    
    # 尝试寻找结论性短语
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
        r'resulting\s*in[:\s]*', r'is[:\s]*([-+]?[\d,\.]+%?)'
    ]
    
    for phrase_pattern in conclusion_phrases:
        conclusion_match = re.search(
            f'{phrase_pattern}(.*?)(?:$|<answer>|<think>|\\n\\n|\\Z)', 
            think_content, 
            re.IGNORECASE | re.DOTALL 
        )
        if conclusion_match:
            conclusion = conclusion_match.group(1).strip()
            if conclusion and re.fullmatch(r'\d+\.', conclusion.split('\n')[0].strip()):
                continue
            
            return _clean_extracted_text(conclusion)
    
    # 尝试寻找最后一个符合数值/百分比格式的字符串
    potential_answers_raw = re.findall(r'([-+]?\s*\(?[\d,\.]+\)?%?)\s*$', think_content, re.MULTILINE)
    if not potential_answers_raw:
        potential_answers_raw = re.findall(r'([-+]?\s*\(?[\d,\.]+\)?%?)', think_content)
    
    if potential_answers_raw:
        for item_raw in reversed(potential_answers_raw):
            item = item_raw.strip()
            if not item: continue
            
            if re.fullmatch(r'(\d+\.|\bstep\s*\d+\b)[:\s]*', item, re.IGNORECASE):
                continue

            cleaned_item = _clean_extracted_text(item)
            
            if cleaned_item and len(cleaned_item) > 0 and not re.fullmatch(r'[^\w\s\d%.-]*', cleaned_item):
                return cleaned_item
                
    # 最后回退：取 <think> 内容的最后一行
    lines = [line for line in think_content.strip().split('\n') if line.strip()]
    if lines:
        return _clean_extracted_text(lines[-1])
    return ""


def calculate_f1_score(prediction: str, ground_truth: str) -> float:
    """计算F1分数，包含更鲁棒的归一化"""
    def normalize_for_f1(text):
        text = text.strip()
        
        # 移除数字中的逗号
        text = text.replace(',', '')
        # 移除负数括号
        if text.startswith('(') and text.endswith(')'):
            text = '-' + text[1:-1]
        
        # 标准化百分号
        text = text.replace('%', ' %').strip()
        text = text.replace(' %', '%')

        # 移除常见的引导词句
        text = re.sub(r'^(the\s*answer\s*is|it\s*was|the\s*value\s*is|resulting\s*in|this\s*represents|the\s*effect\s*is|therefore|so|thus|in\s*conclusion|final\s*answer\s*is|final\s*number\s*is)\s*', '', text, flags=re.IGNORECASE).strip()
        
        # 移除末尾可能的多余标点
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


def calculate_exact_match(prediction: str, ground_truth: str) -> float:
    """计算精确匹配率"""
    def normalize_for_em(text):
        text = text.strip().lower()
        text = text.replace(',', '')
        if text.startswith('(') and text.endswith(')'):
            text = '-' + text[1:-1]
        text = text.replace('%', ' %').strip().replace(' %', '%')
        text = re.sub(r'^(the\s*answer\s*is|it\s*was|the\s*value\s*is|resulting\s*in|this\s*represents|the\s*effect\s*is|therefore|so|thus|in\s*conclusion|final\s*answer\s*is|final\s*number\s*is)\s*', '', text, flags=re.IGNORECASE).strip()
        text = text.rstrip('.')
        return text
    
    return 1.0 if normalize_for_em(prediction) == normalize_for_em(ground_truth) else 0.0


# ===================================================================
# 上下文类型判断和决策逻辑 (基于comprehensive_evaluation_enhanced.py)
# ===================================================================

def determine_context_type(context: str) -> str:
    """判断上下文类型：table, text, 或 mixed"""
    # 检查是否包含表格特征
    table_indicators = [
        r'\|\s*[^|]+\s*\|',  # 表格分隔符
        r'Table\s*\d+',      # 表格标题
        r'Row\s*\d+',        # 行标识
        r'Column\s*\d+',     # 列标识
        r'Header\s*[:\s]',   # 表头
        r'Data\s*[:\s]',     # 数据标识
    ]
    
    text_indicators = [
        r'Paragraph\s*\d+',  # 段落标识
        r'Section\s*\d+',    # 章节标识
        r'Report\s*[:\s]',   # 报告标识
        r'Summary\s*[:\s]',  # 摘要标识
    ]
    
    table_score = sum(len(re.findall(pattern, context, re.IGNORECASE)) for pattern in table_indicators)
    text_score = sum(len(re.findall(pattern, context, re.IGNORECASE)) for pattern in text_indicators)
    
    if table_score > text_score:
        return "table"
    elif text_score > table_score:
        return "text"
    else:
        return "mixed"


def analyze_query_features(query: str) -> Dict[str, Any]:
    """分析查询特征"""
    features = {
        "length": len(query),
        "has_numbers": bool(re.search(r'\d+', query)),
        "has_percentages": bool(re.search(r'\d+%', query)),
        "has_currency": bool(re.search(r'[\$£€]', query)),
        "has_comparison": bool(re.search(r'(higher|lower|more|less|increase|decrease|change|difference)', query, re.IGNORECASE)),
        "has_calculation": bool(re.search(r'(calculate|compute|sum|total|average|mean|percentage)', query, re.IGNORECASE)),
        "question_type": "calculation" if re.search(r'(what\s*is|how\s*much|calculate|compute)', query, re.IGNORECASE) else "extraction"
    }
    return features


def calculate_content_ratio(context: str) -> Dict[str, float]:
    """计算内容比例"""
    total_chars = len(context)
    if total_chars == 0:
        return {"table_ratio": 0.0, "text_ratio": 0.0}
    
    # 简单的启发式方法
    table_chars = len(re.findall(r'[|+\-]', context))  # 表格分隔符
    text_chars = len(re.findall(r'[a-zA-Z]', context))  # 字母字符
    
    table_ratio = table_chars / total_chars if total_chars > 0 else 0.0
    text_ratio = text_chars / total_chars if total_chars > 0 else 0.0
    
    return {"table_ratio": table_ratio, "text_ratio": text_ratio}


def hybrid_decision_enhanced(context: str, query: str) -> Dict[str, Any]:
    """增强的混合决策算法"""
    context_type = determine_context_type(context)
    query_features = analyze_query_features(query)
    content_ratio = calculate_content_ratio(context)
    
    # 决策逻辑
    decision_factors = {
        "context_type_weight": 0.4,
        "query_features_weight": 0.4,
        "content_ratio_weight": 0.2
    }
    
    # 基于上下文类型的分数
    context_scores = {
        "table": 0.8 if context_type == "table" else 0.2,
        "text": 0.8 if context_type == "text" else 0.2,
        "mixed": 0.6
    }
    
    # 基于查询特征的分数
    query_scores = {
        "table": 0.7 if query_features["has_calculation"] or query_features["has_comparison"] else 0.3,
        "text": 0.7 if query_features["question_type"] == "extraction" else 0.3,
        "mixed": 0.5
    }
    
    # 基于内容比例的分数
    ratio_scores = {
        "table": content_ratio["table_ratio"],
        "text": content_ratio["text_ratio"],
        "mixed": 0.5
    }
    
    # 计算最终分数
    final_scores = {}
    for context_type_key in ["table", "text", "mixed"]:
        final_scores[context_type_key] = (
            context_scores[context_type_key] * decision_factors["context_type_weight"] +
            query_scores[context_type_key] * decision_factors["query_features_weight"] +
            ratio_scores[context_type_key] * decision_factors["content_ratio_weight"]
        )
    
    # 选择最高分数的类型
    best_type = max(final_scores.keys(), key=lambda k: final_scores[k])
    
    return {
        "decision": best_type,
        "confidence": final_scores[best_type],
        "scores": final_scores,
        "context_type": context_type,
        "query_features": query_features,
        "content_ratio": content_ratio
    }


# ===================================================================
# 模板加载和格式化 (基于comprehensive_evaluation_enhanced.py)
# ===================================================================

def _load_template_content_from_file(template_file_name: str) -> str:
    """从指定文件中加载Prompt模板的完整字符串内容"""
    template_path = Path("data/prompt_templates") / template_file_name
    try:
        with open(template_path, 'r', encoding='utf-8') as f:
            return f.read().strip()
    except FileNotFoundError:
        print(f"❌ 模板文件未找到: {template_path}，请确保文件存在。")
        sys.exit(1)


def _parse_template_string_to_messages(template_full_string: str, query: str, context: str = "", table_context: str = "", text_context: str = "") -> List[Dict[str, str]]:
    """解析模板字符串为messages格式"""
    messages = []
    
    # 使用正则表达式分割所有部分，并保留分隔符内容
    parts = re.split(r'(===SYSTEM===|===USER===|===ASSISTANT===)', template_full_string, flags=re.DOTALL)
    
    # 移除第一个空字符串（如果存在）和多余的空白
    parts = [p.strip() for p in parts if p.strip()]

    # 遍历 parts 列表，重新组合 role 和 content
    for i in range(0, len(parts), 2):
        if i + 1 < len(parts):
            role_tag_raw = parts[i].strip()
            content = parts[i+1].strip()
            
            role = None
            if role_tag_raw == "===SYSTEM===": role = "system"
            elif role_tag_raw == "===USER===": role = "user"
            elif role_tag_raw == "===ASSISTANT===": role = "assistant"
            
            if role and content:
                # 替换占位符
                if role == "user":
                    content = content.replace('{query}', query)
                    content = content.replace('{context}', context)
                    content = content.replace('{table_context}', table_context)
                    content = content.replace('{text_context}', text_context)
                
                messages.append({"role": role, "content": content})
                
    return messages


def load_and_format_template(template_name: str, context: str, query: str) -> List[Dict[str, str]]:
    """加载并格式化模板"""
    template_full_string = _load_template_content_from_file(template_name)
    return _parse_template_string_to_messages(template_full_string, query, context)


def get_final_prompt(context: str, query: str) -> List[Dict[str, str]]:
    """获取最终的prompt"""
    # 使用混合决策算法
    decision_result = hybrid_decision_enhanced(context, query)
    decision = decision_result["decision"]
    
    # 根据决策选择模板 (AlphaFin使用中文模板)
    if decision == "table":
        template_name = "template_for_table_answer.txt"
    elif decision == "text":
        template_name = "template_for_text_answer.txt"
    else:  # mixed
        template_name = "template_for_hybrid_answer.txt"
    
    try:
        return load_and_format_template(template_name, context, query)
    except Exception as e:
        print(f"⚠️ 模板加载失败，使用默认模板: {e}")
        # 回退到默认模板
        return load_and_format_template("default_template.txt", context, query)


# ===================================================================
# 模型加载和生成器包装类
# ===================================================================

class ModelLoader:
    """负责加载和卸载模型，并提供生成接口"""
    def __init__(self, model_name: str):
        self.model_name = model_name
        self.tokenizer = None
        self.model = None
        self.device = "cuda:0" if torch.cuda.is_available() else "cpu"
        self.is_loaded = False

        if "Fin-R1" in model_name: 
            self.model_path = "/users/sgjfei3/data/huggingface/models--SUFE-AIFLM-Lab--Fin-R1/snapshots/026768c4a015b591b54b240743edeac1de0970fa" 
        elif "Qwen3-8B" in model_name:
            self.model_path = "Qwen/Qwen2.5-7B-Instruct"
        else:
            self.model_path = model_name 
            print(f"⚠️ 模型路径 '{model_name}' 未知，尝试从Hugging Face Hub加载。")

    def load_model(self):
        if self.is_loaded:
            print(f"✅ {self.model_name} 已加载，无需重复加载。")
            return
        
        print(f"🔄 加载模型: {self.model_name} 从 {self.model_path}")
        is_local_path = isinstance(self.model_path, str) and "snapshots" in self.model_path

        tokenizer_args = {"trust_remote_code": True, "local_files_only": is_local_path}
        model_args = {"torch_dtype": torch.float16, "device_map": "auto", "trust_remote_code": True, 
                      "load_in_8bit": True, "local_files_only": is_local_path} 

        try:
            print("🔧 加载tokenizer...")
            self.tokenizer = AutoTokenizer.from_pretrained(self.model_path, **tokenizer_args)
            if self.tokenizer.pad_token is None: 
                self.tokenizer.pad_token = self.tokenizer.eos_token
            if self.tokenizer.pad_token_id is None: 
                self.tokenizer.pad_token_id = self.tokenizer.eos_token_id
            print(f"✅ {self.model_name} Tokenizer加载完成.")

            print("🔧 加载模型...")
            self.model = AutoModelForCausalLM.from_pretrained(self.model_path, **model_args)
            self.model.eval()
            print(f"✅ {self.model_name} 模型加载完成. 设备: {self.model.device}, 量化: 8bit")
            self.is_loaded = True
        except Exception as e:
            print(f"❌ {self.model_name} 模型加载失败: {e}")
            import traceback
            traceback.print_exc()
            self.unload_model()
            raise

    def unload_model(self):
        if not self.is_loaded:
            return
        
        print(f"🗑️ 卸载模型: {self.model_name} 并清理显存...")
        try:
            if self.model:
                self.model.to('cpu')
                del self.model
            if self.tokenizer:
                del self.tokenizer
            self.model = None
            self.tokenizer = None
            torch.cuda.empty_cache()
            gc.collect()
            self.is_loaded = False
            print(f"✅ {self.model_name} 显存已清理。")
        except Exception as e:
            print(f"❌ 卸载 {self.model_name} 时发生错误: {e}")

    def generate(self, prompt_string: str, max_new_tokens: int = 150, do_sample: bool = False, repetition_penalty: float = 1.1) -> str:
        """生成文本"""
        if not self.is_loaded or self.model is None or self.tokenizer is None:
            raise RuntimeError(f"模型 {self.model_name} 未加载。请先调用 load_model()。")

        inputs = self.tokenizer(prompt_string, return_tensors="pt", padding=True, truncation=True).to(self.model.device)
        
        with torch.no_grad():
            outputs = self.model.generate( 
                **inputs,
                max_new_tokens=max_new_tokens,
                do_sample=do_sample,
                pad_token_id=self.tokenizer.pad_token_id,
                eos_token_id=self.tokenizer.eos_token_id,
                repetition_penalty=repetition_penalty
            )
        
        generated_tokens = outputs[0, inputs["input_ids"].shape[1]:]
        generated_text = self.tokenizer.decode(generated_tokens, skip_special_tokens=True) 
        
        return generated_text


def _convert_messages_to_chatml(messages: List[Dict[str, str]]) -> str:
    """将messages列表转换为ChatML格式字符串"""
    if not messages:
        return ""
    
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
    
    formatted_prompt += "<|im_start|>assistant\n" 
    
    return formatted_prompt


# ===================================================================
# 主评估逻辑
# ===================================================================

def run_english_comparison_test():
    print("🚀 模型对比测试开始...")
    
    # 配置要测试的模型
    model_loaders = {
        "Fin-R1": ModelLoader("Fin-R1"),
        "Qwen3-8B": ModelLoader("Qwen3-8B")
    }

    # 测试配置
    data_path = "evaluate_mrr/tatqa_eval_enhanced.jsonl"  # 默认使用TatQA英文数据集
    sample_size = 500  # 随机采样数量
    
    # 加载数据集
    print(f"📊 加载数据集: {data_path}")
    try:
        dataset = []
        with open(data_path, 'r', encoding='utf-8') as f:
            for line in f:
                if line.strip():
                    dataset.append(json.loads(line))
        
        if sample_size > 0 and sample_size < len(dataset):
            import random
            random.seed(42)
            dataset = random.sample(dataset, sample_size)
            print(f"✅ 随机采样 {len(dataset)} 个样本进行评估。")
        else:
            print(f"✅ 加载了全部 {len(dataset)} 个样本进行评估。")
            
    except Exception as e:
        print(f"❌ 数据集加载失败: {e}")
        return

    all_results_data = []

    # 逐个模型进行评估
    for model_name, loader in model_loaders.items():
        current_model_results = []
        total_f1_model = 0.0
        total_em_model = 0.0
        total_generation_time_model = 0.0
        
        try:
            loader.load_model()
            
            pbar = tqdm(dataset, desc=f"评估 {model_name}")
            for i, item in enumerate(pbar):
                # 兼容多种查询字段名
                query = item.get("query", "") or item.get("generated_question", "") or item.get("question", "")
                context_data = item.get("context", "")
                expected_answer = item.get("answer", "")
                doc_id = item.get("doc_id", f"sample_{i}")  # 添加doc_id支持

                # 使用混合决策算法获取prompt
                messages = get_final_prompt(context_data, query)
                
                # 转换为ChatML格式
                prompt_string_for_model = _convert_messages_to_chatml(messages)
                
                start_time = time.time()
                generated_text = loader.generate(
                    prompt_string=prompt_string_for_model,
                    max_new_tokens=150,
                    do_sample=False, 
                    repetition_penalty=1.1
                )
                generation_time = time.time() - start_time
                
                # 使用智能答案提取
                final_answer = extract_final_answer_with_rescue(generated_text)
                
                f1 = calculate_f1_score(final_answer, expected_answer)
                em = calculate_exact_match(final_answer, expected_answer)

                total_f1_model += f1
                total_em_model += em
                total_generation_time_model += generation_time

                current_model_results.append({
                    "model": model_name,
                    "sample_id": i,
                    "doc_id": doc_id,
                    "query": query,
                    "expected_answer": expected_answer,
                    "raw_generated_text": generated_text,
                    "final_answer": final_answer,
                    "f1_score": f1,
                    "exact_match": em,
                    "generation_time": generation_time
                })

            # 打印当前模型的汇总结果
            num_samples_evaluated = len(dataset)
            avg_f1 = total_f1_model / num_samples_evaluated
            avg_em = total_em_model / num_samples_evaluated
            avg_gen_time = total_generation_time_model / num_samples_evaluated

            print(f"\n--- {model_name} 评估总结 ---")
            print(f"总样本数: {num_samples_evaluated}")
            print(f"平均 F1-score: {avg_f1:.4f}")
            print(f"平均 Exact Match: {avg_em:.4f}")
            print(f"平均生成时间: {avg_gen_time:.2f} 秒/样本")
            print("--------------------")
            
            all_results_data.extend(current_model_results)

        except Exception as e:
            print(f"❌ 评估 {model_name} 时发生错误: {e}")
            import traceback
            traceback.print_exc()
        finally:
            loader.unload_model()

    # 评估完成，保存所有结果
    output_filename = f"tatqa_comparison_results_{os.path.basename(data_path).replace('.jsonl', '')}_{time.strftime('%Y%m%d_%H%M%S')}.json"
    with open(output_filename, 'w', encoding='utf-8') as f:
        json.dump(all_results_data, f, ensure_ascii=False, indent=4)
    print(f"\n🎉 评估完成！详细结果已保存到: {output_filename}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="TatQA英文模型对比测试脚本")
    parser.add_argument("--data_path", type=str, default="evaluate_mrr/tatqa_eval_enhanced.jsonl", help="评估数据集文件路径")
    parser.add_argument("--sample_size", type=int, default=500, help="随机采样的样本数量 (0表示评估全部)")
    parser.add_argument("--max_new_tokens", type=int, default=150, help="模型生成最大新Token数")
    parser.add_argument("--do_sample", type=bool, default=False, help="是否使用采样生成")
    parser.add_argument("--repetition_penalty", type=float, default=1.1, help="重复惩罚系数")
    
    args = parser.parse_args()
    run_english_comparison_test()
