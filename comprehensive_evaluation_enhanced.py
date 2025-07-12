#!/usr/bin/env python3
"""
最终版全面评估脚本
集成了混合决策算法、动态prompt路由、智能答案提取和多维度评估。
"""

# 1. 导入必要的库
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

# 2. 环境设置
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
# 3. 核心辅助函数
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

        # 移除常见的引导词句，并处理大小写不敏感（作为救援逻辑的清理）
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

# ===================================================================
# 4. 智能路由算法
# ===================================================================

def determine_context_type(context: str) -> str:
    """根据context内容判断结构类型，基于Table ID和Paragraph ID"""
    has_table_id = "Table ID:" in context
    has_paragraph_id = "Paragraph ID:" in context
    
    # 移除ID标识行，获取纯内容
    content_without_ids = re.sub(r'(Table ID|Paragraph ID):.*?\n', '', context, flags=re.DOTALL)
    # 移除表格结构标识
    content_without_ids = re.sub(r'Headers:.*?\n', '', content_without_ids, flags=re.DOTALL)
    content_without_ids = re.sub(r'Row \d+:.*?\n', '', content_without_ids)
    content_without_ids = re.sub(r'Category:.*?\n', '', content_without_ids)
    
    # 检查是否有有意义的文本内容（长度>20的行）
    has_meaningful_text = any(len(line.strip()) > 20 for line in content_without_ids.split('\n') if line.strip())
    
    # 基于ID存在性进行精确判断
    if has_table_id and has_paragraph_id:
        return "table-text"  # 同时包含表格和段落ID
    elif has_table_id:
        return "table"  # 只有表格ID
    elif has_paragraph_id:
        return "text"   # 只有段落ID
    else:
        # 没有ID标识的情况，回退到内容分析
        if has_meaningful_text:
            return "text"
        else:
            return "unknown"

def analyze_query_features(query: str) -> Dict[str, Any]:
    """分析query特征，更细致地识别问题意图"""
    query_lower = query.lower()
    
    # 识别计算性关键词
    calculation_keywords = [
        'sum', 'total', 'average', 'mean', 'percentage', 'ratio', 'difference', 
        'increase', 'decrease', 'growth', 'change', 'compare', 'calculate', 
        'how much', 'how many', 'what is the', 'value of', 'amount of' # 增加更通用的数值问题词
    ]
    
    # 识别文本性关键词 (定义、解释、描述)
    text_keywords = [
        'describe', 'explain', 'what is', 'what was the effect', 'how is', 'why', 
        'when', 'where', 'who', 'what does', 'consist of', 'what led to', 
        'define', 'meaning of', 'included in', 'comprised of' # 增加更多描述性词
    ]
    
    # 识别列表/枚举性关键词
    list_keywords = ['list', 'name', 'assumptions', 'factors', 'items', 'components', 'types of', 'categories of'] 
    
    is_calc = any(keyword in query_lower for keyword in calculation_keywords)
    is_textual = any(keyword in query_lower for keyword in text_keywords)
    is_list = any(keyword in query_lower for keyword in list_keywords) 
    
    return {'is_calc': is_calc, 'is_textual': is_textual, 'is_list': is_list}

def calculate_content_ratio(context: str) -> Dict[str, float]:
    """计算表格和文本内容的比例"""
    lines = [line.strip() for line in context.split('\n') if line.strip()]
    if not lines:
        return {'table_ratio': 0.0, 'text_ratio': 0.0, 'mixed_ratio': 0.0}
    
    # 表格相关行
    table_lines = 0
    text_lines = 0
    
    for line in lines:
        # 表格标识 - 与determine_context_type保持一致
        if (any(keyword in line.lower() for keyword in ['headers:', 'row', 'column']) or 
            line.lower().startswith('table id:') or
            '|' in line and len(line.split('|')) > 2):  # 包含分隔符的表格行
            table_lines += 1
        # 文本标识 - 排除ID行，只计算实际文本内容
        elif (len(line) > 15 and  # 降低长度要求
              not line.lower().startswith('table id:') and 
              not line.lower().startswith('paragraph id:') and
              not any(keyword in line.lower() for keyword in ['headers:', 'row', 'column', 'category:'])):
            text_lines += 1
    
    total_lines = len(lines)
    table_ratio = table_lines / total_lines if total_lines > 0 else 0.0
    text_ratio = text_lines / total_lines if total_lines > 0 else 0.0
    mixed_ratio = 1.0 - abs(table_ratio - text_ratio)  # 混合程度
    
    return {
        'table_ratio': table_ratio,
        'text_ratio': text_ratio, 
        'mixed_ratio': mixed_ratio
    }

def calculate_decision_confidence(context_type: str, query_features: Dict[str, Any], content_ratio: Dict[str, float]) -> Dict[str, float]:
    """计算决策的置信度"""
    confidence_scores = {
        'table': 0.0,
        'text': 0.0,
        'hybrid': 0.0
    }
    
    # 基于上下文类型的置信度（权重：0.4）
    if context_type == "table":
        confidence_scores['table'] += 0.4
    elif context_type == "text":
        confidence_scores['text'] += 0.4
    elif context_type == "table-text":
        confidence_scores['hybrid'] += 0.4
    
    # 基于查询特征的置信度（权重：0.3）
    if query_features['is_list']:
        confidence_scores['table'] += 0.3
        confidence_scores['hybrid'] += 0.1
    elif query_features['is_calc']:
        confidence_scores['table'] += 0.2
        confidence_scores['hybrid'] += 0.2
    elif query_features['is_textual']:
        confidence_scores['text'] += 0.3
        confidence_scores['hybrid'] += 0.1
    
    # 基于内容比例的置信度（权重：0.3）
    if content_ratio['table_ratio'] > 0.5:  # 降低阈值
        confidence_scores['table'] += 0.3
    elif content_ratio['text_ratio'] > 0.5:  # 降低阈值
        confidence_scores['text'] += 0.3
    else:
        # 如果内容比例不明确，根据上下文类型给予支持
        if context_type == "text" and content_ratio['text_ratio'] > 0:
            confidence_scores['text'] += 0.3
        elif context_type == "table" and content_ratio['table_ratio'] > 0:
            confidence_scores['table'] += 0.3
        else:
            confidence_scores['hybrid'] += 0.3
    
    # 归一化置信度
    total = sum(confidence_scores.values())
    if total > 0:
        for key in confidence_scores:
            confidence_scores[key] /= total
    
    return confidence_scores



def hybrid_decision_enhanced(context: str, query: str) -> Dict[str, Any]:
    """增强版混合决策算法，仅用于英文内容（中文内容应使用multi_stage_chinese_template）"""
    # 这个函数只处理英文内容，中文内容应该直接使用multi_stage_chinese_template
    context_type = determine_context_type(context)
    query_features = analyze_query_features(query)
    content_ratio = calculate_content_ratio(context)
    
    # 计算置信度
    confidence_scores = calculate_decision_confidence(context_type, query_features, content_ratio)
    
    # 获取最高置信度的决策
    best_decision = max(confidence_scores.items(), key=lambda x: x[1])
    
    # 检查是否有多个高置信度候选
    high_confidence_threshold = 0.3
    candidates = [(decision, score) for decision, score in confidence_scores.items() if score >= high_confidence_threshold]
    candidates.sort(key=lambda x: x[1], reverse=True)
    
    # 如果最高置信度不够高，且有多个候选，标记为困难决策
    is_difficult = best_decision[1] < 0.5 and len(candidates) > 1
    
    # 构建决策结果
    decision_result = {
        'primary_decision': best_decision[0],
        'confidence': best_decision[1],
        'is_difficult': is_difficult,
        'candidates': candidates,
        'context_type': context_type,
        'query_features': query_features,
        'content_ratio': content_ratio,
        'confidence_scores': confidence_scores
    }
    
    return decision_result

def hybrid_decision(context: str, query: str) -> str:
    """混合决策算法，基于Table ID和Paragraph ID进行精确路由"""
    # 使用增强版决策算法
    decision_result = hybrid_decision_enhanced(context, query)
    
    # 如果是困难决策，使用更保守的策略
    if decision_result['is_difficult']:
        # 对于困难决策，优先选择hybrid模板
        if decision_result['primary_decision'] in ['table', 'text'] and decision_result['confidence'] < 0.6:
            return "hybrid"
    
    return decision_result['primary_decision']

# ===================================================================
# 5. 动态Prompt加载与路由
# ===================================================================

def _parse_template_string_to_messages(template_full_string: str, query: str, context: str = "", table_context: str = "", text_context: str = "") -> List[Dict[str, str]]:
    """
    解析包含 ===TAG=== 分隔符的模板字符串，并构建消息列表。
    根据传入的 context 类型替换占位符。
    """
    messages = []
    # 使用正则表达式分割所有部分，并保留分隔符内容
    parts = re.split(r'(===SYSTEM===|===USER===|===ASSISTANT===)', template_full_string, flags=re.DOTALL)
    
    # 移除第一个空字符串（如果存在）和多余的空白
    parts = [p.strip() for p in parts if p.strip()]

    # 遍历 parts 列表，重新组合 role 和 content
    for i in range(0, len(parts), 2):
        if i + 1 < len(parts):
            role_tag_raw = parts[i].strip() # 例如 "===SYSTEM==="
            content = parts[i+1].strip() # 标签后的内容
            
            # 提取实际的角色名称
            role = None
            if role_tag_raw == "===SYSTEM===":
                role = "system"
            elif role_tag_raw == "===USER===":
                role = "user"
            elif role_tag_raw == "===ASSISTANT===":
                role = "assistant"
            
            if role and content:
                # 替换占位符 (只针对 'user' 角色消息进行替换)
                if role == "user":
                    content = content.replace('{query}', query)
                    
                    # 处理中文模板的特殊占位符
                    if '{summary}' in content and '{context}' in content:
                        # 中文模板：使用摘要和完整上下文
                        combined_context = f"{table_context}\n{text_context}".strip()
                        summary = combined_context[:500] + "..." if len(combined_context) > 500 else combined_context
                        content = content.replace('{summary}', summary)
                        content = content.replace('{context}', combined_context)
                    else:
                        # 英文模板：处理分离的上下文占位符
                        content = content.replace('{question}', query)
                        
                        if '{table_context}' in content and '{text_context}' in content:
                            content = content.replace('{table_context}', table_context)
                            content = content.replace('{text_context}', text_context)
                        elif '{context}' in content: # 兼容只有 {context} 的模板
                            # 如果是通用 {context} 占位符，且有分离的上下文，则拼接
                            if table_context and text_context:
                                combined_context = f"Table Context:\n{table_context}\n\nText Context:\n{text_context}"
                                content = content.replace('{context}', combined_context.strip())
                            elif table_context:
                                content = content.replace('{context}', f"Table Context:\n{table_context}")
                            elif text_context:
                                content = content.replace('{context}', f"Text Context:\n{text_context}")
                            else: # 如果没有分离上下文，就用原始 context
                                content = content.replace('{context}', context)
                        # 确保没有未替换的上下文占位符（这些应该被前面的逻辑处理掉）
                        content = content.replace('{table_context}', '').replace('{text_context}', '')
                
                messages.append({"role": role, "content": content})
                
    return messages

def load_and_format_template(template_name: str, context: str, query: str) -> List[Dict[str, str]]:
    """
    加载并格式化指定的prompt模板（包含 ===TAG=== 分隔符）。
    该函数不处理上下文分离，通用 {context} 占位符。
    """
    template_path = Path("data/prompt_templates") / template_name
    try:
        with open(template_path, 'r', encoding='utf-8') as f:
            template_full_string = f.read().strip()
    except FileNotFoundError:
        print(f"❌ 模板文件未找到: {template_path}，无法继续。")
        sys.exit(1)
    
    return _parse_template_string_to_messages(template_full_string, query, context=context)

def load_and_format_template_with_separated_context(template_name: str, table_context: str, text_context: str, query: str) -> List[Dict[str, str]]:
    """
    加载并格式化指定的prompt模板（包含 ===TAG=== 分隔符），使用分离的上下文。
    专门处理 {table_context} 和 {text_context} 占位符。
    """
    template_path = Path("data/prompt_templates") / template_name
    try:
        with open(template_path, 'r', encoding='utf-8') as f:
            template_full_string = f.read().strip()
    except FileNotFoundError:
        print(f"❌ 模板文件未找到: {template_path}，无法继续。")
        sys.exit(1)
    
    return _parse_template_string_to_messages(template_full_string, query, table_context=table_context, text_context=text_context)

def get_final_prompt(context: str, query: str) -> List[Dict[str, str]]:
    """基于增强混合决策算法实现的最终Prompt路由，集成上下文分离功能（仅处理英文内容）"""
    # 英文内容使用混合决策算法
    decision_result = hybrid_decision_enhanced(context, query)
    predicted_answer_source = decision_result['primary_decision']
    
    # 记录决策信息用于调试
    if decision_result['is_difficult']:
        print(f"⚠️  困难决策检测: {predicted_answer_source} (置信度: {decision_result['confidence']:.3f})")
        print(f"   候选决策: {decision_result['candidates']}")
    
    if predicted_answer_source == "table":
        template_file = 'template_for_table_answer.txt'
    elif predicted_answer_source == "text":
        template_file = 'template_for_text_answer.txt'
    elif predicted_answer_source == "hybrid":
        template_file = 'template_for_hybrid_answer.txt'
    else: # "unknown" 回退
        template_file = 'template_for_hybrid_answer.txt'
    
    # 英文内容使用上下文分离功能
    if USE_CONTEXT_SEPARATOR:
        try:
            # 分离上下文
            separated = context_separator.separate_context(context)
            
            # 格式化 prompt 参数
            prompt_params = context_separator.format_for_prompt(separated, query)
            
            # 使用分离后的上下文格式化模板
            # load_and_format_template_with_separated_context 会调用 _parse_template_string_to_messages
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
# 6. 核心评估类
# ===================================================================

class ComprehensiveEvaluator:
    def __init__(self, model_name: str, device: str):
        self.model_name = model_name
        self.device = device
        self.max_new_tokens = 4096 # 合理的 max_new_tokens
        print("🔄 加载模型...")
        # LocalLLMGenerator 的初始化保持不变，它会在内部加载模型和 Tokenizer
        self.generator = LocalLLMGenerator(model_name=self.model_name, device=self.device)
        print("✅ 模型加载完成")

    def run_evaluation(self, eval_data: List[Dict[str, Any]]) -> Dict[str, Any]:
        results = []
        start_time = time.time()
        pbar = tqdm(eval_data, desc="🔍 评估样本", unit="个")

        for sample in pbar:
            result = self._evaluate_single_sample(sample)
            results.append(result)
        
        total_time = time.time() - start_time
        print(f"\n✅ 评估完成，总耗时: {total_time:.2f}秒")
        print(f"📊 处理了 {len(results)} 个结果")
        
        analysis = self.analyze_results(results)
        analysis['performance'] = {'total_time': total_time, 'avg_time_per_sample': total_time / len(results) if results else 0}
        return {"results": results, "analysis": analysis}

    def _evaluate_single_sample(self, sample: Dict[str, Any]) -> Dict[str, Any]:
        try:
            messages = get_final_prompt(sample["context"], sample["query"])
            
            # ### 核心：将 messages 列表转换为 Fin-R1 (Qwen2.5) 期望的ChatML格式字符串
            prompt_text = self._convert_messages_to_text(messages) 

            gen_start_time = time.time()
            # generator.generate 期望 List[str]，所以用 [prompt_text] 包裹
            generation_result = self.generator.generate([prompt_text])[0]
            gen_time = time.time() - gen_start_time
            
            final_answer_to_evaluate = extract_final_answer_with_rescue(generation_result)
            evaluation = self._evaluate_quality(final_answer_to_evaluate, sample["answer"])
            
            # 记录路由决策和实际答案来源，便于分析
            decision_result = hybrid_decision_enhanced(sample["context"], sample["query"])
            predicted_source = decision_result['primary_decision']
            actual_source = sample.get("answer_from", "unknown") 

            return {
                "query": sample["query"],
                "expected_answer": sample["answer"],
                "generated_answer": generation_result,      # 原始模型输出
                "extracted_answer": final_answer_to_evaluate, # 经过 extract_final_answer_with_rescue 处理后的答案
                "evaluation": evaluation,
                "answer_from": actual_source, 
                "predicted_answer_from": predicted_source,
                "decision_confidence": decision_result['confidence'],
                "is_difficult_decision": decision_result['is_difficult'],
                "context_type": decision_result['context_type'],
                "content_ratio": decision_result['content_ratio'],
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
        
        # <<< 关键修复：移除或注释掉这一行 >>>
        # 因为Prompt模板的末尾是用户消息的一部分，模型会根据ChatML的规则自动在用户消息后生成助手回应，无需额外添加 <|im_start|>assistant。
        # formatted_prompt += "<|im_start|>assistant\n" 
        
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

        # 分析决策相关指标
        difficult_decisions = [r for r in valid_results if r.get('is_difficult_decision', False)]
        avg_confidence = np.mean([r.get('decision_confidence', 0) for r in valid_results])
        
        analysis = {
            "overall_metrics": {
                "total_samples": len(results),
                "valid_samples": len(valid_results),
                "error_samples": len(error_results),
                "exact_match_rate": (sum(all_em) / len(all_em) * 100) if all_em else 0,
                "avg_f1_score": np.mean(all_f1) if all_f1 else 0,
                "difficult_decisions": len(difficult_decisions),
                "avg_decision_confidence": avg_confidence
            },
            "by_answer_type": {},
            "decision_analysis": {
                "difficult_decisions_count": len(difficult_decisions),
                "difficult_decisions_ratio": len(difficult_decisions) / len(valid_results) if valid_results else 0,
                "avg_confidence": avg_confidence
            }
        }

        types = set(r.get("answer_from", "unknown") for r in results)
        print(f"📊 发现答案类型: {list(types)}")
        
        for t in types:
            subset = [r for r in results if r.get("answer_from", "unknown") == t]
            subset_valid = [r for r in subset if 'evaluation' in r]
            subset_f1 = [r['evaluation']['f1_score'] for r in subset_valid]
            subset_em = [r['evaluation']['exact_match'] for r in subset_valid]
            analysis["by_answer_type"][t] = {
                "count": len(subset),
                "valid_count": len(subset_valid),
                "exact_match_rate": (sum(subset_em) / len(subset_em) * 100) if subset_em else 0,
                "avg_f1_score": np.mean(subset_f1) if subset_f1 else 0
            }
        return analysis

    def print_summary(self, analysis: Dict[str, Any]):
        print("\n" + "="*60)
        print("📊 评估结果摘要")
        print("="*60)
        overall = analysis.get("overall_metrics", {})
        print(f"📈 总体指标:")
        print(f"    - 总样本数: {overall.get('total_samples', 0)}")
        print(f"    - 有效样本数: {overall.get('valid_samples', 0)}")
        print(f"    - 错误样本数: {overall.get('error_samples', 0)}")
        print(f"    - 精确匹配率: {overall.get('exact_match_rate', 0):.2f}%")
        print(f"    - 平均F1分数: {overall.get('avg_f1_score', 0):.4f}")
        print(f"    - 困难决策数: {overall.get('difficult_decisions', 0)}")
        print(f"    - 平均决策置信度: {overall.get('avg_decision_confidence', 0):.3f}")

        # 显示决策分析
        decision_analysis = analysis.get("decision_analysis", {})
        print(f"\n🧠 决策分析:")
        print(f"    - 困难决策比例: {decision_analysis.get('difficult_decisions_ratio', 0):.2%}")
        print(f"    - 平均置信度: {decision_analysis.get('avg_confidence', 0):.3f}")

        by_type = analysis.get("by_answer_type", {})
        print("\n📊 按答案来源类型分析:")
        for type_name, metrics in by_type.items():
            print(f"    - {type_name.upper()} 类型:")
            print(f"      - 总样本数: {metrics.get('count', 0)}")
            print(f"      - 有效样本数: {metrics.get('valid_count', 0)}")
            print(f"      - 精确匹配率: {metrics.get('exact_match_rate', 0):.2f}%")
            print(f"      - 平均F1分数: {metrics.get('avg_f1_score', 0):.4f}")
        print("="*60)

# ===================================================================
# 7. 主函数
# ===================================================================
def main():
    parser = argparse.ArgumentParser(description="最终版全面评估脚本")
    parser.add_argument("--model", type=str, default="SUFE-AIFLM-Lab/Fin-R1", help="要评估的LLM名称")
    parser.add_argument("--data_path", type=str, required=True, help="评估数据集文件路径 (jsonl或json格式)")
    parser.add_argument("--sample_size", type=int, default=None, help="随机采样的样本数量，不提供则评估全部")
    parser.add_argument("--device", type=str, default="cuda", help="设备 (cuda/cpu/auto)")
    args = parser.parse_args()

    # 设备选择逻辑
    if args.device == "auto":
        device = "cuda:0" if torch.cuda.is_available() else "cpu"  # 默认使用cuda:0
    elif args.device == "cuda":
        if not torch.cuda.is_available():
            print("❌ CUDA不可用，回退到CPU")
            device = "cpu"
        else:
            device = "cuda:0"  # 默认使用cuda:0
            print(f"✅ 使用GPU: {torch.cuda.get_device_name(0)}")
            gpu_id = 0
            print(f"GPU内存: {torch.cuda.get_device_properties(gpu_id).total_memory / 1024**3:.1f} GB")
    else:
        device = args.device

    # 1. 加载数据
    try:
        from utils.data_loader import load_json_or_jsonl, sample_data
        eval_data = load_json_or_jsonl(args.data_path)
        
        # 采样
        if args.sample_size and args.sample_size < len(eval_data):
            eval_data = sample_data(eval_data, args.sample_size, 42)
            print(f"✅ 随机采样 {len(eval_data)} 个样本进行评估。")
        else:
            print(f"✅ 加载了全部 {len(eval_data)} 个样本进行评估。")
            
    except Exception as e:
        print(f"❌ 数据加载失败: {e}")
        return

    # 2. 初始化并运行评估器
    evaluator = ComprehensiveEvaluator(model_name=args.model, device=device)
    analysis_results = evaluator.run_evaluation(eval_data)
    
    # 3. 打印和保存结果
    evaluator.print_summary(analysis_results['analysis'])
    output_filename = f"final_evaluation_results_{time.strftime('%Y%m%d_%H%M%S')}.json"
    with open(output_filename, 'w', encoding='utf-8') as f:
        json.dump(analysis_results, f, indent=2, ensure_ascii=False)
    print(f"\n🎉 评估完成！详细结果已保存到: {output_filename}")


if __name__ == "__main__":
    main()