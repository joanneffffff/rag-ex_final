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
from typing import List, Dict, Any, Optional
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
    # 确保你的RAG生成器可以被正确导入
    from xlm.components.generator.local_llm_generator import LocalLLMGenerator
    USE_RAG_GENERATOR = True
    print("✅ 使用RAG系统的LocalLLMGenerator")
except ImportError:
    USE_RAG_GENERATOR = False
    print("⚠️ 无法导入RAG系统的LocalLLMGenerator，脚本将无法运行。")
    sys.exit(1)


# ===================================================================
# 3. 核心辅助函数
# ===================================================================

def extract_final_answer_with_rescue(raw_output: str) -> str:
    """
    从模型的原始输出中智能提取最终答案。
    它首先尝试寻找<answer>标签，如果失败或为空，则启动救援逻辑从<think>标签中提取。
    """
    def _clean_extracted_text(text: str) -> str:
        """对提取出的文本进行通用清理"""
        text = text.strip()
        # 移除模型可能错误复制进来的 Prompt 指令 (假设这些文本不会出现在正确答案中)
        text = text.replace("[重要：只在这里提供最终答案。无解释，无单位，无多余文本。]", "").strip()
        
        # 移除常见的引导词句，并处理大小写不敏感
        text = re.sub(r'^(the\s*answer\s*is|it\s*was|the\s*value\s*is|resulting\s*in|this\s*represents|the\s*effect\s*is|therefore|so|thus|in\s*conclusion|final\s*answer\s*is|final\s*number\s*is)\s*', '', text, flags=re.IGNORECASE).strip()
        
        # 移除末尾可能的多余标点符号，如句号、逗号、分号 (但保留百分号)
        text = re.sub(r'[\.。;,]$', '', text).strip()

        # 标准化百分号 (例如 "percent" -> "%")
        text = re.sub(r'\s*percent\s*', '%', text, flags=re.IGNORECASE).strip()
        
        # 移除常见的货币符号和单位词 (如果你的 expected_answer 不包含这些)
        text = re.sub(r'(\$|million|billion|usd|eur|pounds|£)', '', text, flags=re.IGNORECASE).strip()
        
        # 移除数字中的逗号 (如果你的 expected_answer 不包含逗号)
        text = text.replace(',', '')
        
        # 移除负数括号 (例如 "(33)" -> "-33")
        if text.startswith('(') and text.endswith(')'):
            text = '-' + text[1:-1] # 转换为负数
            
        return text

    # 1. 尝试从 <answer> 标签中提取
    answer_match = re.search(r'<answer>(.*?)</answer>', raw_output, re.DOTALL)
    if answer_match:
        content = answer_match.group(1).strip()
        if content:
            return _clean_extracted_text(content)

    # 2. 如果 <answer> 标签失败或为空，尝试从 <think> 标签中提取
    think_match = re.search(r'<think>(.*?)</think>', raw_output, re.DOTALL)
    if not think_match:
        # 如果连 <think> 标签都没有，尝试提取原始输出的最后一行作为答案
        lines = raw_output.strip().split('\n')
        return _clean_extracted_text(lines[-1]) if lines else ""

    think_content = think_match.group(1)
    
    # --- 2.1. 尝试寻找结论性短语 ---
    conclusion_phrases = [
        r'the\s*final\s*answer\s*is[:\s]*',
        r'the\s*answer\s*is[:\s]*', 
        r'therefore,\s*the\s*answer\s*is[:\s]*', 
        r'the\s*result\s*is[:\s]*', 
        r'equals\s*to[:\s]*', 
        r'is\s*equal\s*to[:\s]*', 
        r'the\s*value\s*is[:\s]*', 
        r'the\s*change\s*is[:\s]*', 
        r'the\s*amount\s*is[:\s]*',
        r'conclusion[:\s]*', 
        r'final\s*extracted\s*value/calculated\s*result[:\s]*',
        r'final\s*number[:\s]*',
        r'adjusted\s*net\s*income\s*is[:\s]*',
        r'percentage\s*change\s*is[:\s]*', 
        r'decreased\s*by[:\s]*', 
        r'increased\s*by[:\s]*',
        r'net\s*change\s*is[:\s]*', # 增加更多通用模式
        r'total\s*is[:\s]*',
        r'resulting\s*in[:\s]*', # 捕获 "resulting in X"
        r'is[:\s]*([-+]?[\d,\.]+%?)' # 捕获"is:"后面直接跟的数字或百分比
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
            if re.fullmatch(r'\d+\.', conclusion.split('\n')[0].strip()):
                continue # 如果第一行是步骤编号，跳过
            
            return _clean_extracted_text(conclusion)
    
    # --- 2.2. 如果结论性短语不匹配，尝试寻找最后一个符合数值/百分比/常见格式的字符串 ---
    # 优先匹配行尾的数字或百分比，因为它们更可能是最终答案
    potential_answers_raw = re.findall(r'[-+]?\s*\(?[\d,\.]+\)?%?\s*$', think_content, re.MULTILINE)
    if not potential_answers_raw:
        # 如果行尾没有，在整个文本中从后往前找所有可能的数字/百分比
        potential_answers_raw = re.findall(r'[-+]?\s*\(?[\d,\.]+\)?%?', think_content)
    
    if potential_answers_raw:
        # 逆序遍历，找到最接近末尾且最可能是答案的有效项
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
        # 1. 标准化百分号 (例如 "percent" -> "%")
        text = text.replace(' percent', '%').replace(' Percent', '%').replace(' PERCENT', '%')
        
        # 2. 移除常见的货币符号、单位词、逗号和括号
        # 🚨 再次强调：这里是否移除取决于你的 expected_answer 格式。
        # 如果 expected_answer 是 "123,456.78"，则不要移除逗号。
        # 如果 expected_answer 是 "$123.45"，则不要移除 $。
        # 建议你的 expected_answer 尽量标准化为不含这些符号的纯数字或约定格式，
        # 这样这里可以统一移除，简化匹配。
        text = re.sub(r'(\$|million|billion|usd|eur|pounds|£|\(|\))', '', text, flags=re.IGNORECASE)
        text = text.replace(',', '') # 假设 expected_answer 和 extracted_answer 都不含数字逗号

        # 3. 移除除了字母数字、空格、小数点、负号和百分号之外的所有字符
        text = re.sub(r'[^\w\s%\.-]', '', text) 

        # 4. 移除常见的引导词句 (对 prediction 和 ground_truth 都进行同样清理)
        text = re.sub(r'^(the\s*answer\s*is|it\s*was|the\s*value\s*is|resulting\s*in|this\s*represents|the\s*effect\s*is|therefore|so|thus|in\s*conclusion|final\s*answer\s*is|final\s*number\s*is)\s*', '', text, flags=re.IGNORECASE).strip()
        
        # 5. 移除末尾可能的多余标点 (例如句号)
        text = text.rstrip('.')
        
        # 6. 小写并分割
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
    """根据context内容判断结构类型"""
    has_table = "Table ID:" in context
    text_content = re.sub(r'Table ID:.*?\n(Headers:.*?\n)?', '', context, flags=re.DOTALL)
    text_content = re.sub(r'Row \d+:.*?\n', '', text_content)
    text_content = re.sub(r'Category:.*?\n', '', text_content)
    has_meaningful_text = any(len(line.strip()) > 20 for line in text_content.split('\n'))

    if has_table and has_meaningful_text: return "table-text"
    elif has_table: return "table"
    else: return "text"

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

def hybrid_decision(context: str, query: str) -> str:
    """混合决策算法，预测答案来源，优化优先级"""
    context_type = determine_context_type(context)
    query_features = analyze_query_features(query)

    # 优先级最高：如果问题明确是列表/枚举，通常直接从表格行名或文本枚举中提取
    if query_features['is_list']:
        if context_type == "table":
            return "table" 
        elif context_type == "text":
            return "text"
        else: # "table-text"
            # 列表问题在混合上下文中，更可能偏向表格的行/列名，或者文本中的枚举
            # 这里的决策可以根据实际数据集中 "list" 答案的来源进行调整
            return "table-text" # 或者 'table' 如果列表主要来自表格

    # 第二优先级：计算性问题，强烈依赖数值数据
    if query_features['is_calc']:
        if context_type == "table" or context_type == "table-text":
            return "table-text" # 即使是纯表格，计算也通常需要更通用的表格处理逻辑
        else: # 纯文本，但问计算，这可能是一个复杂问题或需要从文本中解析数值进行计算
            return "text" # 回退到文本处理，让模型尝试从文本中提取数字并计算

    # 第三优先级：解释性/事实性问题
    if query_features['is_textual']:
        if context_type == "text":
            return "text"
        elif context_type == "table-text":
            # 解释性问题在混合上下文中，优先从文本获取详细描述
            return "text" 
        else: # 纯表格，但问解释，可能来自表格的描述性行/列或表格标题/备注
            return "table" # 这种情况需要你的 'table' 模板能处理描述性问题

    # 默认回退：如果以上规则都不匹配
    # 根据上下文类型进行默认路由
    return context_type # 直接返回识别到的上下文类型，让相应模板处理

# ===================================================================
# 5. 动态Prompt加载与路由
# ===================================================================

def load_and_format_template(template_name: str, context: str, query: str) -> List[Dict[str, str]]:
    """加载并格式化指定的prompt模板"""
    # 模板放在 'data/prompt_templates' 文件夹下
    template_path = Path("data/prompt_templates") / template_name
    try:
        with open(template_path, 'r', encoding='utf-8') as f:
            template_content = f.read().strip()
    except FileNotFoundError:
        print(f"❌ 模板文件未找到: {template_path}，无法继续。")
        sys.exit(1)
    
    system_match = re.search(r'<system>(.*?)</system>', template_content, re.DOTALL)
    system_content = system_match.group(1).strip() if system_match else ""
    user_match = re.search(r'<user>(.*?)</user>', template_content, re.DOTALL)
    user_template = user_match.group(1).strip() if user_match else "Context:\n{context}\n\nQuestion:\n{question}"
    user_content = user_template.replace('{context}', context).replace('{question}', query)
    return [{"role": "system", "content": system_content}, {"role": "user", "content": user_content}]

def get_final_prompt(context: str, query: str) -> List[Dict[str, str]]:
    """基于混合决策算法实现的最终Prompt路由"""
    predicted_answer_source = hybrid_decision(context, query)
    
    if predicted_answer_source == "table":
        template_file = 'template_for_table_answer.txt'
    elif predicted_answer_source == "text":
        template_file = 'template_for_text_answer.txt'
    else: # "table-text"
        template_file = 'template_for_hybrid_answer.txt'
    
    # print(f"  [路由决策] Context: {determine_context_type(context)}, Query: '{query[:30]}...', 使用模板: {template_file}")
    return load_and_format_template(template_file, context, query)

# ===================================================================
# 6. 核心评估类
# ===================================================================

class ComprehensiveEvaluator:
    def __init__(self, model_name: str, device: str):
        self.model_name = model_name
        self.device = device
        self.max_new_tokens = 4096
        print("🔄 加载模型...")
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
            prompt_text = self._convert_messages_to_text(messages)

            gen_start_time = time.time()
            generation_result = self.generator.generate([prompt_text])[0]
            gen_time = time.time() - gen_start_time
            
            final_answer_to_evaluate = extract_final_answer_with_rescue(generation_result)
            evaluation = self._evaluate_quality(final_answer_to_evaluate, sample["answer"])
            
            # 记录路由决策和实际答案来源，便于分析
            predicted_source = hybrid_decision(sample["context"], sample["query"])
            actual_source = sample.get("answer_from", "unknown") # 确保这个字段存在于你的数据集中

            return {
                "query": sample["query"],
                "expected_answer": sample["answer"],
                "generated_answer": generation_result,       # 原始模型输出
                "extracted_answer": final_answer_to_evaluate, # 经过 extract_final_answer_with_rescue 处理后的答案
                "evaluation": evaluation,
                "answer_from": actual_source, # 数据集标注的答案来源
                "predicted_answer_from": predicted_source, # 路由算法预测的答案来源
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
        text = ""
        for message in messages:
            text += f'<{message["role"]}>\n{message["content"]}\n'
        return text

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
            "overall_metrics": {
                "total_samples": len(results),
                "valid_samples": len(valid_results),
                "error_samples": len(error_results),
                "exact_match_rate": (sum(all_em) / len(all_em) * 100) if all_em else 0,
                "avg_f1_score": np.mean(all_f1) if all_f1 else 0
            },
            "by_answer_type": {}
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
        print(f"  - 总样本数: {overall.get('total_samples', 0)}")
        print(f"  - 有效样本数: {overall.get('valid_samples', 0)}")
        print(f"  - 错误样本数: {overall.get('error_samples', 0)}")
        print(f"  - 精确匹配率: {overall.get('exact_match_rate', 0):.2f}%")
        print(f"  - 平均F1分数: {overall.get('avg_f1_score', 0):.4f}")

        by_type = analysis.get("by_answer_type", {})
        print("\n📊 按答案来源类型分析:")
        for type_name, metrics in by_type.items():
            print(f"  - {type_name.upper()} 类型:")
            print(f"    - 总样本数: {metrics.get('count', 0)}")
            print(f"    - 有效样本数: {metrics.get('valid_count', 0)}")
            print(f"    - 精确匹配率: {metrics.get('exact_match_rate', 0):.2f}%")
            print(f"    - 平均F1分数: {metrics.get('avg_f1_score', 0):.4f}")
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
        device = "cuda:1" if torch.cuda.is_available() else "cpu"  # 默认使用cuda:1
    elif args.device == "cuda":
        if not torch.cuda.is_available():
            print("❌ CUDA不可用，回退到CPU")
            device = "cpu"
        else:
            device = "cuda:1"  # 默认使用cuda:1
            print(f"✅ 使用GPU: {torch.cuda.get_device_name(1) if torch.cuda.device_count() > 1 else torch.cuda.get_device_name(0)}")
            gpu_id = 1 if torch.cuda.device_count() > 1 else 0
            print(f"GPU内存: {torch.cuda.get_device_properties(gpu_id).total_memory / 1024**3:.1f} GB")
    else:
        device = args.device

    # 1. 加载数据
    print(f"📖 正在从 {args.data_path} 加载数据...")
    eval_data = []
    
    # 检查文件是否存在
    if not Path(args.data_path).exists():
        print(f"❌ 文件不存在: {args.data_path}")
        return
    
    # 首先尝试作为JSONL格式加载（逐行解析）
    jsonl_success = False
    with open(args.data_path, 'r', encoding='utf-8') as f:
        for line_num, line in enumerate(f, 1):
            line = line.strip()
            if line:  # 跳过空行
                try:
                    eval_data.append(json.loads(line))
                except json.JSONDecodeError as e:
                    print(f"❌ 第{line_num}行JSON解析失败: {e}")
                    continue
    
    # 如果JSONL解析成功，使用结果
    if eval_data:
        print(f"✅ 成功加载为JSONL格式，样本数: {len(eval_data)}")
        jsonl_success = True
    else:
        print("JSONL解析失败，尝试作为单个JSON文件解析...")
    
    # 如果JSONL解析失败，尝试作为单个JSON文件解析
    if not jsonl_success:
        eval_data = []  # 重置数据
        with open(args.data_path, 'r', encoding='utf-8') as f:
            content = f.read()
            try:
                data = json.loads(content)
                if isinstance(data, list):
                    eval_data = data
                    print(f"✅ 成功加载为JSON数组，样本数: {len(eval_data)}")
                elif isinstance(data, dict) and 'results' in data:
                    eval_data = data['results']
                    print(f"✅ 成功加载为JSON对象，样本数: {len(eval_data)}")
                else:
                    print(f"❌ 不支持的JSON格式")
                    return
            except json.JSONDecodeError as e:
                print(f"❌ JSON解析失败: {e}")
                return
    
    if args.sample_size and args.sample_size < len(eval_data):
        np.random.seed(42)
        indices = np.random.choice(len(eval_data), args.sample_size, replace=False)
        eval_data = [eval_data[i] for i in indices]
        print(f"✅ 随机采样 {len(eval_data)} 个样本进行评估。")
    else:
        print(f"✅ 加载了全部 {len(eval_data)} 个样本进行评估。")

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