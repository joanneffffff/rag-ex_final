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
    answer_match = re.search(r'<answer>(.*?)</answer>', raw_output, re.DOTALL)
    if answer_match:
        content = answer_match.group(1).strip()
        if content:
            return content

    think_match = re.search(r'<think>(.*?)</think>', raw_output, re.DOTALL)
    if not think_match:
        lines = raw_output.strip().split('\n')
        return lines[-1].strip() if lines else ""

    think_content = think_match.group(1)
    
    conclusion_phrases = [
        'the answer is', 'the final answer is', 'therefore, the answer is', 
        'the result is', 'equals to', 'is equal to', 'the value is', 
        'the change is', 'the amount is'
    ]
    for phrase in conclusion_phrases:
        # 寻找结论性短语，并捕获后面的内容
        conclusion_match = re.search(
            f'{re.escape(phrase)}\\s*:?\\s*([$()\\d,.;\\w\\s-]+)($|\\.|\\n)', 
            think_content, 
            re.IGNORECASE
        )
        if conclusion_match:
            conclusion = conclusion_match.group(1).strip()
            return re.sub(r'[\.。,]$', '', conclusion).strip()

    numbers = re.findall(r'[-+]?\$?\(?[\d,]+\.?\d*\)?\%?', think_content)
    if numbers:
        last_number = numbers[-1].replace('$', '').replace(',', '').replace('(', '').replace(')', '').strip()
        return last_number
        
    lines = [line for line in think_content.strip().split('\n') if line.strip()]
    return lines[-1].strip() if lines else ""

def calculate_f1_score(prediction: str, ground_truth: str) -> float:
    """计算F1分数"""
    def normalize(text):
        return re.sub(r'[^\w\s]', '', text.lower()).split()
    prediction_tokens = normalize(prediction)
    ground_truth_tokens = normalize(ground_truth)
    if not ground_truth_tokens: return 1.0 if not prediction_tokens else 0.0
    if not prediction_tokens: return 0.0
    common = Counter(prediction_tokens) & Counter(ground_truth_tokens)
    num_same = sum(common.values())
    if num_same == 0: return 0.0
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
    """分析query特征"""
    query_lower = query.lower()
    calculation_keywords = ['sum', 'total', 'average', 'mean', 'percentage', 'ratio', 'difference', 'increase', 'decrease', 'growth', 'change', 'compare', 'calculate']
    text_keywords = ['describe', 'explain', 'what is', 'what was the effect', 'how', 'why', 'when', 'where', 'who', 'what does', 'consist of', 'what led to']
    
    is_calc = any(keyword in query_lower for keyword in calculation_keywords)
    is_textual = any(keyword in query_lower for keyword in text_keywords)
    
    return {'is_calc': is_calc, 'is_textual': is_textual}

def hybrid_decision(context: str, query: str) -> str:
    """混合决策算法，预测答案来源"""
    context_type = determine_context_type(context)
    query_features = analyze_query_features(query)

    if context_type == "text":
        return "text"
    
    # 对于包含表格的context
    if query_features['is_textual'] and not query_features['is_calc']:
        # 如果问题明显是解释性的，答案很可能在文本中，即使表格存在
        return "text" if context_type == "table-text" else "table-text"
    
    if query_features['is_calc']:
         # 如果问题是计算性的，答案很可能需要结合表格和文本
        return "table-text"
    
    # 默认情况，答案更可能直接来自表格
    return "table"


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
        self.max_new_tokens = 2048
        print("🔄 加载模型...")
        self.generator = LocalLLMGenerator(model_name=self.model_name, device=self.device)
        print("✅ 模型加载完成")

    def run_evaluation(self, eval_data: List[Dict[str, Any]]) -> Dict[str, Any]:
        results = []
        start_time = time.time()
        pbar = tqdm(eval_data, desc="🔍 评估样本", unit="个")

        for sample in pbar:
            results.append(self._evaluate_single_sample(sample))
        
        total_time = time.time() - start_time
        print(f"\n✅ 评估完成，总耗时: {total_time:.2f}秒")
        
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
            
            return {
                "query": sample["query"],
                "expected_answer": sample["answer"],
                "generated_answer": generation_result,
                "extracted_answer": final_answer_to_evaluate,
                "evaluation": evaluation,
                "answer_from": sample.get("answer_from", "unknown"),
                "predicted_answer_from": hybrid_decision(sample["context"], sample["query"]),
                "generation_time": gen_time
            }
        except Exception as e:
            return {"query": sample["query"], "expected_answer": sample["answer"], "error": str(e)}

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
        # 你的详细分析逻辑，可以复用之前脚本里的版本
        # 这里提供一个简化的版本
        if not results: return {}
        
        all_f1 = [r['evaluation']['f1_score'] for r in results if 'evaluation' in r]
        all_em = [r['evaluation']['exact_match'] for r in results if 'evaluation' in r]

        analysis = {
            "overall_metrics": {
                "total_samples": len(results),
                "exact_match_rate": (sum(all_em) / len(all_em) * 100) if all_em else 0,
                "avg_f1_score": np.mean(all_f1) if all_f1 else 0
            },
            "by_answer_type": {}
        }

        types = set(r.get("answer_from") for r in results)
        for t in types:
            subset = [r for r in results if r.get("answer_from") == t]
            subset_f1 = [r['evaluation']['f1_score'] for r in subset if 'evaluation' in r]
            subset_em = [r['evaluation']['exact_match'] for r in subset if 'evaluation' in r]
            analysis["by_answer_type"][t] = {
                "count": len(subset),
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
        print(f"  - 精确匹配率: {overall.get('exact_match_rate', 0):.2f}%")
        print(f"  - 平均F1分数: {overall.get('avg_f1_score', 0):.4f}")

        by_type = analysis.get("by_answer_type", {})
        print("\n📊 按答案来源类型分析:")
        for type_name, metrics in by_type.items():
            print(f"  - {type_name.upper()} 类型 ({metrics.get('count', 0)} 样本):")
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
        device = "cuda" if torch.cuda.is_available() else "cpu"
    elif args.device == "cuda":
        if not torch.cuda.is_available():
            print("❌ CUDA不可用，回退到CPU")
            device = "cpu"
        else:
            device = "cuda"
            print(f"✅ 使用GPU: {torch.cuda.get_device_name()}")
            print(f"GPU内存: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB")
    else:
        device = args.device

    # 1. 加载数据
    print(f"📖 正在从 {args.data_path} 加载数据...")
    eval_data = []
    with open(args.data_path, 'r', encoding='utf-8') as f:
        # 兼容 .json 和 .jsonl
        content = f.read()
        try:
            # 尝试解析为单个JSON数组
            data = json.loads(content)
            if isinstance(data, list):
                eval_data = data
            # 如果是JSON对象，并且有 'results' 键
            elif isinstance(data, dict) and 'results' in data:
                eval_data = data['results']
        except json.JSONDecodeError:
            # 如果失败，按jsonl格式逐行解析
            f.seek(0)
            for line in f:
                eval_data.append(json.loads(line))
    
    if args.sample_size and args.sample_size < len(eval_data):
        np.random.seed(42)
        indices = np.random.choice(len(eval_data), args.sample_size, replace=False)
        eval_data = [eval_data[i] for i in indices]
        print(f"✅ 随机采样 {len(eval_data)} 个样本进行评估。")
    else:
        print(f"✅ 加载了全部 {len(eval_data)} 个样本进行评估。")

    # 2. 初始化并运行评估器
    evaluator = ComprehensiveEvaluator(model_name=args.model, device=args.device)
    analysis_results = evaluator.run_evaluation(eval_data)
    
    # 3. 打印和保存结果
    evaluator.print_summary(analysis_results)
    output_filename = f"final_evaluation_results_{time.strftime('%Y%m%d_%H%M%S')}.json"
    with open(output_filename, 'w', encoding='utf-8') as f:
        json.dump(analysis_results, f, indent=2, ensure_ascii=False)
    print(f"\n🎉 评估完成！详细结果已保存到: {output_filename}")


if __name__ == "__main__":
    main()