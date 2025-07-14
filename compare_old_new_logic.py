#!/usr/bin/env python3
"""
对比旧逻辑和新逻辑的差异
"""

import re
from collections import Counter
from typing import List

NOT_FOUND_REPLY_ENGLISH = "I cannot find the answer in the provided context."

# 旧逻辑（简化版本）
def normalize_answer_english_old(s: str) -> str:
    """旧版本的英文答案标准化"""
    if not s:
        return ""
    s = ' '.join(s.split())
    s = re.sub(r'[^\w\s]', '', s)
    return s.strip().lower()

def get_tokens_english_old(s: str) -> List[str]:
    """旧版本的英文token获取"""
    return s.split()

def calculate_f1_score_old(prediction: str, ground_truth: str) -> float:
    """旧版本的F1计算"""
    pred_tokens = set(get_tokens_english_old(normalize_answer_english_old(prediction)))
    gt_tokens = set(get_tokens_english_old(normalize_answer_english_old(ground_truth)))
    
    if not gt_tokens:
        return 1.0 if not pred_tokens else 0.0
    
    intersection = pred_tokens & gt_tokens
    precision = len(intersection) / len(pred_tokens) if pred_tokens else 0.0
    recall = len(intersection) / len(gt_tokens)
    
    if precision + recall == 0:
        return 0.0
    
    return 2 * precision * recall / (precision + recall)

def calculate_exact_match_old(prediction: str, ground_truth: str) -> float:
    """旧版本的EM计算"""
    pred_normalized = normalize_answer_english_old(prediction)
    gt_normalized = normalize_answer_english_old(ground_truth)
    return 1.0 if pred_normalized == gt_normalized else 0.0

# 新逻辑
def _shared_text_standardizer_english(text: str) -> str:
    """
    Helper function to standardize English text for both answer extraction and F1 score calculation.
    Strictly follows the rules from the English Prompt Template.
    """
    text = text.strip()
    
    # Lowercase all text
    text = text.lower()

    # 递归替换所有 \text{...} 为 ...（保留内容）
    while True:
        new_text = re.sub(r'\\text\{([^}]*)\}', r'\1', text, flags=re.DOTALL)
        if new_text == text:
            break
        text = new_text
    # 其余 LaTeX 格式直接去掉
    text = re.sub(r'\\[a-zA-Z]+\{[^}]*\}', '', text)
    text = re.sub(r'\\[a-zA-Z]+', '', text)
    
    # Remove currency symbols and common unit words based on prompt rule
    text = re.sub(r'\b(million|billion|thousand|trillion|usd|eur|gbp|m|b)\b', '', text, flags=re.IGNORECASE).strip()
    text = re.sub(r'[\$£€]', '', text).strip()

    # Remove commas from numbers
    text = text.replace(',', '')

    # Handle negative numbers in parentheses (e.g., "(33)" -> "-33")
    if text.startswith('(') and text.endswith(')'):
        text = '-' + text[1:-1]
    
    # Normalize percentages
    text = text.replace(' percent', '%').replace('pct', '%')
    text = re.sub(r'(\d+\.?\d*)\s*%', r'\1%', text)
    
    # Remove common introductory phrases
    text = re.sub(r'^(the\s*answer\s*is|it\s*was|the\s*value\s*is|resulting\s*in|this\s*represents|the\s*effect\s*is|therefore|so|thus|in\s*conclusion|final\s*answer\s*is|final\s*number\s*is)\s*', '', text, flags=re.IGNORECASE).strip()
    
    # Remove trailing punctuation
    if text.endswith('%'):
        text = re.sub(r'[\.,;]$', '', text).strip()
    else:
        text = re.sub(r'[\.,;%]$', '', text).strip() 
    
    # Final cleanup of whitespace
    text = ' '.join(text.split()).strip()

    return text

def calculate_f1_score_new(prediction: str, ground_truth: str) -> float:
    """新版本的F1计算"""
    
    normalized_prediction = _shared_text_standardizer_english(prediction).lower()
    normalized_ground_truth = _shared_text_standardizer_english(ground_truth).lower()

    # Handle cases where the model explicitly states "I cannot find the answer..."
    if normalized_prediction == NOT_FOUND_REPLY_ENGLISH.lower():
        return 1.0 if normalized_ground_truth == NOT_FOUND_REPLY_ENGLISH.lower() else 0.0
    
    # Handle cases where the ground truth is "I cannot find the answer...", but the model gave a factual answer (which is an error)
    if normalized_ground_truth == NOT_FOUND_REPLY_ENGLISH.lower():
        return 0.0

    prediction_tokens = normalized_prediction.split()
    ground_truth_tokens = normalized_ground_truth.split()

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

def calculate_exact_match_new(prediction: str, ground_truth: str) -> float:
    """新版本的EM计算"""
    return 1.0 if _shared_text_standardizer_english(prediction).lower() == _shared_text_standardizer_english(ground_truth).lower() else 0.0

def process_answer(answer: str) -> str:
    """移除答案中的[Reranker: Enabled]文本"""
    if not answer:
        return answer
    answer = re.sub(r'^\[Reranker: Enabled\]\s*', '', answer)
    return answer.strip()

# 测试用例
test_cases = [
    {
        "prediction": "[Reranker: Enabled] 1%",
        "ground_truth": "$0.3 million",
        "description": "货币和单位词处理"
    },
    {
        "prediction": "[Reranker: Enabled] The answer is 12.5%",
        "ground_truth": "12.5%",
        "description": "移除前缀短语"
    },
    {
        "prediction": "[Reranker: Enabled] I cannot find the answer in the provided context.",
        "ground_truth": "geographic distribution of pretax income from continuing operations",
        "description": "无法找到答案的情况"
    },
    {
        "prediction": "[Reranker: Enabled] I cannot find the answer in the provided context.",
        "ground_truth": "I cannot find the answer in the provided context.",
        "description": "双方都无法找到答案"
    },
    {
        "prediction": "[Reranker: Enabled] $1,234.56",
        "ground_truth": "1234.56",
        "description": "数字格式标准化"
    }
]

print("🔍 对比旧逻辑和新逻辑")
print("=" * 80)

for i, test_case in enumerate(test_cases, 1):
    print(f"\n📋 测试用例 {i}: {test_case['description']}")
    print(f"原始预测: '{test_case['prediction']}'")
    print(f"原始真实: '{test_case['ground_truth']}'")
    
    # 移除前缀
    cleaned_prediction = process_answer(test_case['prediction'])
    print(f"清理后预测: '{cleaned_prediction}'")
    
    # 旧逻辑
    f1_old = calculate_f1_score_old(cleaned_prediction, test_case['ground_truth'])
    em_old = calculate_exact_match_old(cleaned_prediction, test_case['ground_truth'])
    
    # 新逻辑
    f1_new = calculate_f1_score_new(cleaned_prediction, test_case['ground_truth'])
    em_new = calculate_exact_match_new(cleaned_prediction, test_case['ground_truth'])
    
    print(f"\n📊 旧逻辑结果:")
    print(f"  标准化预测: '{normalize_answer_english_old(cleaned_prediction)}'")
    print(f"  标准化真实: '{normalize_answer_english_old(test_case['ground_truth'])}'")
    print(f"  F1: {f1_old:.4f}")
    print(f"  EM: {em_old:.4f}")
    
    print(f"\n📊 新逻辑结果:")
    print(f"  标准化预测: '{_shared_text_standardizer_english(cleaned_prediction)}'")
    print(f"  标准化真实: '{_shared_text_standardizer_english(test_case['ground_truth'])}'")
    print(f"  F1: {f1_new:.4f}")
    print(f"  EM: {em_new:.4f}")
    
    print(f"\n📈 变化:")
    print(f"  F1变化: {f1_new - f1_old:+.4f}")
    print(f"  EM变化: {em_new - em_old:+.4f}")

print("\n✅ 对比完成") 