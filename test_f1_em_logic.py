#!/usr/bin/env python3
"""
测试F1和EM计算逻辑
"""

import re
from collections import Counter

NOT_FOUND_REPLY_ENGLISH = "I cannot find the answer in the provided context."

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

def calculate_f1_score(prediction: str, ground_truth: str) -> float:
    """Calculates F1-score based on token overlap for English."""
    
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

def calculate_exact_match(prediction: str, ground_truth: str) -> float:
    """Calculates Exact Match score for English."""
    return 1.0 if _shared_text_standardizer_english(prediction).lower() == _shared_text_standardizer_english(ground_truth).lower() else 0.0

def process_answer(answer: str) -> str:
    """
    移除答案中的[Reranker: Enabled]文本
    """
    if not answer:
        return answer
    
    # 移除[Reranker: Enabled]前缀
    answer = re.sub(r'^\[Reranker: Enabled\]\s*', '', answer)
    
    return answer.strip()

# 测试用例
test_cases = [
    {
        "prediction": "[Reranker: Enabled] 1%",
        "ground_truth": "$0.3 million",
        "description": "移除前缀后的F1计算"
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
    }
]

print("🧪 测试F1和EM计算逻辑")
print("=" * 60)

for i, test_case in enumerate(test_cases, 1):
    print(f"\n📋 测试用例 {i}: {test_case['description']}")
    print(f"预测: '{test_case['prediction']}'")
    print(f"真实: '{test_case['ground_truth']}'")
    
    # 移除前缀
    cleaned_prediction = process_answer(test_case['prediction'])
    print(f"清理后预测: '{cleaned_prediction}'")
    
    # 计算分数
    f1 = calculate_f1_score(cleaned_prediction, test_case['ground_truth'])
    em = calculate_exact_match(cleaned_prediction, test_case['ground_truth'])
    
    print(f"F1: {f1:.4f}")
    print(f"EM: {em:.4f}")
    
    # 显示标准化后的文本
    normalized_pred = _shared_text_standardizer_english(cleaned_prediction)
    normalized_gt = _shared_text_standardizer_english(test_case['ground_truth'])
    print(f"标准化预测: '{normalized_pred}'")
    print(f"标准化真实: '{normalized_gt}'")

print("\n✅ 测试完成") 