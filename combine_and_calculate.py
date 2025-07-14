#!/usr/bin/env python3
"""
合并所有批次文件并计算整体的F1和EM分数
"""

import json
import os
import re
from pathlib import Path
from typing import Dict, List, Any
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

def calculate_f1_score(prediction: str, ground_truth: str, language: str = "english") -> float:
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

def calculate_exact_match(prediction: str, ground_truth: str, language: str = "english") -> float:
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

def load_and_process_batch(file_path: str) -> List[Dict[str, Any]]:
    """
    加载并处理单个批次文件
    """
    print(f"📁 加载文件: {file_path}")
    
    with open(file_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    processed_samples = []
    
    for sample in data["data"]:
        original_answer = sample.get("answer", "")
        expected_answer = sample.get("expected_answer", "")
        language = sample.get("language", "english")  # 强制使用英文处理
        
        # 移除[Reranker: Enabled]文本
        cleaned_answer = process_answer(original_answer)
        
        # 重新计算F1和EM分数 - 使用英文处理
        f1_score = calculate_f1_score(cleaned_answer, expected_answer, "english")
        exact_match = calculate_exact_match(cleaned_answer, expected_answer, "english")
        
        # 创建处理后的样本
        processed_sample = {
            "sample_id": sample.get("sample_id", 0),
            "query": sample.get("query", ""),
            "summary_context": sample.get("summary_context", ""),
            "answer": cleaned_answer,
            "expected_answer": expected_answer,
            "f1": f1_score,
            "em": exact_match,
            "processing_time": sample.get("processing_time", 0.0),
            "generation_time": sample.get("generation_time", 0.0),
            "token_count": sample.get("token_count", 0),
            "success": sample.get("success", True),
            "language": "english",  # 强制设置为英文
            "auto_stock_prediction": sample.get("auto_stock_prediction", False)
        }
        
        processed_samples.append(processed_sample)
    
    print(f"✅ 处理完成: {len(processed_samples)} 个样本")
    return processed_samples

def combine_all_batches(data_dir: str) -> Dict[str, Any]:
    """
    合并所有批次文件
    """
    print(f"🚀 开始合并目录: {data_dir}")
    
    if not os.path.exists(data_dir):
        print(f"❌ 目录不存在: {data_dir}")
        return {}
    
    # 获取所有批次文件
    batch_files = []
    for file in os.listdir(data_dir):
        if file.startswith('batch_') and file.endswith('.json'):
            batch_files.append(os.path.join(data_dir, file))
    
    batch_files.sort()  # 按文件名排序
    
    print(f"📋 找到 {len(batch_files)} 个批次文件")
    
    # 合并所有样本
    all_samples = []
    total_processing_time = 0.0
    total_generation_time = 0.0
    total_token_count = 0
    successful_samples = 0
    failed_samples = 0
    stock_prediction_samples = 0
    
    for file_path in batch_files:
        samples = load_and_process_batch(file_path)
        all_samples.extend(samples)
        
        # 累计统计
        for sample in samples:
            total_processing_time += sample.get("processing_time", 0.0)
            total_generation_time += sample.get("generation_time", 0.0)
            total_token_count += sample.get("token_count", 0)
            
            if sample.get("success", True):
                successful_samples += 1
            else:
                failed_samples += 1
            
            if sample.get("auto_stock_prediction", False):
                stock_prediction_samples += 1
    
    # 计算总体指标
    total_samples = len(all_samples)
    total_f1 = sum(sample["f1"] for sample in all_samples)
    total_em = sum(sample["em"] for sample in all_samples)
    
    avg_f1 = total_f1 / total_samples if total_samples > 0 else 0.0
    avg_em = total_em / total_samples if total_samples > 0 else 0.0
    avg_processing_time = total_processing_time / total_samples if total_samples > 0 else 0.0
    avg_generation_time = total_generation_time / total_samples if total_samples > 0 else 0.0
    avg_token_count = total_token_count / total_samples if total_samples > 0 else 0.0
    
    # 构建合并结果
    combined_result = {
        "timestamp": "2025-07-14 00:00:00",
        "data_path": "evaluate_mrr/tatqa_eval_balanced_100.jsonl",
        "total_samples": total_samples,
        "successful_samples": successful_samples,
        "failed_samples": failed_samples,
        "success_rate": (successful_samples / total_samples * 100) if total_samples > 0 else 0.0,
        "avg_f1_score": avg_f1,
        "avg_exact_match": avg_em,
        "avg_processing_time": avg_processing_time,
        "total_processing_time": total_processing_time,
        "avg_generation_time": avg_generation_time,
        "avg_token_count": avg_token_count,
        "total_token_count": total_token_count,
        "stock_prediction_samples": stock_prediction_samples,
        "reranker_enabled": False,  # 已移除[Reranker: Enabled]前缀
        "stock_prediction_enabled": stock_prediction_samples > 0,
        "auto_detected_stock_prediction": stock_prediction_samples,
        "data": all_samples
    }
    
    return combined_result

def save_combined_result(result: Dict[str, Any], output_file: str):
    """
    保存合并结果
    """
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(result, f, ensure_ascii=False, indent=2)
    
    print(f"💾 合并结果保存到: {output_file}")

def print_summary(result: Dict[str, Any]):
    """
    打印结果摘要
    """
    print("\n" + "=" * 80)
    print("📊 数据集测试结果汇总")
    print("=" * 80)
    print(f"📁 数据路径: {result['data_path']}")
    print(f"🌍 语言: english")
    print(f"📈 总样本数: {result['total_samples']}")
    print(f"✅ 成功样本数: {result['successful_samples']}")
    print(f"❌ 失败样本数: {result['failed_samples']}")
    print(f"📊 成功率: {result['success_rate']:.2f}%")
    print(f"🎯 平均F1-score: {result['avg_f1_score']:.4f}")
    print(f"🎯 平均Exact Match: {result['avg_exact_match']:.4f}")
    print(f"⏱️ 平均处理时间: {result['avg_processing_time']:.2f}秒")
    print(f"⏱️ 总处理时间: {result['total_processing_time']:.2f}秒")
    print(f"⏱️ 平均生成时间: {result['avg_generation_time']:.2f}秒")
    print(f"🔢 平均Token数: {result['avg_token_count']:.1f}")
    print(f"🔢 总Token数: {result['total_token_count']}")
    print(f"🔮 重排序器: {'启用' if result['reranker_enabled'] else '禁用'}")
    print(f"🔮 股票预测: {'启用' if result['stock_prediction_enabled'] else '禁用'}")
    print(f"🔮 自动检测股票预测: {result['auto_detected_stock_prediction']} 个")
    print("=" * 80)

def main():
    """
    主函数
    """
    data_dir = "raw_data_tatqa_eval_balanced_100"
    output_file = "combined_tatqa_results_reranker_removed.json"
    
    print("🔧 开始合并所有批次文件...")
    print("📝 任务: 移除[Reranker: Enabled]文本并计算整体F1和EM分数")
    print("=" * 60)
    
    # 合并所有批次
    result = combine_all_batches(data_dir)
    
    if result:
        # 保存合并结果
        save_combined_result(result, output_file)
        
        # 打印摘要
        print_summary(result)
        
        print("\n🎉 合并完成！")
    else:
        print("❌ 合并失败！")

if __name__ == "__main__":
    main() 