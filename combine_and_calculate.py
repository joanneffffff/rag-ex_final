#!/usr/bin/env python3
"""
合并所有批次文件并计算整体的F1和EM分数
使用中文LLM评估文件中的计算逻辑
"""

import json
import os
import re
import jieba
from pathlib import Path
from typing import Dict, List, Any
from collections import Counter

NOT_FOUND_REPLY_ENGLISH = "I cannot find the answer in the provided context."

def normalize_answer_chinese(s: str) -> str:
    """
    针对中文进行答案归一化：移除标点、转换全角字符为半角、去除多余空格、分词并小写。
    与llm_comparison/chinese_llm_evaluation.py保持完全一致。
    """
    if not s:
        return ""

    s = s.strip().lower()
    s = s.replace('，', ',').replace('。', '.').replace('！', '!').replace('？', '?').replace('；', ';')
    s = s.replace('（', '(').replace('）', ')')

    punctuation_pattern = r'[!"#$%&\'()*+,-./:;<=>?@[\]^_`{|}~“”‘’【】『』《》—…·～「」～￥%#@！&（）《》]'
    s = re.sub(punctuation_pattern, '', s)

    import jieba
    tokens = list(jieba.cut(s))
    normalized_tokens = [token for token in tokens if token.strip()]
    return " ".join(normalized_tokens)

def get_tokens_chinese(s: str) -> List[str]:
    """使用jieba分词获取中文token列表"""
    return list(jieba.cut(s))

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

def calculate_f1_score_chinese(prediction: str, ground_truth: str) -> float:
    """计算F1-score，支持中文和英文"""
    pred_tokens = set(get_tokens_chinese(normalize_answer_chinese(prediction)))
    gt_tokens = set(get_tokens_chinese(normalize_answer_chinese(ground_truth)))
    
    if not gt_tokens:
        return 1.0 if not pred_tokens else 0.0
    
    intersection = pred_tokens & gt_tokens
    precision = len(intersection) / len(pred_tokens) if pred_tokens else 0.0
    recall = len(intersection) / len(gt_tokens)
    
    if precision + recall == 0:
        return 0.0
    
    return 2 * precision * recall / (precision + recall)

def calculate_exact_match_chinese(prediction: str, ground_truth: str) -> float:
    """计算Exact Match，支持中文和英文"""
    pred_normalized = normalize_answer_chinese(prediction)
    gt_normalized = normalize_answer_chinese(ground_truth)
    
    return 1.0 if pred_normalized == gt_normalized else 0.0

def calculate_f1_score_english(prediction: str, ground_truth: str) -> float:
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

def calculate_exact_match_english(prediction: str, ground_truth: str) -> float:
    """Calculates Exact Match score for English."""
    return 1.0 if _shared_text_standardizer_english(prediction).lower() == _shared_text_standardizer_english(ground_truth).lower() else 0.0

def calculate_f1_score(prediction: str, ground_truth: str, language: str = "chinese") -> float:
    """统一的F1分数计算函数，根据语言选择不同的计算方法"""
    if language == "chinese":
        return calculate_f1_score_chinese(prediction, ground_truth)
    else:
        return calculate_f1_score_english(prediction, ground_truth)

def calculate_exact_match(prediction: str, ground_truth: str, language: str = "chinese") -> float:
    """统一的精确匹配计算函数，根据语言选择不同的计算方法"""
    if language == "chinese":
        return calculate_exact_match_chinese(prediction, ground_truth)
    else:
        return calculate_exact_match_english(prediction, ground_truth)

def process_answer(answer: str) -> str:
    """
    移除答案中的[Reranker: Enabled]前缀、"解析"及其后面的内容，以及"【解释】"及其后面的内容
    """
    if not answer:
        return answer
    
    # 移除[Reranker: Enabled]前缀
    answer = re.sub(r'^\[Reranker: Enabled\]\s*', '', answer)
    
    # 移除"解析"及其后面的所有内容
    parse_index = answer.find("解析")
    if parse_index != -1:
        answer = answer[:parse_index]
    
    # 移除"【解释】"及其后面的所有内容
    explanation_index = answer.find("【解释】")
    if explanation_index != -1:
        answer = answer[:explanation_index]
    
    return answer.strip()

def detect_language(text: str) -> str:
    """
    检测文本语言，简单的中英文检测
    """
    # 检查是否包含中文字符
    chinese_chars = sum(1 for char in text if '\u4e00' <= char <= '\u9fff')
    total_chars = len([char for char in text if char.isalpha() or '\u4e00' <= char <= '\u9fff'])
    
    if total_chars > 0 and chinese_chars / total_chars > 0.3:  # 如果超过30%是中文字符，认为是中文
        return "chinese"
    else:
        return "english"

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
        
        # 检测语言
        language = sample.get("language", "chinese")  # 默认使用中文
        if language == "auto":
            # 如果语言是auto，则自动检测
            query = sample.get("query", "")
            language = detect_language(query)
        
        # 移除[Reranker: Enabled]文本
        cleaned_answer = process_answer(original_answer)
        
        # 重新计算F1和EM分数 - 根据语言选择计算方法
        f1_score = calculate_f1_score(cleaned_answer, expected_answer, language)
        exact_match = calculate_exact_match(cleaned_answer, expected_answer, language)
        
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
            "language": language,
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
    
    # 计算整体指标
    if all_samples:
        # 按语言分组计算
        chinese_samples = [s for s in all_samples if s.get("language") == "chinese"]
        english_samples = [s for s in all_samples if s.get("language") == "english"]
        
        # 总体指标
        overall_f1 = sum(s.get("f1", 0.0) for s in all_samples) / len(all_samples)
        overall_em = sum(s.get("em", 0.0) for s in all_samples) / len(all_samples)
        
        # 中文样本指标
        chinese_f1 = 0.0
        chinese_em = 0.0
        if chinese_samples:
            chinese_f1 = sum(s.get("f1", 0.0) for s in chinese_samples) / len(chinese_samples)
            chinese_em = sum(s.get("em", 0.0) for s in chinese_samples) / len(chinese_samples)
        
        # 英文样本指标
        english_f1 = 0.0
        english_em = 0.0
        if english_samples:
            english_f1 = sum(s.get("f1", 0.0) for s in english_samples) / len(english_samples)
            english_em = sum(s.get("em", 0.0) for s in english_samples) / len(english_samples)
        
        result = {
            "timestamp": "2025-07-14 00:00:00",  # 使用当前时间戳
            "total_samples": len(all_samples),
            "successful_samples": successful_samples,
            "failed_samples": failed_samples,
            "stock_prediction_samples": stock_prediction_samples,
            "overall_metrics": {
                "f1_score": overall_f1,
                "exact_match": overall_em
            },
            "chinese_metrics": {
                "sample_count": len(chinese_samples),
                "f1_score": chinese_f1,
                "exact_match": chinese_em
            },
            "english_metrics": {
                "sample_count": len(english_samples),
                "f1_score": english_f1,
                "exact_match": english_em
            },
            "performance_metrics": {
                "total_processing_time": total_processing_time,
                "total_generation_time": total_generation_time,
                "total_token_count": total_token_count,
                "avg_processing_time": total_processing_time / len(all_samples) if all_samples else 0.0,
                "avg_generation_time": total_generation_time / len(all_samples) if all_samples else 0.0,
                "avg_token_count": total_token_count / len(all_samples) if all_samples else 0.0
            },
            "data": all_samples
        }
    else:
        result = {
            "timestamp": "",
            "total_samples": 0,
            "successful_samples": 0,
            "failed_samples": 0,
            "stock_prediction_samples": 0,
            "overall_metrics": {
                "f1_score": 0.0,
                "exact_match": 0.0
            },
            "chinese_metrics": {
                "sample_count": 0,
                "f1_score": 0.0,
                "exact_match": 0.0
            },
            "english_metrics": {
                "sample_count": 0,
                "f1_score": 0.0,
                "exact_match": 0.0
            },
            "performance_metrics": {
                "total_processing_time": 0.0,
                "total_generation_time": 0.0,
                "total_token_count": 0,
                "avg_processing_time": 0.0,
                "avg_generation_time": 0.0,
                "avg_token_count": 0.0
            },
            "data": []
        }
    
    return result

def save_combined_result(result: Dict[str, Any], output_file: str):
    """保存合并结果到文件"""
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(result, f, ensure_ascii=False, indent=2)
    print(f"💾 结果已保存到: {output_file}")

def print_summary(result: Dict[str, Any]):
    """打印结果摘要"""
    print("\n" + "="*60)
    print("📊 评估结果摘要")
    print("="*60)
    
    print(f"📈 总体指标:")
    print(f"   总样本数: {result['total_samples']}")
    print(f"   成功样本: {result['successful_samples']}")
    print(f"   失败样本: {result['failed_samples']}")
    print(f"   股票预测样本: {result['stock_prediction_samples']}")
    
    print(f"\n🎯 整体性能:")
    print(f"   F1分数: {result['overall_metrics']['f1_score']:.4f}")
    print(f"   精确匹配: {result['overall_metrics']['exact_match']:.4f}")
    
    print(f"\n🇨🇳 中文样本 ({result['chinese_metrics']['sample_count']} 个):")
    print(f"   F1分数: {result['chinese_metrics']['f1_score']:.4f}")
    print(f"   精确匹配: {result['chinese_metrics']['exact_match']:.4f}")
    
    print(f"\n🇺🇸 英文样本 ({result['english_metrics']['sample_count']} 个):")
    print(f"   F1分数: {result['english_metrics']['f1_score']:.4f}")
    print(f"   精确匹配: {result['english_metrics']['exact_match']:.4f}")
    
    print(f"\n⏱️ 性能指标:")
    print(f"   总处理时间: {result['performance_metrics']['total_processing_time']:.2f}秒")
    print(f"   总生成时间: {result['performance_metrics']['total_generation_time']:.2f}秒")
    print(f"   总Token数: {result['performance_metrics']['total_token_count']}")
    print(f"   平均处理时间: {result['performance_metrics']['avg_processing_time']:.2f}秒")
    print(f"   平均生成时间: {result['performance_metrics']['avg_generation_time']:.2f}秒")
    print(f"   平均Token数: {result['performance_metrics']['avg_token_count']:.1f}")
    
    print("="*60)

def main():
    """主函数"""
    # 设置数据目录
    data_dir = "comprehensive_evaluation_results/raw_data_alphafin_eval_samples_updated"
    
    print("🚀 开始重新计算F1和EM指标...")
    print(f"📁 数据目录: {data_dir}")
    
    # 合并所有批次
    result = combine_all_batches(data_dir)
    
    if result:
        # 生成输出文件名
        timestamp = result.get("timestamp", "").replace(" ", "_").replace(":", "-")
        output_file = f"comprehensive_evaluation_results/combined_results_recalculated_{timestamp}.json"
        
        # 保存结果
        save_combined_result(result, output_file)
        
        # 打印摘要
        print_summary(result)
        
        print(f"\n✅ 重新计算完成！结果已保存到: {output_file}")
    else:
        print("❌ 处理失败，没有生成结果")

if __name__ == "__main__":
    main() 