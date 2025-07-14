#!/usr/bin/env python3
"""
处理原始数据，移除[Reranker: Enabled]文本并重新计算F1和EM分数
"""

import json
import os
import re
from pathlib import Path
from typing import Dict, List, Any

def normalize_answer_chinese(s: str) -> str:
    """
    标准化中文答案
    """
    if not s:
        return ""
    
    # 移除"解析"及其后的内容
    if "解析" in s:
        s = s.split("解析")[0]
    
    # 移除"【解释】"及其后的内容
    if "【解释】" in s:
        s = s.split("【解释】")[0]
    
    # 移除"【解释】"及其后的内容
    if "解释" in s:
        s = s.split("解释")[0]
    
    # 清理空白字符
    s = re.sub(r'\s+', ' ', s.strip())
    
    return s

def get_tokens_chinese(s: str) -> List[str]:
    """
    中文分词
    """
    if not s:
        return []
    return list(s)

def calculate_f1_score(prediction: str, ground_truth: str, language: str = "chinese") -> float:
    """
    计算F1分数
    """
    if language == "chinese":
        pred_tokens = get_tokens_chinese(prediction)
        truth_tokens = get_tokens_chinese(ground_truth)
    else:
        pred_tokens = prediction.lower().split()
        truth_tokens = ground_truth.lower().split()
    
    if not pred_tokens or not truth_tokens:
        return 0.0
    
    # 计算交集
    common = set(pred_tokens) & set(truth_tokens)
    
    if not common:
        return 0.0
    
    precision = len(common) / len(pred_tokens)
    recall = len(common) / len(truth_tokens)
    
    if precision + recall == 0:
        return 0.0
    
    return 2 * precision * recall / (precision + recall)

def calculate_exact_match(prediction: str, ground_truth: str, language: str = "chinese") -> float:
    """
    计算精确匹配分数
    """
    if language == "chinese":
        pred_normalized = normalize_answer_chinese(prediction)
        truth_normalized = normalize_answer_chinese(ground_truth)
    else:
        pred_normalized = prediction.lower().strip()
        truth_normalized = ground_truth.lower().strip()
    
    return 1.0 if pred_normalized == truth_normalized else 0.0

def process_answer(answer: str) -> str:
    """
    移除答案中的[Reranker: Enabled]文本
    """
    if not answer:
        return answer
    
    # 移除[Reranker: Enabled]前缀
    answer = re.sub(r'^\[Reranker: Enabled\]\s*', '', answer)
    
    return answer.strip()

def process_batch_file(file_path: str) -> Dict[str, Any]:
    """
    处理单个批次文件
    """
    print(f"📁 处理文件: {file_path}")
    
    with open(file_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    # 统计信息
    total_samples = len(data["data"])
    processed_samples = 0
    total_f1 = 0.0
    total_em = 0.0
    
    # 处理每个样本
    for sample in data["data"]:
        original_answer = sample.get("answer", "")
        expected_answer = sample.get("expected_answer", "")
        language = sample.get("language", "chinese")
        
        # 移除[Reranker: Enabled]文本
        cleaned_answer = process_answer(original_answer)
        
        # 重新计算F1和EM分数
        f1_score = calculate_f1_score(cleaned_answer, expected_answer, language)
        exact_match = calculate_exact_match(cleaned_answer, expected_answer, language)
        
        # 更新样本数据
        sample["answer"] = cleaned_answer
        sample["f1"] = f1_score
        sample["em"] = exact_match
        
        # 累计统计
        total_f1 += f1_score
        total_em += exact_match
        processed_samples += 1
    
    # 计算平均值
    avg_f1 = total_f1 / processed_samples if processed_samples > 0 else 0.0
    avg_em = total_em / processed_samples if processed_samples > 0 else 0.0
    
    # 更新文件统计信息
    data["avg_f1"] = avg_f1
    data["avg_em"] = avg_em
    data["processed_samples"] = processed_samples
    
    # 保存处理后的文件
    output_path = file_path.replace('.json', '_processed.json')
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(data, f, ensure_ascii=False, indent=2)
    
    print(f"✅ 处理完成: {processed_samples} 个样本")
    print(f"📊 平均F1: {avg_f1:.4f}")
    print(f"📊 平均EM: {avg_em:.4f}")
    print(f"💾 保存到: {output_path}")
    
    return {
        "file": file_path,
        "processed_samples": processed_samples,
        "avg_f1": avg_f1,
        "avg_em": avg_em
    }

def process_all_batches(data_dir: str):
    """
    处理所有批次文件
    """
    print(f"🚀 开始处理目录: {data_dir}")
    
    if not os.path.exists(data_dir):
        print(f"❌ 目录不存在: {data_dir}")
        return
    
    # 获取所有批次文件
    batch_files = []
    for file in os.listdir(data_dir):
        if file.startswith('batch_') and file.endswith('.json'):
            batch_files.append(os.path.join(data_dir, file))
    
    batch_files.sort()  # 按文件名排序
    
    print(f"📋 找到 {len(batch_files)} 个批次文件")
    
    # 处理所有文件
    results = []
    total_samples = 0
    total_f1 = 0.0
    total_em = 0.0
    
    for file_path in batch_files:
        result = process_batch_file(file_path)
        results.append(result)
        
        total_samples += result["processed_samples"]
        total_f1 += result["avg_f1"] * result["processed_samples"]
        total_em += result["avg_em"] * result["processed_samples"]
    
    # 计算总体统计
    overall_avg_f1 = total_f1 / total_samples if total_samples > 0 else 0.0
    overall_avg_em = total_em / total_samples if total_samples > 0 else 0.0
    
    print("\n" + "=" * 60)
    print("📊 总体处理结果:")
    print(f"   总样本数: {total_samples}")
    print(f"   平均F1: {overall_avg_f1:.4f}")
    print(f"   平均EM: {overall_avg_em:.4f}")
    print("=" * 60)
    
    # 保存总体结果
    summary = {
        "total_samples": total_samples,
        "overall_avg_f1": overall_avg_f1,
        "overall_avg_em": overall_avg_em,
        "batch_results": results
    }
    
    summary_file = os.path.join(data_dir, "processing_summary.json")
    with open(summary_file, 'w', encoding='utf-8') as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)
    
    print(f"📁 总体结果保存到: {summary_file}")

def main():
    """
    主函数
    """
    data_dir = "raw_data_alphafin_eval_samples_updated"
    
    print("🔧 开始处理原始数据...")
    print("📝 任务: 移除[Reranker: Enabled]文本并重新计算F1和EM分数")
    print("=" * 60)
    
    process_all_batches(data_dir)
    
    print("\n🎉 处理完成！")

if __name__ == "__main__":
    main() 