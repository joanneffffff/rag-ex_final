#!/usr/bin/env python3
"""
测试F1和EM计算逻辑
"""

import json
import re
from typing import List

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
    
    # 移除"解释"及其后的内容
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

def test_sample_calculation():
    """
    测试单个样本的计算
    """
    print("🧪 测试F1和EM计算逻辑...")
    
    # 从原始数据中取一个样本进行测试
    sample_data = {
        "answer": "[Reranker: Enabled] 中国中冶601618.SH在2017年6月30日的资产负债表显示其净资产为76,421,499,000.0元。 此信息直接来源于摘要部分的第一条记录，并得到了详细内容的支持，其中\"股东权益合计\"的具体数值为76,421,499,000.0元，且\"负债及股东权益总计\"为40,269,525,600.0元，进一步验证了净资产的正确性。 无须额外计算，因为净资产已直接给出。",
        "expected_answer": "根据资产负债表数据，该公司的净资产为76,421,499,000.0元。"
    }
    
    original_answer = sample_data["answer"]
    expected_answer = sample_data["expected_answer"]
    
    print(f"📝 原始答案: {original_answer[:100]}...")
    print(f"📝 期望答案: {expected_answer}")
    
    # 移除[Reranker: Enabled]
    cleaned_answer = process_answer(original_answer)
    print(f"🧹 清理后答案: {cleaned_answer[:100]}...")
    
    # 计算F1
    f1_score = calculate_f1_score(cleaned_answer, expected_answer, "chinese")
    print(f"🎯 F1分数: {f1_score:.4f}")
    
    # 计算EM
    em_score = calculate_exact_match(cleaned_answer, expected_answer, "chinese")
    print(f"🎯 EM分数: {em_score:.4f}")
    
    # 测试标准化
    pred_normalized = normalize_answer_chinese(cleaned_answer)
    truth_normalized = normalize_answer_chinese(expected_answer)
    print(f"📊 标准化预测: {pred_normalized[:100]}...")
    print(f"📊 标准化期望: {truth_normalized}")
    
    return f1_score, em_score

def test_multiple_samples():
    """
    测试多个样本
    """
    print("\n🔍 测试多个样本...")
    
    # 加载一个批次文件进行测试
    import os
    batch_files = [f for f in os.listdir("raw_data_alphafin_eval_samples_updated") 
                   if f.startswith('batch_') and f.endswith('.json')]
    
    if batch_files:
        test_file = os.path.join("raw_data_alphafin_eval_samples_updated", batch_files[0])
        
        with open(test_file, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        total_f1 = 0.0
        total_em = 0.0
        sample_count = 0
        
        for sample in data["data"][:5]:  # 只测试前5个样本
            original_answer = sample.get("answer", "")
            expected_answer = sample.get("expected_answer", "")
            
            cleaned_answer = process_answer(original_answer)
            f1_score = calculate_f1_score(cleaned_answer, expected_answer, "chinese")
            em_score = calculate_exact_match(cleaned_answer, expected_answer, "chinese")
            
            print(f"样本 {sample_count + 1}: F1={f1_score:.4f}, EM={em_score:.4f}")
            
            total_f1 += f1_score
            total_em += em_score
            sample_count += 1
        
        avg_f1 = total_f1 / sample_count if sample_count > 0 else 0.0
        avg_em = total_em / sample_count if sample_count > 0 else 0.0
        
        print(f"\n📊 前5个样本平均: F1={avg_f1:.4f}, EM={avg_em:.4f}")

def main():
    """
    主函数
    """
    print("🔧 开始测试F1和EM计算逻辑...")
    print("=" * 60)
    
    # 测试单个样本
    f1, em = test_sample_calculation()
    
    # 测试多个样本
    test_multiple_samples()
    
    print("\n" + "=" * 60)
    print("📋 测试完成")
    print("💡 如果结果与原始结果差异很大，可能需要调整计算逻辑")

if __name__ == "__main__":
    main() 