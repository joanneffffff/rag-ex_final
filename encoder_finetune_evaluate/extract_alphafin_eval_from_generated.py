#!/usr/bin/env python3
"""
从AlphaFin生成数据中提取与原始评估数据匹配的样本
用于评估生成数据的质量
"""

import json
import argparse
from pathlib import Path
from typing import List, Dict
import re
from difflib import SequenceMatcher

def normalize_text(text: str) -> str:
    """标准化文本用于比较"""
    if not text:
        return ""
    # 移除多余空格和换行
    text = re.sub(r'\s+', ' ', text.strip())
    # 转换为小写
    text = text.lower()
    return text

def calculate_similarity(text1: str, text2: str) -> float:
    """计算两个文本的相似度"""
    if not text1 or not text2:
        return 0.0
    
    norm1 = normalize_text(text1)
    norm2 = normalize_text(text2)
    
    if not norm1 or not norm2:
        return 0.0
    
    # 使用序列匹配器计算相似度
    similarity = SequenceMatcher(None, norm1, norm2).ratio()
    return similarity

def extract_key_info(text: str) -> Dict[str, str]:
    """提取文本中的关键信息"""
    info = {
        "company": "",
        "stock_code": "",
        "date": "",
        "numbers": []
    }
    
    # 提取公司名称和股票代码
    company_pattern = r'([^（]+)（([0-9]{6}）)'
    match = re.search(company_pattern, text)
    if match:
        info["company"] = match.group(1).strip()
        info["stock_code"] = match.group(2).strip()
    
    # 提取日期
    date_pattern = r'(\d{4}-\d{2}-\d{2})'
    dates = re.findall(date_pattern, text)
    if dates:
        info["date"] = dates[0]
    
    # 提取数字
    number_pattern = r'\d+\.?\d*'
    numbers = re.findall(number_pattern, text)
    info["numbers"] = [float(n) for n in numbers[:10]]  # 只取前10个数字
    
    return info

def calculate_structured_similarity(original: Dict, generated: Dict) -> float:
    """计算结构化相似度"""
    original_info = extract_key_info(original.get("question", "") + " " + original.get("context", ""))
    generated_info = extract_key_info(generated.get("question", "") + " " + generated.get("context", ""))
    
    # 公司名称相似度
    company_sim = calculate_similarity(original_info["company"], generated_info["company"])
    
    # 股票代码匹配
    stock_match = 1.0 if original_info["stock_code"] == generated_info["stock_code"] else 0.0
    
    # 日期匹配
    date_match = 1.0 if original_info["date"] == generated_info["date"] else 0.0
    
    # 数字相似度（取前5个数字比较）
    number_sim = 0.0
    if original_info["numbers"] and generated_info["numbers"]:
        common_count = 0
        for orig_num in original_info["numbers"][:5]:
            for gen_num in generated_info["numbers"][:5]:
                if abs(float(orig_num) - float(gen_num)) < 0.01:  # 允许小的数值差异
                    common_count += 1
        number_sim = common_count / min(len(original_info["numbers"][:5]), len(generated_info["numbers"][:5]))
    
    # 加权平均
    structured_sim = (
        company_sim * 0.3 +
        stock_match * 0.3 +
        date_match * 0.2 +
        number_sim * 0.2
    )
    
    return structured_sim

def find_matching_samples(original_eval_data: List[Dict], generated_data: List[Dict], 
                         similarity_threshold: float = 0.4) -> List[Dict]:
    """从生成数据中找到与原始评估数据匹配的样本"""
    print(f"🔍 开始匹配样本...")
    print(f"原始评估数据: {len(original_eval_data)} 个样本")
    print(f"生成数据: {len(generated_data)} 个样本")
    
    matched_samples = []
    unmatched_count = 0
    
    for i, original_sample in enumerate(original_eval_data):
        if i % 100 == 0:
            print(f"处理进度: {i}/{len(original_eval_data)}")
        
        original_question = original_sample.get("query", "")  # 注意：原始数据使用"query"字段
        original_context = original_sample.get("context", "")
        original_answer = original_sample.get("answer", "")
        
        best_match = None
        best_similarity = 0.0
        
        # 在生成数据中寻找最佳匹配
        for generated_sample in generated_data:
            generated_question = generated_sample.get("question", "")
            generated_context = generated_sample.get("context", "")
            generated_answer = generated_sample.get("answer", "")
            
            # 计算问题相似度
            question_similarity = calculate_similarity(original_question, generated_question)
            
            # 计算上下文相似度
            context_similarity = calculate_similarity(original_context, generated_context)
            
            # 计算答案相似度
            answer_similarity = calculate_similarity(original_answer, generated_answer)
            
            # 计算结构化相似度
            structured_similarity = calculate_structured_similarity(
                {"question": original_question, "context": original_context},
                {"question": generated_question, "context": generated_context}
            )
            
            # 综合相似度（加权平均）
            overall_similarity = (
                question_similarity * 0.3 + 
                context_similarity * 0.2 + 
                answer_similarity * 0.2 +
                structured_similarity * 0.3
            )
            
            if overall_similarity > best_similarity:
                best_similarity = overall_similarity
                best_match = {
                    "generated_sample": generated_sample,
                    "similarity_scores": {
                        "question": question_similarity,
                        "context": context_similarity,
                        "answer": answer_similarity,
                        "structured": structured_similarity,
                        "overall": overall_similarity
                    }
                }
        
        # 如果找到足够相似的匹配
        if best_match and best_match["similarity_scores"]["overall"] >= similarity_threshold:
            matched_sample = {
                "original_sample": original_sample,
                "matched_sample": best_match["generated_sample"],
                "similarity_scores": best_match["similarity_scores"],
                "match_quality": "high" if best_match["similarity_scores"]["overall"] >= 0.7 else "medium"
            }
            matched_samples.append(matched_sample)
        else:
            unmatched_count += 1
            if unmatched_count <= 5:  # 只显示前5个未匹配的样本
                print(f"未找到匹配: 原始问题='{original_question[:50]}...' (最佳相似度: {best_similarity:.3f})")
    
    print(f"✅ 匹配完成:")
    print(f"  - 成功匹配: {len(matched_samples)} 个样本")
    print(f"  - 未匹配: {unmatched_count} 个样本")
    print(f"  - 匹配率: {len(matched_samples)/len(original_eval_data)*100:.1f}%")
    
    return matched_samples

def analyze_matching_quality(matched_samples: List[Dict]):
    """分析匹配质量"""
    print(f"\n📊 匹配质量分析:")
    
    high_quality = [s for s in matched_samples if s["match_quality"] == "high"]
    medium_quality = [s for s in matched_samples if s["match_quality"] == "medium"]
    
    print(f"  - 高质量匹配 (≥0.7): {len(high_quality)} 个")
    print(f"  - 中等质量匹配 (0.4-0.7): {len(medium_quality)} 个")
    
    if matched_samples:
        avg_similarities = {
            "question": sum(s["similarity_scores"]["question"] for s in matched_samples) / len(matched_samples),
            "context": sum(s["similarity_scores"]["context"] for s in matched_samples) / len(matched_samples),
            "answer": sum(s["similarity_scores"]["answer"] for s in matched_samples) / len(matched_samples),
            "structured": sum(s["similarity_scores"]["structured"] for s in matched_samples) / len(matched_samples),
            "overall": sum(s["similarity_scores"]["overall"] for s in matched_samples) / len(matched_samples)
        }
        
        print(f"  - 平均相似度:")
        print(f"    问题: {avg_similarities['question']:.3f}")
        print(f"    上下文: {avg_similarities['context']:.3f}")
        print(f"    答案: {avg_similarities['answer']:.3f}")
        print(f"    结构化: {avg_similarities['structured']:.3f}")
        print(f"    综合: {avg_similarities['overall']:.3f}")

def save_matched_samples(matched_samples: List[Dict], output_file: str):
    """保存匹配的样本"""
    print(f"\n💾 保存匹配样本到: {output_file}")
    
    # 转换为评估格式
    eval_samples = []
    for match in matched_samples:
        generated_sample = match["matched_sample"]
        
        # 提取评估所需字段
        eval_sample = {
            "query": generated_sample.get("question", ""),
            "context": generated_sample.get("context", ""),
            "answer": generated_sample.get("answer", ""),
            "doc_id": generated_sample.get("doc_id", ""),
            "relevant_doc_ids": generated_sample.get("relevant_doc_ids", []),
            "answer_from": generated_sample.get("answer_from", "unknown"),
            "similarity_scores": match["similarity_scores"],
            "match_quality": match["match_quality"]
        }
        eval_samples.append(eval_sample)
    
    # 保存为JSONL格式
    with open(output_file, 'w', encoding='utf-8') as f:
        for sample in eval_samples:
            f.write(json.dumps(sample, ensure_ascii=False) + '\n')
    
    print(f"✅ 保存了 {len(eval_samples)} 个评估样本")

def show_examples(matched_samples: List[Dict], num_examples: int = 3):
    """显示匹配示例"""
    print(f"\n📝 匹配示例 (显示前{num_examples}个):")
    
    for i, match in enumerate(matched_samples[:num_examples]):
        original = match["original_sample"]
        generated = match["matched_sample"]
        scores = match["similarity_scores"]
        
        print(f"\n--- 示例 {i+1} ---")
        print(f"匹配质量: {match['match_quality']}")
        print(f"综合相似度: {scores['overall']:.3f}")
        print(f"原始问题: {original.get('query', '')[:100]}...")
        print(f"生成问题: {generated.get('question', '')[:100]}...")
        print(f"原始答案: {original.get('answer', '')[:50]}...")
        print(f"生成答案: {generated.get('answer', '')[:50]}...")

def main():
    parser = argparse.ArgumentParser(description="从AlphaFin生成数据中提取评估样本")
    parser.add_argument("--original_eval", type=str, 
                       default="evaluate_mrr/alphafin_eval.jsonl",
                       help="原始评估数据文件")
    parser.add_argument("--generated_data", type=str,
                       default="data/alphafin/alphafin_merged_generated_qa_full_dedup.json",
                       help="LLM生成的数据文件")
    parser.add_argument("--output", type=str,
                       default="evaluate_mrr/alphafin_eval_from_generated.jsonl",
                       help="输出文件路径")
    parser.add_argument("--similarity_threshold", type=float, default=0.4,
                       help="相似度阈值")
    parser.add_argument("--show_examples", action="store_true",
                       help="显示匹配示例")
    
    args = parser.parse_args()
    
    print("🚀 开始从AlphaFin生成数据中提取评估样本")
    print(f"📊 配置:")
    print(f"  - 原始评估数据: {args.original_eval}")
    print(f"  - 生成数据: {args.generated_data}")
    print(f"  - 输出文件: {args.output}")
    print(f"  - 相似度阈值: {args.similarity_threshold}")
    
    # 检查文件是否存在
    if not Path(args.original_eval).exists():
        print(f"❌ 原始评估数据文件不存在: {args.original_eval}")
        return
    
    if not Path(args.generated_data).exists():
        print(f"❌ 生成数据文件不存在: {args.generated_data}")
        return
    
    # 加载原始评估数据
    print(f"\n📖 加载原始评估数据: {args.original_eval}")
    original_eval_data = []
    with open(args.original_eval, 'r', encoding='utf-8') as f:
        for line in f:
            original_eval_data.append(json.loads(line))
    print(f"✅ 加载了 {len(original_eval_data)} 个原始评估样本")
    
    # 加载生成数据
    print(f"\n📖 加载生成数据: {args.generated_data}")
    generated_data = []
    
    # 尝试不同的加载方式
    try:
        # 首先尝试作为JSONL格式加载
        with open(args.generated_data, 'r', encoding='utf-8') as f:
            for line in f:
                line = line.strip()
                if line:
                    generated_data.append(json.loads(line))
        print(f"✅ 作为JSONL格式加载了 {len(generated_data)} 个生成样本")
    except json.JSONDecodeError:
        # 如果JSONL失败，尝试作为单个JSON文件加载
        generated_data = []
        with open(args.generated_data, 'r', encoding='utf-8') as f:
            content = f.read()
            data = json.loads(content)
            if isinstance(data, list):
                generated_data = data
                print(f"✅ 作为JSON数组加载了 {len(generated_data)} 个生成样本")
            elif isinstance(data, dict):
                # 如果是字典，可能包含数据列表
                for key, value in data.items():
                    if isinstance(value, list):
                        generated_data = value
                        print(f"✅ 从JSON对象中加载了 {len(generated_data)} 个生成样本 (键: {key})")
                        break
                if not generated_data:
                    print(f"❌ 无法从JSON对象中找到数据列表")
                    return
            else:
                print(f"❌ 不支持的JSON格式: {type(data)}")
                return
    
    # 查找匹配样本
    matched_samples = find_matching_samples(original_eval_data, generated_data, args.similarity_threshold)
    
    if not matched_samples:
        print(f"❌ 没有找到匹配的样本，请降低相似度阈值")
        return
    
    # 分析匹配质量
    analyze_matching_quality(matched_samples)
    
    # 显示示例
    if args.show_examples:
        show_examples(matched_samples)
    
    # 保存匹配样本
    save_matched_samples(matched_samples, args.output)
    
    print(f"\n🎉 完成！提取了 {len(matched_samples)} 个评估样本")

if __name__ == "__main__":
    main() 