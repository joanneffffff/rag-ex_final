#!/usr/bin/env python3
"""
优化Few-Shot COT策略
在有限Token预算内最大化Few-Shot COT的价值
"""

import json
from typing import List, Dict, Any
from collections import Counter

def load_bad_samples(file_path: str = "comprehensive_evaluation_100_samples.json") -> List[Dict[str, Any]]:
    """加载并筛选低质量样本"""
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
            results = data.get("results", [])
    except FileNotFoundError:
        print(f"❌ 未找到文件: {file_path}")
        return []
    
    # 筛选质量分数低于0.5的样本
    bad_samples = []
    for result in results:
        quality_score = result.get("evaluation", {}).get("quality_score", 0)
        if quality_score < 0.5:
            bad_samples.append(result)
    
    return bad_samples

def categorize_failure_patterns(bad_samples: List[Dict[str, Any]]) -> Dict[str, List[Dict[str, Any]]]:
    """分类失败模式"""
    patterns = {
        "复杂计算": [],
        "多跳推理": [],
        "表格理解": [],
        "实体抽取": [],
        "数值提取": [],
        "其他": []
    }
    
    for sample in bad_samples:
        query = sample.get("query", "").lower()
        context = sample.get("context", "").lower()
        expected_answer = sample.get("expected_answer", "")
        
        # 复杂计算类
        if any(keyword in query for keyword in ["calculate", "compute", "sum", "total", "difference", "percentage", "average", "net"]):
            if "percentage" in query or "average" in query:
                patterns["复杂计算"].append(sample)
            else:
                patterns["复杂计算"].append(sample)
        
        # 多跳推理类
        elif any(keyword in query for keyword in ["respectively", "both", "and", "or", "compare", "which"]):
            patterns["多跳推理"].append(sample)
        
        # 表格理解类
        elif "table id:" in context:
            patterns["表格理解"].append(sample)
        
        # 实体抽取类
        elif any(keyword in query for keyword in ["what does", "method", "company", "name"]):
            patterns["实体抽取"].append(sample)
        
        # 数值提取类
        elif any(keyword in query for keyword in ["what is", "how much", "amount", "value"]):
            patterns["数值提取"].append(sample)
        
        else:
            patterns["其他"].append(sample)
    
    return patterns

def select_representative_samples(patterns: Dict[str, List[Dict[str, Any]]], max_samples: int = 5) -> List[Dict[str, Any]]:
    """选择最具代表性的样本"""
    selected_samples = []
    
    # 按失败模式的重要性排序
    priority_order = ["复杂计算", "多跳推理", "表格理解", "实体抽取", "数值提取"]
    
    for pattern in priority_order:
        samples = patterns.get(pattern, [])
        if samples:
            # 选择质量分数最低的样本（最需要改进的）
            samples.sort(key=lambda x: x.get("evaluation", {}).get("quality_score", 0))
            selected_samples.append({
                "pattern": pattern,
                "sample": samples[0],
                "priority": len(priority_order) - priority_order.index(pattern)
            })
    
    # 按优先级排序并限制数量
    selected_samples.sort(key=lambda x: x["priority"], reverse=True)
    return selected_samples[:max_samples]

def create_optimized_cot_examples(selected_samples: List[Dict[str, Any]]) -> str:
    """创建优化的Few-Shot COT示例"""
    cot_examples = """Below are some examples of how to reason step by step and extract the final answer. For your answer, only output the final A: part, do not repeat the reasoning.

"""
    
    for i, item in enumerate(selected_samples, 1):
        sample = item["sample"]
        pattern = item["pattern"]
        
        # 精简context - 只保留必要信息
        context = sample.get("context", "")
        if "table id:" in context.lower():
            # 对于表格，只保留关键行
            lines = context.split('\n')
            table_lines = [line for line in lines if 'is ' in line and ('$' in line or '%' in line or any(char.isdigit() for char in line))]
            context = '\n'.join(table_lines[:6])  # 限制表格行数
        
        # 精简question
        question = sample.get("query", "")
        
        # 创建精简的Thought过程
        expected_answer = sample.get("expected_answer", "")
        thought = create_optimized_thought(question, context, expected_answer, pattern)
        
        # 构建示例
        cot_examples += f"""Q: {question}
Context: {context[:200]}{'...' if len(context) > 200 else ''}
Thought: {thought}
A: {expected_answer}

"""
    
    return cot_examples

def create_optimized_thought(question: str, context: str, expected_answer: str, pattern: str) -> str:
    """创建优化的Thought过程"""
    if pattern == "复杂计算":
        if "percentage" in question.lower():
            return "Extract values from context, calculate percentage: (new-old)/old*100"
        elif "average" in question.lower():
            return "Extract values, calculate average: sum/count"
        else:
            return "Extract values, perform required calculation"
    
    elif pattern == "多跳推理":
        return "Identify multiple entities, extract values for each, compare or combine as needed"
    
    elif pattern == "表格理解":
        return "Locate relevant rows/columns in table, extract specific values"
    
    elif pattern == "实体抽取":
        return "Find definition or description in context, extract key information"
    
    elif pattern == "数值提取":
        return "Locate specific value in context, extract exact number"
    
    else:
        return "Extract relevant information from context"

def estimate_token_count(text: str) -> int:
    """估算Token数量（粗略估算）"""
    # 简单估算：英文约4个字符1个token
    return len(text) // 4

def main():
    """主函数"""
    print("🎯 优化Few-Shot COT策略分析")
    print("="*60)
    
    # 加载bad samples
    bad_samples = load_bad_samples()
    if not bad_samples:
        print("❌ 没有找到低质量样本")
        return
    
    print(f"✅ 找到 {len(bad_samples)} 个低质量样本")
    
    # 分类失败模式
    patterns = categorize_failure_patterns(bad_samples)
    
    print(f"\n📊 失败模式分布:")
    for pattern, samples in patterns.items():
        if samples:
            print(f"   {pattern}: {len(samples)} 个样本")
    
    # 选择代表性样本
    selected_samples = select_representative_samples(patterns, max_samples=5)
    
    print(f"\n🎯 选择的代表性样本:")
    for i, item in enumerate(selected_samples, 1):
        sample = item["sample"]
        pattern = item["pattern"]
        quality_score = sample.get("evaluation", {}).get("quality_score", 0)
        print(f"   {i}. {pattern}: 质量分数 {quality_score:.3f}")
        print(f"      问题: {sample.get('query', '')[:80]}...")
        print(f"      期望答案: {sample.get('expected_answer', '')}")
    
    # 创建优化的COT示例
    optimized_cot = create_optimized_cot_examples(selected_samples)
    
    # 估算Token使用量
    estimated_tokens = estimate_token_count(optimized_cot)
    
    print(f"\n📝 优化的Few-Shot COT示例:")
    print(f"   估算Token数量: {estimated_tokens}")
    print(f"   示例数量: {len(selected_samples)}")
    print(f"   是否在预算内: {'✅' if estimated_tokens < 800 else '❌'}")
    
    print(f"\n{optimized_cot}")
    
    # 保存优化结果
    optimization_result = {
        "selected_samples": [
            {
                "pattern": item["pattern"],
                "sample_id": item["sample"].get("sample_id", "unknown"),
                "query": item["sample"].get("query", ""),
                "expected_answer": item["sample"].get("expected_answer", ""),
                "quality_score": item["sample"].get("evaluation", {}).get("quality_score", 0)
            }
            for item in selected_samples
        ],
        "optimized_cot": optimized_cot,
        "estimated_tokens": estimated_tokens,
        "total_bad_samples": len(bad_samples)
    }
    
    with open("cot_optimization_result.json", "w", encoding="utf-8") as f:
        json.dump(optimization_result, f, ensure_ascii=False, indent=2)
    
    print(f"\n✅ 优化结果已保存到 cot_optimization_result.json")

if __name__ == "__main__":
    main() 