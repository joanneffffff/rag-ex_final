#!/usr/bin/env python3
"""
详细模板英文测试脚本
使用完整的Chain-of-Thought示例进行TATQA评估
"""

# 临时关闭warnings，避免transformers参数警告
import warnings
warnings.filterwarnings("ignore")

# 更精确地过滤transformers生成参数警告
import logging
logging.getLogger("transformers.generation.utils").setLevel(logging.ERROR)

import json
import re
import torch
import numpy as np
from pathlib import Path
from typing import List, Dict, Any, Optional
import random
import time
from difflib import SequenceMatcher

# 添加项目根目录到路径
import sys
sys.path.append(str(Path(__file__).parent))

# 导入RAG系统的LocalLLMGenerator
try:
    from xlm.components.generator.local_llm_generator import LocalLLMGenerator
    USE_RAG_GENERATOR = True
    print("✅ 使用RAG系统的LocalLLMGenerator")
except ImportError:
    USE_RAG_GENERATOR = False
    print("⚠️ 无法导入RAG系统的LocalLLMGenerator，使用备用方案")
    from transformers import AutoTokenizer, AutoModelForCausalLM
    from transformers.utils.quantization_config import BitsAndBytesConfig

from test_english_template import LLMTemplateTester, load_sample_data

def get_detailed_english_prompt_messages(context_content: str, question_text: str, summary_content: Optional[str] = None) -> List[Dict[str, str]]:
    """
    生成 LLM 期望的 messages 列表。
    这是为高难度 TATQA 数据集特化的高性能模板，核心是利用思维链 (Chain-of-Thought) 引导模型进行复杂推理。
    """
    
    # TATQA专用高精度模板
    system_message_content = """You are a world-class quantitative financial analyst AI. Your mission is to solve complex financial questions with extreme precision, based on a given context that may include both tables and text. You must emulate the thinking process of an expert analyst before giving the final answer.

### Core Directives

1.  **Reasoning Process (Internal Thought)**: For every question, you MUST first perform a step-by-step reasoning process, like the examples below. Break down the question, identify necessary data from the table and text, formulate the calculation, and derive the solution. This is your internal monologue.
2.  **Final Output (Your Public Answer)**: Your final, visible output MUST BE the answer ONLY. It should be stripped of all reasoning, explanations, units (unless asked), and introductory phrases. The thinking process is for you to arrive at the correct answer, but it should not be part of your final output.
3.  **Output Format**:
    * For numerical or list-based answers, separate items with a semicolon and a space (e.g., `Value1; Value2`).
    * For text-based answers, provide only the minimal, essential phrase.
    * If the answer is impossible to find, state exactly: `The answer cannot be found in the provided context.`

### Annotated Reasoning Examples (Chain-of-Thought Demonstration)

---
**Example 1: Multi-Step Calculation**
**Q**: What was the percentage increase / (decrease) in capital expenditure from 2018 to 2019?
**Context**:
Table: Capital expenditures 1: 2019 is $2,807; 2018 is $2,790.
**Thought**:
1.  **Objective**: Calculate the percentage change in capital expenditure between 2018 and 2019.
2.  **Data Extraction**:
    * New Value (2019): 2,807
    * Old Value (2018): 2,790
3.  **Formula**: Percentage Change = ((New Value - Old Value) / Old Value) * 100
4.  **Calculation**:
    * Change = 2,807 - 2,790 = 17
    * Ratio = 17 / 2,790 ≈ 0.006093
    * Percentage = 0.006093 * 100 ≈ 0.6093%
5.  **Final Formatting**: The question asks for a percentage. Rounding to two decimal places is standard.
**A**: 0.61%

---
**Example 2: Table and Text Integration**
**Q**: What was the adjusted operating income, excluding one-time restructuring charges?
**Context**:
Text: "Our adjusted metrics provide a clearer view of core performance by excluding special items, such as restructuring charges."
Table: Operating Income: $500M; Restructuring Charges: $20M.
**Thought**:
1.  **Objective**: Find the adjusted operating income.
2.  **Definition**: The text defines "adjusted" as excluding (i.e., adding back) restructuring charges to the reported operating income.
3.  **Data Extraction**:
    * Operating Income: 500
    * Restructuring Charges: 20
4.  **Formula**: Adjusted Operating Income = Reported Operating Income + Restructuring Charges
5.  **Calculation**: 500 + 20 = 520
6.  **Final Formatting**: Provide the final number.
**A**: 520

---
**Example 3: Filtering and Aggregation**
**Q**: What is the total R&D and G&A expense for the year ended July 27, 2019?
**Context**:
Table (Columns: Expense Type, July 27, 2019, July 28, 2018)
Row_R&D: $6,577; $6,332
Row_Sales: $9,571; $9,242
Row_G&A: $1,827; $2,144
**Thought**:
1.  **Objective**: Sum the R&D and G&A expenses for the specific year 2019.
2.  **Filtering**: I need to focus only on the column "July 27, 2019" and the rows "R&D" and "G&A".
3.  **Data Extraction**:
    * R&D expense for 2019: 6,577
    * G&A expense for 2019: 1,827
4.  **Formula**: Total = R&D + G&A
5.  **Calculation**: 6,577 + 1,827 = 8,404
6.  **Final Formatting**: Provide the final number.
**A**: 8404
---"""

    user_message = f"""Context:
{context_content}

Question:
{question_text}

A:"""

    return [
        {"role": "system", "content": system_message_content},
        {"role": "user", "content": user_message}
    ]

def main():
    print("🚀 详细模板英文测试开始")
    print("使用完整的Chain-of-Thought示例")
    print("="*60)
    
    # 初始化LLM测试器，增加max_length
    tester = LLMTemplateTester(
        model_name="SUFE-AIFLM-Lab/Fin-R1",
        device="auto"
    )
    tester.max_length = 4096  # 增加max_length以支持详细模板
    
    try:
        tester.load_model()
    except Exception as e:
        print(f"❌ 模型加载失败: {e}")
        return
    
    # 加载样本数据
    sample_data = load_sample_data()
    print(f"✅ 加载了 {len(sample_data)} 个测试样本")
    
    # 存储所有结果
    all_results = []
    
    # 测试详细模板
    template_name_to_test = "Detailed English Template with CoT"
    
    for i, sample in enumerate(sample_data):
        print(f"\n--- 样本 {i+1} ---")
        print(f"问题: {sample['question']}")
        print(f"预期答案: {sample['answer']}")

        # 构建详细Prompt消息列表
        messages_for_llm = get_detailed_english_prompt_messages(
            context_content=sample["context"], 
            question_text=sample["question"],
            summary_content=sample["context"]
        )
        
        # 调用LLM生成回答
        generation_result = tester.generate_response(messages_for_llm)
        
        # 将本次测试的结果添加到all_results列表
        result_for_analysis = {
            "template_name": template_name_to_test,
            "template": messages_for_llm,
            "context": sample["context"],
            "question": sample["question"],
            "expected_answer": sample["answer"],
            "template_length": len(tester._convert_messages_to_text(messages_for_llm)),
            "generation": generation_result,
            "evaluation": tester.evaluate_answer_quality(
                generated_answer=generation_result["cleaned_answer"],
                expected_answer=sample["answer"],
                context=sample["context"],
                question=sample["question"]
            )
        }
        all_results.append(result_for_analysis)

        # 打印详细结果和调试信息
        print(f"\n--- 发送给模型的完整Prompt ---")
        print(tester._convert_messages_to_text(messages_for_llm))
        print(f"--- Prompt 结束 ---")
        
        print(f"\n✅ {tester.model_name} 原始回答 (后处理前):")
        print(f"{'='*50}")
        print(generation_result["generated_answer"].strip())
        print(f"{'='*50}")
        
        print(f"\n✅ {tester.model_name} 后处理回答 (最终):")
        print(f"{'='*50}")
        print(generation_result["cleaned_answer"])
        print(f"{'='*50}")
        print(f"📏 最终长度: {len(generation_result['cleaned_answer'])} 字符")
        
        # 获取评估结果
        evaluation_result = result_for_analysis["evaluation"]
        
        print(f"\n📊 评估结果:")
        print(f"   - 质量分数: {evaluation_result['quality_score']:.3f}")
        print(f"   - 精确匹配: {evaluation_result['exact_match']}")
        print(f"   - 语义相似度: {evaluation_result['semantic_similarity']:.3f}")
        if evaluation_result['format_violations']:
            print(f"   - 格式违规: {', '.join(evaluation_result['format_violations'])}")

    # 分析结果
    print(f"\n📊 分析 {len(all_results)} 个测试结果...")
    
    # 计算平均指标
    avg_quality = sum(r["evaluation"]["quality_score"] for r in all_results) / len(all_results)
    avg_time = sum(r["generation"]["generation_time"] for r in all_results) / len(all_results)
    exact_match_rate = sum(1 for r in all_results if r["evaluation"]["exact_match"]) / len(all_results)
    format_violation_rate = sum(1 for r in all_results if r["evaluation"]["format_violations"]) / len(all_results)
    
    print(f"\n📊 {template_name_to_test} 总结:")
    print(f"   平均质量分数: {avg_quality:.3f}")
    print(f"   精确匹配率: {exact_match_rate:.3f}")
    print(f"   平均生成时间: {avg_time:.2f}s")
    print(f"   格式违规率: {format_violation_rate:.3f}")
    
    # 保存详细结果
    output_file = "detailed_template_test_results.json"
    with open(output_file, "w", encoding="utf-8") as f:
        json.dump({
            "results": all_results,
            "summary": {
                "template_name": template_name_to_test,
                "avg_quality": avg_quality,
                "avg_time": avg_time,
                "exact_match_rate": exact_match_rate,
                "format_violation_rate": format_violation_rate
            },
            "timestamp": time.time()
        }, f, indent=2, ensure_ascii=False)
    
    print(f"\n✅ 详细结果已保存到: {output_file}")
    print("🎉 详细模板测试完成！")

if __name__ == "__main__":
    main() 