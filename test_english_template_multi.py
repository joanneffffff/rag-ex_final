#!/usr/bin/env python3
"""
多模板英文测试脚本
快速比较不同RAG模板的效果
"""

import json
import time
from pathlib import Path
from typing import List, Dict, Any, Optional

# 添加项目根目录到路径
import sys
sys.path.append(str(Path(__file__).parent))

# 导入测试器
from test_english_template import LLMTemplateTester, load_sample_data

def load_template(template_path: str) -> Optional[str]:
    """加载模板文件"""
    try:
        with open(template_path, 'r', encoding='utf-8') as f:
            return f.read()
    except FileNotFoundError:
        print(f"❌ 未找到模板文件: {template_path}")
        return None

def create_messages_from_template(template_content: str, context: str, question: str) -> List[Dict[str, str]]:
    """从模板创建消息列表"""
    if "===SYSTEM===" in template_content and "===USER===" in template_content:
        system_part = template_content.split("===SYSTEM===")[1].split("===USER===")[0].strip()
        user_part = template_content.split("===USER===")[1].strip()
        
        # 替换user部分中的占位符
        user_message = user_part.replace("{context}", context).replace("{question}", question)
        system_message = system_part
    else:
        # 如果模板格式不正确，使用整个内容作为system消息
        system_message = template_content
        user_message = f"""Context:
{context}

Question:
{question}

A:"""
    
    return [
        {"role": "system", "content": system_message},
        {"role": "user", "content": user_message}
    ]

def test_single_template(tester: LLMTemplateTester, template_name: str, template_content: str, 
                        sample_data: List[Dict[str, Any]], max_samples: int = 3) -> Dict[str, Any]:
    """测试单个模板"""
    print(f"\n🧪 测试模板: {template_name}")
    print("="*50)
    
    results = []
    
    for i, sample in enumerate(sample_data[:max_samples]):
        print(f"\n--- 样本 {i+1} ---")
        print(f"问题: {sample['question']}")
        print(f"预期答案: {sample['answer']}")
        
        # 创建消息
        messages = create_messages_from_template(template_content, sample["context"], sample["question"])
        
        # 生成回答
        generation_result = tester.generate_response(messages)
        
        # 评估
        evaluation = tester.evaluate_answer_quality(
            generated_answer=generation_result["cleaned_answer"],
            expected_answer=sample["answer"],
            context=sample["context"],
            question=sample["question"]
        )
        
        result = {
            "template_name": template_name,
            "sample_id": i + 1,
            "question": sample["question"],
            "expected_answer": sample["answer"],
            "generated_answer": generation_result["cleaned_answer"],
            "raw_answer": generation_result["generated_answer"],
            "quality_score": evaluation["quality_score"],
            "exact_match": evaluation["exact_match"],
            "semantic_similarity": evaluation["semantic_similarity"],
            "format_violations": evaluation["format_violations"],
            "generation_time": generation_result["generation_time"]
        }
        
        results.append(result)
        
        # 打印结果
        print(f"✅ 生成答案: {generation_result['cleaned_answer']}")
        print(f"📊 质量分数: {evaluation['quality_score']:.3f}")
        print(f"📊 精确匹配: {evaluation['exact_match']}")
        if evaluation['format_violations']:
            print(f"⚠️ 格式违规: {evaluation['format_violations']}")
    
    # 计算平均指标
    avg_quality = sum(r["quality_score"] for r in results) / len(results)
    avg_time = sum(r["generation_time"] for r in results) / len(results)
    exact_match_rate = sum(1 for r in results if r["exact_match"]) / len(results)
    format_violation_rate = sum(1 for r in results if r["format_violations"]) / len(results)
    
    print(f"\n📊 {template_name} 总结:")
    print(f"   平均质量分数: {avg_quality:.3f}")
    print(f"   精确匹配率: {exact_match_rate:.3f}")
    print(f"   平均生成时间: {avg_time:.2f}s")
    print(f"   格式违规率: {format_violation_rate:.3f}")
    
    return {
        "template_name": template_name,
        "results": results,
        "summary": {
            "avg_quality": avg_quality,
            "avg_time": avg_time,
            "exact_match_rate": exact_match_rate,
            "format_violation_rate": format_violation_rate
        }
    }

def main():
    """主函数"""
    print("🚀 多模板英文测试开始")
    
    # 定义要测试的模板
    templates = {
        "Original": "data/prompt_templates/rag_english_template.txt",
        "Optimized": "data/prompt_templates/rag_english_template_optimized.txt", 
        "Minimal": "data/prompt_templates/rag_english_template_minimal.txt",
        "Focused": "data/prompt_templates/rag_english_template_focused.txt"
    }
    
    # 初始化测试器
    tester = LLMTemplateTester(
        model_name="SUFE-AIFLM-Lab/Fin-R1",
        device="auto"
    )
    
    try:
        tester.load_model()
    except Exception as e:
        print(f"❌ 模型加载失败: {e}")
        return
    
    # 加载样本数据
    sample_data = load_sample_data()
    print(f"✅ 加载了 {len(sample_data)} 个测试样本")
    
    # 测试所有模板
    all_results = []
    
    for template_name, template_path in templates.items():
        template_content = load_template(template_path)
        if template_content:
            result = test_single_template(tester, template_name, template_content, sample_data, max_samples=3)
            all_results.append(result)
        else:
            print(f"⚠️ 跳过模板: {template_name}")
    
    # 比较结果
    print(f"\n🏆 模板效果对比")
    print("="*60)
    
    for result in all_results:
        summary = result["summary"]
        print(f"\n{result['template_name']:12}:")
        print(f"   质量分数: {summary['avg_quality']:.3f}")
        print(f"   精确匹配: {summary['exact_match_rate']:.3f}")
        print(f"   生成时间: {summary['avg_time']:.2f}s")
        print(f"   格式违规: {summary['format_violation_rate']:.3f}")
    
    # 找出最佳模板
    best_quality = max(all_results, key=lambda x: x["summary"]["avg_quality"])
    best_time = min(all_results, key=lambda x: x["summary"]["avg_time"])
    best_format = min(all_results, key=lambda x: x["summary"]["format_violation_rate"])
    
    print(f"\n🎯 最佳模板推荐:")
    print(f"   最佳质量: {best_quality['template_name']} ({best_quality['summary']['avg_quality']:.3f})")
    print(f"   最快速度: {best_time['template_name']} ({best_time['summary']['avg_time']:.2f}s)")
    print(f"   最少违规: {best_format['template_name']} ({best_format['summary']['format_violation_rate']:.3f})")
    
    # 保存结果
    output_file = "multi_template_test_results.json"
    with open(output_file, "w", encoding="utf-8") as f:
        json.dump({
            "results": all_results,
            "timestamp": time.time()
        }, f, indent=2, ensure_ascii=False)
    
    print(f"\n✅ 详细结果已保存到: {output_file}")
    print("🎉 多模板测试完成！")

if __name__ == "__main__":
    main() 