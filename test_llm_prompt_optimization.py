#!/usr/bin/env python3
"""
测试和优化LLM prompt模板
分析发送给生成器的内容和prompt模板效果
"""

import sys
import os
import json
from pathlib import Path
from typing import List, Dict, Any

# 添加项目根目录到路径
sys.path.append(str(Path(__file__).parent))

def test_current_prompt():
    """测试当前的prompt模板"""
    print("=== 测试当前Prompt模板 ===")
    
    # 导入多阶段检索系统
    from alphafin_data_process.multi_stage_retrieval_final import MultiStageRetrievalSystem
    from xlm.components.prompt_templates.template_loader import template_loader
    
    # 初始化系统
    data_path = Path("data/alphafin/alphafin_merged_generated_qa.json")
    retrieval_system = MultiStageRetrievalSystem(
        data_path=data_path,
        dataset_type="chinese",
        use_existing_config=True
    )
    
    # 测试查询
    test_query = "德赛电池(000049)的下一季度收益预测如何？"
    print(f"测试查询: {test_query}")
    
    # 执行检索
    results = retrieval_system.search(
        query=test_query,
        company_name="德赛电池",
        stock_code="000049",
        top_k=5
    )
    
    # 分析结果
    print(f"\n检索到 {len(results.get('retrieved_documents', []))} 个文档")
    
    # 查看前3个文档的内容
    for i, doc in enumerate(results.get('retrieved_documents', [])[:3]):
        print(f"\n--- 文档 {i+1} ---")
        print(f"公司: {doc.get('company_name', 'N/A')}")
        print(f"股票代码: {doc.get('stock_code', 'N/A')}")
        print(f"摘要: {doc.get('summary', 'N/A')[:200]}...")
        print(f"相似度分数: {doc.get('combined_score', 0):.4f}")
    
    # 查看LLM生成的答案
    llm_answer = results.get('llm_answer', 'N/A')
    print(f"\n=== LLM生成的答案 ===")
    print(llm_answer)
    
    return results

def analyze_prompt_content():
    """分析发送给LLM的prompt内容"""
    print("\n=== 分析Prompt内容 ===")
    
    # 获取当前prompt模板
    from xlm.components.prompt_templates.template_loader import template_loader
    
    # 模拟上下文和查询
    context = """
    德赛电池(000049)2023年第三季度报告显示，公司实现营业收入45.67亿元，同比增长12.3%；
    净利润为3.24亿元，同比增长8.7%。公司预计2023年第四季度营业收入将达到48-52亿元，
    净利润预计为3.5-4.0亿元。公司表示，受益于新能源汽车市场的持续增长，动力电池业务
    保持良好发展态势。
    """
    
    query = "德赛电池(000049)的下一季度收益预测如何？"
    
    # 使用当前模板
    current_prompt = template_loader.format_template(
        "multi_stage_chinese_template",
        context=context,
        query=query
    )
    
    print("当前Prompt模板:")
    print("=" * 50)
    print(current_prompt)
    print("=" * 50)
    
    return current_prompt

def test_optimized_prompts():
    """测试优化的prompt模板"""
    print("\n=== 测试优化的Prompt模板 ===")
    
    # 模拟上下文和查询
    context = """
    德赛电池(000049)2023年第三季度报告显示，公司实现营业收入45.67亿元，同比增长12.3%；
    净利润为3.24亿元，同比增长8.7%。公司预计2023年第四季度营业收入将达到48-52亿元，
    净利润预计为3.5-4.0亿元。公司表示，受益于新能源汽车市场的持续增长，动力电池业务
    保持良好发展态势。
    """
    
    query = "德赛电池(000049)的下一季度收益预测如何？"
    
    # 优化后的prompt模板
    optimized_prompts = {
        "template_1": f"""你是一位专业的金融分析师。请基于以下上下文信息，准确回答用户问题。

要求：
1. 只使用提供的信息，不要添加外部知识
2. 回答要具体、准确，包含具体数字和百分比
3. 用简洁的中文表达，不超过100字
4. 不要添加任何格式标记或额外说明

上下文：{context}

问题：{query}

回答：""",
        
        "template_2": f"""基于以下财务信息，回答用户问题。

**重要：请提供具体、准确的财务数据，包含数字和百分比。回答要简洁直接。**

财务信息：{context}

问题：{query}

回答：""",
        
        "template_3": f"""你是一位专业的财务分析师。请根据以下公司财务数据回答问题。

**要求：**
- 提供具体的财务数字和预测数据
- 回答要准确、简洁
- 重点关注收益预测相关数据

公司财务数据：{context}

问题：{query}

回答：""",
        
        "template_4": f"""请基于以下上下文信息回答问题。

**回答要求：**
1. 提取具体的财务预测数据
2. 包含数字、百分比等关键信息
3. 用一句话总结核心要点
4. 不要添加任何格式标记

上下文：{context}

问题：{query}

回答："""
    }
    
    # 测试每个模板
    for name, prompt in optimized_prompts.items():
        print(f"\n--- {name} ---")
        print("=" * 50)
        print(prompt)
        print("=" * 50)
    
    return optimized_prompts

def create_optimized_template_file():
    """创建优化的prompt模板文件"""
    print("\n=== 创建优化的Prompt模板文件 ===")
    
    optimized_template = """你是一位专业的金融分析师。请基于以下上下文信息，准确回答用户问题。

**回答要求：**
1. 只使用提供的信息，不要添加外部知识
2. 提供具体的财务数字、预测数据和百分比
3. 回答要准确、简洁，不超过100字
4. 重点关注收益、收入、利润等财务指标
5. 不要添加任何格式标记、编号或额外说明

**示例回答格式：**
- 根据预测，公司下一季度营业收入预计为X亿元，净利润预计为Y亿元
- 相比上季度，预计增长/下降Z%

上下文：{context}

问题：{query}

回答："""
    
    # 保存到文件
    template_path = Path("data/prompt_templates/multi_stage_chinese_optimized.txt")
    with open(template_path, 'w', encoding='utf-8') as f:
        f.write(optimized_template)
    
    print(f"优化模板已保存到: {template_path}")
    return template_path

def test_with_optimized_template():
    """使用优化后的模板测试"""
    print("\n=== 使用优化模板测试 ===")
    
    # 创建优化模板
    template_path = create_optimized_template_file()
    
    # 重新加载模板
    from xlm.components.prompt_templates.template_loader import template_loader
    
    # 模拟测试
    context = """
    德赛电池(000049)2023年第三季度报告显示，公司实现营业收入45.67亿元，同比增长12.3%；
    净利润为3.24亿元，同比增长8.7%。公司预计2023年第四季度营业收入将达到48-52亿元，
    净利润预计为3.5-4.0亿元。公司表示，受益于新能源汽车市场的持续增长，动力电池业务
    保持良好发展态势。
    """
    
    query = "德赛电池(000049)的下一季度收益预测如何？"
    
    # 使用优化模板
    optimized_prompt = template_loader.format_template(
        "multi_stage_chinese_optimized",
        context=context,
        query=query
    )
    
    print("优化后的Prompt:")
    print("=" * 50)
    print(optimized_prompt)
    print("=" * 50)
    
    return optimized_prompt

def main():
    """主函数"""
    print("开始LLM Prompt优化测试...")
    
    # 1. 测试当前prompt
    try:
        results = test_current_prompt()
    except Exception as e:
        print(f"测试当前prompt失败: {e}")
    
    # 2. 分析prompt内容
    analyze_prompt_content()
    
    # 3. 测试优化模板
    test_optimized_prompts()
    
    # 4. 创建并测试优化模板
    test_with_optimized_template()
    
    print("\n=== 测试完成 ===")
    print("建议：")
    print("1. 当前prompt模板过于简单，缺乏具体的指导")
    print("2. 优化模板增加了财务分析的专业性指导")
    print("3. 建议使用优化模板替换当前模板")

if __name__ == "__main__":
    main() 