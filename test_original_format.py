#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from xlm.components.generator.local_llm_generator import LocalLLMGenerator

def test_original_format():
    """测试Fin-R1模型现在接收完整的原始Prompt"""
    
    print("=== 测试Fin-R1模型接收完整Prompt ===")
    
    # 创建生成器实例
    generator = LocalLLMGenerator()
    
    # 构造完整的原始Prompt（模拟多阶段检索系统的输出）
    original_prompt = """你是一位专业的金融分析师，擅长分析公司财务报告并回答相关问题。

请基于以下公司财务报告片段，回答用户的问题。你的回答必须：
1. 准确、客观，基于提供的财务数据
2. 使用专业术语，但确保易于理解
3. 如果信息不足，明确指出并说明需要哪些额外信息
4. 提供具体的数字和百分比支持你的分析

【公司财务报告片段】
---
2023年年度报告显示，公司营业收入达到1,234.56亿元，同比增长15.3%。净利润为123.45亿元，同比增长12.8%。毛利率为25.6%，较上年同期提升1.2个百分点。研发投入为98.76亿元，占营业收入比例为8.0%，同比增长20.1%。经营活动现金流净额为156.78亿元，同比增长18.5%。资产负债率为45.2%，较上年同期下降2.1个百分点。
---

【用户问题】
请分析公司的财务状况和盈利能力，并评估其未来发展前景。
---

请基于上述信息提供详细的分析。"""

    print(f"原始Prompt长度: {len(original_prompt)} 字符")
    print(f"原始Prompt内容预览:\n{original_prompt[:200]}...")
    print("-" * 50)
    
    # 调用生成器
    try:
        responses = generator.generate([original_prompt])
        print(f"生成器响应: {responses[0]}")
    except Exception as e:
        print(f"生成器调用失败: {e}")

if __name__ == "__main__":
    test_original_format() 