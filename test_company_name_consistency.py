#!/usr/bin/env python3
"""
测试公司名称一致性修复效果
"""

import sys
import os
from pathlib import Path

# 添加项目根目录到路径
sys.path.append(str(Path(__file__).parent))

def test_company_name_consistency():
    """测试公司名称一致性修复"""
    print("=== 测试公司名称一致性修复 ===")
    
    # 导入多阶段检索系统
    from alphafin_data_process.multi_stage_retrieval_final import MultiStageRetrievalSystem
    
    # 初始化系统
    data_path = Path("data/alphafin/alphafin_merged_generated_qa.json")
    retrieval_system = MultiStageRetrievalSystem(
        data_path=data_path,
        dataset_type="chinese",
        use_existing_config=True
    )
    
    # 测试查询列表
    test_queries = [
        "德赛电池（000049）2021年利润持续增长的主要原因是什么？",
        "德赛电池(000049)的下一季度收益预测如何？",
        "中国平安（601318）的保险业务发展情况？",
        "比亚迪（002594）的电动汽车销量如何？",
    ]
    
    print(f"测试 {len(test_queries)} 个查询的公司名称一致性...")
    
    for i, query in enumerate(test_queries, 1):
        print(f"\n{'='*60}")
        print(f"测试 {i}: {query}")
        print(f"{'='*60}")
        
        # 提取公司名称和股票代码
        from xlm.utils.stock_info_extractor import extract_stock_info
        company_name, stock_code = extract_stock_info(query)
        
        print(f"提取的公司名称: {company_name}")
        print(f"提取的股票代码: {stock_code}")
        
        # 执行检索
        results = retrieval_system.search(
            query=query,
            company_name=company_name,
            stock_code=stock_code,
            top_k=5
        )
        
        # 检查LLM生成的答案
        if isinstance(results, dict) and 'llm_answer' in results:
            llm_answer = results['llm_answer']
            print(f"\nLLM生成的答案:")
            print(f"'{llm_answer}'")
            
            # 检查公司名称一致性
            check_company_name_consistency(query, llm_answer, company_name)
        else:
            print("❌ 未获取到LLM答案")
    
    print(f"\n{'='*60}")
    print("测试完成")
    print(f"{'='*60}")

def check_company_name_consistency(query: str, answer: str, expected_company: str):
    """检查公司名称一致性"""
    print(f"\n🔍 公司名称一致性检查:")
    
    # 检查原始公司名称是否在答案中
    if expected_company and expected_company in answer:
        print(f"✅ 原始公司名称 '{expected_company}' 在答案中正确保持")
    else:
        print(f"❌ 原始公司名称 '{expected_company}' 在答案中缺失")
    
    # 检查是否有翻译问题
    translation_issues = []
    
    # 德赛电池相关检查
    if "德赛" in expected_company:
        if "battery" in answer.lower() or "Battery" in answer:
            translation_issues.append("德赛电池被翻译为battery")
        if "德赛 battery" in answer or "德赛 Battery" in answer:
            translation_issues.append("德赛电池被部分翻译")
    
    # 中国平安相关检查
    if "中国平安" in expected_company:
        if "ping an" in answer.lower() or "Ping An" in answer:
            translation_issues.append("中国平安被翻译为Ping An")
    
    # 比亚迪相关检查
    if "比亚迪" in expected_company:
        if "byd" in answer.lower() or "BYD" in answer:
            translation_issues.append("比亚迪被翻译为BYD")
    
    if translation_issues:
        print(f"❌ 发现翻译问题:")
        for issue in translation_issues:
            print(f"   - {issue}")
    else:
        print(f"✅ 未发现翻译问题")
    
    # 检查语言一致性
    chinese_chars = sum(1 for char in answer if '\u4e00' <= char <= '\u9fff')
    english_words = len([word for word in answer.split() if word.isalpha() and word.isascii()])
    
    print(f"📊 语言统计:")
    print(f"   中文字符数: {chinese_chars}")
    print(f"   英文单词数: {english_words}")
    
    if chinese_chars > english_words:
        print(f"✅ 答案以中文为主，语言一致性良好")
    else:
        print(f"⚠️  答案中英文混合，可能存在语言不一致问题")

def test_prompt_template():
    """测试prompt模板是否包含公司名称保护指令"""
    print("\n=== 测试Prompt模板 ===")
    
    from xlm.components.prompt_templates.template_loader import template_loader
    
    # 测试模板格式化
    test_context = "德赛电池（000049）的业绩预告超出预期..."
    test_query = "德赛电池（000049）2021年利润持续增长的主要原因是什么？"
    test_summary = test_context[:200] + "..."
    
    prompt = template_loader.format_template(
        "multi_stage_chinese_template",
        summary=test_summary,
        context=test_context,
        query=test_query
    )
    
    if prompt:
        # 检查是否包含公司名称保护指令
        protection_keywords = [
            "严格禁止将中文公司名称翻译为英文",
            "必须保持原始的中文公司名称不变",
            "公司名称处理"
        ]
        
        found_protections = []
        for keyword in protection_keywords:
            if keyword in prompt:
                found_protections.append(keyword)
        
        if found_protections:
            print(f"✅ Prompt模板包含公司名称保护指令:")
            for protection in found_protections:
                print(f"   - {protection}")
        else:
            print(f"❌ Prompt模板缺少公司名称保护指令")
        
        print(f"📏 Prompt长度: {len(prompt)} 字符")
    else:
        print(f"❌ Prompt模板格式化失败")

if __name__ == "__main__":
    # 测试prompt模板
    test_prompt_template()
    
    # 测试公司名称一致性
    test_company_name_consistency() 