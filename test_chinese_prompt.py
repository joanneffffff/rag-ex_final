#!/usr/bin/env python3
"""
测试中文查询的prompt模板
"""

from xlm.components.prompt_templates.template_loader import template_loader

def test_chinese_prompt():
    """测试中文prompt模板"""
    print("=== 中文查询Prompt模板测试 ===\n")
    
    # 1. 显示原始模板
    print("1. 原始模板内容:")
    template = template_loader.get_template("multi_stage_chinese_template")
    print(template)
    print("\n" + "="*50 + "\n")
    
    # 2. 模拟实际使用场景
    print("2. 实际使用示例:")
    
    # 模拟上下文和查询
    context = """
    德赛电池（000049）的业绩预告超出预期，主要得益于iPhone 12 Pro Max需求强劲和新品盈利能力提升。
    预计2021年利润将持续增长，源于A客户的业务成长、非手机业务的增长以及并表比例的增加。
    尽管安卓业务受到H客户销量下滑的影响，但新荣耀新品的发布预计将缓解这一影响。
    """
    
    query = "德赛电池（000049）2021年利润持续增长的主要原因是什么？"
    summary = context[:200] + "..." if len(context) > 200 else context
    # 格式化模板
    formatted_prompt = template_loader.format_template(
        "multi_stage_chinese_template",
        summary=summary,
        context=context,
        query=query
    )
    
    print("格式化后的完整prompt:")
    print(formatted_prompt)
    print("\n" + "="*50 + "\n")
    
    # 3. 分析模板特点
    print("3. 模板特点分析:")
    print("- 模板类型: 多阶段检索中文模板")
    print("- 文件位置: data/prompt_templates/multi_stage_chinese_template.txt")
    print("- 参数: context, query")
    print("- 特点:")
    print("  * 强调直接、纯粹的回答")
    print("  * 禁止自我反思和思考过程")
    print("  * 禁止格式标记和元评论")
    print("  * 要求简洁结束，不带引导语")
    print("  * 结构清晰：上下文 -> 问题 -> 回答")

if __name__ == "__main__":
    test_chinese_prompt() 