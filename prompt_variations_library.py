#!/usr/bin/env python3
"""
Prompt 变体库
包含多种不同的 Prompt 设计，用于测试和优化 Generator LLM 响应
"""

def get_prompt_variations(context, summary, query):
    """获取所有 Prompt 变体"""
    
    variations = {
        "极简版": f"""基于以下信息回答问题：

{summary}

{context}

问题：{query}
答案：""",
        
        "标准版": f"""你是一位金融分析师。请基于以下信息回答问题：

摘要：{summary}

详细内容：{context}

问题：{query}

回答：""",
        
        "指令版": f"""你是一位金融分析师。请严格按照以下要求回答：

要求：
1. 基于提供的财务信息回答
2. 回答简洁，控制在2-3句话内
3. 如果信息不足，回答"根据现有信息，无法提供此项信息。"
4. 不要包含任何格式标记或额外说明

信息：
{summary}

{context}

问题：{query}

回答：""",
        
        "分析版": f"""作为金融分析师，请分析以下财务数据并回答问题：

财务摘要：{summary}

详细数据：{context}

分析问题：{query}

分析结果：""",
        
        "专业版": f"""你是一位专业的金融分析师，擅长分析公司财务报告。

请基于以下公司财务报告信息，准确回答用户问题：

【财务报告摘要】
{summary}

【详细财务数据】
{context}

【用户问题】
{query}

请提供准确、简洁的分析回答：""",
        
        "问答版": f"""基于以下财务信息回答问题：

{summary}

{context}

问题：{query}

答案：""",
        
        "总结版": f"""请基于以下信息总结回答：

{summary}

{context}

问题：{query}

总结：""",
        
        "直接版": f"""{summary}

{context}

问题：{query}

回答：""",
        
        "结构化版": f"""你是一位金融分析师。

背景信息：
{summary}

详细数据：
{context}

用户问题：{query}

专业分析：""",
        
        "简洁指令版": f"""你是金融分析师。基于以下信息简洁回答：

{summary}

{context}

问题：{query}

回答：""",
        
        "无角色版": f"""基于以下财务信息回答问题：

{summary}

{context}

问题：{query}

答案：""",
        
        "中文指令版": f"""你是一位金融分析师。请用中文回答以下问题：

{summary}

{context}

问题：{query}

回答：""",
        
        "英文指令版": f"""You are a financial analyst. Please answer the following question based on the provided information:

{summary}

{context}

Question: {query}

Answer:""",
        
        "混合版": f"""你是一位金融分析师。请基于以下信息回答问题：

财务摘要：{summary}

详细内容：{context}

问题：{query}

回答：""",
        
        "强调版": f"""你是一位专业的金融分析师。

请基于以下信息准确回答用户问题：

{summary}

{context}

问题：{query}

请提供准确回答：""",
        
        "引导版": f"""作为金融分析师，请根据以下信息分析并回答问题：

{summary}

{context}

问题：{query}

分析回答：""",
        
        "模板版": f"""你是一位金融分析师。

信息摘要：{summary}

详细信息：{context}

用户问题：{query}

分析回答：""",
        
        "对话版": f"""你是一位金融分析师，正在与用户进行对话。

用户提供了以下信息：
{summary}

{context}

用户问：{query}

你的回答：""",
        
        "任务版": f"""任务：基于提供的财务信息回答用户问题

角色：金融分析师

信息：
{summary}

{context}

问题：{query}

回答：""",
        
        "格式版": f"""你是一位金融分析师。

【信息摘要】
{summary}

【详细信息】
{context}

【用户问题】
{query}

【分析回答】
""",
        
        "纯文本版": f"""{summary}

{context}

问题：{query}

答案："""
    }
    
    return variations

def get_parameter_variations():
    """获取不同的参数组合"""
    
    parameters = {
        "超保守": {
            "temperature": 0.05,
            "top_p": 0.6,
            "max_new_tokens": 150,
            "description": "最保守的设置，生成最稳定的回答"
        },
        "保守": {
            "temperature": 0.1,
            "top_p": 0.7,
            "max_new_tokens": 200,
            "description": "保守设置，平衡稳定性和创造性"
        },
        "平衡": {
            "temperature": 0.2,
            "top_p": 0.8,
            "max_new_tokens": 300,
            "description": "平衡设置，当前默认配置"
        },
        "创造性": {
            "temperature": 0.3,
            "top_p": 0.9,
            "max_new_tokens": 400,
            "description": "创造性设置，允许更多变化"
        },
        "高创造性": {
            "temperature": 0.4,
            "top_p": 0.95,
            "max_new_tokens": 500,
            "description": "高创造性设置，最大变化"
        },
        "精确": {
            "temperature": 0.05,
            "top_p": 0.5,
            "max_new_tokens": 100,
            "description": "精确设置，最短最准确的回答"
        },
        "详细": {
            "temperature": 0.15,
            "top_p": 0.85,
            "max_new_tokens": 600,
            "description": "详细设置，允许更长的回答"
        }
    }
    
    return parameters

def get_test_scenarios():
    """获取测试场景"""
    
    scenarios = {
        "德赛电池利润增长": {
            "context": """
            德赛电池（000049）2021年业绩预告显示，公司预计实现归属于上市公司股东的净利润为6.5亿元至7.5亿元，
            同比增长11.02%至28.23%。业绩增长的主要原因是：
            1. iPhone 12 Pro Max等高端产品需求强劲，带动公司电池业务增长
            2. 新产品盈利能力提升，毛利率改善
            3. A客户业务持续成长，非手机业务稳步增长
            4. 并表比例增加，贡献业绩增量
            """,
            "summary": "德赛电池2021年业绩增长主要受益于iPhone 12 Pro Max需求强劲和新品盈利能力提升。",
            "query": "德赛电池（000049）2021年利润持续增长的主要原因是什么？"
        },
        "用友网络现金流": {
            "context": """
            用友网络2019年年度报告显示，公司每股经营活动产生的现金流量净额为0.85元，
            较上年同期增长12.5%。主要原因是：
            1. 云服务业务收入增长，现金流入增加
            2. 成本控制优化，现金流出减少
            3. 应收账款管理改善，回款速度提升
            """,
            "summary": "用友网络2019年每股经营活动现金流量净额为0.85元，同比增长12.5%。",
            "query": "用友网络2019年的每股经营活动产生的现金流量净额是多少？"
        },
        "首钢股份业绩": {
            "context": """
            首钢股份2020年业绩报告显示，公司实现营业收入1,234.56亿元，同比下降8.5%；
            净利润为45.67亿元，同比下降15.2%。主要受疫情影响，钢铁需求下降，
            但公司通过降本增效，保持了相对稳定的盈利能力。
            """,
            "summary": "首钢股份2020年业绩受疫情影响有所下降，但通过降本增效保持稳定。",
            "query": "首钢股份的业绩表现如何？"
        }
    }
    
    return scenarios

if __name__ == "__main__":
    # 测试 Prompt 变体
    test_context = "测试上下文"
    test_summary = "测试摘要"
    test_query = "测试问题"
    
    variations = get_prompt_variations(test_context, test_summary, test_query)
    parameters = get_parameter_variations()
    scenarios = get_test_scenarios()
    
    print(f"Prompt 变体数量: {len(variations)}")
    print(f"参数组合数量: {len(parameters)}")
    print(f"测试场景数量: {len(scenarios)}")
    
    print("\nPrompt 变体列表:")
    for name in variations.keys():
        print(f"  - {name}")
    
    print("\n参数组合列表:")
    for name, params in parameters.items():
        print(f"  - {name}: {params['description']}")
    
    print("\n测试场景列表:")
    for name in scenarios.keys():
        print(f"  - {name}") 