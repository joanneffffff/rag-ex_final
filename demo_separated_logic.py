#!/usr/bin/env python3
"""
演示分离的检索和生成逻辑
"""

def demo_separated_logic():
    """演示分离的检索和生成逻辑"""
    print("=== 分离的检索和生成逻辑演示 ===")
    
    # 模拟用户输入
    original_query = "德赛电池(000049)"
    
    print(f"用户原始查询: {original_query}")
    print()
    
    # 模拟股票预测instruction
    instruction = f"请根据下方提供的该股票相关研报与数据，对该股票的下个月的涨跌，进行预测，请给出明确的答案，\"涨\" 或者 \"跌\"。同时给出这个股票下月的涨跌概率，分别是:极大，较大，中上，一般。\n\n问题：{original_query}"
    
    print("生成的股票预测instruction:")
    print(instruction)
    print()
    
    print("处理逻辑:")
    print("1. 检索阶段: 使用原始查询 '德赛电池(000049)' 进行文档检索")
    print("   - 关键词提取: 公司名称='德赛电池', 股票代码='000049'")
    print("   - 元数据过滤: 根据公司名称和股票代码过滤相关文档")
    print("   - FAISS检索: 在过滤后的文档中进行向量相似度检索")
    print("   - 重排序: 使用原始查询对检索结果进行重排序")
    print()
    print("2. 生成阶段: 使用instruction进行答案生成")
    print("   - 将检索到的文档作为上下文")
    print("   - 使用专业的股票预测instruction指导生成")
    print("   - 生成包含涨跌预测和概率的详细分析")
    print()
    
    print("优势:")
    print("- 检索更精准: 使用简洁的原始查询进行检索，避免instruction中的复杂描述干扰检索")
    print("- 生成更专业: 使用专业的instruction指导生成，确保输出格式和内容质量")
    print("- 逻辑清晰: 检索和生成职责分离，便于调试和优化")

if __name__ == "__main__":
    demo_separated_logic() 