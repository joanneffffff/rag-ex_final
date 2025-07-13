#!/usr/bin/env python3
"""
股票预测功能演示
"""

def demo_stock_prediction_logic():
    """演示股票预测逻辑"""
    print("=== 股票预测功能演示 ===")
    
    # 模拟问题
    original_question = "德赛电池(000049)"
    
    print(f"原始问题: {original_question}")
    print()
    
    # 模拟股票预测instruction
    instruction = f"请根据下方提供的该股票相关研报与数据，对该股票的下个月的涨跌，进行预测，请给出明确的答案，\"涨\" 或者 \"跌\"。同时给出这个股票下月的涨跌概率，分别是:极大，较大，中上，一般。\n\n问题：{original_question}"
    
    print("生成的股票预测instruction:")
    print(instruction)
    print()
    
    print("功能说明:")
    print("1. 当用户勾选'股票预测模式 (仅中文查询)'复选框时")
    print("2. 系统会自动将用户的查询转换为股票预测instruction")
    print("3. 使用转换后的instruction进行RAG检索和答案生成")
    print("4. 返回专业的股票涨跌预测结果")
    print()
    
    print("使用场景:")
    print("- 用户输入: '德赛电池'")
    print("- 系统转换: '请根据下方提供的该股票相关研报与数据，对该股票的下个月的涨跌，进行预测...'")
    print("- 最终输出: 包含涨跌预测和概率的详细分析")

if __name__ == "__main__":
    demo_stock_prediction_logic() 