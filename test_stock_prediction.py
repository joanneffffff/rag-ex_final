#!/usr/bin/env python3
"""
测试股票预测功能
"""

from xlm.ui.optimized_rag_ui import OptimizedRagUI

def test_stock_prediction():
    """测试股票预测功能"""
    print("开始测试股票预测功能...")
    
    # 初始化UI
    ui = OptimizedRagUI()
    
    # 测试问题
    test_questions = [
        "德赛电池(000049)",
        "用友网络",
        "下月股价能否上涨?",
        "这个股票怎么样？"
    ]
    
    print("\n=== 测试股票预测功能 ===")
    for i, question in enumerate(test_questions, 1):
        print(f"\n测试 {i}: {question}")
        
        # 测试普通模式
        print("普通模式:")
        try:
            answer, context = ui._process_question(question, "Both", True, False)
            print(f"答案: {answer[:200]}...")
        except Exception as e:
            print(f"普通模式失败: {e}")
        
        # 测试股票预测模式
        print("股票预测模式:")
        try:
            answer, context = ui._process_question(question, "Both", True, True)
            print(f"答案: {answer[:200]}...")
        except Exception as e:
            print(f"股票预测模式失败: {e}")
        
        print("-" * 50)

if __name__ == "__main__":
    test_stock_prediction() 