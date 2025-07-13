#!/usr/bin/env python3
"""
测试股票预测模式的日志输出
"""

def test_stock_prediction_logs():
    """测试股票预测模式的日志输出"""
    print("=== 股票预测模式日志测试 ===")
    
    # 模拟用户输入
    test_queries = [
        "德赛电池(000049)",
        "用友网络",
        "下月股价能否上涨?",
        "这个股票怎么样？"
    ]
    
    print("\n模拟股票预测instruction生成过程:")
    print("=" * 60)
    
    for i, query in enumerate(test_queries, 1):
        print(f"\n📝 测试 {i}: 原始查询: '{query}'")
        
        # 模拟instruction生成
        instruction = f"请根据下方提供的该股票相关研报与数据，对该股票的下个月的涨跌，进行预测，请给出明确的答案，\"涨\" 或者 \"跌\"。同时给出这个股票下月的涨跌概率，分别是:极大，较大，中上，一般。\n\n问题：{query}"
        
        print(f"📋 生成的instruction:")
        print(f"   {instruction}")
        print(f"🔄 查询转换完成:")
        print(f"   - 检索使用: '{query}'")
        print(f"   - 生成使用: '{instruction[:100]}{'...' if len(instruction) > 100 else ''}'")
        print("-" * 60)
    
    print("\n🎯 日志输出说明:")
    print("1. 🔍 [股票预测模式] - 标识进入股票预测处理流程")
    print("2. 📝 [股票预测模式] - 显示原始用户查询")
    print("3. 📋 [股票预测模式] - 显示生成的完整instruction")
    print("4. 🔄 [股票预测模式] - 显示查询转换结果")
    print("5. 🚀 [分离模式] - 标识进入分离的检索和生成流程")
    print("6. 📊 [分离模式] - 显示处理策略说明")

if __name__ == "__main__":
    test_stock_prediction_logs() 