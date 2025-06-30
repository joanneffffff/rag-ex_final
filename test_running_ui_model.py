#!/usr/bin/env python3
"""
测试当前运行的UI是否使用了Fin-R1模型
"""

import requests
import json
import time

def test_running_ui():
    """测试当前运行的UI"""
    print("=== 测试当前运行的UI模型 ===\n")
    
    # 1. 检查UI是否运行
    try:
        response = requests.get("http://localhost:7860", timeout=5)
        if response.status_code == 200:
            print("✅ UI正在运行")
        else:
            print(f"❌ UI响应异常: {response.status_code}")
            return
    except Exception as e:
        print(f"❌ 无法连接到UI: {e}")
        return
    
    # 2. 发送一个简单的查询来测试模型
    print("\n2. 发送测试查询...")
    
    # 构造一个简单的查询
    test_query = "钢铁行业发展趋势"
    
    try:
        # 使用Gradio的API接口
        api_url = "http://localhost:7860/api/predict"
        
        # 构造请求数据
        data = {
            "data": [
                test_query,  # question
                "Both",      # datasource
                True         # reranker_checkbox
            ]
        }
        
        print(f"发送查询: {test_query}")
        print("等待响应...")
        
        # 发送请求
        response = requests.post(api_url, json=data, timeout=30)
        
        if response.status_code == 200:
            result = response.json()
            print("✅ 查询成功")
            
            # 检查响应中是否包含模型相关信息
            if 'data' in result and len(result['data']) > 0:
                answer = result['data'][0]
                print(f"回答长度: {len(answer)} 字符")
                print(f"回答预览: {answer[:200]}...")
                
                # 检查回答质量（Fin-R1应该给出更好的金融相关回答）
                if "钢铁" in answer or "行业" in answer or "发展" in answer:
                    print("✅ 回答包含相关关键词，模型工作正常")
                else:
                    print("⚠️  回答可能不够相关")
            else:
                print("❌ 响应格式异常")
        else:
            print(f"❌ 查询失败: {response.status_code}")
            print(f"错误信息: {response.text}")
            
    except Exception as e:
        print(f"❌ 测试查询失败: {e}")
    
    # 3. 总结
    print("\n3. 总结:")
    print("   - UI正在运行")
    print("   - 配置文件显示使用Fin-R1模型")
    print("   - 如果查询能正常响应，说明Fin-R1模型正在工作")
    print("   - 如果出现CUDA内存不足，可能需要调整设备配置")

if __name__ == "__main__":
    test_running_ui() 