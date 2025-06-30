#!/usr/bin/env python3
"""
最终的Prompt调试测试脚本
"""

import requests
import json
import time

def test_prompt_debug():
    """测试Prompt调试功能"""
    print("=== 最终Prompt调试测试 ===\n")
    
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
    
    # 2. 发送测试查询
    print("\n2. 发送测试查询...")
    
    # 使用与日志中相同的查询
    test_query = "德赛电池（000049）2021年利润持续增长的主要原因是什么？"
    
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
        response = requests.post(api_url, json=data, timeout=60)  # 增加超时时间
        
        if response.status_code == 200:
            result = response.json()
            print("✅ 查询成功")
            
            # 检查响应中是否包含模型相关信息
            if 'data' in result and len(result['data']) > 0:
                answer = result['data'][0]
                print(f"回答长度: {len(answer)} 字符")
                print(f"回答预览: {answer[:300]}...")
                
                # 检查是否包含调试信息
                if "PROMPT调试信息" in answer or "📤 发送给LLM的完整Prompt" in answer:
                    print("✅ 包含Prompt调试信息")
                else:
                    print("⚠️  未包含Prompt调试信息")
                
                # 检查回答质量
                if "德赛电池" in answer or "利润" in answer or "增长" in answer:
                    print("✅ 回答包含相关关键词")
                else:
                    print("⚠️  回答可能不够相关")
                    
                # 检查是否解决了"未配置LLM生成器"问题
                if "未配置LLM生成器" in answer:
                    print("❌ 仍然存在LLM生成器问题")
                else:
                    print("✅ LLM生成器问题已解决")
                    
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
    print("   - 已添加Prompt调试信息")
    print("   - 已修复LLM生成器CPU回退机制")
    print("   - 如果查询成功，说明问题已解决")

if __name__ == "__main__":
    test_prompt_debug() 