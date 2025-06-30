#!/usr/bin/env python3
"""
简单LLM测试脚本
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

def test_simple_llm():
    """简单测试LLM生成器"""
    print("开始简单LLM测试...")
    
    try:
        from xlm.components.generator.local_llm_generator import LocalLLMGenerator
        
        # 初始化生成器
        print("初始化LLM生成器...")
        generator = LocalLLMGenerator()
        print(f"生成器初始化成功: {generator.model_name}")
        
        # 简单测试
        test_prompt = "你好，请简单介绍一下自己。"
        print(f"测试Prompt: {test_prompt}")
        
        responses = generator.generate([test_prompt])
        response = responses[0] if responses else "无响应"
        
        print(f"生成结果: {response}")
        return True
        
    except Exception as e:
        print(f"测试失败: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    test_simple_llm() 