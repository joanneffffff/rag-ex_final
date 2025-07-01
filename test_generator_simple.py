#!/usr/bin/env python3
"""
简单的生成器测试脚本
直接测试生成器并显示回答内容
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from config.parameters import Config
from xlm.components.generator.local_llm_generator import LocalLLMGenerator

def test_generator():
    """测试生成器并显示回答"""
    
    print("🚀 开始测试生成器...")
    
    # 创建配置
    config = Config()
    print(f"📋 当前配置:")
    print(f"   - 模型: {config.generator.model_name}")
    print(f"   - 设备: {config.generator.device}")
    print(f"   - max_new_tokens: {config.generator.max_new_tokens}")
    print(f"   - 量化: {config.generator.use_quantization} ({config.generator.quantization_type})")
    
    try:
        # 创建生成器
        print("\n🔧 正在加载生成器...")
        generator = LocalLLMGenerator()
        
        # 测试问题
        test_questions = [
            "请简要分析德赛电池2021年的财务状况。",
            "德赛电池的主要业务是什么？"
        ]
        
        print(f"\n📝 测试问题:")
        for i, question in enumerate(test_questions, 1):
            print(f"   {i}. {question}")
        
        print(f"\n🤖 开始生成回答...")
        
        # 生成回答
        responses = generator.generate(test_questions)
        
        print(f"\n✅ 生成完成！")
        print(f"=" * 80)
        
        # 显示回答
        for i, (question, response) in enumerate(zip(test_questions, responses), 1):
            print(f"\n📋 问题 {i}: {question}")
            print(f"🤖 回答 {i}:")
            print(f"{'='*40}")
            print(response)
            print(f"{'='*40}")
            print(f"📏 回答长度: {len(response)} 字符")
            print(f"📊 Token数: 约 {len(response.split())} 个词")
        
        print(f"\n🎯 测试完成！")
        
    except Exception as e:
        print(f"❌ 测试失败: {str(e)}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    test_generator() 