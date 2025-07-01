#!/usr/bin/env python3
"""
快速测试脚本 - 直接获取生成器回答
"""

import sys
import os
sys.path.append('.')

def quick_test():
    print("🚀 快速测试开始...")
    
    try:
        # 导入配置
        from config.parameters import Config
        config = Config()
        
        print(f"📋 模型: {config.generator.model_name}")
        print(f"📋 设备: {config.generator.device}")
        print(f"📋 Token数: {config.generator.max_new_tokens}")
        
        # 导入生成器
        from xlm.components.generator.local_llm_generator import LocalLLMGenerator
        
        print("🔧 加载生成器...")
        generator = LocalLLMGenerator()
        
        # 简单测试
        test_prompt = "请用一句话回答：德赛电池的主要业务是什么？"
        print(f"\n📝 测试问题: {test_prompt}")
        
        print("🤖 生成中...")
        responses = generator.generate([test_prompt])
        
        print(f"\n✅ 回答:")
        print(f"{'='*50}")
        print(responses[0])
        print(f"{'='*50}")
        
    except Exception as e:
        print(f"❌ 错误: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    quick_test() 