#!/usr/bin/env python3
"""
测试Fin-R1配置是否正确
"""

import sys
from pathlib import Path

# Add parent directory to path
sys.path.append(str(Path(__file__).parent))

from config.parameters import Config
from xlm.components.generator.local_llm_generator import LocalLLMGenerator

def test_finr1_config():
    print("🔧 测试Fin-R1配置...")
    
    # 加载配置
    config = Config()
    
    print(f"📋 模型名称: {config.generator.model_name}")
    print(f"📋 设备: {config.generator.device}")
    print(f"📋 量化: {config.generator.use_quantization} ({config.generator.quantization_type})")
    print(f"📋 max_new_tokens: {config.generator.max_new_tokens}")
    print(f"📋 do_sample: {config.generator.do_sample}")
    print(f"📋 repetition_penalty: {config.generator.repetition_penalty}")
    print(f"📋 eos_token_id: {config.generator.eos_token_id}")
    
    # 测试生成器初始化
    try:
        print("\n🔧 初始化生成器...")
        generator = LocalLLMGenerator()
        
        print(f"✅ 生成器初始化成功")
        print(f"📋 模型设备: {next(generator.model.parameters()).device}")
        print(f"📋 模型名称: {generator.model_name}")
        
        # 测试简单生成
        test_prompt = "你好，请简单介绍一下自己。"
        print(f"\n🧪 测试生成: {test_prompt}")
        
        responses = generator.generate([test_prompt])
        print(f"✅ 生成成功: {responses[0][:100]}...")
        
    except Exception as e:
        print(f"❌ 错误: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    test_finr1_config() 