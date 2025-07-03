#!/usr/bin/env python3
"""
测试GPU配置脚本
验证模型是否正确使用GPU
"""

import torch
import sys
import os

# 添加项目根目录到Python路径
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from xlm.components.generator.local_llm_generator import LocalLLMGenerator
from config.parameters import Config

def test_gpu_configuration():
    """测试GPU配置"""
    print("=" * 60)
    print("🔧 GPU配置测试")
    print("=" * 60)
    
    # 1. 检查CUDA可用性
    print("1. 检查CUDA可用性:")
    print(f"   - CUDA可用: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"   - GPU数量: {torch.cuda.device_count()}")
        for i in range(torch.cuda.device_count()):
            gpu_name = torch.cuda.get_device_name(i)
            gpu_memory = torch.cuda.get_device_properties(i).total_memory / 1024**3
            print(f"   - GPU {i}: {gpu_name} ({gpu_memory:.1f} GB)")
    
    # 2. 检查配置
    print("\n2. 检查配置:")
    config = Config()
    print(f"   - 生成器设备: {config.generator.device}")
    print(f"   - 使用量化: {config.generator.use_quantization}")
    print(f"   - 量化类型: {config.generator.quantization_type}")
    print(f"   - 模型名称: {config.generator.model_name}")
    
    # 3. 测试模型加载
    print("\n3. 测试模型加载:")
    try:
        print("   - 正在加载模型...")
        generator = LocalLLMGenerator(device="cuda")
        
        print(f"   - 模型设备: {generator.device}")
        print(f"   - 模型量化: {generator.use_quantization}")
        print(f"   - 量化类型: {generator.quantization_type}")
        
        # 检查模型是否在GPU上
        model_device = next(generator.model.parameters()).device
        print(f"   - 模型实际设备: {model_device}")
        
        if model_device.type == 'cuda':
            print("   ✅ 模型成功加载到GPU")
        else:
            print("   ❌ 模型未加载到GPU")
            
    except Exception as e:
        print(f"   ❌ 模型加载失败: {e}")
        return False
    
    # 4. 测试生成
    print("\n4. 测试生成:")
    try:
        test_prompt = "请简单介绍一下人工智能。"
        print(f"   - 测试提示: {test_prompt}")
        
        response = generator.generate([test_prompt])
        print(f"   - 生成响应: {response[0][:100]}...")
        print("   ✅ 生成测试成功")
        
    except Exception as e:
        print(f"   ❌ 生成测试失败: {e}")
        return False
    
    print("\n" + "=" * 60)
    print("🎉 GPU配置测试完成")
    print("=" * 60)
    return True

if __name__ == "__main__":
    success = test_gpu_configuration()
    sys.exit(0 if success else 1) 