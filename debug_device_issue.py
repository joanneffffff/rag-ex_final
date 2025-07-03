#!/usr/bin/env python3
"""
调试为什么LocalLLMGenerator没有使用GPU
"""

import torch
from config.parameters import config

def debug_device_issue():
    """调试设备问题"""
    print("🔍 调试设备问题...")
    print("="*60)
    
    # 检查CUDA状态
    print(f"CUDA available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"GPU count: {torch.cuda.device_count()}")
        for i in range(torch.cuda.device_count()):
            print(f"GPU {i}: {torch.cuda.get_device_name(i)}")
    
    print("\n📋 配置文件中的设备设置:")
    print(f"Generator device: {config.generator.device}")
    print(f"Generator use_quantization: {config.generator.use_quantization}")
    print(f"Generator quantization_type: {config.generator.quantization_type}")
    
    print("\n🔧 测试LocalLLMGenerator初始化...")
    print("="*60)
    
    try:
        from xlm.components.generator.local_llm_generator import LocalLLMGenerator
        
        # 测试不同的设备设置
        test_configs = [
            ("auto", "自动检测"),
            ("cuda", "CUDA默认"),
            ("cuda:0", "CUDA:0"),
            ("cuda:1", "CUDA:1"),
            ("cpu", "CPU")
        ]
        
        for device, description in test_configs:
            print(f"\n🧪 测试设备: {device} ({description})")
            try:
                # 创建LocalLLMGenerator实例
                generator = LocalLLMGenerator(
                    model_name="SUFE-AIFLM-Lab/Fin-R1",
                    device=device
                )
                
                # 检查模型实际使用的设备
                model_device = next(generator.model.parameters()).device
                print(f"  ✅ 初始化成功")
                print(f"  📍 模型实际设备: {model_device}")
                print(f"  📍 配置的设备: {generator.device}")
                
                # 如果成功使用GPU，就使用这个配置
                if model_device.type == 'cuda':
                    print(f"  🎉 成功使用GPU: {model_device}")
                    break
                    
            except Exception as e:
                print(f"  ❌ 初始化失败: {str(e)[:100]}...")
                
    except ImportError as e:
        print(f"❌ 无法导入LocalLLMGenerator: {e}")
    
    print("\n💡 建议:")
    print("="*60)
    print("1. 如果GPU可用但模型仍使用CPU，可能是内存不足")
    print("2. 尝试启用量化: use_quantization=True, quantization_type='4bit'")
    print("3. 检查是否有其他进程占用GPU内存")
    print("4. 尝试使用不同的GPU设备: cuda:0, cuda:1等")

if __name__ == "__main__":
    debug_device_issue() 