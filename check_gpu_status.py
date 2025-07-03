#!/usr/bin/env python3
"""
检查GPU状态和配置
"""

import torch
import os
from config.parameters import config

def check_gpu_status():
    """检查GPU状态"""
    print("🔍 检查GPU状态...")
    print("="*50)
    
    # 检查CUDA是否可用
    print(f"CUDA available: {torch.cuda.is_available()}")
    
    if torch.cuda.is_available():
        # 检查GPU数量
        gpu_count = torch.cuda.device_count()
        print(f"GPU count: {gpu_count}")
        
        # 检查每个GPU的信息
        for i in range(gpu_count):
            gpu_name = torch.cuda.get_device_name(i)
            gpu_memory = torch.cuda.get_device_properties(i).total_memory / 1024**3
            print(f"GPU {i}: {gpu_name}, Memory: {gpu_memory:.1f} GB")
        
        # 检查当前设备
        current_device = torch.cuda.current_device()
        print(f"Current CUDA device: {current_device}")
        
        # 检查默认设备
        default_device = torch.cuda.get_device_name(0)
        print(f"Default CUDA device: {default_device}")
    else:
        print("❌ CUDA不可用")
    
    print("\n🔧 检查配置...")
    print("="*50)
    
    # 检查生成器配置
    generator_config = config.generator
    print(f"Generator device: {generator_config.device}")
    print(f"Generator use_quantization: {generator_config.use_quantization}")
    print(f"Generator quantization_type: {generator_config.quantization_type}")
    print(f"Generator max_new_tokens: {generator_config.max_new_tokens}")
    print(f"Generator max_generation_time: {generator_config.max_generation_time}")
    
    # 检查编码器配置
    encoder_config = config.encoder
    print(f"Encoder device: {encoder_config.device}")
    
    # 检查重排序器配置
    reranker_config = config.reranker
    print(f"Reranker device: {reranker_config.device}")
    
    print("\n🚀 测试GPU访问...")
    print("="*50)
    
    if torch.cuda.is_available():
        try:
            # 测试GPU 0
            device_0 = torch.device("cuda:0")
            test_tensor_0 = torch.randn(100, 100).to(device_0)
            print(f"✅ GPU 0 访问成功: {test_tensor_0.device}")
            
            # 测试GPU 1 (如果存在)
            if torch.cuda.device_count() > 1:
                device_1 = torch.device("cuda:1")
                test_tensor_1 = torch.randn(100, 100).to(device_1)
                print(f"✅ GPU 1 访问成功: {test_tensor_1.device}")
            else:
                print("⚠️ 只有一个GPU，无法测试GPU 1")
                
        except Exception as e:
            print(f"❌ GPU访问失败: {e}")
    else:
        print("❌ 无法测试GPU访问，CUDA不可用")

if __name__ == "__main__":
    check_gpu_status() 