#!/usr/bin/env python3
"""
快速调试LocalLLMGenerator的设备设置
"""

from xlm.components.generator.local_llm_generator import LocalLLMGenerator

def quick_debug():
    print("🔍 快速调试设备设置...")
    
    # 创建LocalLLMGenerator实例
    generator = LocalLLMGenerator(
        model_name="SUFE-AIFLM-Lab/Fin-R1",
        device="cuda"  # 明确指定cuda
    )
    
    print(f"📋 设备设置:")
    print(f"  - self.device: {generator.device}")
    print(f"  - self.use_quantization: {generator.use_quantization}")
    print(f"  - self.quantization_type: {generator.quantization_type}")
    
    # 检查条件
    condition = generator.use_quantization and generator.device and generator.device.startswith('cuda')
    print(f"  - 量化条件: {condition}")
    
    # 检查模型实际设备
    model_device = next(generator.model.parameters()).device
    print(f"  - 模型实际设备: {model_device}")

if __name__ == "__main__":
    quick_debug() 