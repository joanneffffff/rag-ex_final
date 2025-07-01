#!/usr/bin/env python3
"""
8bit 量化内存问题分析
详细分析为什么 8bit 量化在 CUDA:1 上无法使用
"""

import torch
import gc
import os
import sys
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig

def analyze_gpu_memory_status():
    """分析 GPU 内存状态"""
    print("🔍 GPU 内存状态分析")
    print("=" * 50)
    
    if torch.cuda.is_available():
        for device_id in [0, 1]:
            gpu_memory = torch.cuda.get_device_properties(device_id).total_memory
            allocated_memory = torch.cuda.memory_allocated(device_id)
            reserved_memory = torch.cuda.memory_reserved(device_id)
            free_memory = gpu_memory - allocated_memory
            
            print(f"📊 GPU {device_id}:")
            print(f"   - 总内存: {gpu_memory / 1024**3:.1f}GB")
            print(f"   - 已分配: {allocated_memory / 1024**3:.1f}GB")
            print(f"   - 已保留: {reserved_memory / 1024**3:.1f}GB")
            print(f"   - 可用内存: {free_memory / 1024**3:.1f}GB")
            print(f"   - 利用率: {(allocated_memory / gpu_memory) * 100:.1f}%")
            print()

def test_memory_allocation_sizes():
    """测试不同大小的内存分配"""
    print("🔍 内存分配测试")
    print("=" * 50)
    
    if torch.cuda.is_available():
        # 测试不同大小的内存块
        sizes_mb = [100, 500, 1000, 2000, 3000, 4000, 5000, 6000, 7000, 8000]
        
        for size_mb in sizes_mb:
            try:
                size_bytes = size_mb * 1024 * 1024
                # 尝试分配 float16 张量（模拟模型权重）
                tensor = torch.empty(size_bytes // 2, dtype=torch.float16, device='cuda:1')
                print(f"✅ 成功分配 {size_mb}MB 内存块")
                del tensor
                torch.cuda.empty_cache()
            except RuntimeError as e:
                print(f"❌ 无法分配 {size_mb}MB 内存块: {e}")
                break

def estimate_model_memory_requirements():
    """估算模型内存需求"""
    print("🔍 模型内存需求估算")
    print("=" * 50)
    
    # Fin-R1 模型参数估算
    model_name = "SUFE-AIFLM-Lab/Fin-R1"
    
    # 常见的模型大小估算
    model_sizes = {
        "7B": 7 * 1024**3,  # 7B 参数
        "13B": 13 * 1024**3,  # 13B 参数
        "30B": 30 * 1024**3,  # 30B 参数
    }
    
    print("📊 不同精度下的内存需求估算:")
    
    for model_size_name, param_count in model_sizes.items():
        print(f"\n🔧 {model_size_name} 模型:")
        
        # FP32 (32位浮点)
        fp32_memory = param_count * 4  # 4 bytes per parameter
        print(f"   - FP32: {fp32_memory / 1024**3:.1f}GB")
        
        # FP16 (16位浮点)
        fp16_memory = param_count * 2  # 2 bytes per parameter
        print(f"   - FP16: {fp16_memory / 1024**3:.1f}GB")
        
        # INT8 (8位整数)
        int8_memory = param_count * 1  # 1 byte per parameter
        print(f"   - INT8: {int8_memory / 1024**3:.1f}GB")
        
        # INT4 (4位整数)
        int4_memory = param_count * 0.5  # 0.5 bytes per parameter
        print(f"   - INT4: {int4_memory / 1024**3:.1f}GB")
        
        # 额外开销（激活值、梯度等）
        overhead_ratio = 0.3  # 30% 额外开销
        total_8bit = int8_memory * (1 + overhead_ratio)
        total_4bit = int4_memory * (1 + overhead_ratio)
        
        print(f"   - INT8 (含开销): {total_8bit / 1024**3:.1f}GB")
        print(f"   - INT4 (含开销): {total_4bit / 1024**3:.1f}GB")

def test_8bit_loading_attempt():
    """尝试加载 8bit 模型并分析失败原因"""
    print("🔍 8bit 模型加载测试")
    print("=" * 50)
    
    model_name = "SUFE-AIFLM-Lab/Fin-R1"
    cache_dir = "/users/sgjfei3/data/huggingface"
    
    try:
        print("🔧 步骤 1: 加载 tokenizer...")
        tokenizer = AutoTokenizer.from_pretrained(
            model_name,
            cache_dir=cache_dir,
            local_files_only=True
        )
        
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
        
        print("✅ Tokenizer 加载成功")
        
        print("\n🔧 步骤 2: 配置 8bit 量化...")
        quantization_config = BitsAndBytesConfig(
            load_in_8bit=True,
            llm_int8_threshold=6.0,
            llm_int8_has_fp16_weight=False,
        )
        
        print("✅ 8bit 量化配置完成")
        
        print("\n🔧 步骤 3: 尝试加载 8bit 模型...")
        print("📊 加载前 GPU 1 内存状态:")
        gpu_memory = torch.cuda.get_device_properties(1).total_memory
        allocated_memory = torch.cuda.memory_allocated(1)
        free_memory = gpu_memory - allocated_memory
        print(f"   - 可用内存: {free_memory / 1024**3:.1f}GB")
        
        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            cache_dir=cache_dir,
            quantization_config=quantization_config,
            device_map="auto",
            torch_dtype=torch.float16,
            low_cpu_mem_usage=True,
            use_cache=False,
            local_files_only=True
        )
        
        print("✅ 8bit 模型加载成功")
        
        # 检查加载后的内存使用
        allocated_memory_after = torch.cuda.memory_allocated(1)
        memory_used = allocated_memory_after - allocated_memory
        print(f"📊 模型占用内存: {memory_used / 1024**3:.1f}GB")
        
        return True
        
    except Exception as e:
        print(f"❌ 8bit 加载失败: {e}")
        
        # 分析错误信息
        error_str = str(e)
        if "out of memory" in error_str.lower():
            print("\n🔍 OOM 错误分析:")
            print("   - 错误类型: CUDA 内存不足")
            print("   - 可能原因:")
            print("     * 8bit 量化仍需要大量内存")
            print("     * 模型大小超出可用内存")
            print("     * 内存碎片化导致无法分配连续大块")
        
        elif "cuda" in error_str.lower():
            print("\n🔍 CUDA 错误分析:")
            print("   - 错误类型: CUDA 相关错误")
            print("   - 可能原因:")
            print("     * CUDA 版本兼容性问题")
            print("     * 驱动程序问题")
            print("     * 硬件限制")
        
        return False

def analyze_memory_fragmentation():
    """分析内存碎片化问题"""
    print("🔍 内存碎片化分析")
    print("=" * 50)
    
    if torch.cuda.is_available():
        # 模拟内存碎片化情况
        print("🔧 模拟内存分配模式...")
        
        # 分配一些小的内存块
        small_tensors = []
        for i in range(10):
            try:
                tensor = torch.empty(100 * 1024 * 1024, dtype=torch.float16, device='cuda:1')  # 100MB
                small_tensors.append(tensor)
                print(f"✅ 分配小块 {i+1}: 100MB")
            except:
                break
        
        # 尝试分配大块内存
        try:
            large_tensor = torch.empty(8 * 1024 * 1024 * 1024 // 2, dtype=torch.float16, device='cuda:1')  # 8GB
            print("✅ 成功分配 8GB 大块内存")
            del large_tensor
        except RuntimeError as e:
            print(f"❌ 无法分配 8GB 大块内存: {e}")
            print("💡 这表明存在内存碎片化问题")
        
        # 清理小内存块
        for tensor in small_tensors:
            del tensor
        torch.cuda.empty_cache()
        
        # 再次尝试分配大块内存
        try:
            large_tensor = torch.empty(8 * 1024 * 1024 * 1024 // 2, dtype=torch.float16, device='cuda:1')  # 8GB
            print("✅ 清理后成功分配 8GB 大块内存")
            del large_tensor
        except RuntimeError as e:
            print(f"❌ 清理后仍无法分配 8GB 大块内存: {e}")

def main():
    """主函数"""
    print("🎯 8bit 量化内存问题分析")
    print("=" * 60)
    
    # 1. 分析当前 GPU 内存状态
    analyze_gpu_memory_status()
    
    # 2. 估算模型内存需求
    estimate_model_memory_requirements()
    
    # 3. 测试内存分配
    test_memory_allocation_sizes()
    
    # 4. 分析内存碎片化
    analyze_memory_fragmentation()
    
    # 5. 尝试加载 8bit 模型
    print("\n" + "=" * 60)
    success = test_8bit_loading_attempt()
    
    # 6. 总结分析
    print("\n" + "=" * 60)
    print("📊 分析总结")
    print("=" * 60)
    
    if success:
        print("✅ 8bit 量化可以正常工作")
    else:
        print("❌ 8bit 量化无法在 CUDA:1 上使用")
        print("\n🔍 主要原因:")
        print("1. **内存不足**: 8bit 量化仍需要大量内存")
        print("2. **其他进程占用**: PID 587879 正在使用 17GB+ 内存")
        print("3. **内存碎片化**: 无法分配连续的大块内存")
        print("4. **模型大小**: Fin-R1 模型较大，8bit 量化后仍超出可用内存")
        
        print("\n💡 解决方案:")
        print("1. **使用 4bit 量化**: 内存需求减半")
        print("2. **等待其他进程结束**: 释放 GPU 内存")
        print("3. **使用 CPU 回退**: 避免 GPU 内存限制")
        print("4. **优化内存分配**: 使用更激进的内存优化策略")

if __name__ == "__main__":
    main() 