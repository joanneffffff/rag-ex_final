#!/usr/bin/env python3
"""
GPU 内存优化测试脚本
专门针对 CUDA:1 内存不足问题
"""

import torch
import gc
import os
import sys
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig

def clear_gpu_memory():
    """清理 GPU 内存"""
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        gc.collect()
        print("✅ GPU 内存清理完成")

def check_gpu_memory(device_id=1):
    """检查 GPU 内存状态"""
    if torch.cuda.is_available():
        gpu_memory = torch.cuda.get_device_properties(device_id).total_memory
        allocated_memory = torch.cuda.memory_allocated(device_id)
        reserved_memory = torch.cuda.memory_reserved(device_id)
        free_memory = gpu_memory - allocated_memory
        
        print(f"📊 GPU {device_id} 内存状态:")
        print(f"   - 总内存: {gpu_memory / 1024**3:.1f}GB")
        print(f"   - 已分配: {allocated_memory / 1024**3:.1f}GB")
        print(f"   - 已保留: {reserved_memory / 1024**3:.1f}GB")
        print(f"   - 可用内存: {free_memory / 1024**3:.1f}GB")
        
        return free_memory
    return 0

def test_aggressive_memory_optimization():
    """测试激进的内存优化策略"""
    print("🚀 开始激进内存优化测试...")
    
    # 设置环境变量
    os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'expandable_segments:True,max_split_size_mb:64'
    os.environ['TOKENIZERS_PARALLELISM'] = 'false'
    
    # 清理内存
    clear_gpu_memory()
    check_gpu_memory(1)
    
    model_name = "SUFE-AIFLM-Lab/Fin-R1"
    cache_dir = "/users/sgjfei3/data/huggingface"
    
    try:
        print("\n🔧 步骤 1: 加载 tokenizer...")
        tokenizer = AutoTokenizer.from_pretrained(
            model_name,
            cache_dir=cache_dir,
            use_fast=True,
            local_files_only=True
        )
        
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
        
        print("✅ Tokenizer 加载成功")
        clear_gpu_memory()
        
        print("\n🔧 步骤 2: 配置 4bit 量化...")
        # 使用更激进的 4bit 量化配置
        quantization_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.float16,
            bnb_4bit_use_double_quant=True,  # 启用双重量化
            llm_int8_threshold=6.0,
            llm_int8_has_fp16_weight=False,
        )
        
        print("✅ 量化配置完成")
        
        print("\n🔧 步骤 3: 加载模型 (4bit 量化)...")
        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            cache_dir=cache_dir,
            quantization_config=quantization_config,
            device_map="auto",  # 让 transformers 自动管理设备分配
            torch_dtype=torch.float16,
            low_cpu_mem_usage=True,
            use_cache=False,  # 禁用 KV 缓存以节省内存
            local_files_only=True
        )
        
        print("✅ 模型加载成功")
        check_gpu_memory(1)
        
        # 测试生成
        print("\n🔧 步骤 4: 测试生成...")
        test_prompt = "请简要介绍一下金融分析的基本方法。"
        
        inputs = tokenizer(test_prompt, return_tensors="pt", truncation=True, max_length=512)
        
        # 移动到正确的设备
        if hasattr(model, 'device'):
            device = model.device
        else:
            device = "cuda:1"
        
        inputs = {k: v.to(device) for k, v in inputs.items()}
        
        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=100,
                do_sample=False,
                temperature=0.1,
                repetition_penalty=1.1,
                pad_token_id=tokenizer.pad_token_id,
                eos_token_id=tokenizer.eos_token_id,
            )
        
        response = tokenizer.decode(outputs[0], skip_special_tokens=True)
        print(f"✅ 生成成功: {response}")
        
        return True
        
    except Exception as e:
        print(f"❌ 加载失败: {e}")
        return False

def test_cpu_fallback():
    """测试 CPU 回退方案"""
    print("\n🔄 测试 CPU 回退方案...")
    
    model_name = "SUFE-AIFLM-Lab/Fin-R1"
    cache_dir = "/users/sgjfei3/data/huggingface"
    
    try:
        print("🔧 在 CPU 上加载模型...")
        tokenizer = AutoTokenizer.from_pretrained(
            model_name,
            cache_dir=cache_dir,
            local_files_only=True
        )
        
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
        
        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            cache_dir=cache_dir,
            device_map="cpu",
            torch_dtype=torch.float32,
            low_cpu_mem_usage=True,
            local_files_only=True
        )
        
        print("✅ CPU 模型加载成功")
        
        # 测试生成
        test_prompt = "请简要介绍一下金融分析的基本方法。"
        inputs = tokenizer(test_prompt, return_tensors="pt", truncation=True, max_length=512)
        
        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=100,
                do_sample=False,
                temperature=0.1,
                repetition_penalty=1.1,
                pad_token_id=tokenizer.pad_token_id,
                eos_token_id=tokenizer.eos_token_id,
            )
        
        response = tokenizer.decode(outputs[0], skip_special_tokens=True)
        print(f"✅ CPU 生成成功: {response}")
        
        return True
        
    except Exception as e:
        print(f"❌ CPU 加载失败: {e}")
        return False

def main():
    """主函数"""
    print("🎯 GPU 内存优化测试")
    print("=" * 50)
    
    # 检查当前 GPU 状态
    print("📊 当前 GPU 状态:")
    check_gpu_memory(1)
    
    # 测试激进内存优化
    success = test_aggressive_memory_optimization()
    
    if not success:
        print("\n⚠️ GPU 加载失败，尝试 CPU 回退...")
        test_cpu_fallback()
    
    print("\n🏁 测试完成")

if __name__ == "__main__":
    main() 