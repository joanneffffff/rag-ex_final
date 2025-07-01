#!/usr/bin/env python3
"""
GPU 内存监控脚本
实时监控 GPU 内存使用情况
"""

import torch
import time
import psutil
import os

def get_gpu_memory_info(device_id=1):
    """获取 GPU 内存信息"""
    if torch.cuda.is_available():
        gpu_memory = torch.cuda.get_device_properties(device_id).total_memory
        allocated_memory = torch.cuda.memory_allocated(device_id)
        reserved_memory = torch.cuda.memory_reserved(device_id)
        free_memory = gpu_memory - allocated_memory
        
        return {
            'total': gpu_memory / 1024**3,
            'allocated': allocated_memory / 1024**3,
            'reserved': reserved_memory / 1024**3,
            'free': free_memory / 1024**3,
            'utilization': (allocated_memory / gpu_memory) * 100
        }
    return None

def get_process_memory_info():
    """获取当前进程内存信息"""
    process = psutil.Process()
    memory_info = process.memory_info()
    
    return {
        'rss': memory_info.rss / 1024**3,  # GB
        'vms': memory_info.vms / 1024**3,  # GB
        'percent': process.memory_percent()
    }

def monitor_memory(interval=5, duration=60):
    """监控内存使用情况"""
    print(f"🔍 开始监控 GPU 内存 (间隔: {interval}秒, 持续时间: {duration}秒)")
    print("=" * 80)
    
    start_time = time.time()
    iteration = 0
    
    while time.time() - start_time < duration:
        iteration += 1
        current_time = time.strftime("%H:%M:%S")
        
        print(f"\n⏰ [{current_time}] 第 {iteration} 次检查:")
        
        # GPU 内存信息
        gpu_info = get_gpu_memory_info(1)
        if gpu_info:
            print(f"📊 GPU 1 内存:")
            print(f"   - 总内存: {gpu_info['total']:.1f}GB")
            print(f"   - 已分配: {gpu_info['allocated']:.1f}GB")
            print(f"   - 已保留: {gpu_info['reserved']:.1f}GB")
            print(f"   - 可用: {gpu_info['free']:.1f}GB")
            print(f"   - 利用率: {gpu_info['utilization']:.1f}%")
        
        # 进程内存信息
        process_info = get_process_memory_info()
        print(f"💻 进程内存:")
        print(f"   - RSS: {process_info['rss']:.1f}GB")
        print(f"   - VMS: {process_info['vms']:.1f}GB")
        print(f"   - 占用率: {process_info['percent']:.1f}%")
        
        # 检查是否有其他进程使用 GPU
        try:
            import subprocess
            result = subprocess.run(['nvidia-smi', '--query-compute-apps=pid,used_memory', '--format=csv,noheader,nounits'], 
                                  capture_output=True, text=True)
            if result.stdout.strip():
                print("🔍 其他 GPU 进程:")
                for line in result.stdout.strip().split('\n'):
                    if line.strip():
                        parts = line.split(', ')
                        if len(parts) == 2:
                            pid, memory = parts
                            print(f"   - PID {pid}: {memory} MiB")
        except:
            pass
        
        time.sleep(interval)
    
    print("\n🏁 监控完成")

def analyze_memory_fragmentation():
    """分析内存碎片化情况"""
    print("🔍 分析 GPU 内存碎片化...")
    
    if torch.cuda.is_available():
        # 尝试分配不同大小的内存块
        sizes = [64, 128, 256, 512, 1024, 2048]  # MB
        
        for size_mb in sizes:
            try:
                size_bytes = size_mb * 1024 * 1024
                tensor = torch.empty(size_bytes // 4, dtype=torch.float32, device='cuda:1')
                print(f"✅ 成功分配 {size_mb}MB 内存块")
                del tensor
                torch.cuda.empty_cache()
            except RuntimeError as e:
                print(f"❌ 无法分配 {size_mb}MB 内存块: {e}")
                break

def main():
    """主函数"""
    print("🎯 GPU 内存监控工具")
    print("=" * 50)
    
    # 检查 GPU 可用性
    if not torch.cuda.is_available():
        print("❌ CUDA 不可用")
        return
    
    print(f"✅ 检测到 {torch.cuda.device_count()} 个 GPU")
    
    # 显示当前状态
    print("\n📊 当前 GPU 状态:")
    gpu_info = get_gpu_memory_info(1)
    if gpu_info:
        print(f"GPU 1: {gpu_info['total']:.1f}GB 总内存, {gpu_info['free']:.1f}GB 可用")
    
    # 分析内存碎片化
    print("\n🔍 内存碎片化分析:")
    analyze_memory_fragmentation()
    
    # 开始监控
    print("\n" + "=" * 50)
    monitor_memory(interval=3, duration=30)

if __name__ == "__main__":
    main() 