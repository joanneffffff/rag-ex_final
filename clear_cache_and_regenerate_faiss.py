#!/usr/bin/env python3
"""
清除缓存并重新生成FAISS索引
解决英文数据变化导致的FAISS索引不匹配问题
"""

import os
import shutil
import glob
from pathlib import Path

def clear_cache_and_regenerate():
    """清除缓存并重新生成FAISS索引"""
    
    print("🧹 开始清除缓存并重新生成FAISS索引...")
    
    # 1. 清除缓存目录
    cache_dirs = [
        "cache",
        "data/faiss_indexes",
        "xlm/cache"
    ]
    
    for cache_dir in cache_dirs:
        if os.path.exists(cache_dir):
            print(f"🗑️ 清除缓存目录: {cache_dir}")
            try:
                shutil.rmtree(cache_dir)
                print(f"✅ 成功清除: {cache_dir}")
            except Exception as e:
                print(f"⚠️ 清除失败: {cache_dir}, 错误: {e}")
        else:
            print(f"ℹ️ 缓存目录不存在: {cache_dir}")
    
    # 2. 查找并清除所有FAISS相关文件
    faiss_patterns = [
        "**/*.faiss",
        "**/*.bin",
        "**/*.npy",
        "**/faiss_index*"
    ]
    
    for pattern in faiss_patterns:
        files = glob.glob(pattern, recursive=True)
        for file in files:
            if os.path.isfile(file):
                try:
                    os.remove(file)
                    print(f"🗑️ 删除文件: {file}")
                except Exception as e:
                    print(f"⚠️ 删除失败: {file}, 错误: {e}")
    
    # 3. 创建必要的目录
    for cache_dir in cache_dirs:
        os.makedirs(cache_dir, exist_ok=True)
        print(f"📁 创建目录: {cache_dir}")
    
    print("\n🎉 缓存清除完成！")
    print("\n📋 下一步操作:")
    print("1. 重新启动RAG系统")
    print("2. 系统会自动重新生成FAISS索引")
    print("3. 英文文档数量: 5398 (修复后的数据)")
    print("4. 中文文档数量: 26591 (保持不变)")
    
    return True

if __name__ == "__main__":
    clear_cache_and_regenerate() 