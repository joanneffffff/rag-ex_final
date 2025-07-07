#!/usr/bin/env python3
"""
强制重新生成FAISS索引
解决英文嵌入向量为空的问题
"""

import os
import shutil
import glob
from pathlib import Path

def force_regenerate_faiss():
    """强制重新生成FAISS索引"""
    
    print("🔄 强制重新生成FAISS索引...")
    
    # 1. 清除所有缓存
    print("1. 清除所有缓存...")
    cache_dirs = [
        "cache",
        "data/faiss_indexes", 
        "xlm/cache"
    ]
    
    for cache_dir in cache_dirs:
        if os.path.exists(cache_dir):
            print(f"🗑️ 清除缓存目录: {cache_dir}")
            shutil.rmtree(cache_dir)
    
    # 2. 查找并删除所有FAISS相关文件
    print("2. 删除所有FAISS相关文件...")
    faiss_patterns = [
        "**/*.faiss",
        "**/*.bin", 
        "**/*.npy",
        "**/faiss_index*",
        "**/cache/**/*.npy",
        "**/cache/**/*.faiss"
    ]
    
    for pattern in faiss_patterns:
        files = glob.glob(pattern, recursive=True)
        for file in files:
            if os.path.isfile(file):
                try:
                    os.remove(file)
                    print(f"🗑️ 删除: {file}")
                except Exception as e:
                    print(f"⚠️ 删除失败: {file}, 错误: {e}")
    
    # 3. 创建必要的目录
    print("3. 创建必要的目录...")
    for cache_dir in cache_dirs:
        os.makedirs(cache_dir, exist_ok=True)
        print(f"📁 创建: {cache_dir}")
    
    # 4. 检查知识库文件
    print("4. 检查知识库文件...")
    knowledge_base = "data/unified/tatqa_knowledge_base_combined.jsonl"
    if os.path.exists(knowledge_base):
        # 统计文档数量
        doc_count = 0
        with open(knowledge_base, 'r', encoding='utf-8') as f:
            for line in f:
                if line.strip():
                    doc_count += 1
        print(f"✅ 知识库文件存在，文档数: {doc_count}")
    else:
        print(f"❌ 知识库文件不存在: {knowledge_base}")
        return False
    
    print("\n🎉 缓存清除完成！")
    print("\n📋 下一步操作:")
    print("1. 重新启动RAG系统")
    print("2. 系统会强制重新生成所有FAISS索引")
    print("3. 英文文档数量: 5398")
    print("4. 中文文档数量: 26591")
    print("5. 预计需要几分钟时间生成索引")
    
    return True

if __name__ == "__main__":
    force_regenerate_faiss() 