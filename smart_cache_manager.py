#!/usr/bin/env python3
"""
智能缓存管理器 - 自动检测和修复缓存问题
"""

import os
import sys
import shutil
import numpy as np
import faiss
from pathlib import Path
sys.path.append(str(Path(__file__).parent))

from xlm.utils.dual_language_loader import DualLanguageLoader
from config.parameters import Config

class SmartCacheManager:
    def __init__(self):
        self.config = Config()
        self.cache_dir = self.config.encoder.cache_dir
        self.data_loader = DualLanguageLoader()
        
    def check_cache_health(self):
        """检查缓存健康状态"""
        print("=== 缓存健康检查 ===")
        
        # 检查缓存目录
        if not os.path.exists(self.cache_dir):
            print(f"❌ 缓存目录不存在: {self.cache_dir}")
            return False
            
        print(f"✅ 缓存目录存在: {self.cache_dir}")
        
        # 检查英文数据缓存
        english_cache_ok = self._check_english_cache()
        
        # 检查中文数据缓存
        chinese_cache_ok = self._check_chinese_cache()
        
        return english_cache_ok and chinese_cache_ok
    
    def _check_english_cache(self):
        """检查英文缓存"""
        print(f"\n--- 检查英文缓存 ---")
        
        # 加载英文数据
        try:
            english_docs = self.data_loader.load_tatqa_context_only(self.config.data.english_data_path)
            print(f"英文文档数量: {len(english_docs)}")
            
            if not english_docs:
                print("❌ 没有英文文档，无法检查缓存")
                return False
                
        except Exception as e:
            print(f"❌ 加载英文数据失败: {e}")
            return False
        
        # 生成缓存键
        from xlm.components.encoder.finbert import FinbertEncoder
        encoder_en = FinbertEncoder(
            model_name=self.config.encoder.english_model_path,
            cache_dir=self.config.encoder.cache_dir,
            device=self.config.encoder.device
        )
        
        cache_key = self._get_cache_key(english_docs, str(encoder_en.model_name))
        embeddings_path = self._get_cache_path(cache_key, "npy")
        index_path = self._get_cache_path(cache_key, "faiss")
        
        print(f"缓存键: {cache_key}")
        print(f"嵌入向量路径: {embeddings_path}")
        print(f"FAISS索引路径: {index_path}")
        
        # 检查文件是否存在
        embeddings_exist = os.path.exists(embeddings_path)
        index_exist = os.path.exists(index_path)
        
        print(f"嵌入向量文件存在: {embeddings_exist}")
        print(f"FAISS索引文件存在: {index_exist}")
        
        if not embeddings_exist or not index_exist:
            print("❌ 英文缓存文件不完整")
            return False
        
        # 检查嵌入向量有效性
        try:
            embeddings = np.load(embeddings_path)
            print(f"嵌入向量形状: {embeddings.shape}")
            
            if embeddings.shape[0] != len(english_docs):
                print(f"❌ 文档数量不匹配: 缓存={embeddings.shape[0]}, 当前={len(english_docs)}")
                return False
                
            if embeddings.size == 0:
                print("❌ 嵌入向量为空")
                return False
                
        except Exception as e:
            print(f"❌ 嵌入向量读取失败: {e}")
            return False
        
        # 检查FAISS索引有效性
        try:
            index = faiss.read_index(index_path)
            if hasattr(index, 'ntotal'):
                print(f"FAISS索引文档数: {index.ntotal}")
                
                if index.ntotal != len(english_docs):
                    print(f"❌ FAISS索引大小不匹配: 缓存={index.ntotal}, 当前={len(english_docs)}")
                    return False
                    
                if index.ntotal == 0:
                    print("❌ FAISS索引为空")
                    return False
                    
        except Exception as e:
            print(f"❌ FAISS索引读取失败: {e}")
            return False
        
        print("✅ 英文缓存健康")
        return True
    
    def _check_chinese_cache(self):
        """检查中文缓存"""
        print(f"\n--- 检查中文缓存 ---")
        
        # 加载中文数据
        try:
            chinese_docs = self.data_loader.load_alphafin_data(self.config.data.chinese_data_path)
            print(f"中文文档数量: {len(chinese_docs)}")
            
            if not chinese_docs:
                print("❌ 没有中文文档，无法检查缓存")
                return False
                
        except Exception as e:
            print(f"❌ 加载中文数据失败: {e}")
            return False
        
        # 生成缓存键
        from xlm.components.encoder.finbert import FinbertEncoder
        encoder_ch = FinbertEncoder(
            model_name=self.config.encoder.chinese_model_path,
            cache_dir=self.config.encoder.cache_dir,
            device=self.config.encoder.device
        )
        
        cache_key = self._get_cache_key(chinese_docs, str(encoder_ch.model_name))
        embeddings_path = self._get_cache_path(cache_key, "npy")
        index_path = self._get_cache_path(cache_key, "faiss")
        
        print(f"缓存键: {cache_key}")
        print(f"嵌入向量路径: {embeddings_path}")
        print(f"FAISS索引路径: {index_path}")
        
        # 检查文件是否存在
        embeddings_exist = os.path.exists(embeddings_path)
        index_exist = os.path.exists(index_path)
        
        print(f"嵌入向量文件存在: {embeddings_exist}")
        print(f"FAISS索引文件存在: {index_exist}")
        
        if not embeddings_exist or not index_exist:
            print("❌ 中文缓存文件不完整")
            return False
        
        # 检查嵌入向量有效性
        try:
            embeddings = np.load(embeddings_path)
            print(f"嵌入向量形状: {embeddings.shape}")
            
            if embeddings.shape[0] != len(chinese_docs):
                print(f"❌ 文档数量不匹配: 缓存={embeddings.shape[0]}, 当前={len(chinese_docs)}")
                return False
                
            if embeddings.size == 0:
                print("❌ 嵌入向量为空")
                return False
                
        except Exception as e:
            print(f"❌ 嵌入向量读取失败: {e}")
            return False
        
        # 检查FAISS索引有效性
        try:
            index = faiss.read_index(index_path)
            if hasattr(index, 'ntotal'):
                print(f"FAISS索引文档数: {index.ntotal}")
                
                if index.ntotal != len(chinese_docs):
                    print(f"❌ FAISS索引大小不匹配: 缓存={index.ntotal}, 当前={len(chinese_docs)}")
                    return False
                    
                if index.ntotal == 0:
                    print("❌ FAISS索引为空")
                    return False
                    
        except Exception as e:
            print(f"❌ FAISS索引读取失败: {e}")
            return False
        
        print("✅ 中文缓存健康")
        return True
    
    def _get_cache_key(self, documents, encoder_name):
        """生成缓存键"""
        import hashlib
        
        # 创建文档内容的哈希
        content_hash = hashlib.md5()
        for doc in documents:
            content_hash.update(doc.content.encode('utf-8'))
        
        # 只使用编码器名称的最后部分，避免路径问题
        encoder_basename = os.path.basename(encoder_name)
        
        # 结合编码器名称和文档数量
        cache_key = f"{encoder_basename}_{len(documents)}_{content_hash.hexdigest()[:16]}"
        return cache_key
    
    def _get_cache_path(self, cache_key, suffix):
        """获取缓存文件路径"""
        return os.path.join(self.cache_dir, f"{cache_key}.{suffix}")
    
    def clean_all_cache(self):
        """清理所有缓存"""
        print("=== 清理所有缓存 ===")
        
        if not os.path.exists(self.cache_dir):
            print("缓存目录不存在，无需清理")
            return
        
        try:
            # 删除所有.npy和.faiss文件
            removed_count = 0
            for file in os.listdir(self.cache_dir):
                if file.endswith(('.npy', '.faiss')):
                    file_path = os.path.join(self.cache_dir, file)
                    os.remove(file_path)
                    removed_count += 1
                    print(f"删除: {file}")
            
            print(f"✅ 清理完成，删除了 {removed_count} 个缓存文件")
            
        except Exception as e:
            print(f"❌ 清理缓存失败: {e}")
    
    def repair_cache(self):
        """修复缓存问题"""
        print("=== 修复缓存问题 ===")
        
        # 检查缓存健康状态
        if self.check_cache_health():
            print("✅ 缓存健康，无需修复")
            return True
        
        print("🔄 检测到缓存问题，开始修复...")
        
        # 清理所有缓存
        self.clean_all_cache()
        
        print("✅ 缓存修复完成，下次启动RAG系统时将重新生成缓存")
        return True

def main():
    """主函数"""
    manager = SmartCacheManager()
    
    print("智能缓存管理器")
    print("1. 检查缓存健康状态")
    print("2. 清理所有缓存")
    print("3. 修复缓存问题")
    
    choice = input("请选择操作 (1-3): ").strip()
    
    if choice == "1":
        manager.check_cache_health()
    elif choice == "2":
        manager.clean_all_cache()
    elif choice == "3":
        manager.repair_cache()
    else:
        print("无效选择")

if __name__ == "__main__":
    main() 