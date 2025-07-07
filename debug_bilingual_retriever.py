#!/usr/bin/env python3
"""
调试BilingualRetriever编码过程的脚本
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from xlm.components.encoder.finbert import FinbertEncoder
from xlm.utils.dual_language_loader import DualLanguageLoader
from config.parameters import Config
import torch

def test_bilingual_encoding():
    print("=== 调试BilingualRetriever编码过程 ===")
    
    config = Config()
    
    # 1. 初始化编码器
    print("1. 初始化英文编码器...")
    encoder_en = FinbertEncoder(
        model_name=config.encoder.english_model_path,
        cache_dir=config.encoder.cache_dir,
        device=config.encoder.device
    )
    print("✅ 编码器初始化完成")
    
    # 2. 加载英文数据
    print("\n2. 加载英文数据...")
    data_loader = DualLanguageLoader()
    english_docs = data_loader.load_tatqa_context_only(config.data.english_data_path)
    print(f"✅ 加载了 {len(english_docs)} 个英文文档")
    
    if not english_docs:
        print("❌ 没有加载到英文文档")
        return
    
    # 3. 模拟BilingualRetriever的编码过程
    print("\n3. 模拟BilingualRetriever编码过程...")
    
    # 检查文档类型
    print(f"   第一个文档类型: {type(english_docs[0])}")
    print(f"   第一个文档content类型: {type(english_docs[0].content)}")
    print(f"   第一个文档content长度: {len(english_docs[0].content)}")
    print(f"   第一个文档content预览: {english_docs[0].content[:100]}...")
    
    # 准备批量文本
    batch_texts = []
    for i, doc in enumerate(english_docs):
        # 英文使用content字段
        batch_texts.append(doc.content)
        
        # 打印前几个文档的内容预览
        if i < 3:
            content_preview = batch_texts[-1][:100] + "..." if len(batch_texts[-1]) > 100 else batch_texts[-1]
            print(f"   文档 {i+1} 内容预览: {content_preview}")
    
    print(f"   准备编码 {len(batch_texts)} 个文本")
    
    # 测试编码器
    try:
        print("   测试编码器...")
        test_text = batch_texts[0] if batch_texts else "test"
        test_embedding = encoder_en.encode([test_text])
        print(f"   ✅ 测试编码成功，嵌入维度: {test_embedding.shape}")
    except Exception as e:
        print(f"   ❌ 编码器测试失败: {e}")
        import traceback
        traceback.print_exc()
        return
    
    # 批量编码
    try:
        print("   开始批量编码...")
        batch_size = 32
        embeddings = encoder_en.encode(texts=batch_texts, batch_size=batch_size, show_progress_bar=True)
        print(f"   ✅ 编码完成，嵌入向量形状: {embeddings.shape}")
        
        # 检查嵌入向量
        print(f"   嵌入向量统计:")
        print(f"     最小值: {embeddings.min():.4f}")
        print(f"     最大值: {embeddings.max():.4f}")
        print(f"     平均值: {embeddings.mean():.4f}")
        print(f"     标准差: {embeddings.std():.4f}")
        
    except Exception as e:
        print(f"   ❌ 批量编码过程中发生错误: {e}")
        import traceback
        traceback.print_exc()
        return
    
    # 4. 测试小批量编码
    print("\n4. 测试小批量编码...")
    try:
        small_batch = batch_texts[:10]  # 只测试前10个
        small_embeddings = encoder_en.encode(texts=small_batch, batch_size=5, show_progress_bar=True)
        print(f"   ✅ 小批量编码成功，形状: {small_embeddings.shape}")
    except Exception as e:
        print(f"   ❌ 小批量编码失败: {e}")
        import traceback
        traceback.print_exc()
    
    print("\n=== 调试完成 ===")

if __name__ == "__main__":
    test_bilingual_encoding() 