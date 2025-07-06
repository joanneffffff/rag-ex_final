#!/usr/bin/env python3
"""
测试英文编码器的脚本
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from xlm.components.encoder.finbert import FinbertEncoder
from xlm.utils.dual_language_loader import DualLanguageLoader
from config.parameters import Config

def main():
    print("=== 测试英文编码器 ===")
    
    config = Config()
    
    # 1. 测试编码器初始化
    print("1. 初始化英文编码器...")
    try:
        encoder = FinbertEncoder(
            model_name=config.encoder.english_model_path,
            cache_dir=config.encoder.cache_dir,
            device=config.encoder.device
        )
        print(f"✅ 编码器初始化成功: {encoder.model_name}")
        print(f"设备: {encoder.device}")
    except Exception as e:
        print(f"❌ 编码器初始化失败: {e}")
        import traceback
        traceback.print_exc()
        return
    
    # 2. 测试单个文本编码
    print("\n2. 测试单个文本编码...")
    test_text = "How was internally developed software capitalised?"
    try:
        embedding = encoder.encode([test_text])
        print(f"✅ 单个文本编码成功，形状: {embedding.shape}")
    except Exception as e:
        print(f"❌ 单个文本编码失败: {e}")
        import traceback
        traceback.print_exc()
        return
    
    # 3. 加载英文数据
    print("\n3. 加载英文数据...")
    data_loader = DualLanguageLoader()
    english_docs = data_loader.load_tatqa_context_only(config.data.english_data_path)
    print(f"✅ 加载了 {len(english_docs)} 个英文文档")
    
    if not english_docs:
        print("❌ 没有英文文档")
        return
    
    # 4. 测试批量编码
    print("\n4. 测试批量编码...")
    test_docs = english_docs[:5]  # 只测试前5个文档
    test_texts = [doc.content for doc in test_docs]
    
    try:
        embeddings = encoder.encode(texts=test_texts, batch_size=2, show_progress_bar=True)
        print(f"✅ 批量编码成功，形状: {embeddings.shape}")
    except Exception as e:
        print(f"❌ 批量编码失败: {e}")
        import traceback
        traceback.print_exc()
        return
    
    print("\n=== 测试完成 ===")
    print("英文编码器工作正常！")

if __name__ == "__main__":
    main() 