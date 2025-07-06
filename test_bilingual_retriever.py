#!/usr/bin/env python3
"""
测试BilingualRetriever初始化的脚本
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from xlm.components.retriever.bilingual_retriever import BilingualRetriever
from xlm.components.encoder.finbert import FinbertEncoder
from xlm.utils.dual_language_loader import DualLanguageLoader
from config.parameters import Config

def main():
    print("=== 测试BilingualRetriever初始化 ===")
    
    config = Config()
    
    # 1. 初始化编码器
    print("1. 初始化编码器...")
    encoder_en = FinbertEncoder(
        model_name=config.encoder.english_model_path,
        cache_dir=config.encoder.cache_dir,
        device=config.encoder.device
    )
    encoder_ch = FinbertEncoder(
        model_name=config.encoder.chinese_model_path,
        cache_dir=config.encoder.cache_dir,
        device=config.encoder.device
    )
    print("✅ 编码器初始化完成")
    
    # 2. 加载数据
    print("\n2. 加载数据...")
    data_loader = DualLanguageLoader()
    english_docs = data_loader.load_tatqa_context_only(config.data.english_data_path)
    chinese_docs = data_loader.load_alphafin_data(config.data.chinese_data_path)
    print(f"✅ 数据加载完成: {len(english_docs)} 个英文文档, {len(chinese_docs)} 个中文文档")
    
    # 3. 初始化BilingualRetriever（强制重新计算嵌入）
    print("\n3. 初始化BilingualRetriever...")
    retriever = BilingualRetriever(
        encoder_en=encoder_en,
        encoder_ch=encoder_ch,
        corpus_documents_en=english_docs,
        corpus_documents_ch=chinese_docs,
        use_faiss=True,
        use_gpu=True,
        batch_size=32,
        cache_dir=config.encoder.cache_dir,
        use_existing_embedding_index=False  # 强制重新计算
    )
    print("✅ BilingualRetriever初始化完成")
    
    # 4. 检查状态
    print("\n4. 检查最终状态...")
    print(f"英文嵌入向量: {retriever.corpus_embeddings_en.shape if retriever.corpus_embeddings_en is not None else 'None'}")
    print(f"中文嵌入向量: {retriever.corpus_embeddings_ch.shape if retriever.corpus_embeddings_ch is not None else 'None'}")
    print(f"英文FAISS索引: {'已初始化' if retriever.index_en else '未初始化'}")
    print(f"中文FAISS索引: {'已初始化' if retriever.index_ch else '未初始化'}")
    
    # 5. 测试检索
    print("\n5. 测试检索...")
    test_query = "How was internally developed software capitalised?"
    try:
        result = retriever.retrieve(
            text=test_query,
            top_k=5,
            return_scores=True,
            language="en"
        )
        if isinstance(result, tuple):
            docs, scores = result
            print(f"✅ 检索成功: {len(docs)} 个文档")
            for i, (doc, score) in enumerate(zip(docs[:3], scores[:3])):
                print(f"  文档 {i+1} (分数: {score:.4f}): {doc.content[:100]}...")
        else:
            print(f"✅ 检索成功: {len(result)} 个文档")
    except Exception as e:
        print(f"❌ 检索失败: {e}")
        import traceback
        traceback.print_exc()
    
    print("\n=== 测试完成 ===")

if __name__ == "__main__":
    main() 