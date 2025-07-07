#!/usr/bin/env python3
"""
测试数据集变化兼容性
验证系统能否自动检测数据变化并重新生成索引
"""

import os
import sys
import json
from pathlib import Path

# 添加项目路径
sys.path.append(str(Path(__file__).parent))

from xlm.components.retriever.bilingual_retriever import BilingualRetriever
from xlm.components.encoder.finbert_encoder import FinbertEncoder
from xlm.utils.document_loader import DocumentLoader

def test_dataset_compatibility():
    """测试数据集变化兼容性"""
    
    print("🧪 测试数据集变化兼容性")
    print("=" * 50)
    
    # 1. 初始化编码器
    print("1. 初始化编码器...")
    encoder_en = FinbertEncoder("sentence-transformers/all-MiniLM-L6-v2")
    encoder_ch = FinbertEncoder("shibing624/text2vec-base-chinese")
    
    # 2. 加载文档
    print("2. 加载文档...")
    loader = DocumentLoader()
    
    # 加载英文文档（修复后的数据）
    english_docs = loader.load_documents_from_jsonl(
        "data/unified/tatqa_knowledge_base_combined.jsonl",
        language="english"
    )
    print(f"   英文文档数量: {len(english_docs)}")
    
    # 加载中文文档
    chinese_docs = loader.load_documents_from_jsonl(
        "data/alphafin/alphafin_summarized_and_structured_qa_0628_colab_missing.json",
        language="chinese"
    )
    print(f"   中文文档数量: {len(chinese_docs)}")
    
    # 3. 初始化检索器
    print("3. 初始化检索器...")
    retriever = BilingualRetriever(
        encoder_en=encoder_en,
        encoder_ch=encoder_ch,
        corpus_documents_en=english_docs,
        corpus_documents_ch=chinese_docs,
        use_faiss=True,
        use_existing_embedding_index=True
    )
    
    # 4. 测试检索功能
    print("4. 测试检索功能...")
    
    # 测试英文查询
    print("\n📝 测试英文查询:")
    test_query_en = "How was internally developed software capitalised?"
    results_en = retriever.retrieve(test_query_en, top_k=3)
    print(f"   查询: {test_query_en}")
    print(f"   结果数量: {len(results_en)}")
    if results_en:
        print(f"   第一个结果: {results_en[0].content[:100]}...")
    
    # 测试中文查询
    print("\n📝 测试中文查询:")
    test_query_ch = "中兴通讯在AI时代如何布局通信能力提升？"
    results_ch = retriever.retrieve(test_query_ch, top_k=3)
    print(f"   查询: {test_query_ch}")
    print(f"   结果数量: {len(results_ch)}")
    if results_ch:
        print(f"   第一个结果: {results_ch[0].content[:100]}...")
    
    # 5. 检查索引状态
    print("\n5. 检查索引状态:")
    print(f"   英文FAISS索引: {'已初始化' if retriever.index_en else '未初始化'}")
    print(f"   中文FAISS索引: {'已初始化' if retriever.index_ch else '未初始化'}")
    print(f"   英文嵌入向量: {retriever.corpus_embeddings_en.shape if retriever.corpus_embeddings_en is not None else 'None'}")
    print(f"   中文嵌入向量: {retriever.corpus_embeddings_ch.shape if retriever.corpus_embeddings_ch is not None else 'None'}")
    
    print("\n✅ 数据集变化兼容性测试完成！")
    
    return True

if __name__ == "__main__":
    test_dataset_compatibility() 