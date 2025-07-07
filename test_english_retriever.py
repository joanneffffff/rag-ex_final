#!/usr/bin/env python3
"""
专门测试BilingualRetriever英文初始化的脚本
"""

import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent))

from xlm.components.retriever.bilingual_retriever import BilingualRetriever
from xlm.components.encoder.finbert import FinbertEncoder
from xlm.utils.dual_language_loader import DualLanguageLoader
from config.parameters import Config

def test_english_retriever():
    """测试BilingualRetriever的英文初始化"""
    print("=== 测试BilingualRetriever英文初始化 ===")
    
    # 1. 加载配置
    config = Config()
    print(f"1. 配置信息:")
    print(f"   英文数据路径: {config.data.english_data_path}")
    print(f"   英文编码器路径: {config.encoder.english_model_path}")
    print(f"   缓存目录: {config.encoder.cache_dir}")
    
    # 2. 加载英文数据
    print(f"\n2. 加载英文数据...")
    data_loader = DualLanguageLoader()
    
    try:
        english_docs = data_loader.load_tatqa_context_only(config.data.english_data_path)
        print(f"   加载的英文文档数量: {len(english_docs)}")
        
        if not english_docs:
            print("   ❌ 没有英文文档，无法继续测试")
            return
            
        # 检查第一个文档
        first_doc = english_docs[0]
        print(f"   第一个文档:")
        print(f"     - content长度: {len(first_doc.content)}")
        print(f"     - content预览: {first_doc.content[:100]}...")
        print(f"     - language: {first_doc.metadata.language}")
        
    except Exception as e:
        print(f"   ❌ 加载英文数据失败: {e}")
        return
    
    # 3. 加载英文编码器
    print(f"\n3. 加载英文编码器...")
    try:
        encoder_en = FinbertEncoder(
            model_name=config.encoder.english_model_path,
            cache_dir=config.encoder.cache_dir,
            device=config.encoder.device
        )
        print(f"   ✅ 英文编码器加载成功")
        print(f"   模型名称: {encoder_en.model_name}")
        print(f"   设备: {encoder_en.device}")
        
    except Exception as e:
        print(f"   ❌ 英文编码器加载失败: {e}")
        return
    
    # 4. 创建BilingualRetriever（只测试英文）
    print(f"\n4. 创建BilingualRetriever（英文文档）...")
    try:
        # 创建一个虚拟的中文编码器（不会被使用）
        encoder_ch = FinbertEncoder(
            model_name=config.encoder.chinese_model_path,
            cache_dir=config.encoder.cache_dir,
            device=config.encoder.device
        )
        
        retriever = BilingualRetriever(
            encoder_en=encoder_en,
            encoder_ch=encoder_ch,
            corpus_documents_en=english_docs,
            corpus_documents_ch=[],  # 空的中文文档列表
            use_faiss=True,
            use_gpu=True,
            batch_size=32,
            cache_dir=config.encoder.cache_dir,
            use_existing_embedding_index=False  # 强制重新计算
        )
        print(f"   ✅ BilingualRetriever创建成功")
        
        # 检查嵌入向量状态
        print(f"   英文嵌入向量形状: {retriever.corpus_embeddings_en.shape if retriever.corpus_embeddings_en is not None else 'None'}")
        print(f"   英文FAISS索引: {'已初始化' if retriever.index_en else '未初始化'}")
        
        if retriever.corpus_embeddings_en is not None and retriever.corpus_embeddings_en.shape[0] > 0:
            print(f"   ✅ 英文嵌入向量生成成功，文档数: {retriever.corpus_embeddings_en.shape[0]}")
        else:
            print(f"   ❌ 英文嵌入向量生成失败或为空")
            
    except Exception as e:
        print(f"   ❌ BilingualRetriever创建失败: {e}")
        import traceback
        traceback.print_exc()
        return
    
    # 5. 测试检索功能
    print(f"\n5. 测试英文检索功能...")
    try:
        test_query = "What is the revenue?"
        results = retriever.retrieve(test_query, top_k=3, language='en')
        print(f"   查询: {test_query}")
        print(f"   检索结果数量: {len(results)}")
        
        if results:
            for i, doc in enumerate(results[:2]):
                if hasattr(doc, 'content'):
                    print(f"   结果{i+1}: {doc.content[:100]}...")
                else:
                    print(f"   结果{i+1}: {str(doc)[:100]}...")
        else:
            print(f"   ⚠️ 没有检索到结果")
            
    except Exception as e:
        print(f"   ❌ 检索测试失败: {e}")

if __name__ == "__main__":
    test_english_retriever() 