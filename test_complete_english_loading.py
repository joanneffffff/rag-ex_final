#!/usr/bin/env python3
"""
测试完整的英文数据加载和编码流程
验证从数据加载到BilingualRetriever初始化的完整流程
"""

import sys
import os
from pathlib import Path
import traceback

# 添加项目根目录到Python路径
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

def test_data_loader():
    """测试数据加载器"""
    print("=" * 80)
    print("🔍 测试数据加载器")
    print("=" * 80)
    
    try:
        from xlm.utils.dual_language_loader import DualLanguageLoader
        from config.parameters import Config
        
        config = Config()
        data_loader = DualLanguageLoader()
        
        print(f"📁 英文数据路径: {config.data.english_data_path}")
        
        # 测试英文数据加载
        print("\n📊 加载英文数据...")
        english_docs = data_loader.load_tatqa_context_only(config.data.english_data_path)
        
        print(f"✅ 英文数据加载成功，文档数量: {len(english_docs)}")
        
        if english_docs:
            print(f"\n📋 前3个英文文档示例:")
            for i, doc in enumerate(english_docs[:3]):
                content = doc.content[:100] + "..." if len(doc.content) > 100 else doc.content
                print(f"  {i+1}. 文档类型: {type(doc)}")
                print(f"     内容长度: {len(doc.content)}")
                print(f"     内容预览: {content}")
                print(f"     元数据: {doc.metadata}")
        
        return english_docs
        
    except Exception as e:
        print(f"❌ 数据加载失败: {e}")
        traceback.print_exc()
        return None

def test_encoder_creation():
    """测试编码器创建"""
    print("\n" + "=" * 80)
    print("🔍 测试编码器创建")
    print("=" * 80)
    
    try:
        from xlm.components.encoder.finbert import FinbertEncoder
        from config.parameters import Config
        
        config = Config()
        
        print(f"📁 英文编码器路径: {config.encoder.english_model_path}")
        print(f"📁 中文编码器路径: {config.encoder.chinese_model_path}")
        
        # 创建英文编码器
        print("\n📊 创建英文编码器...")
        encoder_en = FinbertEncoder(
            model_name=config.encoder.english_model_path,
            cache_dir=config.encoder.cache_dir,
            device=config.encoder.device
        )
        print(f"✅ 英文编码器创建成功")
        print(f"   模型名称: {encoder_en.model_name}")
        print(f"   设备: {encoder_en.device}")
        print(f"   嵌入维度: {encoder_en.get_embedding_dimension()}")
        
        # 创建中文编码器
        print("\n📊 创建中文编码器...")
        encoder_ch = FinbertEncoder(
            model_name=config.encoder.chinese_model_path,
            cache_dir=config.encoder.cache_dir,
            device=config.encoder.device
        )
        print(f"✅ 中文编码器创建成功")
        print(f"   模型名称: {encoder_ch.model_name}")
        print(f"   设备: {encoder_ch.device}")
        print(f"   嵌入维度: {encoder_ch.get_embedding_dimension()}")
        
        return encoder_en, encoder_ch
        
    except Exception as e:
        print(f"❌ 编码器创建失败: {e}")
        traceback.print_exc()
        return None, None

def test_bilingual_retriever(english_docs, encoder_en, encoder_ch):
    """测试BilingualRetriever初始化"""
    print("\n" + "=" * 80)
    print("🔍 测试BilingualRetriever初始化")
    print("=" * 80)
    
    try:
        from xlm.components.retriever.bilingual_retriever import BilingualRetriever
        from config.parameters import Config
        
        config = Config()
        
        print(f"📊 英文文档数量: {len(english_docs) if english_docs else 0}")
        print(f"📊 中文文档数量: 0 (测试中不加载中文)")
        
        # 创建BilingualRetriever
        print("\n📊 创建BilingualRetriever...")
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
        
        print(f"✅ BilingualRetriever创建成功")
        print(f"   英文嵌入向量形状: {retriever.corpus_embeddings_en.shape if retriever.corpus_embeddings_en is not None else 'None'}")
        print(f"   中文嵌入向量形状: {retriever.corpus_embeddings_ch.shape if retriever.corpus_embeddings_ch is not None else 'None'}")
        print(f"   英文FAISS索引: {'已初始化' if retriever.index_en else '未初始化'}")
        print(f"   中文FAISS索引: {'已初始化' if retriever.index_ch else '未初始化'}")
        
        return retriever
        
    except Exception as e:
        print(f"❌ BilingualRetriever创建失败: {e}")
        traceback.print_exc()
        return None

def test_retrieval(retriever):
    """测试检索功能"""
    print("\n" + "=" * 80)
    print("🔍 测试检索功能")
    print("=" * 80)
    
    try:
        test_query = "How was internally developed software capitalised?"
        print(f"📝 测试查询: {test_query}")
        
        # 执行检索
        results = retriever.retrieve(
            text=test_query,
            top_k=5,
            language="en"
        )
        
        print(f"✅ 检索成功，结果数量: {len(results)}")
        
        if results:
            print(f"\n📋 检索结果示例:")
            for i, doc in enumerate(results[:3]):
                content = doc.content[:100] + "..." if len(doc.content) > 100 else doc.content
                print(f"  {i+1}. 内容预览: {content}")
        
        return True
        
    except Exception as e:
        print(f"❌ 检索失败: {e}")
        traceback.print_exc()
        return False

if __name__ == "__main__":
    print("🚀 开始完整英文数据加载和编码流程测试")
    
    # 测试1: 数据加载
    english_docs = test_data_loader()
    if not english_docs:
        print("❌ 数据加载失败，停止测试")
        sys.exit(1)
    
    # 测试2: 编码器创建
    encoder_en, encoder_ch = test_encoder_creation()
    if not encoder_en or not encoder_ch:
        print("❌ 编码器创建失败，停止测试")
        sys.exit(1)
    
    # 测试3: BilingualRetriever初始化
    retriever = test_bilingual_retriever(english_docs, encoder_en, encoder_ch)
    if not retriever:
        print("❌ BilingualRetriever初始化失败，停止测试")
        sys.exit(1)
    
    # 测试4: 检索功能
    retrieval_success = test_retrieval(retriever)
    
    print("\n" + "=" * 80)
    print("📋 测试总结")
    print("=" * 80)
    
    if retrieval_success:
        print("✅ 所有测试通过！英文嵌入向量问题已解决")
        print("✅ 完整流程正常工作")
    else:
        print("❌ 检索测试失败，需要进一步检查")
    
    print("=" * 80) 