#!/usr/bin/env python3
"""
测试UI中上下文重复问题
"""

import sys
from pathlib import Path

# 添加项目根目录到路径
sys.path.append(str(Path(__file__).parent))

from xlm.ui.optimized_rag_ui import OptimizedRagUI
from config.parameters import Config

def test_ui_context_duplicates():
    """测试UI中上下文重复问题"""
    print("=" * 80)
    print("测试UI中上下文重复问题")
    print("=" * 80)
    
    try:
        # 1. 初始化UI
        print("\n1. 初始化UI...")
        config = Config()
        ui = OptimizedRagUI(
            cache_dir=config.encoder.cache_dir,
            use_faiss=config.retriever.use_faiss,
            enable_reranker=config.reranker.enabled,
            use_existing_embedding_index=config.retriever.use_existing_embedding_index,
            max_alphafin_chunks=config.retriever.max_alphafin_chunks
        )
        print("✅ UI初始化成功")
        
        # 2. 测试英文查询
        print("\n2. 测试英文查询...")
        test_question = "What is the revenue of Apple in 2023?"
        
        # 模拟UI处理流程
        print(f"查询: {test_question}")
        
        # 检测语言
        try:
            from langdetect import detect
            language = detect(test_question)
            is_chinese = language.startswith('zh')
        except:
            is_chinese = any('\u4e00' <= char <= '\u9fff' for char in test_question)
        
        print(f"检测语言: {'中文' if is_chinese else '英文'}")
        
        # 3. 获取检索结果
        print("\n3. 获取检索结果...")
        if is_chinese:
            # 中文查询 - 使用多阶段检索系统
            if hasattr(ui, 'chinese_retrieval_system') and ui.chinese_retrieval_system:
                faiss_results = ui.chinese_retrieval_system.retrieve(test_question, top_k=10)
                retrieved_documents = []
                retriever_scores = []
                for doc_idx, score in faiss_results:
                    original_doc = ui.chinese_retrieval_system.data[doc_idx]
                    chunks = ui.chinese_retrieval_system.doc_to_chunks_mapping.get(doc_idx, [])
                    if chunks:
                        from xlm.dto.dto import DocumentWithMetadata, DocumentMetadata
                        doc = DocumentWithMetadata(
                            content=chunks[0],
                            metadata=DocumentMetadata(
                                source=str(original_doc.get('company_name', '')),
                                language="chinese",
                                doc_id=str(original_doc.get('doc_id', str(doc_idx)))
                            )
                        )
                        retrieved_documents.append(doc)
                        retriever_scores.append(score)
            else:
                print("❌ 中文检索系统未初始化")
                return
        else:
            # 英文查询 - 使用双语检索器
            if hasattr(ui, 'retriever'):
                retrieved_documents, retriever_scores = ui.retriever.retrieve(
                    text=test_question,
                    top_k=10,
                    return_scores=True
                )
            else:
                print("❌ 检索器未初始化")
                return
        
        print(f"检索到 {len(retrieved_documents)} 个文档")
        
        # 4. 检查重复内容
        print("\n4. 检查重复内容...")
        content_hashes = set()
        duplicate_count = 0
        
        for i, (doc, score) in enumerate(zip(retrieved_documents, retriever_scores)):
            content = doc.content if hasattr(doc, 'content') else str(doc)
            content_hash = hash(content)
            
            if content_hash in content_hashes:
                duplicate_count += 1
                print(f"❌ 发现重复文档 {i+1}:")
                print(f"   分数: {score:.4f}")
                print(f"   内容前100字符: {content[:100]}...")
                print(f"   重复哈希: {content_hash}")
            else:
                content_hashes.add(content_hash)
                print(f"✅ 文档 {i+1}: 唯一")
                print(f"   分数: {score:.4f}")
                print(f"   内容前100字符: {content[:100]}...")
        
        print(f"\n重复统计:")
        print(f"   总文档数: {len(retrieved_documents)}")
        print(f"   唯一文档数: {len(content_hashes)}")
        print(f"   重复文档数: {duplicate_count}")
        print(f"   重复率: {duplicate_count/len(retrieved_documents)*100:.2f}%")
        
        # 5. 检查UI去重逻辑
        print("\n5. 检查UI去重逻辑...")
        unique_docs = []
        seen_hashes = set()
        
        for doc, score in zip(retrieved_documents, retriever_scores):
            if hasattr(doc, 'content'):
                content = doc.content
            else:
                content = str(doc)
            h = hash(content)
            if h not in seen_hashes:
                unique_docs.append((doc, score))
                seen_hashes.add(h)
            if len(unique_docs) >= ui.config.retriever.rerank_top_k:
                break
        
        print(f"UI去重后文档数: {len(unique_docs)}")
        print(f"UI去重效果: {len(retrieved_documents) - len(unique_docs)} 个重复被移除")
        
        # 6. 检查HTML生成
        print("\n6. 检查HTML生成...")
        ui_docs = []
        for doc, score in unique_docs:
            if getattr(doc.metadata, 'language', '') == 'chinese':
                doc_id = str(getattr(doc.metadata, 'origin_doc_id', '') or getattr(doc.metadata, 'doc_id', '')).strip()
                raw_context = ui.docid2context.get(doc_id, "")
                if not raw_context:
                    raw_context = doc.content
            else:
                raw_context = doc.content
            preview_content = raw_context[:200] + "..." if len(raw_context) > 200 else raw_context
            ui_docs.append((doc, score, preview_content, raw_context))
        
        print(f"UI文档数: {len(ui_docs)}")
        
        # 检查UI文档中的重复
        ui_content_hashes = set()
        ui_duplicate_count = 0
        
        for i, (doc, score, preview, raw) in enumerate(ui_docs):
            content_hash = hash(raw)
            if content_hash in ui_content_hashes:
                ui_duplicate_count += 1
                print(f"❌ UI中发现重复文档 {i+1}:")
                print(f"   分数: {score:.4f}")
                print(f"   预览: {preview[:50]}...")
            else:
                ui_content_hashes.add(content_hash)
                print(f"✅ UI文档 {i+1}: 唯一")
                print(f"   分数: {score:.4f}")
                print(f"   预览: {preview[:50]}...")
        
        print(f"\nUI重复统计:")
        print(f"   UI文档数: {len(ui_docs)}")
        print(f"   UI唯一文档数: {len(ui_content_hashes)}")
        print(f"   UI重复文档数: {ui_duplicate_count}")
        
        # 7. 总结
        print("\n7. 总结:")
        if duplicate_count == 0 and ui_duplicate_count == 0:
            print("✅ 未发现上下文重复问题")
        else:
            print("❌ 发现上下文重复问题:")
            if duplicate_count > 0:
                print(f"   - 检索阶段重复: {duplicate_count} 个")
            if ui_duplicate_count > 0:
                print(f"   - UI显示阶段重复: {ui_duplicate_count} 个")
            print("建议检查:")
            print("   1. 数据源中是否存在重复内容")
            print("   2. 检索器是否返回了重复文档")
            print("   3. UI去重逻辑是否正常工作")
        
    except Exception as e:
        print(f"❌ 测试失败: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    test_ui_context_duplicates() 