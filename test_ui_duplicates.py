#!/usr/bin/env python3
"""
测试UI中的重复context问题
"""

import sys
from pathlib import Path
import json

# 添加项目根目录到路径
sys.path.append(str(Path(__file__).parent))

def test_knowledge_base_duplicates():
    """测试知识库中的重复情况"""
    print("=" * 80)
    print("测试知识库重复情况")
    print("=" * 80)
    
    # 检查知识库文件
    knowledge_base_file = "data/unified/tatqa_knowledge_base_combined.jsonl"
    
    if not Path(knowledge_base_file).exists():
        print(f"❌ 知识库文件不存在: {knowledge_base_file}")
        return
    
    print(f"📖 检查知识库文件: {knowledge_base_file}")
    
    # 读取知识库
    documents = []
    with open(knowledge_base_file, 'r', encoding='utf-8') as f:
        for line_num, line in enumerate(f, 1):
            if line.strip():
                try:
                    doc = json.loads(line)
                    documents.append(doc)
                except json.JSONDecodeError as e:
                    print(f"⚠️ 第{line_num}行JSON解析错误: {e}")
                    continue
    
    print(f"📊 知识库总文档数: {len(documents)}")
    
    # 检查context重复
    context_groups = {}
    for doc in documents:
        context = doc.get('context', '')
        if context:
            normalized_context = ' '.join(context.split())
            if normalized_context not in context_groups:
                context_groups[normalized_context] = []
            context_groups[normalized_context].append(doc)
    
    # 找出重复的context
    duplicates = {context: docs for context, docs in context_groups.items() if len(docs) > 1}
    
    print(f"\n📋 知识库重复统计:")
    print(f"   - 唯一context数量: {len(context_groups)}")
    print(f"   - 有重复的context数量: {len(duplicates)}")
    print(f"   - 重复文档总数: {sum(len(docs) for docs in duplicates.values())}")
    
    if duplicates:
        print(f"\n🔍 知识库重复详情:")
        for i, (context, docs) in enumerate(list(duplicates.items())[:5]):  # 只显示前5个
            print(f"\n重复组 {i+1} (共{len(docs)}个文档):")
            print(f"Context预览: {context[:100]}...")
            
            for j, doc in enumerate(docs):
                doc_id = doc.get('doc_id', '')
                source = doc.get('source', '')
                print(f"  {j+1}. {doc_id} ({source})")
    else:
        print("✅ 知识库中没有重复的context")

def test_retriever_duplicates():
    """测试检索器返回的重复情况"""
    print("\n" + "=" * 80)
    print("测试检索器重复情况")
    print("=" * 80)
    
    try:
        from xlm.ui.optimized_rag_ui import OptimizedRagUI
        from config.parameters import Config
        
        # 初始化UI
        print("🔄 初始化UI...")
        config = Config()
        ui = OptimizedRagUI(
            cache_dir=config.encoder.cache_dir,
            use_faiss=config.retriever.use_faiss,
            enable_reranker=config.reranker.enabled,
            use_existing_embedding_index=config.retriever.use_existing_embedding_index,
            max_alphafin_chunks=config.retriever.max_alphafin_chunks
        )
        print("✅ UI初始化成功")
        
        # 测试英文查询
        test_questions = [
            "What is the revenue of Apple in 2023?",
            "What are the financial highlights?",
            "What is the net income?"
        ]
        
        for question in test_questions:
            print(f"\n🔍 测试查询: {question}")
            
            # 检测语言
            try:
                from langdetect import detect
                language = detect(question)
                is_chinese = language.startswith('zh')
            except:
                is_chinese = any('\u4e00' <= char <= '\u9fff' for char in question)
            
            print(f"检测语言: {'中文' if is_chinese else '英文'}")
            
            # 获取检索结果
            if is_chinese:
                if hasattr(ui, 'chinese_retrieval_system') and ui.chinese_retrieval_system:
                    # 中文检索系统返回的是(doc_idx, score)元组
                    faiss_results = ui.chinese_retrieval_system.retrieve(question, top_k=10)
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
                    continue
            else:
                if hasattr(ui, 'retriever'):
                    # 英文使用BilingualRetriever的retrieve方法
                    result = ui.retriever.retrieve(
                        text=question,
                        top_k=10,
                        return_scores=True
                    )
                    if isinstance(result, tuple):
                        retrieved_documents, retriever_scores = result
                    else:
                        retrieved_documents = result
                        retriever_scores = [1.0] * len(result) if result else []
                else:
                    print("❌ 检索器未初始化")
                    continue
            
            print(f"检索到 {len(retrieved_documents)} 个文档")
            
            # 检查重复内容
            content_hashes = set()
            duplicate_count = 0
            duplicate_contents = []
            
            for i, (doc, score) in enumerate(zip(retrieved_documents, retriever_scores)):
                if hasattr(doc, 'content'):
                    content = doc.content
                else:
                    content = str(doc)
                content_hash = hash(content)
                
                if content_hash in content_hashes:
                    duplicate_count += 1
                    duplicate_contents.append({
                        'index': i,
                        'score': score,
                        'content_preview': content[:100],
                        'content_hash': content_hash
                    })
                else:
                    content_hashes.add(content_hash)
            
            print(f"重复统计:")
            print(f"   总文档数: {len(retrieved_documents)}")
            print(f"   唯一文档数: {len(content_hashes)}")
            print(f"   重复文档数: {duplicate_count}")
            if retrieved_documents:
                print(f"   重复率: {duplicate_count/len(retrieved_documents)*100:.2f}%")
            else:
                print(f"   重复率: 0.00%")
            
            if duplicate_contents:
                print(f"重复内容详情:")
                for dup in duplicate_contents[:3]:  # 只显示前3个
                    print(f"   文档 {dup['index']+1}: 分数={dup['score']:.4f}, 内容={dup['content_preview']}...")
    
    except Exception as e:
        print(f"❌ 测试失败: {e}")
        import traceback
        traceback.print_exc()

def test_ui_processing_duplicates():
    """测试UI处理过程中的重复情况"""
    print("\n" + "=" * 80)
    print("测试UI处理重复情况")
    print("=" * 80)
    
    try:
        from xlm.ui.optimized_rag_ui import OptimizedRagUI
        from config.parameters import Config
        
        # 初始化UI
        print("🔄 初始化UI...")
        config = Config()
        ui = OptimizedRagUI(
            cache_dir=config.encoder.cache_dir,
            use_faiss=config.retriever.use_faiss,
            enable_reranker=config.reranker.enabled,
            use_existing_embedding_index=config.retriever.use_existing_embedding_index,
            max_alphafin_chunks=config.retriever.max_alphafin_chunks
        )
        print("✅ UI初始化成功")
        
        # 测试查询
        test_question = "What is the revenue of Apple in 2023?"
        print(f"\n🔍 测试查询: {test_question}")
        
        # 模拟UI处理流程
        try:
            # 检测语言
            from langdetect import detect
            language = detect(test_question)
            is_chinese = language.startswith('zh')
        except:
            is_chinese = any('\u4e00' <= char <= '\u9fff' for char in test_question)
        
        print(f"检测语言: {'中文' if is_chinese else '英文'}")
        
        # 获取检索结果
        if is_chinese:
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
            if hasattr(ui, 'retriever'):
                # 英文使用BilingualRetriever的retrieve方法
                result = ui.retriever.retrieve(
                    text=test_question,
                    top_k=10,
                    return_scores=True
                )
                if isinstance(result, tuple):
                    retrieved_documents, retriever_scores = result
                else:
                    retrieved_documents = result
                    retriever_scores = [1.0] * len(result) if result else []
            else:
                print("❌ 检索器未初始化")
                return
        
        print(f"检索到 {len(retrieved_documents)} 个文档")
        
        # 模拟UI去重逻辑
        print("\n🔄 模拟UI去重逻辑...")
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
        
        # 检查UI文档处理
        print("\n🔄 检查UI文档处理...")
        ui_docs = []
        seen_ui_hashes = set()
        seen_table_ids = set()
        seen_paragraph_ids = set()
        
        for doc, score in unique_docs:
            if getattr(doc.metadata, 'language', '') == 'chinese':
                doc_id = str(getattr(doc.metadata, 'origin_doc_id', '') or getattr(doc.metadata, 'doc_id', '')).strip()
                raw_context = ui.docid2context.get(doc_id, "")
                if not raw_context:
                    raw_context = doc.content
            else:
                raw_context = doc.content
            
            # 检查内容类型并应用相应的去重逻辑
            has_table_id = "Table ID:" in raw_context
            has_paragraph_id = "Paragraph ID:" in raw_context
            
            if has_table_id:
                import re
                table_id_match = re.search(r'Table ID:\s*([a-f0-9-]+)', raw_context)
                if table_id_match:
                    table_id = table_id_match.group(1)
                    if table_id in seen_table_ids:
                        print(f"跳过重复的Table ID: {table_id}")
                        continue
                    seen_table_ids.add(table_id)
            elif has_paragraph_id:
                import re
                paragraph_id_match = re.search(r'Paragraph ID:\s*([a-f0-9-]+)', raw_context)
                if paragraph_id_match:
                    paragraph_id = paragraph_id_match.group(1)
                    if paragraph_id in seen_paragraph_ids:
                        print(f"跳过重复的Paragraph ID: {paragraph_id}")
                        continue
                    seen_paragraph_ids.add(paragraph_id)
            
            # 对raw_context进行去重检查
            context_hash = hash(raw_context)
            if context_hash in seen_ui_hashes:
                print(f"跳过重复的UI文档")
                continue
            
            seen_ui_hashes.add(context_hash)
            preview_content = raw_context[:200] + "..." if len(raw_context) > 200 else raw_context
            ui_docs.append((doc, score, preview_content, raw_context))
        
        print(f"UI处理后的文档数: {len(ui_docs)}")
        print(f"Table ID去重: {len(seen_table_ids)} 个唯一Table ID")
        print(f"Paragraph ID去重: {len(seen_paragraph_ids)} 个唯一Paragraph ID")
        print(f"Context去重: {len(seen_ui_hashes)} 个唯一Context")
        
    except Exception as e:
        print(f"❌ 测试失败: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    # 1. 测试知识库重复
    test_knowledge_base_duplicates()
    
    # 2. 测试检索器重复
    test_retriever_duplicates()
    
    # 3. 测试UI处理重复
    test_ui_processing_duplicates()
    
    print("\n" + "=" * 80)
    print("测试完成")
    print("=" * 80) 