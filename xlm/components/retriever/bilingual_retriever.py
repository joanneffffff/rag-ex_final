from typing import List, Dict, Tuple, Union
import numpy as np
import torch
import faiss
import os
import pickle
import hashlib
from sentence_transformers.util import semantic_search
from langdetect import detect
from tqdm import tqdm

from xlm.components.encoder.finbert import FinbertEncoder
from xlm.components.retriever.retriever import Retriever
from xlm.dto.dto import DocumentWithMetadata, DocumentMetadata

class BilingualRetriever(Retriever):
    def __init__(
        self,
        encoder_en: FinbertEncoder,
        encoder_ch: FinbertEncoder,
        max_context_length: int = 100,
        num_threads: int = 4,
        corpus_documents_en: List[DocumentWithMetadata] = None,
        corpus_documents_ch: List[DocumentWithMetadata] = None,
        use_faiss: bool = False,
        batch_size: int = 32,
        use_gpu: bool = False,
        cache_dir: str = "cache",
        use_existing_embedding_index: bool = False
    ):
        self.encoder_en = encoder_en
        self.encoder_ch = encoder_ch
        self.max_context_length = max_context_length
        self.__num_threads = num_threads
        self.use_faiss = use_faiss
        self.batch_size = batch_size
        self.use_gpu = use_gpu
        self.cache_dir = cache_dir
        self.use_existing_embedding_index = use_existing_embedding_index

        self.corpus_documents_en = corpus_documents_en or []
        self.corpus_embeddings_en = None
        self.index_en = None

        self.corpus_documents_ch = corpus_documents_ch or []
        self.corpus_embeddings_ch = None
        self.index_ch = None

        # 创建缓存目录
        os.makedirs(self.cache_dir, exist_ok=True)

        # 只在use_existing_embedding_index为True时尝试加载缓存，否则强制重新编码
        if self.use_existing_embedding_index:
            loaded = self._load_cached_embeddings()
            if loaded:
                print("Loaded cached embeddings successfully.")
            else:
                print("未找到有效缓存，自动重新计算embedding...")
                self.use_existing_embedding_index = False
                self._compute_embeddings()
        else:
            self._compute_embeddings()

        # 验证文档类型
        if self.corpus_documents_en:
            for i, doc in enumerate(self.corpus_documents_en):
                if not isinstance(doc, DocumentWithMetadata):
                    print(f"警告: corpus_documents_en[{i}]不是DocumentWithMetadata类型: {type(doc)}")
                elif not hasattr(doc, 'content') or not isinstance(doc.content, str):
                    print(f"警告: corpus_documents_en[{i}]的content字段类型错误: {type(getattr(doc, 'content', None))}")
                    
        if self.corpus_documents_ch:
            for i, doc in enumerate(self.corpus_documents_ch):
                if not isinstance(doc, DocumentWithMetadata):
                    print(f"警告: corpus_documents_ch[{i}]不是DocumentWithMetadata类型: {type(doc)}")
                elif not hasattr(doc, 'content') or not isinstance(doc.content, str):
                    print(f"警告: corpus_documents_ch[{i}]的content字段类型错误: {type(getattr(doc, 'content', None))}")

    def _get_cache_key(self, documents: List[DocumentWithMetadata], encoder_name: str) -> str:
        """生成缓存键，基于文档内容和编码器名称"""
        # 创建文档内容的哈希
        content_hash = hashlib.md5()
        for doc in documents:
            content_hash.update(doc.content.encode('utf-8'))
        
        # 只使用编码器名称的最后部分，避免路径问题
        encoder_basename = os.path.basename(encoder_name)
        
        # 结合编码器名称和文档数量
        cache_key = f"{encoder_basename}_{len(documents)}_{content_hash.hexdigest()[:16]}"
        return cache_key

    def _get_cache_path(self, cache_key: str, suffix: str) -> str:
        """获取缓存文件路径"""
        return os.path.join(self.cache_dir, f"{cache_key}.{suffix}")

    def _load_cached_embeddings(self) -> bool:
        """尝试加载缓存的嵌入向量"""
        try:
            loaded_any = False
            
            # 检查英文文档缓存
            if self.corpus_documents_en:
                cache_key_en = self._get_cache_key(self.corpus_documents_en, str(self.encoder_en.model_name))
                embeddings_path_en = self._get_cache_path(cache_key_en, "npy")
                index_path_en = self._get_cache_path(cache_key_en, "faiss")
                
                if os.path.exists(embeddings_path_en):
                    self.corpus_embeddings_en = np.load(embeddings_path_en)
                    loaded_any = True
                    
                    if self.use_faiss and os.path.exists(index_path_en):
                        self.index_en = faiss.read_index(index_path_en)
                    elif self.use_faiss:
                        self.index_en = self._init_faiss(self.encoder_en, len(self.corpus_documents_en))
                        if self.corpus_embeddings_en is not None:
                            self._add_to_faiss(self.index_en, self.corpus_embeddings_en)
            else:
                # 英文文档为空，不加载任何缓存文件
                # 因为无法验证缓存是否与当前数据匹配
                print("⚠️ 英文文档列表为空，跳过缓存加载以确保数据一致性")
                loaded_any = False

            # 检查中文文档缓存
            if self.corpus_documents_ch:
                cache_key_ch = self._get_cache_key(self.corpus_documents_ch, str(self.encoder_ch.model_name))
                embeddings_path_ch = self._get_cache_path(cache_key_ch, "npy")
                index_path_ch = self._get_cache_path(cache_key_ch, "faiss")
                
                if os.path.exists(embeddings_path_ch):
                    self.corpus_embeddings_ch = np.load(embeddings_path_ch)
                    loaded_any = True
                    
                    if self.use_faiss and os.path.exists(index_path_ch):
                        self.index_ch = faiss.read_index(index_path_ch)
                    elif self.use_faiss:
                        self.index_ch = self._init_faiss(self.encoder_ch, len(self.corpus_documents_ch))
                        if self.corpus_embeddings_ch is not None:
                            self._add_to_faiss(self.index_ch, self.corpus_embeddings_ch)
            else:
                # 中文文档为空，不加载任何缓存文件
                # 因为无法验证缓存是否与当前数据匹配
                print("⚠️ 中文文档列表为空，跳过缓存加载以确保数据一致性")
                loaded_any = False

            return loaded_any
        except Exception as e:
            return False

    def _save_cached_embeddings(self):
        """保存嵌入向量到缓存"""
        try:
            # 保存英文文档嵌入向量
            if self.corpus_documents_en and self.corpus_embeddings_en is not None:
                cache_key_en = self._get_cache_key(self.corpus_documents_en, str(self.encoder_en.model_name))
                embeddings_path_en = self._get_cache_path(cache_key_en, "npy")
                index_path_en = self._get_cache_path(cache_key_en, "faiss")
                # 确保目录存在
                os.makedirs(os.path.dirname(embeddings_path_en), exist_ok=True)
                os.makedirs(os.path.dirname(index_path_en), exist_ok=True)
                np.save(embeddings_path_en, self.corpus_embeddings_en)
                if self.use_faiss and self.index_en:
                    faiss.write_index(self.index_en, index_path_en)

            # 保存中文文档嵌入向量
            if self.corpus_documents_ch and self.corpus_embeddings_ch is not None:
                cache_key_ch = self._get_cache_key(self.corpus_documents_ch, str(self.encoder_ch.model_name))
                embeddings_path_ch = self._get_cache_path(cache_key_ch, "npy")
                index_path_ch = self._get_cache_path(cache_key_ch, "faiss")
                # 确保目录存在
                os.makedirs(os.path.dirname(embeddings_path_ch), exist_ok=True)
                os.makedirs(os.path.dirname(index_path_ch), exist_ok=True)
                np.save(embeddings_path_ch, self.corpus_embeddings_ch)
                if self.use_faiss and self.index_ch:
                    faiss.write_index(self.index_ch, index_path_ch)
        except Exception as e:
            pass

    def _compute_embeddings(self):
        """计算嵌入向量"""
        print("=== 开始计算嵌入向量 ===")
        print(f"use_existing_embedding_index: {self.use_existing_embedding_index}")
        
        if self.use_faiss:
            if self.corpus_documents_en:
                print(f"初始化英文FAISS索引，文档数量: {len(self.corpus_documents_en)}")
                self.index_en = self._init_faiss(self.encoder_en, len(self.corpus_documents_en))
            if self.corpus_documents_ch:
                print(f"初始化中文FAISS索引，文档数量: {len(self.corpus_documents_ch)}")
                self.index_ch = self._init_faiss(self.encoder_ch, len(self.corpus_documents_ch))

        if self.corpus_documents_en:
            print(f"开始编码英文文档，数量: {len(self.corpus_documents_en)}")
            print(f"英文编码器: {self.encoder_en.model_name}")
            print(f"英文编码器设备: {self.encoder_en.device}")
            
            # 检查英文文档内容
            if self.corpus_documents_en:
                first_doc = self.corpus_documents_en[0]
                print(f"第一个英文文档内容预览: {first_doc.content[:100]}...")
            
            self.corpus_embeddings_en = self._batch_encode_corpus(self.corpus_documents_en, self.encoder_en, 'en')
            print(f"英文嵌入向量计算完成，形状: {self.corpus_embeddings_en.shape if self.corpus_embeddings_en is not None else 'None'}")
            
            if self.corpus_embeddings_en is None or self.corpus_embeddings_en.shape[0] == 0:
                print("❌ 英文嵌入向量计算失败！")
            else:
                print("✅ 英文嵌入向量计算成功！")
                
            if self.use_faiss and self.corpus_embeddings_en is not None:
                print("将英文嵌入向量添加到FAISS索引")
                self._add_to_faiss(self.index_en, self.corpus_embeddings_en)
        else:
            print("⚠️ 英文文档列表为空")

        if self.corpus_documents_ch:
            print(f"开始编码中文文档，数量: {len(self.corpus_documents_ch)}")
            self.corpus_embeddings_ch = self._batch_encode_corpus(self.corpus_documents_ch, self.encoder_ch, 'zh')
            print(f"中文嵌入向量计算完成，形状: {self.corpus_embeddings_ch.shape if self.corpus_embeddings_ch is not None else 'None'}")
            if self.use_faiss and self.corpus_embeddings_ch is not None:
                print("将中文嵌入向量添加到FAISS索引")
                self._add_to_faiss(self.index_ch, self.corpus_embeddings_ch)
        else:
            print("⚠️ 中文文档列表为空")
        
        # 保存到缓存
        print("保存嵌入向量到缓存")
        self._save_cached_embeddings()
        
        print("=== 嵌入向量计算完成 ===")
        print(f"最终状态:")
        print(f"  英文嵌入向量: {self.corpus_embeddings_en.shape if self.corpus_embeddings_en is not None else 'None'}")
        print(f"  中文嵌入向量: {self.corpus_embeddings_ch.shape if self.corpus_embeddings_ch is not None else 'None'}")
        pass

    def _init_faiss(self, encoder, corpus_size):
        """Initialize FAISS index"""
        dimension = encoder.get_embedding_dimension()
        
        # Create FAISS index
        if self.use_gpu and faiss.get_num_gpus() > 0:
            res = faiss.StandardGpuResources()
            index = faiss.IndexFlatL2(dimension)
            gpu_index = faiss.index_cpu_to_gpu(res, 0, index)
            return gpu_index
        else:
            if corpus_size < 1000:
                return faiss.IndexFlatL2(dimension)
            else:
                nlist = min(max(int(corpus_size / 100), 4), 1024)
                quantizer = faiss.IndexFlatL2(dimension)
                index = faiss.IndexIVFFlat(quantizer, dimension, nlist)
                return index

    def _batch_encode_corpus(self, documents: List[DocumentWithMetadata], encoder: FinbertEncoder, language: str = None) -> np.ndarray:
        """Encode corpus documents in batches with a progress bar."""
        print(f"=== 开始批量编码语料库 ===")
        print(f"语言: {language}")
        print(f"文档数量: {len(documents)}")
        print(f"编码器: {encoder.model_name}")
        print(f"编码器设备: {encoder.device}")
        
        if not documents:
            print("⚠️ 文档列表为空，返回空数组")
            return np.array([])
        
        batch_texts = []
        for i, doc in enumerate(documents):
            if language == 'zh' and hasattr(doc.metadata, 'summary') and doc.metadata.summary:
                # 中文使用summary字段进行FAISS索引
                batch_texts.append(doc.metadata.summary)
            else:
                # 英文使用content字段
                batch_texts.append(doc.content)
            
            # 打印前几个文档的内容预览
            if i < 3:
                content_preview = batch_texts[-1][:100] + "..." if len(batch_texts[-1]) > 100 else batch_texts[-1]
                print(f"文档 {i+1} 内容预览: {content_preview}")
        
        print(f"准备编码 {len(batch_texts)} 个文本")
        
        # 测试编码器是否正常工作
        try:
            print("测试编码器...")
            test_text = batch_texts[0] if batch_texts else "test"
            test_embedding = encoder.encode([test_text])
            print(f"测试编码成功，嵌入维度: {test_embedding.shape}")
        except Exception as e:
            print(f"❌ 编码器测试失败: {e}")
            import traceback
            traceback.print_exc()
            return np.array([])
        
        try:
            print("开始批量编码...")
            embeddings = encoder.encode(texts=batch_texts, batch_size=self.batch_size, show_progress_bar=True)
            print(f"编码完成，嵌入向量形状: {embeddings.shape}")
            return embeddings
        except Exception as e:
            print(f"❌ 批量编码过程中发生错误: {e}")
            import traceback
            traceback.print_exc()
            return np.array([])
    
    def _add_to_faiss(self, index, embeddings: np.ndarray):
        """Add embeddings to FAISS index in batches"""
        if embeddings is None or embeddings.shape[0] == 0:
            return
        
        embeddings_np = embeddings.astype('float32')
        if not index.is_trained:
            index.train(embeddings_np)
        index.add(embeddings_np)

    def retrieve(
        self,
        text: str,
        top_k: int = 3,
        return_scores: bool = False,
        language: str = None,
    ) -> Union[List[DocumentWithMetadata], Tuple[List[DocumentWithMetadata], List[float]]]:
        # 添加详细的调试信息
        print(f"=== BilingualRetriever.retrieve() 调试信息 ===")
        print(f"查询文本: {text}")
        print(f"英文文档数量: {len(self.corpus_documents_en) if self.corpus_documents_en else 0}")
        print(f"中文文档数量: {len(self.corpus_documents_ch) if self.corpus_documents_ch else 0}")
        print(f"英文嵌入向量形状: {self.corpus_embeddings_en.shape if self.corpus_embeddings_en is not None else 'None'}")
        print(f"中文嵌入向量形状: {self.corpus_embeddings_ch.shape if self.corpus_embeddings_ch is not None else 'None'}")
        print(f"英文FAISS索引: {'已初始化' if self.index_en else '未初始化'}")
        print(f"中文FAISS索引: {'已初始化' if self.index_ch else '未初始化'}")
        
        # 检查self.corpus_documents_en/ch类型
        if hasattr(self, 'corpus_documents_en') and self.corpus_documents_en is not None:
            for i, doc in enumerate(self.corpus_documents_en):
                if not isinstance(doc, DocumentWithMetadata):
                    print(f"警告: corpus_documents_en[{i}]不是DocumentWithMetadata类型: {type(doc)}")
                elif not hasattr(doc, 'content') or not isinstance(doc.content, str):
                    print(f"警告: corpus_documents_en[{i}]的content字段类型错误: {type(getattr(doc, 'content', None))}")
                    
        if hasattr(self, 'corpus_documents_ch') and self.corpus_documents_ch is not None:
            for i, doc in enumerate(self.corpus_documents_ch):
                if not isinstance(doc, DocumentWithMetadata):
                    print(f"警告: corpus_documents_ch[{i}]不是DocumentWithMetadata类型: {type(doc)}")
                elif not hasattr(doc, 'content') or not isinstance(doc.content, str):
                    print(f"警告: corpus_documents_ch[{i}]的content字段类型错误: {type(getattr(doc, 'content', None))}")
                    
        if language is None:
            # 增强的语言检测逻辑
            try:
                lang = detect(text)
                # 检查是否包含中文字符
                chinese_chars = sum(1 for char in text if '\u4e00' <= char <= '\u9fff')
                total_chars = len([char for char in text if char.isalpha() or '\u4e00' <= char <= '\u9fff'])
                
                # 如果包含中文字符且中文比例超过30%，或者langdetect检测为中文，则认为是中文
                if chinese_chars > 0 and (chinese_chars / total_chars > 0.3 or lang.startswith('zh')):
                    language = 'zh'
                else:
                    language = 'en'
            except:
                # 如果langdetect失败，使用字符检测
                chinese_chars = sum(1 for char in text if '\u4e00' <= char <= '\u9fff')
                language = 'zh' if chinese_chars > 0 else 'en'
        
        print(f"检测到的语言: {language}")
        
        if language == 'zh':
            encoder = self.encoder_ch
            corpus_embeddings = self.corpus_embeddings_ch
            corpus_documents = self.corpus_documents_ch
            index = self.index_ch
        else:
            encoder = self.encoder_en
            corpus_embeddings = self.corpus_embeddings_en
            corpus_documents = self.corpus_documents_en
            index = self.index_en
        
        print(f"选择的编码器: {encoder.model_name}")
        print(f"选择的语料库文档数量: {len(corpus_documents) if corpus_documents else 0}")
        print(f"选择的嵌入向量形状: {corpus_embeddings.shape if corpus_embeddings is not None else 'None'}")
        
        if corpus_embeddings is None or corpus_embeddings.shape[0] == 0:
            print("❌ 嵌入向量为空或形状为0，返回空结果")
            if return_scores:
                return [], []
            else:
                return []
        
        query_embeddings = encoder.encode([text])
        print(f"查询嵌入向量形状: {query_embeddings.shape}")
        
        if self.use_faiss and index:
            print("使用FAISS索引进行检索")
            distances, indices = index.search(query_embeddings.astype('float32'), top_k)
            results = []
            for score, idx in zip(distances[0], indices[0]):
                if idx != -1:
                    results.append({'corpus_id': idx, 'score': 1 - score / 2})
            print(f"FAISS检索结果数量: {len(results)}")
        else:
            print("使用语义搜索进行检索")
            # 确保tensor在正确的设备上
            query_tensor = torch.tensor(query_embeddings, device=encoder.device)
            corpus_tensor = torch.tensor(corpus_embeddings, device=encoder.device)
            hits = semantic_search(
                query_tensor,
                corpus_tensor,
                top_k=top_k
            )
            results = hits[0]
            print(f"语义搜索结果数量: {len(results)}")
        
        doc_indices = [hit['corpus_id'] for hit in results]
        scores = [hit['score'] for hit in results]
        raw_documents = [corpus_documents[i] for i in doc_indices]
        
        print(f"最终返回文档数量: {len(raw_documents)}")
        
        # 确保返回的是DocumentWithMetadata对象，统一使用content字段
        documents = []
        for i, doc in enumerate(raw_documents):
            if isinstance(doc, DocumentWithMetadata):
                # 已经是正确的类型，直接使用
                documents.append(doc)
            elif isinstance(doc, dict):
                # 如果是字典，转换为DocumentWithMetadata
                content = doc.get('content', doc.get('context', ''))
                if not isinstance(content, str):
                    # 如果content不是字符串，尝试取context字段或转为字符串
                    content = content.get('context', '') if isinstance(content, dict) and 'context' in content else str(content)
                metadata = DocumentMetadata(
                    source=doc.get('source', 'unknown'),
                    created_at=doc.get('created_at', ''),
                    author=doc.get('author', ''),
                    language=language or 'unknown'
                )
                documents.append(DocumentWithMetadata(content=content, metadata=metadata))
            else:
                # 其他类型，尝试转换为字符串
                print(f"警告: 检索结果[{i}]类型异常: {type(doc)}, 尝试转换")
                content = str(doc) if doc is not None else ""
                metadata = DocumentMetadata(
                    source="unknown",
                    created_at="",
                    author="",
                    language=language or 'unknown'
                )
                documents.append(DocumentWithMetadata(content=content, metadata=metadata))
        
        if return_scores:
            return documents, scores
        else:
            return documents 