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

        # 初始化嵌入向量为空数组，确保即使文档为空也有有效状态
        if self.corpus_documents_en is None:
            self.corpus_documents_en = []
        if self.corpus_documents_ch is None:
            self.corpus_documents_ch = []
            
        print(f"初始化状态: 英文文档 {len(self.corpus_documents_en)} 个, 中文文档 {len(self.corpus_documents_ch)} 个")
        
        # 智能缓存加载：优先尝试缓存，失败时自动回退到重新计算
        if self.use_existing_embedding_index:
            try:
                print("🔄 尝试加载现有缓存...")
                loaded = self._load_cached_embeddings()
                if loaded:
                    print("✅ 缓存加载成功")
                    
                    # 验证加载的嵌入向量是否有效
                    if self._validate_loaded_embeddings():
                        print("✅ 嵌入向量验证通过")
                    else:
                        print("⚠️ 嵌入向量验证失败，重新计算...")
                        self._compute_embeddings()
                else:
                    print("⚠️ 缓存加载失败，重新计算embedding...")
                    self._compute_embeddings()
                    
            except Exception as e:
                print(f"❌ 缓存加载过程中发生错误: {e}")
                print("🔄 自动回退到重新计算embedding...")
                self._compute_embeddings()
        else:
            print("🔄 强制重新计算embedding...")
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

    def _is_cache_valid(self, documents: List[DocumentWithMetadata], cache_key: str) -> bool:
        """检查缓存是否有效（数据是否发生变化）"""
        try:
            # 检查缓存文件是否存在
            embeddings_path = self._get_cache_path(cache_key, "npy")
            index_path = self._get_cache_path(cache_key, "faiss")
            
            if not os.path.exists(embeddings_path) or not os.path.exists(index_path):
                print(f"⚠️ 缓存文件不存在，需要重新生成")
                return False
            
            # 检查嵌入向量维度是否匹配
            if os.path.exists(embeddings_path):
                try:
                    cached_embeddings = np.load(embeddings_path)
                    if cached_embeddings.shape[0] != len(documents):
                        print(f"⚠️ 文档数量不匹配: 缓存={cached_embeddings.shape[0]}, 当前={len(documents)}")
                        return False
                    
                    # 检查嵌入向量是否为空或无效
                    if cached_embeddings.size == 0:
                        print(f"⚠️ 缓存的嵌入向量为空")
                        return False
                        
                except Exception as e:
                    print(f"⚠️ 嵌入向量缓存读取失败: {e}")
                    return False
            
            # 检查FAISS索引是否有效
            if os.path.exists(index_path):
                try:
                    index = faiss.read_index(index_path)
                    if hasattr(index, 'ntotal') and index.ntotal != len(documents):
                        print(f"⚠️ FAISS索引大小不匹配: 缓存={index.ntotal}, 当前={len(documents)}")
                        return False
                        
                    # 检查FAISS索引是否为空
                    if hasattr(index, 'ntotal') and index.ntotal == 0:
                        print(f"⚠️ FAISS索引为空")
                        return False
                        
                except Exception as e:
                    print(f"⚠️ FAISS索引读取失败: {e}")
                    return False
            
            print(f"✅ 缓存验证通过")
            return True
            
        except Exception as e:
            print(f"⚠️ 缓存验证失败: {e}")
            return False

    def _clear_invalid_cache(self, cache_key: str):
        """清除无效的缓存文件"""
        try:
            embeddings_path = self._get_cache_path(cache_key, "npy")
            index_path = self._get_cache_path(cache_key, "faiss")
            
            if os.path.exists(embeddings_path):
                os.remove(embeddings_path)
                print(f"🗑️ 删除无效嵌入向量缓存: {embeddings_path}")
            
            if os.path.exists(index_path):
                os.remove(index_path)
                print(f"🗑️ 删除无效FAISS索引缓存: {index_path}")
                
        except Exception as e:
            print(f"⚠️ 清除缓存失败: {e}")
    
    def _validate_loaded_embeddings(self) -> bool:
        """验证加载的嵌入向量是否有效"""
        try:
            # 验证英文嵌入向量
            if self.corpus_documents_en:
                if self.corpus_embeddings_en is None:
                    print("❌ 英文嵌入向量为None")
                    return False
                
                if self.corpus_embeddings_en.size == 0:
                    print("❌ 英文嵌入向量为空")
                    return False
                
                if self.corpus_embeddings_en.shape[0] != len(self.corpus_documents_en):
                    print(f"❌ 英文嵌入向量维度不匹配: {self.corpus_embeddings_en.shape[0]} != {len(self.corpus_documents_en)}")
                    return False
                
                print(f"✅ 英文嵌入向量有效: {self.corpus_embeddings_en.shape}")
            
            # 验证中文嵌入向量
            if self.corpus_documents_ch:
                if self.corpus_embeddings_ch is None:
                    print("❌ 中文嵌入向量为None")
                    return False
                
                if self.corpus_embeddings_ch.size == 0:
                    print("❌ 中文嵌入向量为空")
                    return False
                
                if self.corpus_embeddings_ch.shape[0] != len(self.corpus_documents_ch):
                    print(f"❌ 中文嵌入向量维度不匹配: {self.corpus_embeddings_ch.shape[0]} != {len(self.corpus_documents_ch)}")
                    return False
                
                print(f"✅ 中文嵌入向量有效: {self.corpus_embeddings_ch.shape}")
            
            # 验证FAISS索引
            if self.use_faiss:
                if self.corpus_documents_en and self.index_en:
                    if not hasattr(self.index_en, 'ntotal') or self.index_en.ntotal == 0:
                        print("❌ 英文FAISS索引为空")
                        return False
                    print(f"✅ 英文FAISS索引有效: {self.index_en.ntotal} 个文档")
                
                if self.corpus_documents_ch and self.index_ch:
                    if not hasattr(self.index_ch, 'ntotal') or self.index_ch.ntotal == 0:
                        print("❌ 中文FAISS索引为空")
                        return False
                    print(f"✅ 中文FAISS索引有效: {self.index_ch.ntotal} 个文档")
            
            return True
            
        except Exception as e:
            print(f"❌ 嵌入向量验证失败: {e}")
            return False

    def _load_cached_embeddings(self) -> bool:
        """尝试加载缓存的嵌入向量，自动检测数据变化"""
        try:
            loaded_any = False
            
            # 检查英文文档缓存
            if self.corpus_documents_en:
                cache_key_en = self._get_cache_key(self.corpus_documents_en, str(self.encoder_en.model_name))
                print(f"🔍 检查英文缓存: {cache_key_en}")
                
                try:
                    if self._is_cache_valid(self.corpus_documents_en, cache_key_en):
                        embeddings_path_en = self._get_cache_path(cache_key_en, "npy")
                        index_path_en = self._get_cache_path(cache_key_en, "faiss")
                        
                        # 尝试加载嵌入向量
                        try:
                            self.corpus_embeddings_en = np.load(embeddings_path_en)
                            print(f"✅ 英文嵌入向量加载成功，形状: {self.corpus_embeddings_en.shape}")
                            loaded_any = True
                        except Exception as e:
                            print(f"❌ 英文嵌入向量加载失败: {e}")
                            self.corpus_embeddings_en = np.array([])
                        
                        # 尝试加载FAISS索引
                        if self.use_faiss and os.path.exists(index_path_en):
                            try:
                                self.index_en = faiss.read_index(index_path_en)
                                print(f"✅ 英文FAISS索引加载成功，文档数: {len(self.corpus_documents_en)}")
                            except Exception as e:
                                print(f"❌ 英文FAISS索引加载失败: {e}")
                                self.index_en = None
                    else:
                        # 清除无效缓存
                        self._clear_invalid_cache(cache_key_en)
                        print(f"🔄 英文数据发生变化，需要重新生成索引")
                        self.corpus_embeddings_en = np.array([])
                        self.index_en = None
                        
                except Exception as e:
                    print(f"❌ 英文缓存验证失败: {e}")
                    self._clear_invalid_cache(cache_key_en)
                    self.corpus_embeddings_en = np.array([])
                    self.index_en = None
            else:
                print("⚠️ 英文文档列表为空，初始化空嵌入向量")
                self.corpus_embeddings_en = np.array([])
                if self.use_faiss:
                    self.index_en = self._init_faiss(self.encoder_en, 0)

            # 检查中文文档缓存
            if self.corpus_documents_ch:
                cache_key_ch = self._get_cache_key(self.corpus_documents_ch, str(self.encoder_ch.model_name))
                print(f"🔍 检查中文缓存: {cache_key_ch}")
                
                try:
                    if self._is_cache_valid(self.corpus_documents_ch, cache_key_ch):
                        embeddings_path_ch = self._get_cache_path(cache_key_ch, "npy")
                        index_path_ch = self._get_cache_path(cache_key_ch, "faiss")
                        
                        # 尝试加载嵌入向量
                        try:
                            self.corpus_embeddings_ch = np.load(embeddings_path_ch)
                            print(f"✅ 中文嵌入向量加载成功，形状: {self.corpus_embeddings_ch.shape}")
                            loaded_any = True
                        except Exception as e:
                            print(f"❌ 中文嵌入向量加载失败: {e}")
                            self.corpus_embeddings_ch = np.array([])
                        
                        # 尝试加载FAISS索引
                        if self.use_faiss and os.path.exists(index_path_ch):
                            try:
                                self.index_ch = faiss.read_index(index_path_ch)
                                print(f"✅ 中文FAISS索引加载成功，文档数: {len(self.corpus_documents_ch)}")
                            except Exception as e:
                                print(f"❌ 中文FAISS索引加载失败: {e}")
                                self.index_ch = None
                    else:
                        # 清除无效缓存
                        self._clear_invalid_cache(cache_key_ch)
                        print(f"🔄 中文数据发生变化，需要重新生成索引")
                        self.corpus_embeddings_ch = np.array([])
                        self.index_ch = None
                        
                except Exception as e:
                    print(f"❌ 中文缓存验证失败: {e}")
                    self._clear_invalid_cache(cache_key_ch)
                    self.corpus_embeddings_ch = np.array([])
                    self.index_ch = None
            else:
                print("⚠️ 中文文档列表为空，初始化空嵌入向量")
                self.corpus_embeddings_ch = np.array([])
                if self.use_faiss:
                    self.index_ch = self._init_faiss(self.encoder_ch, 0)

            return loaded_any
        except Exception as e:
            print(f"❌ 缓存加载失败: {e}")
            # 确保嵌入向量不为None
            if self.corpus_embeddings_en is None:
                self.corpus_embeddings_en = np.array([])
            if self.corpus_embeddings_ch is None:
                self.corpus_embeddings_ch = np.array([])
            return False

    def _save_cached_embeddings(self):
        """保存嵌入向量到缓存"""
        try:
            # 保存英文文档嵌入向量
            if self.corpus_documents_en and self.corpus_embeddings_en is not None and self.corpus_embeddings_en.size > 0:
                try:
                    cache_key_en = self._get_cache_key(self.corpus_documents_en, str(self.encoder_en.model_name))
                    embeddings_path_en = self._get_cache_path(cache_key_en, "npy")
                    index_path_en = self._get_cache_path(cache_key_en, "faiss")
                    
                    # 确保目录存在
                    os.makedirs(os.path.dirname(embeddings_path_en), exist_ok=True)
                    os.makedirs(os.path.dirname(index_path_en), exist_ok=True)
                    
                    # 保存嵌入向量
                    np.save(embeddings_path_en, self.corpus_embeddings_en)
                    print(f"✅ 英文嵌入向量已保存: {embeddings_path_en}")
                    
                    # 保存FAISS索引
                    if self.use_faiss and self.index_en:
                        faiss.write_index(self.index_en, index_path_en)
                        print(f"✅ 英文FAISS索引已保存: {index_path_en}")
                        
                except Exception as e:
                    print(f"❌ 英文缓存保存失败: {e}")

            # 保存中文文档嵌入向量
            if self.corpus_documents_ch and self.corpus_embeddings_ch is not None and self.corpus_embeddings_ch.size > 0:
                try:
                    cache_key_ch = self._get_cache_key(self.corpus_documents_ch, str(self.encoder_ch.model_name))
                    embeddings_path_ch = self._get_cache_path(cache_key_ch, "npy")
                    index_path_ch = self._get_cache_path(cache_key_ch, "faiss")
                    
                    # 确保目录存在
                    os.makedirs(os.path.dirname(embeddings_path_ch), exist_ok=True)
                    os.makedirs(os.path.dirname(index_path_ch), exist_ok=True)
                    
                    # 保存嵌入向量
                    np.save(embeddings_path_ch, self.corpus_embeddings_ch)
                    print(f"✅ 中文嵌入向量已保存: {embeddings_path_ch}")
                    
                    # 保存FAISS索引
                    if self.use_faiss and self.index_ch:
                        faiss.write_index(self.index_ch, index_path_ch)
                        print(f"✅ 中文FAISS索引已保存: {index_path_ch}")
                        
                except Exception as e:
                    print(f"❌ 中文缓存保存失败: {e}")
                    
        except Exception as e:
            print(f"❌ 缓存保存过程中发生错误: {e}")

    def _compute_embeddings(self):
        """计算嵌入向量"""
        print("=== 开始计算嵌入向量 ===")
        print(f"use_existing_embedding_index: {self.use_existing_embedding_index}")
        
        try:
            # 检查是否需要初始化FAISS索引
            if self.use_faiss:
                if self.corpus_documents_en and self.index_en is None:
                    print(f"初始化英文FAISS索引，文档数量: {len(self.corpus_documents_en)}")
                    self.index_en = self._init_faiss(self.encoder_en, len(self.corpus_documents_en))
                if self.corpus_documents_ch and self.index_ch is None:
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
                    self.corpus_embeddings_en = np.array([])  # 确保不为None
                else:
                    print("✅ 英文嵌入向量计算成功！")
                    
                if self.use_faiss and self.corpus_embeddings_en is not None and self.corpus_embeddings_en.shape[0] > 0:
                    print("将英文嵌入向量添加到FAISS索引")
                    self._add_to_faiss(self.index_en, self.corpus_embeddings_en)
            else:
                print("⚠️ 英文文档列表为空，初始化空嵌入向量")
                self.corpus_embeddings_en = np.array([])
                if self.use_faiss and self.index_en is None:
                    self.index_en = self._init_faiss(self.encoder_en, 0)

            if self.corpus_documents_ch:
                print(f"开始编码中文文档，数量: {len(self.corpus_documents_ch)}")
                self.corpus_embeddings_ch = self._batch_encode_corpus(self.corpus_documents_ch, self.encoder_ch, 'zh')
                print(f"中文嵌入向量计算完成，形状: {self.corpus_embeddings_ch.shape if self.corpus_embeddings_ch is not None else 'None'}")
                
                if self.corpus_embeddings_ch is None or self.corpus_embeddings_ch.shape[0] == 0:
                    print("❌ 中文嵌入向量计算失败！")
                    self.corpus_embeddings_ch = np.array([])  # 确保不为None
                else:
                    print("✅ 中文嵌入向量计算成功！")
                    
                if self.use_faiss and self.corpus_embeddings_ch is not None and self.corpus_embeddings_ch.shape[0] > 0:
                    print("将中文嵌入向量添加到FAISS索引")
                    self._add_to_faiss(self.index_ch, self.corpus_embeddings_ch)
            else:
                print("⚠️ 中文文档列表为空，初始化空嵌入向量")
                self.corpus_embeddings_ch = np.array([])
                if self.use_faiss and self.index_ch is None:
                    self.index_ch = self._init_faiss(self.encoder_ch, 0)
            
            # 保存到缓存
            print("保存嵌入向量到缓存")
            try:
                self._save_cached_embeddings()
            except Exception as e:
                print(f"⚠️ 缓存保存失败: {e}")
            
            print("=== 嵌入向量计算完成 ===")
            print(f"最终状态:")
            print(f"  英文嵌入向量: {self.corpus_embeddings_en.shape if self.corpus_embeddings_en is not None else 'None'}")
            print(f"  中文嵌入向量: {self.corpus_embeddings_ch.shape if self.corpus_embeddings_ch is not None else 'None'}")
            
        except Exception as e:
            print(f"❌ 嵌入向量计算过程中发生错误: {e}")
            # 确保嵌入向量不为None
            if self.corpus_embeddings_en is None:
                self.corpus_embeddings_en = np.array([])
            if self.corpus_embeddings_ch is None:
                self.corpus_embeddings_ch = np.array([])
            print("🔄 已重置嵌入向量为空数组")

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
        print(f"文档数量: {len(documents) if documents else 0}")
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
        
        # 添加索引范围检查
        print(f"检索到的索引: {doc_indices}")
        print(f"语料库文档数量: {len(corpus_documents)}")
        
        # 过滤无效索引
        valid_indices = []
        valid_scores = []
        for i, (idx, score) in enumerate(zip(doc_indices, scores)):
            if 0 <= idx < len(corpus_documents):
                valid_indices.append(idx)
                valid_scores.append(score)
            else:
                print(f"警告: 跳过无效索引 {idx} (超出范围 [0, {len(corpus_documents)})")
        
        raw_documents = [corpus_documents[i] for i in valid_indices]
        scores = valid_scores
        
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