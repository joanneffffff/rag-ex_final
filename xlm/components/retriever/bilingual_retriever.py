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

        # Create cache directory
        os.makedirs(self.cache_dir, exist_ok=True)

        # Initialize embedding vectors as empty arrays, ensure valid state even when documents are empty
        if self.corpus_documents_en is None:
            self.corpus_documents_en = []
        if self.corpus_documents_ch is None:
            self.corpus_documents_ch = []
            
        print(f"Initialization status: English documents {len(self.corpus_documents_en)} items, Chinese documents {len(self.corpus_documents_ch)} items")
        
        # Smart cache loading: prioritize cache, automatically fallback to recomputation on failure
        if self.use_existing_embedding_index:
            try:
                print("Attempting to load existing cache...")
                loaded = self._load_cached_embeddings()
                if loaded:
                    print("Cache loaded successfully")
                    
                    # Validate if loaded embedding vectors are valid
                    if self._validate_loaded_embeddings():
                        print("Embedding vector validation passed")
                    else:
                        print("Embedding vector validation failed, recomputing...")
                        self._compute_embeddings()
                else:
                    print("Cache loading failed, recomputing embeddings...")
                    self._compute_embeddings()
                    
            except Exception as e:
                print(f"Error occurred during cache loading: {e}")
                print("Automatically falling back to recomputing embeddings...")
                self._compute_embeddings()
        else:
            print("Forcing recomputation of embeddings...")
            self._compute_embeddings()

        # Validate document types
        if self.corpus_documents_en:
            for i, doc in enumerate(self.corpus_documents_en):
                if not isinstance(doc, DocumentWithMetadata):
                    print(f"Warning: corpus_documents_en[{i}] is not DocumentWithMetadata type: {type(doc)}")
                elif not hasattr(doc, 'content') or not isinstance(doc.content, str):
                    print(f"Warning: corpus_documents_en[{i}] content field type error: {type(getattr(doc, 'content', None))}")
                    
        if self.corpus_documents_ch:
            for i, doc in enumerate(self.corpus_documents_ch):
                if not isinstance(doc, DocumentWithMetadata):
                    print(f"Warning: corpus_documents_ch[{i}] is not DocumentWithMetadata type: {type(doc)}")
                elif not hasattr(doc, 'content') or not isinstance(doc.content, str):
                    print(f"Warning: corpus_documents_ch[{i}] content field type error: {type(getattr(doc, 'content', None))}")

    def _get_cache_key(self, documents: List[DocumentWithMetadata], encoder_name: str) -> str:
        """Generate cache key based on document content and encoder name"""
        # Create hash of document content
        content_hash = hashlib.md5()
        for doc in documents:
            content_hash.update(doc.content.encode('utf-8'))
        
        # Only use the last part of encoder name to avoid path issues
        encoder_basename = os.path.basename(encoder_name)
        
        # Combine encoder name and document count
        cache_key = f"{encoder_basename}_{len(documents)}_{content_hash.hexdigest()[:16]}"
        return cache_key

    def _get_cache_path(self, cache_key: str, suffix: str) -> str:
        """Get cache file path"""
        return os.path.join(self.cache_dir, f"{cache_key}.{suffix}")

    def _is_cache_valid(self, documents: List[DocumentWithMetadata], cache_key: str) -> bool:
        """Check if cache is valid (whether data has changed)"""
        try:
            # Check if cache files exist
            embeddings_path = self._get_cache_path(cache_key, "npy")
            index_path = self._get_cache_path(cache_key, "faiss")
            
            if not os.path.exists(embeddings_path) or not os.path.exists(index_path):
                print(f"Cache files do not exist, need to regenerate")
                return False
            
            # Check if embedding vector dimensions match
            if os.path.exists(embeddings_path):
                try:
                    cached_embeddings = np.load(embeddings_path)
                    if cached_embeddings.shape[0] != len(documents):
                        print(f"Document count mismatch: cache={cached_embeddings.shape[0]}, current={len(documents)}")
                        return False
                    
                    # Check if embedding vector is empty or invalid
                    if cached_embeddings.size == 0:
                        print(f"Cached embedding vector is empty")
                        return False
                        
                except Exception as e:
                    print(f"Failed to read embedding vector cache: {e}")
                    return False
            
            # Check if FAISS index is valid
            if os.path.exists(index_path):
                try:
                    index = faiss.read_index(index_path)
                    if hasattr(index, 'ntotal') and index.ntotal != len(documents):
                        print(f"FAISS index size mismatch: cache={index.ntotal}, current={len(documents)}")
                        return False
                        
                    # Check if FAISS index is empty
                    if hasattr(index, 'ntotal') and index.ntotal == 0:
                        print(f"FAISS index is empty")
                        return False
                        
                except Exception as e:
                    print(f"Failed to read FAISS index: {e}")
                    return False
            
            print(f"Cache validation passed")
            return True
            
        except Exception as e:
            print(f"Cache validation failed: {e}")
            return False

    def _clear_invalid_cache(self, cache_key: str):
        """Clear invalid cache files"""
        try:
            embeddings_path = self._get_cache_path(cache_key, "npy")
            index_path = self._get_cache_path(cache_key, "faiss")
            
            if os.path.exists(embeddings_path):
                os.remove(embeddings_path)
                print(f"Deleted invalid embedding vector cache: {embeddings_path}")
            
            if os.path.exists(index_path):
                os.remove(index_path)
                print(f"Deleted invalid FAISS index cache: {index_path}")
                
        except Exception as e:
            print(f"Failed to clear cache: {e}")
    
    def _validate_loaded_embeddings(self) -> bool:
        """Validate if loaded embedding vectors are valid"""
        try:
            # Validate English embedding vectors
            if self.corpus_documents_en:
                if self.corpus_embeddings_en is None:
                    print("English embedding vector is None")
                    return False
                
                if self.corpus_embeddings_en.size == 0:
                    print("English embedding vector is empty")
                    return False
                
                if self.corpus_embeddings_en.shape[0] != len(self.corpus_documents_en):
                    print(f"English embedding vector dimension mismatch: {self.corpus_embeddings_en.shape[0]} != {len(self.corpus_documents_en)}")
                    return False
                
                print(f"English embedding vector valid: {self.corpus_embeddings_en.shape}")
            
            # Validate Chinese embedding vectors
            if self.corpus_documents_ch:
                if self.corpus_embeddings_ch is None:
                    print("Chinese embedding vector is None")
                    return False
                
                if self.corpus_embeddings_ch.size == 0:
                    print("Chinese embedding vector is empty")
                    return False
                
                if self.corpus_embeddings_ch.shape[0] != len(self.corpus_documents_ch):
                    print(f"Chinese embedding vector dimension mismatch: {self.corpus_embeddings_ch.shape[0]} != {len(self.corpus_documents_ch)}")
                    return False
                
                print(f"Chinese embedding vector valid: {self.corpus_embeddings_ch.shape}")
            
            # Validate FAISS indices
            if self.use_faiss:
                if self.corpus_documents_en and self.index_en:
                    if not hasattr(self.index_en, 'ntotal') or self.index_en.ntotal == 0:
                        print("English FAISS index is empty")
                        return False
                    print(f"English FAISS index valid: {self.index_en.ntotal} documents")
                
                if self.corpus_documents_ch and self.index_ch:
                    if not hasattr(self.index_ch, 'ntotal') or self.index_ch.ntotal == 0:
                        print("Chinese FAISS index is empty")
                        return False
                    print(f"Chinese FAISS index valid: {self.index_ch.ntotal} documents")
            
            return True
            
        except Exception as e:
            print(f"Embedding vector validation failed: {e}")
            return False

    def _load_cached_embeddings(self) -> bool:
        """Try to load cached embedding vectors, automatically detect data changes"""
        try:
            loaded_any = False
            
            # Check English document cache
            if self.corpus_documents_en:
                cache_key_en = self._get_cache_key(self.corpus_documents_en, str(self.encoder_en.model_name))
                print(f"Checking English cache: {cache_key_en}")
                
                try:
                    if self._is_cache_valid(self.corpus_documents_en, cache_key_en):
                        embeddings_path_en = self._get_cache_path(cache_key_en, "npy")
                        index_path_en = self._get_cache_path(cache_key_en, "faiss")
                        
                        # Try to load embedding vectors
                        try:
                            self.corpus_embeddings_en = np.load(embeddings_path_en)
                            print(f"English embedding vector loaded successfully, shape: {self.corpus_embeddings_en.shape}")
                            loaded_any = True
                        except Exception as e:
                            print(f"English embedding vector loading failed: {e}")
                            self.corpus_embeddings_en = np.array([])
                        
                        # Try to load FAISS index
                        if self.use_faiss and os.path.exists(index_path_en):
                            try:
                                self.index_en = faiss.read_index(index_path_en)
                                print(f"English FAISS index loaded successfully, document count: {len(self.corpus_documents_en)}")
                            except Exception as e:
                                print(f"English FAISS index loading failed: {e}")
                                self.index_en = None
                    else:
                        # Clear invalid cache
                        self._clear_invalid_cache(cache_key_en)
                        print(f"English data has changed, need to regenerate index")
                        self.corpus_embeddings_en = np.array([])
                        self.index_en = None
                        
                except Exception as e:
                    print(f"English cache validation failed: {e}")
                    self._clear_invalid_cache(cache_key_en)
                    self.corpus_embeddings_en = np.array([])
                    self.index_en = None
            else:
                print("English document list is empty, initializing empty embedding vector")
                self.corpus_embeddings_en = np.array([])
                if self.use_faiss:
                    self.index_en = self._init_faiss(self.encoder_en, 0)

            # Check Chinese document cache
            if self.corpus_documents_ch:
                cache_key_ch = self._get_cache_key(self.corpus_documents_ch, str(self.encoder_ch.model_name))
                print(f"Checking Chinese cache: {cache_key_ch}")
                
                try:
                    if self._is_cache_valid(self.corpus_documents_ch, cache_key_ch):
                        embeddings_path_ch = self._get_cache_path(cache_key_ch, "npy")
                        index_path_ch = self._get_cache_path(cache_key_ch, "faiss")
                        
                        # Try to load embedding vectors
                        try:
                            self.corpus_embeddings_ch = np.load(embeddings_path_ch)
                            print(f"Chinese embedding vector loaded successfully, shape: {self.corpus_embeddings_ch.shape}")
                            loaded_any = True
                        except Exception as e:
                            print(f"Chinese embedding vector loading failed: {e}")
                            self.corpus_embeddings_ch = np.array([])
                        
                        # Try to load FAISS index
                        if self.use_faiss and os.path.exists(index_path_ch):
                            try:
                                self.index_ch = faiss.read_index(index_path_ch)
                                print(f"Chinese FAISS index loaded successfully, document count: {len(self.corpus_documents_ch)}")
                            except Exception as e:
                                print(f"Chinese FAISS index loading failed: {e}")
                                self.index_ch = None
                    else:
                        # Clear invalid cache
                        self._clear_invalid_cache(cache_key_ch)
                        print(f"Chinese data has changed, need to regenerate index")
                        self.corpus_embeddings_ch = np.array([])
                        self.index_ch = None
                        
                except Exception as e:
                    print(f"Chinese cache validation failed: {e}")
                    self._clear_invalid_cache(cache_key_ch)
                    self.corpus_embeddings_ch = np.array([])
                    self.index_ch = None
            else:
                print("Chinese document list is empty, initializing empty embedding vector")
                self.corpus_embeddings_ch = np.array([])
                if self.use_faiss:
                    self.index_ch = self._init_faiss(self.encoder_ch, 0)

            return loaded_any
        except Exception as e:
            print(f"Cache loading failed: {e}")
            # Ensure embedding vectors are not None
            if self.corpus_embeddings_en is None:
                self.corpus_embeddings_en = np.array([])
            if self.corpus_embeddings_ch is None:
                self.corpus_embeddings_ch = np.array([])
            return False

    def _save_cached_embeddings(self):
        """Save embedding vectors to cache"""
        try:
            # Save English document embedding vectors
            if self.corpus_documents_en and self.corpus_embeddings_en is not None and self.corpus_embeddings_en.size > 0:
                try:
                    cache_key_en = self._get_cache_key(self.corpus_documents_en, str(self.encoder_en.model_name))
                    embeddings_path_en = self._get_cache_path(cache_key_en, "npy")
                    index_path_en = self._get_cache_path(cache_key_en, "faiss")
                    
                    # Ensure directory exists
                    os.makedirs(os.path.dirname(embeddings_path_en), exist_ok=True)
                    os.makedirs(os.path.dirname(index_path_en), exist_ok=True)
                    
                    # Save embedding vectors
                    np.save(embeddings_path_en, self.corpus_embeddings_en)
                    print(f"English embedding vector saved: {embeddings_path_en}")
                    
                    # Save FAISS index
                    if self.use_faiss and self.index_en:
                        faiss.write_index(self.index_en, index_path_en)
                        print(f"English FAISS index saved: {index_path_en}")
                        
                except Exception as e:
                    print(f"English cache saving failed: {e}")

            # Save Chinese document embedding vectors
            if self.corpus_documents_ch and self.corpus_embeddings_ch is not None and self.corpus_embeddings_ch.size > 0:
                try:
                    cache_key_ch = self._get_cache_key(self.corpus_documents_ch, str(self.encoder_ch.model_name))
                    embeddings_path_ch = self._get_cache_path(cache_key_ch, "npy")
                    index_path_ch = self._get_cache_path(cache_key_ch, "faiss")
                    
                    # Ensure directory exists
                    os.makedirs(os.path.dirname(embeddings_path_ch), exist_ok=True)
                    os.makedirs(os.path.dirname(index_path_ch), exist_ok=True)
                    
                    # Save embedding vectors
                    np.save(embeddings_path_ch, self.corpus_embeddings_ch)
                    print(f"Chinese embedding vector saved: {embeddings_path_ch}")
                    
                    # Save FAISS index
                    if self.use_faiss and self.index_ch:
                        faiss.write_index(self.index_ch, index_path_ch)
                        print(f"Chinese FAISS index saved: {index_path_ch}")
                        
                except Exception as e:
                    print(f"Chinese cache saving failed: {e}")
                    
        except Exception as e:
            print(f"Error occurred during cache saving: {e}")

    def _compute_embeddings(self):
        """Compute embedding vectors"""
        print("=== Starting embedding vector computation ===")
        print(f"use_existing_embedding_index: {self.use_existing_embedding_index}")
        
        try:
            # Check if FAISS index initialization is needed
            if self.use_faiss:
                if self.corpus_documents_en and self.index_en is None:
                    print(f"Initializing English FAISS index, document count: {len(self.corpus_documents_en)}")
                    self.index_en = self._init_faiss(self.encoder_en, len(self.corpus_documents_en))
                if self.corpus_documents_ch and self.index_ch is None:
                    print(f"Initializing Chinese FAISS index, document count: {len(self.corpus_documents_ch)}")
                    self.index_ch = self._init_faiss(self.encoder_ch, len(self.corpus_documents_ch))

            if self.corpus_documents_en:
                print(f"Starting to encode English documents, count: {len(self.corpus_documents_en)}")
                print(f"English encoder: {self.encoder_en.model_name}")
                print(f"English encoder device: {self.encoder_en.device}")
                
                # Check English document content
                if self.corpus_documents_en:
                    first_doc = self.corpus_documents_en[0]
                    print(f"First English document content preview: {first_doc.content[:100]}...")
                
                self.corpus_embeddings_en = self._batch_encode_corpus(self.corpus_documents_en, self.encoder_en, 'en')
                print(f"English embedding vector computation completed, shape: {self.corpus_embeddings_en.shape if self.corpus_embeddings_en is not None else 'None'}")
                
                if self.corpus_embeddings_en is None or self.corpus_embeddings_en.shape[0] == 0:
                    print("English embedding vector computation failed!")
                    self.corpus_embeddings_en = np.array([])  # Ensure not None
                else:
                    print("English embedding vector computation successful!")
                    
                if self.use_faiss and self.corpus_embeddings_en is not None and self.corpus_embeddings_en.shape[0] > 0:
                    print("Adding English embedding vectors to FAISS index")
                    self._add_to_faiss(self.index_en, self.corpus_embeddings_en)
            else:
                print("English document list is empty, initializing empty embedding vector")
                self.corpus_embeddings_en = np.array([])
                if self.use_faiss and self.index_en is None:
                    self.index_en = self._init_faiss(self.encoder_en, 0)

            if self.corpus_documents_ch:
                print(f"Starting to encode Chinese documents, count: {len(self.corpus_documents_ch)}")
                self.corpus_embeddings_ch = self._batch_encode_corpus(self.corpus_documents_ch, self.encoder_ch, 'zh')
                print(f"Chinese embedding vector computation completed, shape: {self.corpus_embeddings_ch.shape if self.corpus_embeddings_ch is not None else 'None'}")
                
                if self.corpus_embeddings_ch is None or self.corpus_embeddings_ch.shape[0] == 0:
                    print("Chinese embedding vector computation failed!")
                    self.corpus_embeddings_ch = np.array([])  # Ensure not None
                else:
                    print("Chinese embedding vector computation successful!")
                    
                if self.use_faiss and self.corpus_embeddings_ch is not None and self.corpus_embeddings_ch.shape[0] > 0:
                    print("Adding Chinese embedding vectors to FAISS index")
                    self._add_to_faiss(self.index_ch, self.corpus_embeddings_ch)
            else:
                print("Chinese document list is empty, initializing empty embedding vector")
                self.corpus_embeddings_ch = np.array([])
                if self.use_faiss and self.index_ch is None:
                    self.index_ch = self._init_faiss(self.encoder_ch, 0)
            
            # Save to cache
            print("Saving embedding vectors to cache")
            try:
                self._save_cached_embeddings()
            except Exception as e:
                print(f"Cache saving failed: {e}")
            
            print("=== Embedding vector computation completed ===")
            print(f"Final status:")
            print(f"  English embedding vector: {self.corpus_embeddings_en.shape if self.corpus_embeddings_en is not None else 'None'}")
            print(f"  Chinese embedding vector: {self.corpus_embeddings_ch.shape if self.corpus_embeddings_ch is not None else 'None'}")
            
        except Exception as e:
            print(f"Error occurred during embedding vector computation: {e}")
            # Ensure embedding vectors are not None
            if self.corpus_embeddings_en is None:
                self.corpus_embeddings_en = np.array([])
            if self.corpus_embeddings_ch is None:
                self.corpus_embeddings_ch = np.array([])
            print("Reset embedding vectors to empty arrays")

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
        print(f"=== Starting batch encoding of corpus ===")
        print(f"Language: {language}")
        print(f"Document count: {len(documents) if documents else 0}")
        print(f"Encoder: {encoder.model_name}")
        print(f"Encoder device: {encoder.device}")
        
        if not documents:
            print("Document list is empty, returning empty array")
            return np.array([])
        
        batch_texts = []
        for i, doc in enumerate(documents):
            if language == 'zh' and hasattr(doc.metadata, 'summary') and doc.metadata.summary:
                # Chinese uses summary field for FAISS indexing
                batch_texts.append(doc.metadata.summary)
            else:
                # English uses content field
                batch_texts.append(doc.content)
            
            # Print content preview of first few documents
            if i < 3:
                content_preview = batch_texts[-1][:100] + "..." if len(batch_texts[-1]) > 100 else batch_texts[-1]
                print(f"Document {i+1} content preview: {content_preview}")
        
        print(f"Preparing to encode {len(batch_texts)} texts")
        
        # Test if encoder is working properly
        try:
            print("Testing encoder...")
            test_text = batch_texts[0] if batch_texts else "test"
            test_embedding = encoder.encode([test_text])
            print(f"Test encoding successful, embedding dimension: {test_embedding.shape}")
        except Exception as e:
            print(f"Encoder test failed: {e}")
            import traceback
            traceback.print_exc()
            return np.array([])
        
        try:
            print("Starting batch encoding...")
            embeddings = encoder.encode(texts=batch_texts, batch_size=self.batch_size, show_progress_bar=True)
            print(f"Encoding completed, embedding vector shape: {embeddings.shape}")
            return embeddings
        except Exception as e:
            print(f"Error occurred during batch encoding: {e}")
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
        # Add detailed debug information
        print(f"=== BilingualRetriever.retrieve() Debug Information ===")
        print(f"Query text: {text}")
        print(f"English document count: {len(self.corpus_documents_en) if self.corpus_documents_en else 0}")
        print(f"Chinese document count: {len(self.corpus_documents_ch) if self.corpus_documents_ch else 0}")
        print(f"English embedding vector shape: {self.corpus_embeddings_en.shape if self.corpus_embeddings_en is not None else 'None'}")
        print(f"Chinese embedding vector shape: {self.corpus_embeddings_ch.shape if self.corpus_embeddings_ch is not None else 'None'}")
        print(f"English FAISS index: {'Initialized' if self.index_en else 'Not initialized'}")
        print(f"Chinese FAISS index: {'Initialized' if self.index_ch else 'Not initialized'}")
        
        # Check self.corpus_documents_en/ch types
        if hasattr(self, 'corpus_documents_en') and self.corpus_documents_en is not None:
            for i, doc in enumerate(self.corpus_documents_en):
                if not isinstance(doc, DocumentWithMetadata):
                    print(f"Warning: corpus_documents_en[{i}] is not DocumentWithMetadata type: {type(doc)}")
                elif not hasattr(doc, 'content') or not isinstance(doc.content, str):
                    print(f"Warning: corpus_documents_en[{i}] content field type error: {type(getattr(doc, 'content', None))}")
                    
        if hasattr(self, 'corpus_documents_ch') and self.corpus_documents_ch is not None:
            for i, doc in enumerate(self.corpus_documents_ch):
                if not isinstance(doc, DocumentWithMetadata):
                    print(f"Warning: corpus_documents_ch[{i}] is not DocumentWithMetadata type: {type(doc)}")
                elif not hasattr(doc, 'content') or not isinstance(doc.content, str):
                    print(f"Warning: corpus_documents_ch[{i}] content field type error: {type(getattr(doc, 'content', None))}")
                    
        if language is None:
            # Enhanced language detection logic
            try:
                lang = detect(text)
                # Check if contains Chinese characters
                chinese_chars = sum(1 for char in text if '\u4e00' <= char <= '\u9fff')
                total_chars = len([char for char in text if char.isalpha() or '\u4e00' <= char <= '\u9fff'])
                
                # If contains Chinese characters and Chinese ratio exceeds 30%, or langdetect detects as Chinese, consider as Chinese
                if chinese_chars > 0 and (chinese_chars / total_chars > 0.3 or lang.startswith('zh')):
                    language = 'zh'
                else:
                    language = 'en'
            except:
                # If langdetect fails, use character detection
                chinese_chars = sum(1 for char in text if '\u4e00' <= char <= '\u9fff')
                language = 'zh' if chinese_chars > 0 else 'en'
        
        print(f"Detected language: {language}")
        
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
        
        print(f"Selected encoder: {encoder.model_name}")
        print(f"Selected corpus document count: {len(corpus_documents) if corpus_documents else 0}")
        print(f"Selected embedding vector shape: {corpus_embeddings.shape if corpus_embeddings is not None else 'None'}")
        
        if corpus_embeddings is None or corpus_embeddings.shape[0] == 0:
            print("Embedding vector is empty or shape is 0, returning empty result")
            if return_scores:
                return [], []
            else:
                return []
        
        query_embeddings = encoder.encode([text])
        print(f"Query embedding vector shape: {query_embeddings.shape}")
        
        if self.use_faiss and index:
            print("Using FAISS index for retrieval")
            distances, indices = index.search(query_embeddings.astype('float32'), top_k)
            results = []
            for score, idx in zip(distances[0], indices[0]):
                if idx != -1:
                    results.append({'corpus_id': idx, 'score': 1 - score / 2})
            print(f"FAISS retrieval result count: {len(results)}")
        else:
            print("Using semantic search for retrieval")
            # Ensure tensor is on correct device
            query_tensor = torch.tensor(query_embeddings, device=encoder.device)
            corpus_tensor = torch.tensor(corpus_embeddings, device=encoder.device)
            hits = semantic_search(
                query_tensor,
                corpus_tensor,
                top_k=top_k
            )
            results = hits[0]
            print(f"Semantic search result count: {len(results)}")
        
        doc_indices = [hit['corpus_id'] for hit in results]
        scores = [hit['score'] for hit in results]
        
        # Add index range check
        print(f"Retrieved indices: {doc_indices}")
        print(f"Corpus document count: {len(corpus_documents)}")
        
        # Filter invalid indices
        valid_indices = []
        valid_scores = []
        for i, (idx, score) in enumerate(zip(doc_indices, scores)):
            if 0 <= idx < len(corpus_documents):
                valid_indices.append(idx)
                valid_scores.append(score)
            else:
                print(f"Warning: Skipping invalid index {idx} (out of range [0, {len(corpus_documents)})")
        
        raw_documents = [corpus_documents[i] for i in valid_indices]
        scores = valid_scores
        
        print(f"Final returned document count: {len(raw_documents)}")
        
        # Ensure returned objects are DocumentWithMetadata, unified use of content field
        documents = []
        for i, doc in enumerate(raw_documents):
            if isinstance(doc, DocumentWithMetadata):
                # Already correct type, use directly
                documents.append(doc)
            elif isinstance(doc, dict):
                # If it's a dictionary, convert to DocumentWithMetadata
                content = doc.get('content', doc.get('context', ''))
                if not isinstance(content, str):
                    # If content is not a string, try to get context field or convert to string
                    content = content.get('context', '') if isinstance(content, dict) and 'context' in content else str(content)
                metadata = DocumentMetadata(
                    source=doc.get('source', 'unknown'),
                    created_at=doc.get('created_at', ''),
                    author=doc.get('author', ''),
                    language=language or 'unknown'
                )
                documents.append(DocumentWithMetadata(content=content, metadata=metadata))
            else:
                # Other types, try to convert to string
                print(f"Warning: Retrieval result[{i}] type exception: {type(doc)}, attempting conversion")
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