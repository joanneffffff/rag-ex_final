import json
import pickle
from pathlib import Path
from typing import List, Dict, Optional, Tuple
from collections import defaultdict
import re
from datetime import datetime
import sys
import numpy as np

try:
    import faiss
    from sentence_transformers import SentenceTransformer
    import torch
except ImportError as e:
    print("Please install required dependencies: pip install faiss-cpu sentence-transformers torch")
    print(f"Error: {e}")
    exit(1)

# Import existing QwenReranker
try:
    sys.path.append(str(Path(__file__).parent.parent))
    from xlm.components.retriever.reranker import QwenReranker
    from config.parameters import Config, DEFAULT_CACHE_DIR
except ImportError as e:
    print(f"Cannot import QwenReranker or Config: {e}")
    print("Please ensure the xlm directory structure is correct")
    exit(1)

from xlm.components.prompt_templates.template_loader import template_loader

def load_json_or_jsonl(file_path: Path) -> List[Dict]:
    """
    Load JSON or JSONL format file
    
    Args:
        file_path: File path
        
    Returns:
        Data list
    """
    print(f"Loading data file: {file_path}")
    
    try:
        # Try to load as JSON first
        with open(file_path, 'r', encoding='utf-8') as f:
            try:
                data = json.load(f)
                print(f"Successfully loaded JSON format file, {len(data)} records")
                return data
            except json.JSONDecodeError as e:
                print(f"JSON format parsing failed: {e}")
                print("Trying to load as JSONL format...")
                
                # Reset file pointer
                f.seek(0)
                
                # Try to load as JSONL
                data = []
                for line_num, line in enumerate(f, 1):
                    line = line.strip()
                    if line:  # Skip empty lines
                        try:
                            item = json.loads(line)
                            data.append(item)
                        except json.JSONDecodeError as line_error:
                            print(f"Warning: JSON parsing failed on line {line_num}: {line_error}")
                            print(f"Problem line content: {line[:100]}...")
                            continue
                
                print(f"Successfully loaded JSONL format file, {len(data)} records")
                return data
                
    except FileNotFoundError:
        print(f"File not found: {file_path}")
        return []
    except Exception as e:
        print(f"Error reading file: {e}")
        return []

class MultiStageRetrievalSystem:
    """
    Multi-stage retrieval system:
    1. Pre-filtering: Based on metadata (only supported for Chinese data)
    2. FAISS retrieval: Based on generated_question and summary to generate a unified embedding index
    3. Reranker: Based on original_context using Qwen3-0.6B for reranking
    
    Support English and Chinese datasets, using existing configuration models
    - Chinese data (AlphaFin): Support metadata pre-filtering + FAISS + Qwen reranking
    - English data (TatQA): Only support FAISS + Qwen reranking (no metadata)
    """
    
    def __init__(self, data_path: Path, dataset_type: str = "chinese", use_existing_config: bool = True):
        """
        Initialize multi-stage retrieval system
        
        Args:
            data_path: Data file path
            dataset_type: Dataset type ("chinese" or "english")
            use_existing_config: Whether to use existing configuration
        """
        self.data_path = data_path
        self.dataset_type = dataset_type
        self.use_existing_config = use_existing_config
        
        # Initialize components
        self.data = []
        self.embedding_model = None
        self.faiss_index = None
        self.valid_indices = []
        self.qwen_reranker = None
        self.llm_generator = None
        
        # Metadata index
        self.metadata_index = {
            'company_name': {},
            'stock_code': {},
            'report_date': {},
            'company_stock': {}
        }
        
        # Stock code and company name mapping
        self.stock_company_mapping = {}
        self.company_stock_mapping = {}
        
        # Configuration
        self.config = None
        self.model_name = None
        
        # Document to chunks mapping (for reranking)
        self.doc_to_chunks_mapping = {}
        
        # Load configuration
        if use_existing_config:
            self._load_config()
        
        # Load data
        self._load_data()
        
        # Load stock code and company name mapping
        self._load_stock_company_mapping()
        
        # Build metadata index
        self._build_metadata_index()
        
        # Initialize embedding model
        self._init_embedding_model()
        
        # Build FAISS index
        self._build_faiss_index()
        
        # Initialize reranker
        self._init_qwen_reranker()
        
        # Initialize LLM generator
        self._init_llm_generator()
    
    def _load_config(self):
        """Load configuration file"""
        try:
            from config.parameters import Config
            self.config = Config()
            
            # Select encoder based on dataset type
            if self.dataset_type == "chinese":
                # Use Chinese encoder
                self.model_name = self.config.encoder.chinese_model_path
                print(f"Using Chinese encoder: {self.model_name}")
            else:
                # Use English encoder
                self.model_name = self.config.encoder.english_model_path
                print(f"Using English encoder: {self.model_name}")
            
            print("Using existing configuration to initialize multi-stage retrieval system")
        except Exception as e:
            print(f"Loading configuration failed: {e}")
            # Fallback to default model
            if self.dataset_type == "chinese":
                self.model_name = "distiluse-base-multilingual-cased-v2"
                print(f"Using default Chinese encoder: {self.model_name}")
            else:
                self.model_name = "all-MiniLM-L6-v2"
                print(f"Using default English encoder: {self.model_name}")
    
    def _load_data(self):
        """Load data"""
        print("Loading data...")
        
        # Load original AlphaFin data for FAISS index
        print("Loading original AlphaFin data for FAISS index...")
        self.data = load_json_or_jsonl(self.data_path)
        print(f"Loaded {len(self.data)} original records")
        
        # Build doc_id to chunks mapping
        print("Building doc_id to chunks mapping...")
        self.doc_to_chunks_mapping = {}
        
        for doc_idx, record in enumerate(self.data):
            if self.dataset_type == "chinese":
                # For Chinese data, generate chunks
                original_context = record.get('original_context', '')
                company_name = record.get('company_name', '公司')
                
                if original_context:
                    # Use convert_json_context_to_natural_language_chunks function
                    from xlm.utils.optimized_data_loader import convert_json_context_to_natural_language_chunks
                    chunks = convert_json_context_to_natural_language_chunks(original_context, company_name)
                    
                    if chunks:
                        self.doc_to_chunks_mapping[doc_idx] = chunks
                    else:
                        # If no chunks, use summary as fallback
                        self.doc_to_chunks_mapping[doc_idx] = [record.get('summary', '')]
                else:
                    # If no original_context, use summary
                    self.doc_to_chunks_mapping[doc_idx] = [record.get('summary', '')]
            else:
                # English data, use context or content
                context = record.get('context', '') or record.get('content', '')
                self.doc_to_chunks_mapping[doc_idx] = [context]
        
        print(f"Built {len(self.doc_to_chunks_mapping)} doc_id to chunks mapping")
        
        # Count total chunks
        total_chunks = sum(len(chunks) for chunks in self.doc_to_chunks_mapping.values())
        print(f"Generated {total_chunks} chunks for reranking")
        
        # Use original data as main data
        # self.data = self.original_data # original_data is removed
        
        print(f"Dataset type: {self.dataset_type}")
        
        # Check data format
        if self.data and isinstance(self.data[0], dict):
            sample_record = self.data[0]
            print(f"Data fields: {list(sample_record.keys())}")
            
            # Check if there are metadata fields
            has_metadata = any(field in sample_record for field in ['company_name', 'stock_code', 'report_date'])
            print(f"Contains metadata fields: {has_metadata}")
    
    def _load_stock_company_mapping(self):
        """Load stock code and company name mapping file"""
        if self.dataset_type == "chinese":
            # Try to load mapping file from multiple paths
            possible_paths = [
                Path("data/astock_code_company_name.csv"),
                Path(__file__).parent.parent / "data" / "astock_code_company_name.csv",
                Path(__file__).parent / "data" / "astock_code_company_name.csv"
            ]
            
            mapping_path = None
            for path in possible_paths:
                if path.exists():
                    mapping_path = path
                    break
            
            if mapping_path:
                try:
                    import pandas as pd
                    df = pd.read_csv(mapping_path, encoding='utf-8')
                    
                    # Build bidirectional mapping
                    for _, row in df.iterrows():
                        stock_code = str(row['stock_code']).strip()
                        company_name = str(row['company_name']).strip()
                        
                        if stock_code and company_name:
                            # Stock code -> company name
                            self.stock_company_mapping[stock_code] = company_name
                            # Company name -> stock code
                            self.company_stock_mapping[company_name] = stock_code
                    
                    print(f"Successfully loaded stock code and company name mapping file: {mapping_path}")
                    print(f"Stock code mapping count: {len(self.stock_company_mapping)}")
                    print(f"Company name mapping count: {len(self.company_stock_mapping)}")
                    
                except Exception as e:
                    print(f"Loading stock code and company name mapping file failed: {e}")
                    print(f"File path: {mapping_path}")
                    self.stock_company_mapping = {}
                    self.company_stock_mapping = {}
            else:
                print("Stock code and company name mapping file does not exist")
                print(f"Tried paths: {[str(p) for p in possible_paths]}")
                self.stock_company_mapping = {}
                self.company_stock_mapping = {}
        else:
            print("English dataset, do not load stock code and company name mapping file")
            self.stock_company_mapping = {}
            self.company_stock_mapping = {}

    def _build_metadata_index(self):
        """Build metadata index for pre-filtering (only supported for Chinese data)"""
        if self.dataset_type != "chinese":
            print("Non-Chinese dataset, skip building metadata index")
            return
            
        print("Building metadata index...")
        
        # Check if there are metadata fields
        if not self.data:
            print("Data format does not support metadata index")
            return
            
        # Check data format
        if hasattr(self.data[0], 'content'):
            # DocumentWithMetadata format
            print("Using DocumentWithMetadata format, skip building metadata index")
            print("Note: chunk-level data does not support metadata pre-filtering")
            return
        elif isinstance(self.data[0], dict):
            # Dictionary format
            sample_record = self.data[0]
            has_metadata = any(field in sample_record for field in ['company_name', 'stock_code', 'report_date'])
            
            if not has_metadata:
                print("Data does not contain metadata fields, skip building metadata index")
                return
            
            # Index by company name
            self.metadata_index['company_name'] = defaultdict(list)
            # Index by stock code
            self.metadata_index['stock_code'] = defaultdict(list)
            # Index by report date
            self.metadata_index['report_date'] = defaultdict(list)
            # Index by company name + stock code combination
            self.metadata_index['company_stock'] = defaultdict(list)
            
            for idx, record in enumerate(self.data):
                # Index by company name
                if record.get('company_name'):
                    company_name = record['company_name'].strip().lower()
                    self.metadata_index['company_name'][company_name].append(idx)
                
                # Index by stock code
                if record.get('stock_code'):
                    stock_code = str(record['stock_code']).strip().lower()
                    self.metadata_index['stock_code'][stock_code].append(idx)
                
                # Index by report date
                if record.get('report_date'):
                    report_date = record['report_date'].strip()
                    self.metadata_index['report_date'][report_date].append(idx)
                
                # Index by company name + stock code combination
                if record.get('company_name') and record.get('stock_code'):
                    company_name = record['company_name'].strip().lower()
                    stock_code = str(record['stock_code']).strip().lower()
                    key = f"{company_name}_{stock_code}"
                    self.metadata_index['company_stock'][key].append(idx)
            
            print(f"Metadata index built:")
            print(f"  - Company name: {len(self.metadata_index['company_name'])}")
            print(f"  - Stock code: {len(self.metadata_index['stock_code'])}")
            print(f"  - Report date: {len(self.metadata_index['report_date'])}")
            print(f"  - Company + stock code combination: {len(self.metadata_index['company_stock'])}")
    
    def _init_embedding_model(self):
        """Initialize sentence embedding model"""
        print(f"Loading embedding model: {self.model_name}")
        print(f"Model type: {'Multilingual encoder' if self.dataset_type == 'chinese' else 'English encoder'}")
        
        # Use existing configuration cache directory
        cache_dir = None
        if self.config:
            cache_dir = self.config.encoder.cache_dir
            print(f"Using configuration cache directory: {cache_dir}")
        
        try:
            # Get device setting from configuration
            device = "cuda:0"  # Default value
            if self.config and hasattr(self.config, 'encoder'):
                device = self.config.encoder.device or "cuda:0"
            
            # Check if it is a finetuned model path
            if "finetuned" in self.model_name or "models/" in self.model_name:
                print(f"Detected finetuned model, using FinbertEncoder...")
                from xlm.components.encoder.finbert import FinbertEncoder
                self.embedding_model = FinbertEncoder(
                    model_name=self.model_name,
                    cache_dir=cache_dir,
                    device=device
                )
                print(f"Finetuned model loaded ({device})")
            else:
                # Use SentenceTransformer to load HuggingFace model
                from sentence_transformers import SentenceTransformer
                self.embedding_model = SentenceTransformer(self.model_name, cache_folder=cache_dir)
                # Move model to specified device
                if hasattr(self.embedding_model, 'to'):
                    self.embedding_model.to(device)
                print(f"HuggingFace model loaded ({device})")
        except Exception as e:
            print(f"Embedding model loading failed: {e}")
            print("Trying to use default model...")
            try:
                # Get device setting from configuration
                device = "cuda:0"  # Default value
                if self.config and hasattr(self.config, 'encoder'):
                    device = self.config.encoder.device or "cuda:0"
                
                # Fallback to default model
                if self.dataset_type == "chinese":
                    fallback_model = "distiluse-base-multilingual-cased-v2"
                else:
                    fallback_model = "all-MiniLM-L6-v2"
                print(f"Using fallback model: {fallback_model}")
                from sentence_transformers import SentenceTransformer
                self.embedding_model = SentenceTransformer(fallback_model)
                # Move model to specified device
                if hasattr(self.embedding_model, 'to'):
                    self.embedding_model.to(device)
                print(f"Fallback model loaded ({device})")
            except Exception as e2:
                print(f"Fallback model loading failed: {e2}")
                self.embedding_model = None
    
    def _build_faiss_index(self):
        """Build FAISS index"""
        if self.embedding_model is None:
            print("Embedding model not initialized, skip building FAISS index")
            return
            
        print("Building FAISS index...")
        print("Chinese data: use summary field for vector encoding")
        print("English data: use context/content field for vector encoding")
        
        # Prepare texts for embedding
        texts_for_embedding = []
        valid_indices = []
        
        for idx, record in enumerate(self.data):
            # Select different text combination strategies based on dataset type
            if self.dataset_type == "chinese":
                # Chinese data: only use summary
                summary = record.get('summary', '')
                
                if summary:
                    texts_for_embedding.append(summary)
                    valid_indices.append(idx)
                else:
                    continue
            else:
                # English data: use context or content field
                context = record.get('context', '') or record.get('content', '')
                
                if context:
                    texts_for_embedding.append(context)
                    valid_indices.append(idx)
                else:
                    continue
        
        if not texts_for_embedding:
            print("No valid text for embedding")
            return
        
        # Generate embeddings
        print(f"Encoding {len(texts_for_embedding)} texts...")
        embeddings = self.embedding_model.encode(texts_for_embedding, show_progress_bar=True)
        
        # Build FAISS index
        dimension = embeddings.shape[1]
        self.faiss_index = faiss.IndexFlatIP(dimension)  # Use inner product similarity
        self.faiss_index.add(embeddings.astype('float32'))
        
        # Save mapping of valid indices
        self.valid_indices = valid_indices
        
        print(f"FAISS index built, dimension: {dimension}")
        print(f"Valid indices count: {len(self.valid_indices)}")
        print(f"Based on summary, for coarse-grained retrieval")
    
    def _init_qwen_reranker(self):
        """Initialize Qwen reranker"""
        print("Initializing Qwen reranker...")
        try:
            # Use existing configuration
            model_name = "Qwen/Qwen3-Reranker-0.6B"
            cache_dir = DEFAULT_CACHE_DIR  # Use DEFAULT_CACHE_DIR
            use_quantization = True
            quantization_type = "4bit"
            
            if self.config:
                model_name = self.config.reranker.model_name
                cache_dir = self.config.reranker.cache_dir or DEFAULT_CACHE_DIR  # 确保不为None
                use_quantization = self.config.reranker.use_quantization
                quantization_type = self.config.reranker.quantization_type
            
            print(f"Using reranker configuration: {model_name}")
            print(f"Cache directory: {cache_dir}")
            print(f"Quantization: {use_quantization} ({quantization_type})")
            
            # Get device setting from configuration
            device = "cpu"  # Default use CPU
            if self.config and hasattr(self.config, 'reranker'):
                device = self.config.reranker.device or "cpu"
            
            # Use existing QwenReranker
            self.qwen_reranker = QwenReranker(
                model_name=model_name,
                device=device,
                cache_dir=cache_dir,
                use_quantization=use_quantization,
                quantization_type=quantization_type
            )
            print(f"Qwen reranker initialized ({device})")
        except Exception as e:
            print(f"Qwen reranker initialization failed: {e}")
            self.qwen_reranker = None
    
    def _init_llm_generator(self):
        """Initialize LLM generator - using shared resource manager"""
        print("Initializing LLM generator...")
        try:
            # Try to use shared resource manager
            try:
                from xlm.utils.shared_resource_manager import shared_resource_manager
                
                # Use parameters from configuration
                model_name = None
                cache_dir = None
                device = None
                use_quantization = None
                quantization_type = None
                
                if self.config and hasattr(self.config, 'generator'):
                    model_name = self.config.generator.model_name
                    cache_dir = self.config.generator.cache_dir
                    device = self.config.generator.device
                    use_quantization = self.config.generator.use_quantization
                    quantization_type = self.config.generator.quantization_type
                
                # Try to get LLM generator from shared resource manager
                self.llm_generator = shared_resource_manager.get_llm_generator(
                    model_name=model_name,
                    cache_dir=cache_dir,
                    device=device,
                    use_quantization=use_quantization,
                    quantization_type=quantization_type
                )
                
                if self.llm_generator:
                    print("Using shared LLM generator")
                    return
                else:
                    print("Shared LLM generator retrieval failed, fallback to independent initialization")
                    
            except ImportError:
                print("Shared resource manager not available, fallback to independent initialization")
            
            # Fallback to independent initialization
            from xlm.components.generator.local_llm_generator import LocalLLMGenerator
            
            # Use parameters from configuration
            model_name = None
            cache_dir = None
            device = None
            use_quantization = None
            quantization_type = None
            
            if self.config and hasattr(self.config, 'generator'):
                model_name = self.config.generator.model_name
                cache_dir = self.config.generator.cache_dir
                device = self.config.generator.device
                use_quantization = self.config.generator.use_quantization
                quantization_type = self.config.generator.quantization_type
            
            # First try GPU mode
            try:
                print(f"Trying GPU mode to load LLM generator: {device}")
                self.llm_generator = LocalLLMGenerator(
                    model_name=model_name,
                    cache_dir=cache_dir,
                    device=device,
                    use_quantization=use_quantization,
                    quantization_type=quantization_type
                )
                print("LLM generator GPU mode initialization completed")
            except Exception as gpu_error:
                print(f"GPU mode loading failed: {gpu_error}")
                print("Fallback to CPU mode...")
                
                # Fallback to CPU mode
                try:
                    self.llm_generator = LocalLLMGenerator(
                        model_name=model_name,
                        cache_dir=cache_dir,
                        device="cpu",  # Force use CPU
                        use_quantization=False,  # CPU mode does not use quantization
                        quantization_type=None
                    )
                    print("LLM generator CPU mode initialization completed")
                except Exception as cpu_error:
                    print(f"CPU mode also failed: {cpu_error}")
                    self.llm_generator = None
                    
        except Exception as e:
            print(f"LLM generator initialization failed: {e}")
            self.llm_generator = None
    
    def pre_filter(self, 
                   company_name: Optional[str] = None,
                   stock_code: Optional[str] = None,
                   report_date: Optional[str] = None,
                   max_candidates: int = 1000) -> List[int]:
        """
        Pre-filter based on metadata (only supported for Chinese data)
        When using pre-filter, automatically enable stock code and company name mapping to improve matching accuracy
        
        Args:
            company_name: company name
            stock_code: stock code
            report_date: report date
            max_candidates: maximum number of candidates
            
        Returns:
            List of candidate record indices
        """
        if self.dataset_type != "chinese":
            print("Non-Chinese dataset, skip pre-filter")
            return list(range(len(self.data)))
        
        print("Starting metadata pre-filter...")
        print("Automatically enable stock code and company name mapping to improve matching accuracy")
        
        # If no filtering conditions are provided, return all records
        if not any([company_name, stock_code, report_date]):
            print("No filtering conditions, return all records")
            return list(range(len(self.data)))
        
        # Use stock code and company name mapping for enhanced matching
        enhanced_company_name = company_name
        enhanced_stock_code = stock_code
        
        # Automatically enable mapping to improve matching accuracy
        # If stock code is provided, try to get the corresponding company name
        if stock_code and not company_name:
            mapped_company = self.stock_company_mapping.get(stock_code)
            if mapped_company:
                enhanced_company_name = mapped_company
                print(f"Found company name through stock code mapping: {stock_code} -> {mapped_company}")
        
        # If company name is provided, try to get the corresponding stock code
        if company_name and not stock_code:
            mapped_stock = self.company_stock_mapping.get(company_name)
            if mapped_stock:
                enhanced_stock_code = mapped_stock
                print(f"Found stock code through company name mapping: {company_name} -> {mapped_stock}")
        
        # Prioritize using combined index (company name + stock code)
        if enhanced_company_name and enhanced_stock_code:
            company_name_lower = enhanced_company_name.strip().lower()
            stock_code_lower = str(enhanced_stock_code).strip().lower()
            key = f"{company_name_lower}_{stock_code_lower}"
            
            if key in self.metadata_index['company_stock']:
                indices = self.metadata_index['company_stock'][key]
                print(f"Combined filtering: company '{enhanced_company_name}' + stock '{enhanced_stock_code}' matches {len(indices)} records")
                return indices[:max_candidates]
            else:
                print(f"Combined filtering: company '{enhanced_company_name}' + stock '{enhanced_stock_code}' no matching records")
                # If combined matching fails, try single matching
                return self._fallback_filter(enhanced_company_name, enhanced_stock_code, report_date, max_candidates)
        
        # If only company name is provided
        elif enhanced_company_name:
            company_name_lower = enhanced_company_name.strip().lower()
            if company_name_lower in self.metadata_index['company_name']:
                indices = self.metadata_index['company_name'][company_name_lower]
                print(f"Company name filtering: '{enhanced_company_name}' matches {len(indices)} records")
                return indices[:max_candidates]
            else:
                print(f"Company name filtering: '{enhanced_company_name}' no matching records")
                # Try fuzzy matching
                return self._fuzzy_company_match(enhanced_company_name, max_candidates)
        
        # If only stock code is provided
        elif enhanced_stock_code:
            stock_code_lower = str(enhanced_stock_code).strip().lower()
            if stock_code_lower in self.metadata_index['stock_code']:
                indices = self.metadata_index['stock_code'][stock_code_lower]
                print(f"Stock code filtering: '{enhanced_stock_code}' matches {len(indices)} records")
                return indices[:max_candidates]
            else:
                print(f"Stock code filtering: '{enhanced_stock_code}' no matching records")
                return []
        
        # If only report date is provided
        elif report_date:
            report_date_str = report_date.strip()
            if report_date_str in self.metadata_index['report_date']:
                indices = self.metadata_index['report_date'][report_date_str]
                print(f"Report date filtering: '{report_date}' matches {len(indices)} records")
                return indices[:max_candidates]
            else:
                print(f"Report date filtering: '{report_date}' no matching records")
                return []
        
        print("Pre-filter completed, candidate document count: 0")
        return []
    
    def _fallback_filter(self, company_name: Optional[str], stock_code: Optional[str], 
                        report_date: Optional[str], max_candidates: int) -> List[int]:
        """Fallback strategy when combined matching fails"""
        print("Combined matching failed, try single matching...")
        
        all_indices = set()
        
        # Try company name matching
        if company_name:
            company_name_lower = company_name.strip().lower()
            if company_name_lower in self.metadata_index['company_name']:
                indices = self.metadata_index['company_name'][company_name_lower]
                all_indices.update(indices)
                print(f"Fallback company name matching: {len(indices)} records")
        
        # Try stock code matching
        if stock_code:
            stock_code_lower = str(stock_code).strip().lower()
            if stock_code_lower in self.metadata_index['stock_code']:
                indices = self.metadata_index['stock_code'][stock_code_lower]
                all_indices.update(indices)
                print(f"Fallback stock code matching: {len(indices)} records")
        
        # Try report date matching
        if report_date:
            report_date_str = report_date.strip()
            if report_date_str in self.metadata_index['report_date']:
                indices = self.metadata_index['report_date'][report_date_str]
                all_indices.update(indices)
                print(f"Fallback report date matching: {len(indices)} records")
        
        result = list(all_indices)[:max_candidates]
        print(f"Fallback strategy total matching: {len(result)} records")
        return result
    
    def _fuzzy_company_match(self, company_name: str, max_candidates: int) -> List[int]:
        """Fuzzy company name matching"""
        print(f"Trying fuzzy matching company name: {company_name}")
        
        company_name_lower = company_name.strip().lower()
        all_indices = set()
        
        # Find records containing the company name in the metadata index
        for indexed_name, indices in self.metadata_index['company_name'].items():
            if (company_name_lower in indexed_name or 
                indexed_name in company_name_lower or
                any(word in indexed_name for word in company_name_lower.split())):
                all_indices.update(indices)
                print(f"Fuzzy matching: '{indexed_name}' -> {len(indices)} records")
        
        result = list(all_indices)[:max_candidates]
        print(f"Fuzzy matching total result: {len(result)} records")
        return result
    
    def faiss_search(self, query: str, candidate_indices: List[int], top_k: int = 100) -> List[Tuple[int, float]]:
        """
        Use FAISS for vector retrieval
        
        Args:
            query: query text
            candidate_indices: candidate record indices
            top_k: return top k results
            
        Returns:
            List of (index, similarity score)
        """
        if self.faiss_index is None:
            print("FAISS index not initialized")
            return []
        
        print(f"Starting FAISS retrieval, candidate document count: {len(candidate_indices)}")
        
        # If there are no candidate documents, return empty result
        if not candidate_indices:
            print("No candidate documents, return empty result")
            return []
        
        # Generate query embedding
        try:
            query_embedding = self.embedding_model.encode([query])
            print(f"Query embedding generation completed, dimension: {query_embedding.shape}")
        except Exception as e:
            print(f"Query embedding generation failed: {e}")
            return []
        
        # Use existing FAISS index for efficient search
        print("Using existing FAISS index for efficient search")
        
        try:
            # Search on the complete FAISS index, then filter candidate documents
            search_k = min(top_k * 5, len(self.valid_indices))  # Search more to ensure coverage of candidate documents
            scores, indices = self.faiss_index.search(query_embedding.astype('float32'), search_k)
            
            # Map FAISS index back to original data index, and limit to candidate documents
            results = []
            candidate_set = set(candidate_indices)
            
            for faiss_idx, score in zip(indices[0], scores[0]):
                if faiss_idx < len(self.valid_indices):
                    original_idx = self.valid_indices[faiss_idx]
                    # Check if it is in the candidate list
                    if original_idx in candidate_set:
                        results.append((original_idx, float(score)))
                        # If enough candidates are found, end early
                        if len(results) >= top_k:
                            break
            
            print(f"FAISS retrieval completed, valid results: {len(results)} records")
            return results
            
        except Exception as e:
            print(f"FAISS retrieval failed: {e}")
            return []
    
    def rerank(self, 
               query: str, 
               candidate_results: List[Tuple[int, float]], 
               top_k: int = 10) -> List[Tuple[int, float, float]]:  # Changed to 10, consistent with configuration file
        """
        Use Qwen reranker to rerank candidate results
        
        Args:
            query: query text
            candidate_results: candidate results list [(doc_idx, faiss_score), ...]
            top_k: return top k results
            
        Returns:
            List of (doc_idx, faiss_score, reranker_score), ...]
        """
        if not self.qwen_reranker or not candidate_results:
            print("Reranker not available or no candidate results")
            return [(idx, score, 0.0) for idx, score in candidate_results[:top_k]]
        
        print(f"Starting reranking {len(candidate_results)} candidate results...")
        
        # Prepare documents for reranking - use doc_id to chunks mapping
        docs_for_rerank = []
        doc_to_rerank_mapping = []
        
        for doc_idx, faiss_score in candidate_results:
            if doc_idx in self.doc_to_chunks_mapping:
                chunks = self.doc_to_chunks_mapping[doc_idx]
                for chunk in chunks:
                    if chunk.strip():  # Skip empty chunk
                        docs_for_rerank.append(chunk)
                        doc_to_rerank_mapping.append((doc_idx, faiss_score))
            else:
                # If no mapping is found, use original data
                if doc_idx < len(self.data):
                    record = self.data[doc_idx]
                    if self.dataset_type == "chinese":
                        content = record.get('summary', '')
                    else:
                        content = record.get('context', '') or record.get('content', '')
                    if content.strip():
                        docs_for_rerank.append(content)
                        doc_to_rerank_mapping.append((doc_idx, faiss_score))
        
        print(f"Preparing reranking {len(docs_for_rerank)} chunks...")
        
        if not docs_for_rerank:
            print("No documents to rerank")
            return [(idx, score, 0.0) for idx, score in candidate_results[:top_k]]
        
        # Use Qwen reranker to rerank
        try:
            # Prepare doc_ids for new method
            doc_ids = []
            for doc_idx, faiss_score in candidate_results:
                if doc_idx < len(self.data):
                    doc_id = self.data[doc_idx].get('doc_id', f'doc_{doc_idx}')
                    doc_ids.append(doc_id)
                else:
                    doc_ids.append(f'doc_{doc_idx}')
            
            # Use new rerank_with_doc_ids method
            reranked_results = self.qwen_reranker.rerank_with_doc_ids(
                query=query,
                documents=docs_for_rerank,
                doc_ids=doc_ids,
                batch_size=1  # Reduce to 1 to avoid GPU memory issues
            )
            print(f"Reranker processing completed, returning {len(reranked_results)} results")
        except Exception as e:
            print(f"Reranking failed: {e}")
            # Fallback to original results
            return [(idx, score, 0.0) for idx, score in candidate_results[:top_k]]
        
        # Map reranking results back to original document indices
        final_results = []
        for doc_text, reranker_score, doc_id in reranked_results:
            # Find the corresponding original document index
            for i, (doc_idx, faiss_score) in enumerate(candidate_results):
                if doc_idx < len(self.data) and self.data[doc_idx].get('doc_id') == doc_id:
                    # Combined score: FAISS score + reranker score
                    combined_score = faiss_score + reranker_score
                    final_results.append((doc_idx, faiss_score, combined_score))
                    break
        
        # Sort by combined score
        final_results.sort(key=lambda x: x[2], reverse=True)
        
        print(f"Reranking completed, returning {len(final_results)} results")
        return final_results[:top_k]
    
    def generate_answer(self, query: str, candidate_results: List[Tuple[int, float, float]], top_k_for_context: int = 5) -> str:
        """
        Generate LLM answer - use smart context extraction, significantly reducing the context passed to LLM
        
        Args:
            query: query text
            candidate_results: candidate results list [(doc_idx, faiss_score, reranker_score), ...]
            top_k_for_context: number of candidates for context generation
            
        Returns:
            Generated LLM answer
        """
        if not candidate_results:
            print("No candidate results, cannot generate answer")
            return ""
        
        print(f"Starting to generate LLM answer...")
        print(f"Original query: '{query}'")
        print(f"Query length: {len(query)} characters")
        
        # Use smart context extraction, limit to 2000 characters
        context = self.extract_relevant_context(query, candidate_results, max_chars=2000)
        
        print(f"Smart context extraction length: {len(context)} characters")
        
        # Use LLM generator to generate answer
        if self.llm_generator:
            try:
                # Select prompt template based on dataset type
                if self.dataset_type == "chinese":
                    # Chinese prompt template
                    # Get summary of Top1 document
                    if candidate_results and candidate_results[0][0] < len(self.data):
                        top1_record = self.data[candidate_results[0][0]]
                        summary = top1_record.get('summary', '')
                        if not summary:
                            # If there is no summary field, use the first 200 characters of context
                            summary = context[:200] + "..." if len(context) > 200 else context
                    else:
                        summary = context[:200] + "..." if len(context) > 200 else context
                    
                    prompt = template_loader.format_template(
                        "multi_stage_chinese_template",
                        context=context, 
                        query=query,
                        summary=summary
                    )
                else:
                    # English prompt template
                    prompt = template_loader.format_template(
                        "multi_stage_english_template",
                        context=context, 
                        query=query
                    )
                
                if prompt is None:
                    # Fallback to simple prompt
                    if self.dataset_type == "chinese":
                        prompt = f"Based on the following context, answer the question:\n\n{context}\n\nQuestion: {query}\n\nAnswer:"
                    else:
                        prompt = f"Context: {context}\nQuestion: {query}\nAnswer:"
                
                # ===== Detailed Prompt debugging information =====
                print("\n" + "="*80)
                print("PROMPT debugging information")
                print("="*80)
                print(f"Template name: {'multi_stage_chinese_template' if self.dataset_type == 'chinese' else 'multi_stage_english_template'}")
                print(f"Full prompt length: {len(prompt)} characters")
                print(f"Original query: '{query}'")
                print(f"Query length: {len(query)} characters")
                print(f"Context length: {len(context)} characters")
                print(f"Context first 200 characters: '{context[:200]}...'")
                print(f"Context last 200 characters: '...{context[-200:]}'")
                
                # Check if Prompt is truncated
                if len(prompt) > 10000:
                    print("WARNING: Prompt length exceeds 10000 characters, may be truncated")
                else:
                    print("Prompt length is normal")
                
                # Check if query is in Prompt
                if query in prompt:
                    print("Query is correctly included in Prompt")
                else:
                    print("Query not found in Prompt")
                    print(f"Expected query: '{query}'")
                    print(f"Query part in Prompt: '{prompt.split('Question:')[-1].split('Answer:')[0] if 'Question:' in prompt else 'NOT_FOUND'}'")
                
                # Check if context is in Prompt
                if context[:100] in prompt:
                    print("Context is correctly included in Prompt")
                else:
                    print("Context not found in Prompt")
                
                print("\n" + "="*80)
                print("Full Prompt sent to LLM:")
                print("="*80)
                print(prompt)
                print("="*80)
                print("Prompt ends")
                print("="*80 + "\n")
                
                # Generate answer
                answer = self.llm_generator.generate(texts=[prompt])[0]
                
                # ===== Answer debugging information =====
                print("\n" + "="*80)
                print("LLM generated answer:")
                print("="*80)
                print(answer)
                print("="*80)
                print("Answer ends")
                print("="*80 + "\n")
                
                return answer
            except Exception as e:
                print(f"Error generating answer: {e}")
                return "Error generating answer."
        else:
            return "LLM generator not configured."
    
    def search(self, 
               query: str,
               company_name: Optional[str] = None,
               stock_code: Optional[str] = None,
               report_date: Optional[str] = None,
               top_k: int = 10,
               use_prefilter: bool = True) -> Dict:  # Remove mapping switch parameter
        """
        Full multi-stage retrieval process
        
        Args:
            query: query text
            company_name: company name (optional, only for Chinese data)
            stock_code: stock code (optional, only for Chinese data)
            report_date: report date (optional, only for Chinese data)
            top_k: return top k results
            use_prefilter: whether to use prefilter (default True, mapping is automatically enabled when used)
            
        Returns:
            Retrieval result list
        """
        print(f"\nStarting multi-stage retrieval...")
        print(f"Query: {query}")
        print(f"Dataset type: {self.dataset_type}")
        print(f"Prefilter switch: {'Enabled' if use_prefilter else 'Disabled'}")
        if use_prefilter:
            print("Prefilter mode automatically enables stock code and company name mapping")
        
        if self.dataset_type == "chinese":
            if company_name:
                print(f"Company name: {company_name}")
            if stock_code:
                print(f"Stock code: {stock_code}")
            if report_date:
                print(f"Report date: {report_date}")
        else:
            print("English dataset, metadata filtering is not supported")
        
        # Use configured retrieval parameters
        retrieval_top_k = 100
        rerank_top_k = top_k
        
        if self.config:
            retrieval_top_k = self.config.retriever.retrieval_top_k
            rerank_top_k = self.config.retriever.rerank_top_k
        
        # 1. Pre-filtering (based on switch)
        if use_prefilter and self.dataset_type == "chinese":
            print("Step 1: Enable metadata pre-filtering...")
            candidate_indices = self.pre_filter(company_name, stock_code, report_date) # 预过滤时自动启用映射
            print(f"Pre-filtering result: {len(candidate_indices)} candidate documents")
            
            # If pre-filtering does not find matching documents, fall back to full FAISS retrieval
            if len(candidate_indices) == 0:
                print("Pre-filtering found no results, falling back to full FAISS retrieval...")
                candidate_indices = list(range(len(self.data)))
                print(f"Falling back to full retrieval, candidate documents: {len(candidate_indices)}")
        else:
            print("Step 1: Skip metadata pre-filtering, use full retrieval...")
            candidate_indices = list(range(len(self.data)))
            print(f"Full retrieval, candidate documents: {len(candidate_indices)}")
        
        # 2. FAISS retrieval - based on pre-filtering results, but ensure retrieval of configured candidate number
        print("Step 2: Based on candidate documents, perform FAISS retrieval...")
        # If candidate results are less than the configured retrieval number, use candidate results; otherwise use configured retrieval number
        actual_top_k = min(retrieval_top_k, len(candidate_indices))
        faiss_results = self.faiss_search(query, candidate_indices, top_k=actual_top_k)
        print(f"FAISS retrieval result: {len(faiss_results)} documents")
        final_faiss_results = faiss_results
        
        # 3. Qwen Reranker
        print("Step 3: Start reranking...")
        final_results = self.rerank(query, final_faiss_results, top_k=rerank_top_k)
        print(f"Reranking completed: {len(final_results)} chunks")
        print("Reranker processing completed")
        
        # 4. LLM answer generation - concatenate the top-K1 chunks after reranking as context
        llm_answer = self.generate_answer(query, final_results, top_k_for_context=5)
        
        # 5. Format results
        formatted_results = []
        for idx, faiss_score, combined_score in final_results:
            record = self.data[idx]
            
            # Select different fields based on dataset type
            if hasattr(record, 'content'):
                # DocumentWithMetadata format
                result = {
                    'index': idx,
                    'faiss_score': faiss_score,
                    'combined_score': combined_score,
                    'content': record.content[:200] + '...' if len(record.content) > 200 else record.content,
                    'source': record.metadata.source if hasattr(record.metadata, 'source') else 'unknown',
                    'language': record.metadata.language if hasattr(record.metadata, 'language') else 'unknown'
                }
            else:
                # Dictionary format
                if self.dataset_type == "chinese":
                    # Chinese data: use original_context
                    context = record.get('original_context', '')
                    result = {
                        'index': idx,
                        'faiss_score': faiss_score,
                        'combined_score': combined_score,
                        'context': context[:200] + '...' if len(context) > 200 else context,
                        'company_name': record.get('company_name', ''),
                        'stock_code': record.get('stock_code', ''),
                        'report_date': record.get('report_date', ''),
                        'summary': record.get('summary', '')[:200] + '...' if len(record.get('summary', '')) > 200 else record.get('summary', ''),
                        'generated_question': record.get('generated_question', ''),
                        'original_question': record.get('original_question', ''),
                        'original_answer': record.get('original_answer', '')
                    }
                else:
                    # English data: use context
                    context = record.get('context', '') or record.get('content', '')
                    result = {
                        'index': idx,
                        'faiss_score': faiss_score,
                        'combined_score': combined_score,
                        'context': context[:200] + '...' if len(context) > 200 else context,
                        'question': record.get('question', ''),
                        'answer': record.get('answer', '')
                    }
            
            formatted_results.append(result)
        
        # Add LLM generated answer to results
        final_output = {
            'retrieved_documents': formatted_results,
            'llm_answer': llm_answer,
            'query': query,
            'total_documents': len(formatted_results),
            'use_prefilter': use_prefilter  # Add prefilter status to output
        }
        
        print(f"Retrieval completed, returning {len(formatted_results)} results")
        print(f"LLM answer generation completed")
        return final_output
    
    def save_index(self, output_dir: Path):
        """Save index to file"""
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Save FAISS index
        if self.faiss_index:
            faiss.write_index(self.faiss_index, str(output_dir / "faiss_index.bin"))
        
        # Save metadata index (only for Chinese data)
        if self.dataset_type == "chinese":
            with open(output_dir / "metadata_index.pkl", 'wb') as f:
                pickle.dump(self.metadata_index, f)
        
        # Save valid index mapping
        with open(output_dir / "valid_indices.pkl", 'wb') as f:
            pickle.dump(self.valid_indices, f)
        
        # Save dataset type information
        with open(output_dir / "dataset_info.json", 'w') as f:
            json.dump({
                'dataset_type': self.dataset_type,
                'model_name': self.model_name
            }, f, indent=2)
        
        print(f"Index saved to: {output_dir}")
    
    def load_index(self, index_dir: Path):
        """Load index from file"""
        # Load dataset information
        info_path = index_dir / "dataset_info.json"
        if info_path.exists():
            with open(info_path, 'r') as f:
                info = json.load(f)
                self.dataset_type = info.get('dataset_type', 'chinese')
                self.model_name = info.get('model_name', 'all-MiniLM-L6-v2')
        
        # Load FAISS index
        faiss_path = index_dir / "faiss_index.bin"
        if faiss_path.exists():
            self.faiss_index = faiss.read_index(str(faiss_path))
        
        # Load metadata index (only for Chinese data)
        if self.dataset_type == "chinese":
            metadata_path = index_dir / "metadata_index.pkl"
            if metadata_path.exists():
                with open(metadata_path, 'rb') as f:
                    self.metadata_index = pickle.load(f)
        
        # Load valid index mapping
        valid_indices_path = index_dir / "valid_indices.pkl"
        if valid_indices_path.exists():
            with open(valid_indices_path, 'rb') as f:
                self.valid_indices = pickle.load(f)
        
        print(f"Index loaded from {index_dir}")
        print(f"Dataset type: {self.dataset_type}")

    def extract_relevant_context(self, query: str, candidate_results: List[Tuple[int, float, float]], max_chars: int = 2000) -> str:
        """
        Smartly extract relevant context from the Top1 document
        
        Args:
            query: query text
            candidate_results: candidate results list
            max_chars: maximum character limit
            
        Returns:
            Relevant context extracted from the Top1 document
        """
        print(f"Start smartly extracting relevant context from the Top1 document...")
        print(f"Query: {query}")
        print(f"Candidate documents: {len(candidate_results)}")
        
        if not candidate_results:
            print("No candidate results")
            return ""
        
        # Get Top1 document
        top1_idx, top1_faiss_score, top1_reranker_score = candidate_results[0]
        
        if top1_idx >= len(self.data):
            print(f"Top1 document index out of range: {top1_idx}")
            return ""
        
        record = self.data[top1_idx]
        print(f"Using Top1 document (index: {top1_idx}, FAISS score: {top1_faiss_score:.4f}, Reranker score: {top1_reranker_score:.4f})")
        
        # Get full context of Top1 document
        if self.dataset_type == "chinese":
            full_context = record.get('original_context', '')
            if not full_context:
                full_context = record.get('summary', '')
        else:
            full_context = record.get('context', '') or record.get('content', '')
        
        if not full_context:
            print("Top1 document has no context content")
            return ""
        
        print(f"Top1 document full context length: {len(full_context)} characters")
        
        # Smartly extract relevant context from the Top1 document
        # Extract query keywords
        query_keywords = self._extract_keywords(query)
        print(f"Query keywords: {query_keywords}")
        
        # Smartly extract relevant sentences
        relevant_sentences = self._extract_relevant_sentences(full_context, query_keywords, max_chars_per_doc=max_chars)
        
        # Concatenate context
        context = "\n\n".join(relevant_sentences)
        
        print(f"Top1 document smartly extracted context completed:")
        print(f"   Original length: {len(full_context)} characters")
        print(f"   Extracted length: {len(context)} characters")
        print(f"   Sentence number: {len(relevant_sentences)}")
        print(f"   First 100 characters: {context[:100]}...")
        
        return context
    
    def _extract_keywords(self, query: str) -> List[str]:
        """Extract query keywords"""
        # Simple keyword extraction
        keywords = []
        
        # Extract stock code
        import re
        stock_pattern = r'[A-Z]{2}\d{4}|[A-Z]{2}\d{6}|\d{6}'
        stock_matches = re.findall(stock_pattern, query)
        keywords.extend(stock_matches)
        
        # Extract company name
        company_pattern = r'([A-Za-z\u4e00-\u9fff]+)(?:公司|集团|股份|有限)'
        company_matches = re.findall(company_pattern, query)
        keywords.extend(company_matches)
        
        # Extract year
        year_pattern = r'20\d{2}年'
        year_matches = re.findall(year_pattern, query)
        keywords.extend(year_matches)
        
        # Extract key concepts
        key_concepts = ['利润', '营收', '增长', '业绩', '预测', '原因', '主要', '持续']
        for concept in key_concepts:
            if concept in query:
                keywords.append(concept)
        
        return list(set(keywords))
    
    def _extract_relevant_sentences(self, content: str, keywords: List[str], max_chars_per_doc: int = 800) -> List[str]:
        """Extract sentences most relevant to keywords from the document"""
        if not content or not keywords:
            return []
        
        # Split by sentence
        import re
        sentences = re.split(r'[。！？\n]+', content)
        sentences = [s.strip() for s in sentences if s.strip()]
        
        # Calculate the relevance score of each sentence
        sentence_scores = []
        for sentence in sentences:
            score = 0
            for keyword in keywords:
                if keyword in sentence:
                    score += 1
            # Consider sentence length, avoid too long sentences
            if len(sentence) > 200:
                score *= 0.5
            sentence_scores.append((sentence, score))
        
        # Sort by score
        sentence_scores.sort(key=lambda x: x[1], reverse=True)
        
        # Select the most relevant sentences
        selected_sentences = []
        total_chars = 0
        
        for sentence, score in sentence_scores:
            if score > 0 and total_chars + len(sentence) <= max_chars_per_doc:
                selected_sentences.append(sentence)
                total_chars += len(sentence)
        
        return selected_sentences

def main():
    """Main function - demonstrate multi-stage retrieval system"""
    # Data file path
    data_path = Path("data/alphafin/alphafin_merged_generated_qa.json")
    index_dir = Path("data/alphafin/retrieval_index")
    
    # Initialize retrieval system (Chinese data)
    print("Initializing multi-stage retrieval system (Chinese data)...")
    retrieval_system = MultiStageRetrievalSystem(data_path, dataset_type="chinese")
    
    # Save index (optional)
    retrieval_system.save_index(index_dir)
    
    # Demonstrate retrieval
    print("\n" + "="*50)
    print("Retrieval demonstration")
    print("="*50)
    
    # Example query 1: Search based on company name (only Chinese data supported)
    print("\nExample 1: Search based on company name")
    results1 = retrieval_system.search(
        query="公司业绩表现如何？",
        company_name="中国宝武",
        top_k=5
    )
    
    for i, result in enumerate(results1['retrieved_documents']):
        print(f"\nResult {i+1}:")
        print(f"  Company: {result['company_name']}")
        print(f"  Stock code: {result['stock_code']}")
        print(f"  Summary: {result['summary']}")
        print(f"  Similarity score: {result['combined_score']:.4f}")
    
    # Example query 2: General search (no metadata filtering)
    print("\nExample 2: General search (no metadata filtering)")
    results2 = retrieval_system.search(
        query="钢铁行业发展趋势",
        top_k=5
    )
    
    for i, result in enumerate(results2['retrieved_documents']):
        print(f"\nResult {i+1}:")
        print(f"  Company: {result['company_name']}")
        print(f"  Stock code: {result['stock_code']}")
        print(f"  Summary: {result['summary']}")
        print(f"  Similarity score: {result['combined_score']:.4f}")

if __name__ == '__main__':
    main() 