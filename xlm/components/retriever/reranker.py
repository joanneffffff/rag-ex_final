"""
Reranker component using Qwen3-0.6B model for document reranking
Following official implementation guidelines
"""

from typing import List, Dict, Tuple, Optional, Union
import torch
import numpy as np
from transformers import AutoModelForCausalLM, AutoTokenizer
from transformers.utils.quantization_config import BitsAndBytesConfig
from tqdm import tqdm

class QwenReranker:
    def __init__(
        self,
        model_name: str = "Qwen/Qwen3-Reranker-0.6B",
        device: Optional[str] = None,
        cache_dir: Optional[str] = None,
        use_quantization: bool = True,
        quantization_type: str = "4bit",  # "4bit" or "8bit"
        use_flash_attention: bool = False
    ):
        """
        Initialize Qwen reranker
        
        Args:
            model_name: Model name or path
            device: Device (cpu/cuda)
            cache_dir: Model cache directory
            use_quantization: Whether to use quantization
            quantization_type: Quantization type ("8bit" or "4bit")
            use_flash_attention: Whether to use flash attention
        """
        self.model_name = model_name
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.cache_dir = cache_dir
        
        print(f"\nLoading reranker model: {model_name}")
        print(f"- Device: {self.device}")
        print(f"- Cache directory: {cache_dir}")
        print(f"- Quantization: {use_quantization} ({quantization_type})")
        print(f"- Flash Attention: {use_flash_attention}")
        
        # Configure quantization parameters
        quantization_config = None
        if use_quantization:
            if quantization_type == "8bit":
                quantization_config = BitsAndBytesConfig(load_in_8bit=True)
            elif quantization_type == "4bit":
                quantization_config = BitsAndBytesConfig(
                    load_in_4bit=True,
                    bnb_4bit_quant_type="nf4",
                    bnb_4bit_compute_dtype=torch.float16
                )
        
        # Load tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(
            model_name,
            cache_dir=cache_dir,
            padding_side='left'
        )
        
        # Load model
        model_kwargs = {
            "torch_dtype": torch.float16,
            "cache_dir": cache_dir
        }
        
        if quantization_config:
            model_kwargs["quantization_config"] = quantization_config
        
        if use_flash_attention:
            model_kwargs["attn_implementation"] = "flash_attention_2"
        
        self.model = AutoModelForCausalLM.from_pretrained(
            model_name,
            **model_kwargs
        )
        
        # Move to device
        if quantization_config:
            # Quantized model needs explicit device assignment
            print(f"Quantized model moved to device: {self.device}")
            self.model = self.model.to(self.device)
        else:
            # Non-quantized model needs manual device assignment
            if self.device.startswith("cuda"):
                self.model = self.model.to(self.device)
            else:
                self.model = self.model.to(self.device)
        
        self.model.eval()
        
        # Get special token IDs
        self.token_false_id = self.tokenizer.convert_tokens_to_ids("no")
        self.token_true_id = self.tokenizer.convert_tokens_to_ids("yes")
        
        # Set max length
        self.max_length = 8192
        
        # Set prompt templates (following official implementation)
        self.prefix = "<|im_start|>system\nJudge whether the Document meets the requirements based on the Query and the Instruct provided. Note that the answer can only be \"yes\" or \"no\".<|im_end|>\n<|im_start|>user\n"
        self.suffix = "<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n"
        
        # Pre-encode prefix and suffix tokens (following official implementation)
        self.prefix_tokens = self.tokenizer.encode(self.prefix, add_special_tokens=False)
        self.suffix_tokens = self.tokenizer.encode(self.suffix, add_special_tokens=False)
        
        print("Reranker model loaded successfully")
    
    def format_instruction(self, instruction: Optional[str], query: str, document: str) -> str:
        """
        Format instruction (following official implementation)
        
        Args:
            instruction: Instruction text
            query: Query text
            document: Document text
            
        Returns:
            Formatted instruction string
        """
        if instruction is None:
            instruction = 'Given a web search query, retrieve relevant passages that answer the query'
        
        return f"<Instruct>: {instruction}\n<Query>: {query}\n<Document>: {document}"
    
    def process_inputs(self, pairs: List[str]) -> Dict[str, torch.Tensor]:
        """
        Process inputs (optimized version, directly uses tokenizer.__call__ method)
        
        Args:
            pairs: Formatted instruction string list
            
        Returns:
            Tokenizer output dictionary
        """
        # Add prefix and suffix for each input
        processed_pairs = []
        for pair in pairs:
            # Combine strings directly, rather than pre-encoding tokens
            full_text = self.prefix + pair + self.suffix
            processed_pairs.append(full_text)
        
        # Use tokenizer to process the full string
        inputs = self.tokenizer(
            processed_pairs,
            padding='max_length',
            truncation=True,
            max_length=self.max_length,
            return_tensors="pt"
        )
        
        # Move to the correct device
        for key in inputs:
            inputs[key] = inputs[key].to(self.model.device)
        
        return inputs
    
    @torch.no_grad()
    def compute_logits(self, inputs: Dict[str, torch.Tensor]) -> List[float]:
        """
        Compute logits (following official implementation)
        
        Args:
            inputs: Tokenizer output
            
        Returns:
            Score list
        """
        batch_scores = self.model(**inputs).logits[:, -1, :]
        true_vector = batch_scores[:, self.token_true_id]
        false_vector = batch_scores[:, self.token_false_id]
        batch_scores = torch.stack([false_vector, true_vector], dim=1)
        batch_scores = torch.nn.functional.log_softmax(batch_scores, dim=1)
        scores = batch_scores[:, 1].exp().tolist()
        return scores
    
    def rerank(
        self,
        query: str,
        documents: List[str],
        batch_size: int = 1  # Reduce to 1 to avoid memory issues
    ) -> List[Tuple[str, float]]:
        """
        Rerank documents (optimized memory usage)
        
        Args:
            query: Query text
            documents: Document list
            batch_size: Batch size (default 2 to reduce memory usage)
            
        Returns:
            List of (document, score) tuples
        """
        if not documents:
            return []
        
        # Format all documents
        formatted_pairs = []
        for doc in documents:
            formatted_text = self.format_instruction(None, query, doc)
            formatted_pairs.append((formatted_text, doc))
        
        # Batch reranking (optimized memory usage)
        all_scores = []
        for i in range(0, len(formatted_pairs), batch_size):
            batch_pairs = formatted_pairs[i:i + batch_size]
            batch_texts = [pair[0] for pair in batch_pairs]
            
            try:
                # Process inputs
                inputs = self.process_inputs(batch_texts)
                
                # Compute scores
                batch_scores = self.compute_logits(inputs)
                all_scores.extend(batch_scores)
                
                # Clean up GPU memory
                del inputs
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                    
                # Force garbage collection
                import gc
                gc.collect()
                
                # More frequent memory cleanup
                if torch.cuda.is_available():
                    torch.cuda.synchronize()
                    torch.cuda.empty_cache()
                    
            except RuntimeError as e:
                if "out of memory" in str(e):
                    print(f"GPU memory insufficient, attempting to reduce batch size...")
                    # If memory is insufficient, try a smaller batch size
                    if batch_size > 1:
                        # Recursive call, using a smaller batch size
                        return self.rerank(query, documents, batch_size=batch_size // 2)
                    else:
                        print("Batch size is already minimized, still insufficient memory, attempting CPU processing...")
                        # Finally attempt CPU processing
                        return self._rerank_on_cpu(query, documents)
                else:
                    raise e
        
        # Combine documents and scores
        results = []
        for i, (formatted_text, doc) in enumerate(formatted_pairs):
            results.append((doc, all_scores[i]))
        
        # Sort by score in descending order
        results.sort(key=lambda x: x[1], reverse=True)
        
        return results

    def rerank_with_doc_ids(
        self,
        query: str,
        documents: List[str],
        doc_ids: List[str],
        batch_size: int = 1
    ) -> List[Tuple[str, float, str]]:
        """
        Rerank documents and return doc_id
        
        Args:
            query: Query text
            documents: Document list
            doc_ids: Document ID list (corresponding to documents)
            batch_size: Batch size
            
        Returns:
            List of (document text, score, doc_id) tuples
        """
        if not documents or not doc_ids or len(documents) != len(doc_ids):
            print("Warning: Document list and doc_id list do not match or are empty")
            return []
        
        # Format all documents
        formatted_pairs = []
        for doc, doc_id in zip(documents, doc_ids):
            formatted_text = self.format_instruction(None, query, doc)
            formatted_pairs.append((formatted_text, doc, doc_id))
        
        # Batch reranking
        all_scores = []
        for i in range(0, len(formatted_pairs), batch_size):
            batch_pairs = formatted_pairs[i:i + batch_size]
            batch_texts = [pair[0] for pair in batch_pairs]
            
            try:
                # Process inputs
                inputs = self.process_inputs(batch_texts)
                
                # Compute scores
                batch_scores = self.compute_logits(inputs)
                all_scores.extend(batch_scores)
                
                # Clean up GPU memory
                del inputs
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                    
                # Force garbage collection
                import gc
                gc.collect()
                
                # More frequent memory cleanup
                if torch.cuda.is_available():
                    torch.cuda.synchronize()
                    torch.cuda.empty_cache()
                    
            except RuntimeError as e:
                if "out of memory" in str(e):
                    print(f"GPU memory insufficient, attempting to reduce batch size...")
                    # If memory is insufficient, try a smaller batch size
                    if batch_size > 1:
                        # Recursive call, using a smaller batch size
                        return self.rerank_with_doc_ids(query, documents, doc_ids, batch_size=batch_size // 2)
                    else:
                        print("Batch size is already minimized, still insufficient memory, attempting CPU processing...")
                        # Finally attempt CPU processing
                        return self._rerank_with_doc_ids_on_cpu(query, documents, doc_ids)
                else:
                    raise e
        
        # Combine documents, scores, and doc_id
        results = []
        for i, (formatted_text, doc, doc_id) in enumerate(formatted_pairs):
            results.append((doc, all_scores[i], doc_id))
        
        # Sort by score in descending order
        results.sort(key=lambda x: x[1], reverse=True)
        
        return results

    def _rerank_with_doc_ids_on_cpu(self, query: str, documents: List[str], doc_ids: List[str]) -> List[Tuple[str, float, str]]:
        """
        CPU fallback reranking (when GPU memory is insufficient)
        
        Args:
            query: Query text
            documents: Document list
            doc_ids: Document ID list
            
        Returns:
            List of (document text, score, doc_id) tuples
        """
        print("Using CPU for reranking...")
        
        # Temporarily move model to CPU
        original_device = next(self.model.parameters()).device
        self.model = self.model.cpu()
        
        try:
            # Format all documents
            formatted_pairs = []
            for doc, doc_id in zip(documents, doc_ids):
                formatted_text = self.format_instruction(None, query, doc)
                formatted_pairs.append((formatted_text, doc, doc_id))
            
            # Process one by one (CPU mode)
            all_scores = []
            for formatted_text, doc, doc_id in formatted_pairs:
                inputs = self.process_inputs([formatted_text])
                score = self.compute_logits(inputs)[0]
                all_scores.append(score)
                del inputs
            
            # Combine documents, scores, and doc_id
            results = []
            for i, (formatted_text, doc, doc_id) in enumerate(formatted_pairs):
                results.append((doc, all_scores[i], doc_id))
            
            # Sort by score in descending order
            results.sort(key=lambda x: x[1], reverse=True)
            
            return results
            
        finally:
            # Restore model to original device
            self.model = self.model.to(original_device)
    
    def _rerank_on_cpu(self, query: str, documents: List[str]) -> List[Tuple[str, float]]:
        """
        CPU fallback reranking (when GPU memory is insufficient)
        
        Args:
            query: Query text
            documents: Document list
            
        Returns:
            List of (document, score) tuples
        """
        print("Using CPU for reranking...")
        
        # Temporarily move model to CPU
        original_device = next(self.model.parameters()).device
        self.model = self.model.cpu()
        
        try:
            # Format all documents
            formatted_pairs = []
            for doc in documents:
                formatted_text = self.format_instruction(None, query, doc)
                formatted_pairs.append((formatted_text, doc))
            
            # Process one by one (CPU mode)
            all_scores = []
            for formatted_text, doc in formatted_pairs:
                inputs = self.process_inputs([formatted_text])
                score = self.compute_logits(inputs)[0]
                all_scores.append(score)
                del inputs
            
            # Combine documents and scores
            results = []
            for i, (formatted_text, doc) in enumerate(formatted_pairs):
                results.append((doc, all_scores[i]))
            
            # Sort by score in descending order
            results.sort(key=lambda x: x[1], reverse=True)
            
            return results
            
        finally:
            # Restore model to original device
            self.model = self.model.to(original_device)
    

    
    def rerank_with_metadata(
        self,
        query: str,
        documents_with_metadata: List[Dict],
        batch_size: int = 4
    ) -> List[Dict]:
        """
        Rerank documents with metadata
        
        Args:
            query: Query text
            documents_with_metadata: List of documents with metadata
            batch_size: Batch size
            
        Returns:
            List of reranked document metadata
        """
        if not documents_with_metadata:
            return []
        
        # Extract document text and create a map
        documents = []
        doc_to_metadata_map = {}
        
        for i, doc_metadata in enumerate(documents_with_metadata):
            doc_text = doc_metadata.get('content', doc_metadata.get('text', ''))
            documents.append(doc_text)
            # Use document content as key to map metadata
            doc_to_metadata_map[doc_text] = doc_metadata
        
        # Perform reranking
        reranked_results = self.rerank(query, documents, batch_size)
        
        # Add scores back to metadata, maintaining the order after reranking
        results = []
        for doc_text, score in reranked_results:
            # Find the corresponding original metadata by document content
            original_metadata = doc_to_metadata_map.get(doc_text, {})
            updated_metadata = original_metadata.copy()
            updated_metadata['reranker_score'] = score
            updated_metadata['content'] = doc_text  # Ensure content field exists
            results.append(updated_metadata)
        
        return results 