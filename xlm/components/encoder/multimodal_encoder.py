"""
Multimodal encoder implementation supporting different data types and parallel processing.
"""

from typing import List, Dict, Union, Optional
import torch
from torch.nn.parallel import DataParallel
import numpy as np
from concurrent.futures import ThreadPoolExecutor
from sentence_transformers import SentenceTransformer

from xlm.components.encoder.encoder import Encoder
from xlm.components.encoder.multimodal_encoder2 import TableEncoder, TimeSeriesEncoder
from config.parameters import Config, EncoderConfig

class MultiModalEncoder:
    def __init__(
        self,
        config: Config,
        text_encoder: Optional[Encoder] = None,
        table_encoder: Optional[Encoder] = None,
        time_series_encoder: Optional[Encoder] = None,
        use_enhanced_encoders: bool = False  # New parameter for enhanced encoders
    ):
        self.config = config
        self.use_enhanced_encoders = use_enhanced_encoders
        
        # Initialize encoders based on use_enhanced_encoders flag
        if use_enhanced_encoders:
            self.text_encoder = text_encoder or self._create_text_encoder()
            self.table_encoder = table_encoder or TableEncoder()
            self.time_series_encoder = time_series_encoder or TimeSeriesEncoder()
        else:
            self.text_encoder = text_encoder or self._create_text_encoder()
            self.table_encoder = table_encoder or self._create_table_encoder()
            self.time_series_encoder = time_series_encoder or self._create_time_series_encoder()
        
        # Setup parallel processing
        self.executor = ThreadPoolExecutor(max_workers=config.retriever.num_threads)
        
        # Setup device
        self.device = config.encoder.device or ("cuda" if torch.cuda.is_available() else "cpu")
        
        # Enable DataParallel if multiple GPUs available
        if self.device == "cuda" and torch.cuda.device_count() > 1:
            self.text_encoder.model = DataParallel(self.text_encoder.model)
            if hasattr(self.table_encoder, 'model'):
                self.table_encoder.model = DataParallel(self.table_encoder.model)
            if hasattr(self.time_series_encoder, 'model'):
                self.time_series_encoder.model = DataParallel(self.time_series_encoder.model)

    def _create_text_encoder(self) -> Encoder:
        """Create default text encoder"""
        return Encoder(
            model_name=self.config.encoder.model_name,
            device=self.config.encoder.device,
            cache_dir=self.config.encoder.cache_dir
        )

    def _create_table_encoder(self) -> Encoder:
        """Create default table encoder"""
        # For now, using same architecture as text
        # Could be replaced with specialized table encoder
        return self._create_text_encoder()

    def _create_time_series_encoder(self) -> Encoder:
        """Create default time series encoder"""
        # For now, using same architecture as text
        # Could be replaced with specialized time series encoder
        return self._create_text_encoder()

    def encode_batch(
        self,
        texts: Optional[List[str]] = None,
        tables: Optional[List[str]] = None,
        time_series: Optional[List[str]] = None,
        batch_size: Optional[int] = None
    ) -> Dict[str, np.ndarray]:
        """
        Encode multiple modalities in parallel batches
        
        Args:
            texts: List of text strings
            tables: List of table strings/representations
            time_series: List of time series strings/representations
            batch_size: Optional batch size override
            
        Returns:
            Dictionary of embeddings for each modality
        """
        batch_size = batch_size or self.config.encoder.batch_size
        futures = []
        
        # Submit encoding tasks in parallel
        if texts:
            futures.append(
                self.executor.submit(
                    self._batch_encode,
                    self.text_encoder,
                    texts,
                    batch_size
                )
            )
        
        if tables:
            futures.append(
                self.executor.submit(
                    self._batch_encode,
                    self.table_encoder,
                    tables,
                    batch_size
                )
            )
            
        if time_series:
            futures.append(
                self.executor.submit(
                    self._batch_encode,
                    self.time_series_encoder,
                    time_series,
                    batch_size
                )
            )

        # Collect results
        results = {}
        if texts:
            results['text'] = futures[0].result()
        if tables:
            results['table'] = futures[1 if texts else 0].result()
        if time_series:
            results['time_series'] = futures[-1].result()

        return results

    def _batch_encode(
        self,
        encoder: Union[Encoder, TableEncoder, TimeSeriesEncoder],
        items: List[str],
        batch_size: int
    ) -> np.ndarray:
        """Encode items in batches"""
        all_embeddings = []
        
        for i in range(0, len(items), batch_size):
            batch = items[i:i + batch_size]
            # Handle different encoder types
            if isinstance(encoder, (TableEncoder, TimeSeriesEncoder)):
                embeddings = encoder.encode(batch)
            else:
                embeddings = encoder.encode(batch)
            all_embeddings.extend(embeddings)
            
            # Clear CUDA cache if using GPU
            if self.device == "cuda":
                torch.cuda.empty_cache()
                
        return np.array(all_embeddings)

    def combine_embeddings(
        self,
        embeddings: Dict[str, np.ndarray]
    ) -> np.ndarray:
        """
        Combine embeddings from different modalities
        
        Args:
            embeddings: Dictionary of embeddings per modality
            
        Returns:
            Combined embeddings
        """
        if self.config.modality.combine_method == "weighted_sum":
            # Initialize with zeros matching the shape of first embedding
            first_emb = next(iter(embeddings.values()))
            combined = np.zeros_like(first_emb)
            
            # Add weighted embeddings
            if 'text' in embeddings:
                combined += self.config.modality.text_weight * embeddings['text']
            if 'table' in embeddings:
                combined += self.config.modality.table_weight * embeddings['table']
            if 'time_series' in embeddings:
                combined += self.config.modality.time_series_weight * embeddings['time_series']
                
        elif self.config.modality.combine_method == "concatenate":
            combined = np.concatenate(list(embeddings.values()), axis=-1)
            
        else:
            raise ValueError(f"Unsupported combine method: {self.config.modality.combine_method}")
            
        return combined 

    @property
    def model_name(self):
        # 返回主文本编码器的 model_name
        return getattr(self.text_encoder, "model_name", "multimodal_encoder") 