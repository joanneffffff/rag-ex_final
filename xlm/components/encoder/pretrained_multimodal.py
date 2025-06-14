"""
Pretrained multimodal models for financial data
"""

from typing import List, Dict, Union, Optional
import torch
import numpy as np
from torch import nn
from transformers import AutoModel, AutoTokenizer, AutoProcessor
from sentence_transformers import SentenceTransformer

class PretrainedMultimodalEncoder(nn.Module):
    def __init__(
        self,
        model_name: str = "microsoft/table-transformer",
        use_gpu: bool = True
    ):
        super().__init__()
        self.model_name = model_name
        self.device = torch.device("cuda" if use_gpu and torch.cuda.is_available() else "cpu")
        
        # Load model and processor
        self.model = AutoModel.from_pretrained(model_name)
        self.processor = AutoProcessor.from_pretrained(model_name)
        self.model.to(self.device)
        
        # Initialize text encoder for fusion
        self.text_encoder = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")
        
        # Fusion layer
        self.fusion_layer = nn.Linear(
            self.model.config.hidden_size + self.text_encoder.get_sentence_embedding_dimension(),
            self.text_encoder.get_sentence_embedding_dimension()
        )
    
    def forward(
        self,
        tables: Optional[List[Dict]] = None,
        texts: Optional[List[str]] = None,
        time_series: Optional[List[Dict]] = None
    ) -> torch.Tensor:
        embeddings = []
        
        # Process tables
        if tables:
            table_embeddings = self._process_tables(tables)
            embeddings.append(table_embeddings)
        
        # Process texts
        if texts:
            text_embeddings = self.text_encoder.encode(texts, convert_to_tensor=True)
            embeddings.append(text_embeddings)
        
        # Process time series
        if time_series:
            ts_embeddings = self._process_time_series(time_series)
            embeddings.append(ts_embeddings)
        
        # Fuse embeddings
        if len(embeddings) > 1:
            combined = torch.cat(embeddings, dim=1)
            fused = self.fusion_layer(combined)
            return fused
        else:
            return embeddings[0]
    
    def _process_tables(self, tables: List[Dict]) -> torch.Tensor:
        """Process tables using the pretrained model"""
        processed_tables = []
        for table in tables:
            # Convert table to image or structured format
            if self.model_name == "microsoft/table-transformer":
                # Process as image
                table_image = self._table_to_image(table)
                inputs = self.processor(images=table_image, return_tensors="pt")
            else:
                # Process as structured data
                inputs = self.processor(table, return_tensors="pt")
            
            # Get embeddings
            with torch.no_grad():
                outputs = self.model(**inputs)
                embeddings = outputs.last_hidden_state[:, 0, :]  # [CLS] token
                processed_tables.append(embeddings)
        
        return torch.stack(processed_tables)
    
    def _process_time_series(self, time_series: List[Dict]) -> torch.Tensor:
        """Process time series data"""
        processed_ts = []
        for ts in time_series:
            # Convert to text representation
            ts_text = self._time_series_to_text(ts)
            # Encode using text encoder
            embedding = self.text_encoder.encode(ts_text, convert_to_tensor=True)
            processed_ts.append(embedding)
        
        return torch.stack(processed_ts)
    
    def _table_to_image(self, table: Dict) -> np.ndarray:
        """Convert table to image format for table-transformer"""
        # Implementation depends on specific requirements
        # This is a placeholder
        return np.zeros((100, 100, 3))
    
    def _time_series_to_text(self, time_series: Dict) -> str:
        """Convert time series to text representation"""
        text = "Time Series Data:\n"
        for date, value in time_series.items():
            text += f"{date}: {value}\n"
        return text
    
    def encode(
        self,
        tables: Optional[List[Dict]] = None,
        texts: Optional[List[str]] = None,
        time_series: Optional[List[Dict]] = None
    ) -> np.ndarray:
        with torch.no_grad():
            embeddings = self.forward(tables, texts, time_series)
            return embeddings.cpu().numpy()
    
    def get_embedding_dimension(self) -> int:
        return self.text_encoder.get_sentence_embedding_dimension()

class MultimodalProcessor:
    def __init__(self):
        self.text_processor = FinancialTextProcessor()
        self.table_processor = TableProcessor()
        self.ts_processor = TimeSeriesProcessor()
    
    def process(
        self,
        tables: Optional[List[Dict]] = None,
        texts: Optional[List[str]] = None,
        time_series: Optional[List[Dict]] = None
    ) -> Dict[str, List[str]]:
        """Process all modalities"""
        processed = {}
        
        if tables:
            processed['tables'] = [self.table_processor.process(table) for table in tables]
        if texts:
            processed['texts'] = [self.text_processor.process(text) for text in texts]
        if time_series:
            processed['time_series'] = [self.ts_processor.process(ts) for ts in time_series]
        
        return processed 