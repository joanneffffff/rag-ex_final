import json
from typing import List, Dict, Union
from pathlib import Path
from xlm.utils.financial_data_processor import FinancialDataProcessor
from xlm.dto.dto import DocumentWithMetadata

class FinancialDataLoader:
    def __init__(self, cache_dir: str = "D:/AI/huggingface"):
        self.cache_dir = cache_dir
        self.processor = FinancialDataProcessor(cache_dir=cache_dir)
    
    def load_data(self, data_path: str) -> List[DocumentWithMetadata]:
        """Load and process financial data from file
        
        Args:
            data_path: Path to the data file (JSON format)
        """
        documents = []
        
        # Load data
        with open(data_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        # Process each entry
        if isinstance(data, list):
            for entry in data:
                if 'input' in entry and '_' in entry['input']:  # News data
                    doc = self.processor.process_news(entry)
                    documents.append(doc)
                elif 'instruction' in entry and isinstance(entry['instruction'], list):  # Stock QA data
                    docs = self.processor.process_stock_data(entry)
                    documents.extend(docs)
                    
                    # Process time series if present
                    for inp in entry['input']:
                        try:
                            # Try to extract time series data
                            time_series_data = self._extract_time_series(inp)
                            if time_series_data:
                                stock_code = self.processor._extract_stock_info(inp)['code']
                                doc = self.processor.process_time_series(time_series_data, stock_code)
                                documents.append(doc)
                        except:
                            continue
        
        return documents
    
    def _extract_time_series(self, text: str) -> Union[Dict[str, float], None]:
        """Extract time series data from text if present"""
        import re
        
        # Try to find JSON-like structure
        match = re.search(r'\{.*\}', text)
        if match:
            try:
                data = json.loads(match.group())
                # Check if it's time series data (date-value pairs)
                if all(isinstance(k, str) and isinstance(v, (int, float)) for k, v in data.items()):
                    return data
            except:
                pass
        return None
    
    def save_processed_data(self, documents: List[DocumentWithMetadata], output_path: str):
        """Save processed documents to file"""
        output_dir = Path(output_path).parent
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Convert documents to serializable format
        data = []
        for doc in documents:
            data.append({
                'content': doc.content,
                'metadata': {
                    'source': doc.metadata.source,
                    'created_at': doc.metadata.created_at,
                    'author': doc.metadata.author
                }
            })
        
        # Save to file
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(data, f, ensure_ascii=False, indent=2) 