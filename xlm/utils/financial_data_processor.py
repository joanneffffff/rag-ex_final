import json
from datetime import datetime
from typing import List, Dict, Union, Optional
import pandas as pd
from xlm.dto.dto import DocumentWithMetadata, DocumentMetadata

class FinancialDataProcessor:
    def __init__(self, cache_dir: str = "D:/AI/huggingface"):
        self.cache_dir = cache_dir
    
    def process_news(self, news_data: Dict) -> DocumentWithMetadata:
        """Process news data into document format
        
        Args:
            news_data: Dictionary containing news data with instruction, input, and output
        """
        # Extract timestamp if available
        timestamp = None
        if isinstance(news_data['input'], str) and '_' in news_data['input']:
            timestamp_str = news_data['input'].split('_')[0]
            try:
                timestamp = datetime.strptime(timestamp_str, '%Y-%m-%d %H:%M:%S')
            except ValueError:
                pass
        
        # Combine relevant information
        content = f"Timestamp: {timestamp if timestamp else 'N/A'}\n"
        content += f"News: {news_data['input']}\n"
        if news_data.get('output'):
            content += f"Summary: {news_data['output']}\n"
        
        # Create document
        return DocumentWithMetadata(
            content=content,
            metadata=DocumentMetadata(
                source="financial_news",
                created_at=str(timestamp) if timestamp else "",
                author=""
            )
        )
    
    def process_stock_data(self, stock_data: Dict) -> List[DocumentWithMetadata]:
        """Process stock market data into document format
        
        Args:
            stock_data: Dictionary containing stock data
        """
        documents = []
        
        # Process instruction-based data
        if isinstance(stock_data.get('instruction'), list):
            for i, (inst, inp, out) in enumerate(zip(
                stock_data['instruction'],
                stock_data['input'],
                stock_data['output']
            )):
                # Try to extract stock code and date
                stock_info = self._extract_stock_info(inp)
                
                content = f"Stock: {stock_info['code'] if stock_info else 'Unknown'}\n"
                content += f"Date: {stock_info['date'] if stock_info else 'Unknown'}\n"
                content += f"Question: {inp}\n"
                content += f"Answer: {out}\n"
                
                # Create document
                doc = DocumentWithMetadata(
                    content=content,
                    metadata=DocumentMetadata(
                        source=f"stock_data_{stock_info['code'] if stock_info else i}",
                        created_at=stock_info['date'] if stock_info else "",
                        author=""
                    )
                )
                documents.append(doc)
        
        return documents
    
    def _extract_stock_info(self, text: str) -> Optional[Dict[str, str]]:
        """Extract stock code and date from text"""
        import re
        
        # Try to find stock code (format: xxxxxx.SZ or xxxxxx.SH)
        code_match = re.search(r'(\d{6}\.(SZ|SH))', text)
        # Try to find date (format: YYYY-MM-DD)
        date_match = re.search(r'(\d{4}-\d{2}-\d{2})', text)
        
        if code_match or date_match:
            return {
                'code': code_match.group(1) if code_match else None,
                'date': date_match.group(1) if date_match else None
            }
        return None
    
    def process_time_series(self, data: Dict[str, float], stock_code: str) -> DocumentWithMetadata:
        """Process time series data into document format
        
        Args:
            data: Dictionary of date-value pairs
            stock_code: Stock code
        """
        # Convert to pandas Series for easier analysis
        series = pd.Series(data)
        
        # Calculate basic statistics
        stats = {
            'mean': series.mean(),
            'median': series.median(),
            'std': series.std(),
            'min': series.min(),
            'max': series.max(),
            'trend': 'increasing' if series.iloc[-1] > series.iloc[0] else 'decreasing'
        }
        
        # Create content with both raw data and analysis
        content = f"Stock Code: {stock_code}\n"
        content += f"Time Series Analysis:\n"
        content += f"- Period: {series.index[0]} to {series.index[-1]}\n"
        content += f"- Average: {stats['mean']:.4f}\n"
        content += f"- Median: {stats['median']:.4f}\n"
        content += f"- Standard Deviation: {stats['std']:.4f}\n"
        content += f"- Range: {stats['min']:.4f} to {stats['max']:.4f}\n"
        content += f"- Trend: {stats['trend']}\n"
        content += "\nRaw Data:\n"
        for date, value in data.items():
            content += f"{date}: {value:.4f}\n"
        
        return DocumentWithMetadata(
            content=content,
            metadata=DocumentMetadata(
                source=f"time_series_{stock_code}",
                created_at=series.index[-1],
                author=""
            )
        ) 