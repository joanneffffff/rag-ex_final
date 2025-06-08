from typing import List, Optional, Dict, Any
import os
import json
from datetime import datetime

from xlm.dto.dto import DocumentWithMetadata, DocumentMetadata


class DocumentLoader:
    """用于加载和处理文档的工具类"""
    
    @staticmethod
    def load_text_file(
        file_path: str,
        source: str = "file",
        additional_metadata: Optional[Dict[str, Any]] = None
    ) -> DocumentWithMetadata:
        """从文本文件加载文档"""
        with open(file_path, 'r', encoding='utf-8') as f:
            content = f.read()
            
        file_name = os.path.basename(file_path)
        metadata = DocumentMetadata(
            doc_id=file_name,
            source=source,
            title=file_name,
            date=datetime.fromtimestamp(os.path.getmtime(file_path)).isoformat(),
            custom_metadata=additional_metadata or {}
        )
        
        return DocumentWithMetadata(content=content, metadata=metadata)
    
    @staticmethod
    def load_json_file(
        file_path: str,
        content_field: str = "content",
        metadata_mapping: Optional[Dict[str, str]] = None
    ) -> DocumentWithMetadata:
        """从JSON文件加载文档，支持自定义字段映射"""
        with open(file_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
            
        content = data.get(content_field, "")
        
        # 处理元数据映射
        metadata_dict = {
            "doc_id": os.path.basename(file_path),
            "source": "json"
        }
        
        if metadata_mapping:
            for target_field, source_field in metadata_mapping.items():
                if source_field in data:
                    metadata_dict[target_field] = data[source_field]
                    
        metadata = DocumentMetadata(**metadata_dict)
        return DocumentWithMetadata(content=content, metadata=metadata)
    
    @staticmethod
    def load_directory(
        directory_path: str,
        file_pattern: str = "*.txt",
        recursive: bool = True,
        source: str = "file",
        additional_metadata: Optional[Dict[str, Any]] = None
    ) -> List[DocumentWithMetadata]:
        """从目录加载多个文档"""
        import glob
        
        if recursive:
            pattern = os.path.join(directory_path, "**", file_pattern)
        else:
            pattern = os.path.join(directory_path, file_pattern)
            
        documents = []
        for file_path in glob.glob(pattern, recursive=recursive):
            if file_path.endswith('.json'):
                doc = DocumentLoader.load_json_file(file_path)
            else:
                doc = DocumentLoader.load_text_file(
                    file_path,
                    source=source,
                    additional_metadata=additional_metadata
                )
            documents.append(doc)
            
        return documents 