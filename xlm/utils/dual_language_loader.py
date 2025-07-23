"""
Dual language data loader for Chinese and English documents
"""

import json
from typing import List, Dict, Tuple
from pathlib import Path
from langdetect import detect, LangDetectException
from tqdm import tqdm

from xlm.dto.dto import DocumentWithMetadata, DocumentMetadata

class DualLanguageLoader:
    def __init__(self):
        """Initialize dual language data loader"""
        pass
    
    def detect_language(self, text: str) -> str:
        """
        Detect the language of the text
        
        Args:
            text: Text content
        
        Returns:
            Language identifier ('chinese' or 'english')
        """
        try:
            lang = detect(text)
            if lang.startswith('zh'):
                return 'chinese'
            else:
                return 'english'
        except LangDetectException:
            # Default to English
            return 'english'
    
    def load_alphafin_data(self, file_path: str) -> List[DocumentWithMetadata]:
        """
        Load AlphaFin Chinese data, field mapping:
        - question: generated_question
        - answer: original_answer
        - context: original_context
        - summary: summary (for FAISS index)
        """
        print(f"Loading AlphaFin Chinese data: {file_path}")
        documents = []
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
            for idx, item in enumerate(tqdm(data, desc="Processing AlphaFin data")):
                question = item.get('generated_question', '').strip()
                answer = item.get('original_answer', '').strip()
                # Use the longer one between original_content and context_content
                original_content = item.get('original_content', '').strip()
                context_content = item.get('context', '').strip()
                context = original_content if len(original_content) > len(context_content) else context_content
                summary = item.get('summary', '').strip()
                company_name = item.get('company_name', '')
                stock_code = item.get('stock_code', '')
                report_date = item.get('report_date', '')
                # Skip if summary is empty
                if not summary:
                    continue
                # Assemble metadata
                metadata = DocumentMetadata(
                    source="alphafin",
                    language="chinese",
                    doc_id=str(item.get('doc_id', f"alphafin_{idx}")),  # 使用原始数据文件的doc_id
                    question=question,
                    answer=answer,
                    company_name=company_name,
                    stock_code=stock_code,
                    report_date=report_date,
                    summary=summary
                )
                # The content field is context, summary is stored separately in metadata
                document = DocumentWithMetadata(
                    content=context,
                    metadata=metadata
                )
                documents.append(document)
            print(f"Loaded {len(documents)} AlphaFin documents (summary not empty)")
            return documents
        except Exception as e:
            print(f"Error: Failed to load AlphaFin data: {e}")
            return []

    def get_alphafin_summaries(self, documents: List[DocumentWithMetadata]) -> List[str]:
        """
        Get the list of summary fields for all AlphaFin documents, used for FAISS index
        """
        return [doc.metadata.summary for doc in documents if hasattr(doc.metadata, 'summary') and doc.metadata.summary]
    
    def load_tatqa_data(self, file_path: str) -> List[DocumentWithMetadata]:
        """
        Load TatQA English data
        
        Args:
            file_path: Data file path
        
        Returns:
            List of English documents
        """
        print(f"Loading TatQA English data: {file_path}")
        
        documents = []
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
            
            for idx, item in enumerate(tqdm(data, desc="Processing TatQA data")):
                question = item.get('question', '').strip()
                answer = item.get('answer', '').strip()
                context = item.get('context', '').strip()
                
                if question and answer and context:
                    # Create document metadata
                    metadata = DocumentMetadata(
                        source="tatqa",
                        language="english",
                        doc_id=f"tatqa_{idx}",
                        question=question,
                        answer=answer
                    )
                    
                    # Create document object
                    document = DocumentWithMetadata(
                        content=context,
                        metadata=metadata
                    )
                    
                    documents.append(document)
            
            print(f"Loaded {len(documents)} English documents")
            return documents
            
        except Exception as e:
            print(f"Error: Failed to load TatQA data: {e}")
            return []
    
    def load_jsonl_data(self, file_path: str, language: str = None) -> List[DocumentWithMetadata]:
        """
        Load JSONL format data
        
        Args:
            file_path: Data file path
            language: Specify language, if None, it will be detected automatically
            
        Returns:
            List of documents
        """
        print(f"加载JSONL数据: {file_path}")
        
        documents = []
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                for idx, line in enumerate(tqdm(f, desc="处理JSONL数据")):
                    try:
                        item = json.loads(line.strip())
                        
                        # Extract necessary fields
                        question = item.get('question', '').strip()
                        answer = item.get('answer', '').strip()
                        context = item.get('context', '').strip()
                        
                        if question and answer and context:
                            # Detect language
                            if language is None:
                                detected_lang = self.detect_language(question)
                            else:
                                detected_lang = language
                            
                            # Create document metadata
                            metadata = DocumentMetadata(
                                source="jsonl",
                                language=detected_lang,
                                doc_id=f"jsonl_{idx}",
                                question=question,
                                answer=answer
                            )
                            
                            # Create document object
                            document = DocumentWithMetadata(
                                content=context,
                                metadata=metadata
                            )
                            
                            documents.append(document)
                            
                    except json.JSONDecodeError:
                        print(f"Warning: Skipping invalid JSON line {idx+1}")
                        continue
            
            print(f"Loaded {len(documents)} JSONL documents")
            return documents
            
        except Exception as e:
            print(f"Error: Failed to load JSONL data: {e}")
            return []
    
    def separate_documents_by_language(self, documents: List[DocumentWithMetadata]) -> Tuple[List[DocumentWithMetadata], List[DocumentWithMetadata]]:
        """
        Separate documents by language
        
        Args:
            documents: List of documents
            
        Returns:
            (Chinese document list, English document list)
        """
        chinese_docs = []
        english_docs = []
        
        for doc in documents:
            if doc.metadata.language == 'chinese':
                chinese_docs.append(doc)
            else:
                english_docs.append(doc)
        
        print(f"Separated result: {len(chinese_docs)} Chinese documents, {len(english_docs)} English documents")
        return chinese_docs, english_docs
    
    def load_tatqa_context_only(self, file_path: str) -> List[DocumentWithMetadata]:
        """
        Load only the context field of TAT-QA English data (for FAISS index/retrieval library)
        """
        print(f"Loading TAT-QA context-only data: {file_path}")
        documents = []
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                for idx, line in enumerate(f):
                    try:
                        item = json.loads(line.strip())
                        # Try to get the content field, if it doesn't exist, try the text field, and finally the context field
                        context = (item.get('content', '') or 
                                 item.get('text', '') or 
                                 item.get('context', ''))
                        context = context.strip()
                        if context:
                            metadata = DocumentMetadata(
                                source="tatqa",
                                created_at="",
                                author="",
                                language="english",
                                doc_id=f"tatqa_{idx}"
                            )
                            document = DocumentWithMetadata(
                                content=context,
                                metadata=metadata
                            )
                            documents.append(document)
                    except Exception as e:
                        print(f"Skipping line {idx+1}, reason: {e}")
            print(f"Loaded {len(documents)} TAT-QA context documents")
            return documents
        except Exception as e:
            print(f"Error: Failed to load TAT-QA context data: {e}")
            return []

    def load_dual_language_data(
        self,
        chinese_data_path: str = None,
        english_data_path: str = None,
        jsonl_data_path: str = None
    ) -> Tuple[List[DocumentWithMetadata], List[DocumentWithMetadata]]:
        """
        Load dual language data (English prioritizes context-only method)
        """
        chinese_docs = []
        english_docs = []
        # Load Chinese data
        if chinese_data_path:
            if chinese_data_path.endswith('.json'):
                chinese_docs.extend(self.load_alphafin_data(chinese_data_path))
            elif chinese_data_path.endswith('.jsonl'):
                chinese_docs.extend(self.load_jsonl_data(chinese_data_path, 'chinese'))
        # Load English data (prioritize context-only)
        if english_data_path:
            if english_data_path.endswith('.json'):
                english_docs.extend(self.load_tatqa_context_only(english_data_path))
            elif english_data_path.endswith('.jsonl'):
                english_docs.extend(self.load_tatqa_context_only(english_data_path))
        # Load JSONL data (automatically detect language)
        if jsonl_data_path:
            all_docs = self.load_jsonl_data(jsonl_data_path)
            chinese_temp, english_temp = self.separate_documents_by_language(all_docs)
            chinese_docs.extend(chinese_temp)
            english_docs.extend(english_temp)
        print(f"Total: {len(chinese_docs)} Chinese documents, {len(english_docs)} English documents")
        return chinese_docs, english_docs
    
    def load_context_only_data(self, file_path: str, language: str = None) -> List[DocumentWithMetadata]:
        """
        Optimized data loading method for context field only (for RAG knowledge base)
        
        Args:
            file_path: Data file path
            language: Specify language, if None, it will be detected automatically
            
        Returns:
            List of documents (only context content, simplified metadata)
        """
        print(f"Loading pure context data: {file_path}")
        
        documents = []
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                for idx, line in enumerate(tqdm(f, desc="Processing context data")):
                    try:
                        item = json.loads(line.strip())
                        
                        # Ensure context field exists and is a string
                        context = item.get('context', '')
                        if isinstance(context, str):
                            context = context.strip()
                        else:
                            # If context is not a string, try to convert or skip
                            print(f"Warning: The context in line {idx} is not a string type: {type(context)}")
                            if isinstance(context, dict):
                                # If it's a dictionary, try to extract the context field
                                context = context.get('context', str(context))
                            else:
                                context = str(context)
                        
                        if context:  # Only check if context exists and is not empty
                            # Detect language (use context content instead of query)
                            if language is None:
                                detected_lang = self.detect_language(context)
                            else:
                                detected_lang = language
                            
                            # Create simplified document metadata (only keep necessary fields)
                            metadata = DocumentMetadata(
                                source="context_only",
                                language=detected_lang,
                                doc_id=f"context_{idx}"
                            )
                            
                            # Create document object, ensure content field is a string
                            doc = DocumentWithMetadata(
                                content=context,
                                metadata=metadata
                            )
                            documents.append(doc)
                        else:
                            print(f"Warning: The context in line {idx} is empty, skipping")
                            
                    except json.JSONDecodeError as e:
                        print(f"Warning: JSON parsing failed in line {idx}: {e}")
                        continue
                    except Exception as e:
                        print(f"Warning: Processing failed in line {idx}: {e}")
                        continue
                        
        except FileNotFoundError:
            print(f"Error: File not found: {file_path}")
            return []
        except Exception as e:
            print(f"Error: Failed to read file: {e}")
            return []
        
        print(f"Successfully loaded {len(documents)} context documents")
        return documents 