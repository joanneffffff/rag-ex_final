"""
Comprehensive test script for unified financial data system
"""

import os
import json
import time
from typing import List, Dict
import pytest
from xlm.utils.unified_data_loader import UnifiedDataLoader
from xlm.dto.dto import DocumentWithMetadata

class TestUnifiedSystem:
    @pytest.fixture(scope="class")
    def data_loader(self):
        """Initialize data loader"""
        loader = UnifiedDataLoader(
            data_dir="data",
            cache_dir="D:/AI/huggingface",
            use_faiss=True,
            batch_size=32
        )
        return loader

    @pytest.fixture(scope="class")
    def processed_data(self, data_loader):
        """Load and process all data"""
        data_loader.build_unified_index(save_dir="data/processed")
        return data_loader

    def test_data_loading(self, processed_data):
        """Test data loading and processing"""
        retriever = processed_data.retriever
        assert retriever is not None
        assert len(retriever.corpus_documents) > 0
        
        # Check if we have both types of documents
        sources = set(doc.metadata.source for doc in retriever.corpus_documents)
        assert "tatqa_table" in sources or "tatqa_paragraph" in sources or "tatqa_qa" in sources
        assert "alphafin" in sources

    def test_retrieval_basic(self, processed_data):
        """Test basic retrieval functionality"""
        test_queries = [
            "What was the company's revenue?",
            "How to calculate PE ratio?",
            "What is the profit margin?",
            "Explain the market trend."
        ]
        
        for query in test_queries:
            results = processed_data.retriever.retrieve(text=query, top_k=3)
            assert len(results) > 0
            assert all(isinstance(doc, DocumentWithMetadata) for doc in results)

    def test_retrieval_with_scores(self, processed_data):
        """Test retrieval with similarity scores"""
        query = "How to analyze financial statements?"
        results, scores = processed_data.retriever.retrieve(
            text=query,
            top_k=3,
            return_scores=True
        )
        
        assert len(results) == len(scores)
        assert all(0 <= score <= 1 for score in scores)

    def test_source_specific_retrieval(self, processed_data):
        """Test retrieval from specific data sources"""
        # Test TatQA specific query
        tatqa_query = "What is shown in the table for Q4?"
        results = processed_data.retriever.retrieve(text=tatqa_query, top_k=3)
        assert any("tatqa" in doc.metadata.source for doc in results)
        
        # Test AlphaFin specific query
        alphafin_query = "股票分析师如何分析市场趋势"
        results = processed_data.retriever.retrieve(text=alphafin_query, top_k=3)
        assert any("alphafin" in doc.metadata.source for doc in results)

    def test_data_consistency(self, processed_data):
        """Test data consistency and format"""
        retriever = processed_data.retriever
        
        for doc in retriever.corpus_documents:
            assert hasattr(doc, 'content')
            assert hasattr(doc, 'metadata')
            assert hasattr(doc.metadata, 'source')
            assert len(doc.content.strip()) > 0

    def test_faiss_index(self, processed_data):
        """Test FAISS index functionality"""
        retriever = processed_data.retriever
        assert retriever.use_faiss
        assert retriever.index is not None
        
        # Test if index dimensions match embeddings
        query = "test query"
        query_embedding = retriever.encode_queries(query)
        assert len(query_embedding[0]) == retriever.index.d

def run_manual_test():
    """Run manual test with detailed output"""
    print("\n=== Running Manual Test of Unified System ===")
    
    # Initialize and load data
    print("\n1. Initializing data loader...")
    loader = UnifiedDataLoader(
        data_dir="data",
        cache_dir="D:/AI/huggingface",
        use_faiss=True,
        batch_size=32
    )
    
    # Build index
    print("\n2. Building unified index...")
    start_time = time.time()
    loader.build_unified_index(save_dir="data/processed")
    end_time = time.time()
    print(f"Index building time: {end_time - start_time:.2f} seconds")
    
    # Test queries
    print("\n3. Testing various queries...")
    test_queries = {
        "TatQA Table": "What are the quarterly revenue figures?",
        "TatQA QA": "How is the profit margin calculated?",
        "AlphaFin Financial": "如何分析股票市盈率",
        "AlphaFin News": "最新的市场趋势分析",
        "Cross-domain": "How to evaluate company performance?"
    }
    
    for query_type, query in test_queries.items():
        print(f"\nQuery Type: {query_type}")
        print(f"Query: {query}")
        
        results, scores = loader.retriever.retrieve(
            text=query,
            top_k=2,
            return_scores=True
        )
        
        for i, (doc, score) in enumerate(zip(results, scores)):
            print(f"\nResult {i+1} (Score: {score:.4f})")
            print(f"Source: {doc.metadata.source}")
            print("Content preview:", doc.content[:200] + "..." if len(doc.content) > 200 else doc.content)
            print("-" * 80)

if __name__ == "__main__":
    run_manual_test() 