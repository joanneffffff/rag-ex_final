"""
Test unified data loading and FAISS index building
"""

import os
import time
from xlm.utils.unified_data_loader import UnifiedDataLoader

def test_unified_data_loading():
    """Test loading and processing all data sources"""
    print("\n=== Testing Unified Data Loading ===")
    
    # Initialize loader
    loader = UnifiedDataLoader(
        data_dir="data",
        cache_dir="D:/AI/huggingface",
        use_faiss=True,
        batch_size=32
    )
    
    # Build unified index
    start_time = time.time()
    loader.build_unified_index(save_dir="data/processed")
    end_time = time.time()
    
    print(f"\nTotal processing time: {end_time - start_time:.2f} seconds")
    
    # Test retrieval
    print("\nTesting retrieval...")
    test_queries = [
        "What was the company's revenue in 2020?",
        "How to calculate PE ratio?",
        "What is the market trend analysis?",
        "Explain the profit margin calculation."
    ]
    
    for query in test_queries:
        print(f"\nQuery: {query}")
        results = loader.retriever.retrieve(text=query, top_k=2)
        
        print("Top 2 results:")
        for i, doc in enumerate(results):
            print(f"\n{i+1}. Source: {doc.metadata.source}")
            print("Content preview:", doc.content[:200] + "..." if len(doc.content) > 200 else doc.content)

if __name__ == "__main__":
    test_unified_data_loading() 