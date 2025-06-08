import os
import json
import psutil
import numpy as np
from tqdm import tqdm
from xlm.components.encoder.encoder import Encoder
from xlm.components.retriever.sbert_retriever import SBERTRetriever
from xlm.dto.dto import DocumentWithMetadata, DocumentMetadata

def get_memory_usage():
    """Get current memory usage in MB"""
    process = psutil.Process(os.getpid())
    return process.memory_info().rss / 1024 / 1024

def print_memory_status(stage: str):
    """Print memory usage at a given stage"""
    memory_used = get_memory_usage()
    print(f"\n{stage}:")
    print(f"Memory Usage: {memory_used:.2f} MB")

def generate_test_documents(num_docs: int, doc_size: int = 1000) -> list:
    """Generate test documents
    
    Args:
        num_docs: Number of documents to generate
        doc_size: Approximate size of each document in characters
    """
    documents = []
    for i in range(num_docs):
        # Generate random text
        content = f"Document {i}: " + "test " * (doc_size // 5)
        doc = DocumentWithMetadata(
            content=content,
            metadata=DocumentMetadata(
                source=f"test_doc_{i}",
                created_at="",
                author=""
            )
        )
        documents.append(doc)
    return documents

def test_retriever(use_faiss: bool, num_docs: int = 1000, batch_size: int = 100):
    """Test retriever with different configurations
    
    Args:
        use_faiss: Whether to use FAISS
        num_docs: Number of test documents
        batch_size: Batch size for processing
    """
    print(f"\n{'='*50}")
    print(f"Testing Retriever (FAISS: {use_faiss})")
    print(f"Number of documents: {num_docs}")
    print(f"Batch size: {batch_size}")
    print(f"{'='*50}")
    
    print_memory_status("Initial Memory")
    
    # Initialize encoder
    encoder = Encoder(
        model_name="sentence-transformers/all-MiniLM-L6-v2",
        cache_dir="D:/AI/huggingface"
    )
    print_memory_status("After Encoder Initialization")
    
    # Generate test documents
    print("\nGenerating test documents...")
    documents = generate_test_documents(num_docs)
    print_memory_status("After Document Generation")
    
    # Initialize retriever
    retriever = SBERTRetriever(
        encoder=encoder,
        corpus_documents=[],  # Start empty
        use_faiss=use_faiss
    )
    print_memory_status("After Retriever Initialization")
    
    # Process documents in batches
    print("\nProcessing documents in batches...")
    for i in tqdm(range(0, len(documents), batch_size)):
        batch = documents[i:i+batch_size]
        
        # Encode batch
        batch_embeddings = retriever.encode_corpus(documents=batch)
        
        # Add to retriever
        if use_faiss:
            retriever._add_to_faiss(batch_embeddings)
        else:
            retriever.corpus_embeddings.extend(batch_embeddings)
        retriever.corpus_documents.extend(batch)
        
        if i % (batch_size * 5) == 0:  # Print memory usage every 5 batches
            print_memory_status(f"After Processing {i+len(batch)} Documents")
    
    # Test search
    print("\nTesting search functionality...")
    query = "This is a test query"
    results = retriever.retrieve_documents_with_scores(query, top_k=5)
    print_memory_status("After Search")
    
    # Print search results
    print("\nSearch Results:")
    for item in results[0]:
        print(f"Score: {item['score']:.4f}, Document ID: {item['corpus_id']}")
    
    return get_memory_usage()

if __name__ == "__main__":
    # Test configurations
    configs = [
        {"use_faiss": True, "num_docs": 1000, "batch_size": 100},
        {"use_faiss": False, "num_docs": 1000, "batch_size": 100},
        {"use_faiss": True, "num_docs": 5000, "batch_size": 100},
        {"use_faiss": False, "num_docs": 5000, "batch_size": 100}
    ]
    
    results = []
    for config in configs:
        try:
            memory_used = test_retriever(**config)
            results.append({
                "config": config,
                "memory_used": memory_used,
                "status": "success"
            })
        except Exception as e:
            results.append({
                "config": config,
                "error": str(e),
                "status": "failed"
            })
    
    # Print summary
    print("\n=== Test Summary ===")
    for result in results:
        config = result["config"]
        print(f"\nConfiguration:")
        print(f"- FAISS: {config['use_faiss']}")
        print(f"- Documents: {config['num_docs']}")
        print(f"- Batch Size: {config['batch_size']}")
        if result["status"] == "success":
            print(f"Memory Used: {result['memory_used']:.2f} MB")
        else:
            print(f"Error: {result['error']}") 