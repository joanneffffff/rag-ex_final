"""
Test script for verifying SBERTRetriever functionality
"""

import os
import sys
import logging
import torch
from xlm.components.encoder.encoder import Encoder
from xlm.components.retriever.sbert_retriever import SBERTRetriever
from xlm.dto.dto import DocumentWithMetadata, DocumentMetadata

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def create_test_documents(num_docs: int = 5) -> list[DocumentWithMetadata]:
    """Create test documents"""
    documents = []
    for i in range(num_docs):
        doc = DocumentWithMetadata(
            content=f"This is test document {i} containing some test content for retrieval testing.",
            metadata=DocumentMetadata(
                source=f"test_doc_{i}.txt",
                created_at="2024-03-20",
                author="Test"
            )
        )
        documents.append(doc)
    return documents

def test_basic_retrieval():
    """Test basic document retrieval functionality"""
    logger.info("Testing basic retrieval...")
    try:
        # Initialize components
        encoder = Encoder(
            model_name="sentence-transformers/all-MiniLM-L6-v2",
            cache_dir="D:/AI/huggingface"
        )
        documents = create_test_documents()
        
        # Create retriever
        retriever = SBERTRetriever(
            encoder=encoder,
            corpus_documents=documents,
            batch_size=2  # Small batch size for testing
        )
        
        # Test simple retrieval
        query = "test document 0"
        results = retriever.retrieve(text=query, top_k=2)
        
        logger.info(f"Query: {query}")
        logger.info(f"Number of results: {len(results)}")
        logger.info(f"Top result content: {results[0].content}")
        
        # Test with scores
        results, scores = retriever.retrieve(text=query, top_k=2, return_scores=True)
        logger.info(f"Top result score: {scores[0]:.4f}")
        
        return True
    except Exception as e:
        logger.error(f"Error in basic retrieval test: {str(e)}")
        return False

def test_faiss_retrieval():
    """Test retrieval with FAISS enabled"""
    logger.info("Testing FAISS retrieval...")
    try:
        encoder = Encoder(
            model_name="sentence-transformers/all-MiniLM-L6-v2",
            cache_dir="D:/AI/huggingface"
        )
        # Create more test documents for FAISS
        documents = create_test_documents(num_docs=100)
        
        retriever = SBERTRetriever(
            encoder=encoder,
            corpus_documents=documents,
            use_faiss=True,
            batch_size=32,  # Larger batch size for more documents
            num_threads=4
        )
        
        # Test multiple queries
        queries = [
            "test document",
            "content for retrieval",
            "document 0"
        ]
        
        for query in queries:
            results = retriever.retrieve(text=query, top_k=3)
            logger.info(f"\nQuery: {query}")
            logger.info(f"Number of results: {len(results)}")
            for i, doc in enumerate(results):
                logger.info(f"Result {i+1}: {doc.content[:50]}...")
            
        return True
    except Exception as e:
        logger.error(f"Error in FAISS retrieval test: {str(e)}")
        return False

def test_corpus_update():
    """Test corpus update functionality"""
    logger.info("Testing corpus update...")
    try:
        encoder = Encoder(
            model_name="sentence-transformers/all-MiniLM-L6-v2",
            cache_dir="D:/AI/huggingface"
        )
        initial_documents = create_test_documents(num_docs=2)
        
        retriever = SBERTRetriever(
            encoder=encoder,
            corpus_documents=initial_documents
        )
        
        # Test initial retrieval
        query = "test document"
        initial_results = retriever.retrieve(text=query, top_k=2)
        logger.info(f"Initial corpus size: {len(retriever.corpus_documents)}")
        logger.info(f"Initial results: {len(initial_results)}")
        
        # Update corpus
        new_documents = create_test_documents(num_docs=3)
        retriever.update_corpus(new_documents)
        
        # Test retrieval after update
        updated_results = retriever.retrieve(text=query, top_k=2)
        logger.info(f"Updated corpus size: {len(retriever.corpus_documents)}")
        logger.info(f"Updated results: {len(updated_results)}")
        
        return True
    except Exception as e:
        logger.error(f"Error in corpus update test: {str(e)}")
        return False

def test_batch_processing():
    """Test batch processing with larger corpus"""
    logger.info("Testing batch processing...")
    try:
        encoder = Encoder(
            model_name="sentence-transformers/all-MiniLM-L6-v2",
            cache_dir="D:/AI/huggingface"
        )
        documents = create_test_documents(num_docs=20)
        
        retriever = SBERTRetriever(
            encoder=encoder,
            corpus_documents=documents,
            batch_size=5
        )
        
        logger.info(f"Corpus size: {len(retriever.corpus_documents)}")
        logger.info(f"Embeddings shape: {len(retriever.corpus_embeddings)}")
        
        # Test retrieval
        query = "test document"
        results = retriever.retrieve(text=query, top_k=3)
        logger.info(f"Successfully retrieved {len(results)} documents")
        
        return True
    except Exception as e:
        logger.error(f"Error in batch processing test: {str(e)}")
        return False

def run_all_tests():
    """Run all tests and return overall status"""
    logger.info("Running SBERTRetriever tests...")
    
    test_results = {
        "Basic Retrieval": test_basic_retrieval(),
        "FAISS Retrieval": test_faiss_retrieval(),
        "Corpus Update": test_corpus_update(),
        "Batch Processing": test_batch_processing()
    }
    
    # Print summary
    logger.info("\nTest Results Summary:")
    for test_name, passed in test_results.items():
        status = "PASSED" if passed else "FAILED"
        logger.info(f"{test_name}: {status}")
    
    # Return overall status
    return all(test_results.values())

if __name__ == "__main__":
    success = run_all_tests()
    sys.exit(0 if success else 1) 