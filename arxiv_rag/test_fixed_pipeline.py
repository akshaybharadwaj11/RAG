#!/usr/bin/env python3
"""
Quick test to verify the fixed pipeline works
Downloads just 5 papers to test everything end-to-end
"""

import logging
import shutil
from pathlib import Path
import sys

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def clean_test_data():
    """Clean previous test data"""
    logger.info("Cleaning previous test data...")
    for dir_path in ["data/raw", "data/embeddings"]:
        if Path(dir_path).exists():
            shutil.rmtree(dir_path)
    Path("data/raw").mkdir(parents=True, exist_ok=True)
    Path("data/embeddings/chromadb").mkdir(parents=True, exist_ok=True)
    logger.info("✓ Clean complete")

def test_download():
    """Test downloading 5 papers"""
    logger.info("\n" + "="*60)
    logger.info("TEST 1: Download 5 Papers")
    logger.info("="*60)
    
    from src.ingestion.arxiv_fetcher import ArxivFetcher
    
    fetcher = ArxivFetcher(
        categories=["cs.AI"],
        max_results=5
    )
    
    papers = fetcher.fetch_papers(Path("data/raw"))
    
    if not papers:
        logger.error("✗ No papers fetched!")
        return False
    
    successful = sum(1 for p in papers if p.get("pdf_path"))
    logger.info(f"\n✓ Downloaded {successful}/{len(papers)} PDFs")
    
    if successful == 0:
        logger.error("✗ All downloads failed!")
        return False
    
    if successful < len(papers):
        logger.warning(f"⚠ Only {successful}/{len(papers)} PDFs downloaded (some failed)")
    
    return True, papers

def test_processing(papers):
    """Test processing papers into chunks"""
    logger.info("\n" + "="*60)
    logger.info("TEST 2: Process Papers into Chunks")
    logger.info("="*60)
    
    from src.ingestion.processor import DocumentProcessor
    
    processor = DocumentProcessor(
        chunk_size=1000,
        chunk_overlap=200
    )
    
    chunks = processor.process_papers(papers)
    
    if not chunks:
        logger.error("✗ No chunks created!")
        return False
    
    logger.info(f"\n✓ Created {len(chunks)} chunks")
    
    # Show sample
    logger.info("\nSample chunk:")
    logger.info(f"  Paper: {chunks[0]['metadata']['title'][:60]}")
    logger.info(f"  Text preview: {chunks[0]['text'][:150]}...")
    
    return True, chunks

def test_vector_store(chunks):
    """Test creating vector store"""
    logger.info("\n" + "="*60)
    logger.info("TEST 3: Create Vector Store")
    logger.info("="*60)
    
    from src.vectorstore.embeddings import EmbeddingGenerator
    from src.vectorstore.store import VectorStore
    
    logger.info("Loading embedding model...")
    embedding_gen = EmbeddingGenerator("sentence-transformers/all-mpnet-base-v2")
    
    logger.info("Creating vector store...")
    vector_store = VectorStore(
        collection_name="arxiv_papers",
        persist_directory="data/embeddings/chromadb",
        embedding_function=embedding_gen
    )
    
    logger.info("Adding documents...")
    vector_store.add_documents(chunks, batch_size=32)
    
    count = vector_store.collection.count()
    logger.info(f"\n✓ Vector store created with {count} chunks")
    
    return True, vector_store, embedding_gen

def test_query(vector_store, embedding_gen):
    """Test querying the system"""
    logger.info("\n" + "="*60)
    logger.info("TEST 4: Query System")
    logger.info("="*60)
    
    test_queries = [
        "What is artificial intelligence?",
        "Explain machine learning",
        "What are neural networks?"
    ]
    
    for query in test_queries:
        logger.info(f"\nQuery: '{query}'")
        
        query_embedding = embedding_gen.generate_embeddings([query])
        results = vector_store.collection.query(
            query_embeddings=query_embedding.tolist(),
            n_results=2
        )
        
        if results["ids"][0]:
            for i in range(len(results['ids'][0])):
                title = results['metadatas'][0][i]['title']
                similarity = 1 - results['distances'][0][i]
                logger.info(f"  [{similarity:.3f}] {title[:60]}")
        else:
            logger.warning("  No results found")
    
    logger.info("\n✓ Query test complete")
    return True

def run_complete_test():
    """Run complete pipeline test"""
    print("""
╔══════════════════════════════════════════════════════════════╗
║          TESTING FIXED PIPELINE (5 papers)                   ║
╚══════════════════════════════════════════════════════════════╝

This will test:
1. Download 5 papers with fixed PDF downloader
2. Process papers into chunks
3. Create vector store
4. Test queries

""")
    
    try:
        # Clean
        clean_test_data()
        
        # Test 1: Download
        success, papers = test_download()
        if not success:
            return False
        
        # Test 2: Process
        success, chunks = test_processing(papers)
        if not success:
            return False
        
        # Test 3: Vector store
        success, vector_store, embedding_gen = test_vector_store(chunks)
        if not success:
            return False
        
        # Test 4: Query
        success = test_query(vector_store, embedding_gen)
        if not success:
            return False
        
        # Success!
        print("\n" + "="*60)
        print("✓✓✓ ALL TESTS PASSED! ✓✓✓")
        print("="*60)
        print(f"""
Your RAG system is working correctly!

Summary:
- Papers downloaded: {len(papers)}
- Chunks created: {len(chunks)}
- Vector store size: {vector_store.collection.count()}

NEXT STEPS:
1. Run full pipeline with more papers:
   python main.py
   
2. Start the API:
   uvicorn src.api.app:app --reload
   
3. Use the dashboard:
   streamlit run dashboard.py
   
4. Validate with:
   python quick_check.py
""")
        return True
        
    except Exception as e:
        logger.error(f"\n✗ Test failed with error: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = run_complete_test()
    sys.exit(0 if success else 1)