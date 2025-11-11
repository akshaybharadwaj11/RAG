# ============================================================================
# STEP-BY-STEP FIX GUIDE
# Get your RAG system up and running properly
# ============================================================================

"""
PROBLEM DIAGNOSIS:
✗ Only 8 papers fetched (expected ~1000)
✗ 0 PDFs downloaded (all failed)
✗ Vector store empty (nothing to index)
✗ ChromaDB telemetry errors

SOLUTIONS:
✓ Updated ChromaDB client (PersistentClient instead of Client)
✓ Robust PDF downloader with retries
✓ Better error handling
✓ Disabled telemetry
"""

# ============================================================================
# STEP 1: Clean Up Old Data
# ============================================================================

"""
# Run this first to start fresh
import shutil
from pathlib import Path

# Remove old data
for dir_path in ["data/raw", "data/embeddings"]:
    if Path(dir_path).exists():
        shutil.rmtree(dir_path)
        print(f"✓ Cleaned {dir_path}")

# Create fresh directories
Path("data/raw").mkdir(parents=True, exist_ok=True)
Path("data/embeddings/chromadb").mkdir(parents=True, exist_ok=True)
print("✓ Created fresh directories")
"""

# ============================================================================
# STEP 2: Update Dependencies
# ============================================================================

"""
# Make sure you have the latest versions
pip install --upgrade chromadb requests arxiv
pip install PyPDF2 sentence-transformers pandas numpy

# Verify versions
python -c "import chromadb; print(f'ChromaDB: {chromadb.__version__}')"
python -c "import requests; print(f'Requests: {requests.__version__}')"
"""

# ============================================================================
# STEP 3: Test ArXiv API First (Small Scale)
# ============================================================================

import arxiv
import logging
from pathlib import Path
import requests
import time

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def test_arxiv_download():
    """Test with just 3 papers to verify it works"""
    logger.info("Testing ArXiv download with 3 papers...")
    
    output_dir = Path("data/raw")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Search for just 3 papers
    search = arxiv.Search(
        query="cat:cs.AI",
        max_results=3,
        sort_by=arxiv.SortCriterion.SubmittedDate
    )
    
    successful = 0
    for i, result in enumerate(search.results(), 1):
        logger.info(f"\n[{i}/3] Processing: {result.title[:60]}...")
        
        # Download PDF
        pdf_filename = result.entry_id.split('/')[-1].replace('.', '_') + ".pdf"
        pdf_path = output_dir / pdf_filename
        
        try:
            # Use requests for robust download
            response = requests.get(result.pdf_url, timeout=30, stream=True)
            response.raise_for_status()
            
            with open(pdf_path, 'wb') as f:
                for chunk in response.iter_content(chunk_size=8192):
                    f.write(chunk)
            
            # Check file
            file_size = pdf_path.stat().st_size
            if file_size > 1000:
                logger.info(f"✓ Downloaded: {file_size / 1024:.1f} KB")
                successful += 1
            else:
                logger.warning(f"✗ File too small: {file_size} bytes")
                
        except Exception as e:
            logger.error(f"✗ Download failed: {e}")
        
        time.sleep(1)  # Rate limiting
    
    logger.info(f"\n{'='*60}")
    logger.info(f"Test complete: {successful}/3 PDFs downloaded successfully")
    logger.info(f"{'='*60}\n")
    
    if successful >= 2:
        logger.info("✓ ArXiv download working! Ready for full pipeline.")
        return True
    else:
        logger.error("✗ Downloads not working. Check your internet connection.")
        return False

# ============================================================================
# STEP 4: Updated Main Pipeline (Start Small, Then Scale)
# ============================================================================

def run_pipeline_incremental():
    """Run pipeline starting small, then scale up"""
    import yaml
    from pathlib import Path
    
    # Load configuration
    with open("config/settings.yaml") as f:
        config = yaml.safe_load(f)
    
    # PHASE 1: Test with 20 papers
    logger.info("\n" + "="*60)
    logger.info("PHASE 1: Testing with 20 papers")
    logger.info("="*60 + "\n")
    
    from src.ingestion.arxiv_fetcher import ArxivFetcher
    
    fetcher = ArxivFetcher(
        categories=config['data']['categories'],
        max_results=20  # Start small!
    )
    papers = fetcher.fetch_papers(Path(config['data']['raw_dir']))
    
    if not papers:
        logger.error("No papers fetched. Stopping.")
        return
    
    pdfs_downloaded = sum(1 for p in papers if p.get('pdf_path'))
    logger.info(f"Phase 1 result: {pdfs_downloaded}/{len(papers)} PDFs downloaded")
    
    if pdfs_downloaded < len(papers) * 0.5:
        logger.error("More than 50% of downloads failed. Check your connection.")
        return
    
    # PHASE 2: Process papers
    logger.info("\n" + "="*60)
    logger.info("PHASE 2: Processing papers into chunks")
    logger.info("="*60 + "\n")
    
    from src.ingestion.processor import DocumentProcessor
    
    processor = DocumentProcessor(
        chunk_size=config['data']['chunk_size'],
        chunk_overlap=config['data']['chunk_overlap']
    )
    chunks = processor.process_papers(papers)
    
    if not chunks:
        logger.error("No chunks created. Check PDF processing.")
        return
    
    logger.info(f"Phase 2 result: {len(chunks)} chunks created")
    
    # PHASE 3: Generate embeddings and store
    logger.info("\n" + "="*60)
    logger.info("PHASE 3: Creating vector store")
    logger.info("="*60 + "\n")
    
    from src.vectorstore.embeddings import EmbeddingGenerator
    from src.vectorstore.store import VectorStore
    
    embedding_gen = EmbeddingGenerator(config['embeddings']['model'])
    vector_store = VectorStore(
        collection_name=config['vectorstore']['collection_name'],
        persist_directory=config['vectorstore']['persist_directory'],
        embedding_function=embedding_gen
    )
    
    vector_store.add_documents(chunks, batch_size=config['embeddings']['batch_size'])
    
    logger.info("\n" + "="*60)
    logger.info("✓ PIPELINE COMPLETE!")
    logger.info(f"✓ {len(papers)} papers processed")
    logger.info(f"✓ {len(chunks)} chunks indexed")
    logger.info("="*60 + "\n")
    
    # PHASE 4: Validate
    logger.info("Running validation...")
    from validate_ingestion import IngestionValidator
    
    validator = IngestionValidator()
    results = validator.run_full_validation()
    
    if results["overall_status"] == "HEALTHY":
        logger.info("\n✓✓✓ SYSTEM HEALTHY! You can now scale to 1000 papers ✓✓✓\n")
        return True
    else:
        logger.error("\n✗ Issues found. Review the validation output above.\n")
        return False

# ============================================================================
# STEP 5: Scale to Full Dataset
# ============================================================================

def scale_to_full():
    """Once small test works, scale to 1000 papers"""
    logger.info("Scaling to full 1000 papers...")
    
    # Update config
    import yaml
    with open("config/settings.yaml") as f:
        config = yaml.safe_load(f)
    
    config['data']['max_papers'] = 1000
    
    with open("config/settings.yaml", 'w') as f:
        yaml.dump(config, f)
    
    # Run full pipeline
    run_pipeline_incremental()

# ============================================================================
# STEP 6: Quick Test Query
# ============================================================================

def test_query_system():
    """Test that retrieval works"""
    logger.info("\n" + "="*60)
    logger.info("Testing Query System")
    logger.info("="*60 + "\n")
    
    from src.vectorstore.embeddings import EmbeddingGenerator
    from explore_data import DataExplorer
    
    explorer = DataExplorer()
    embedding_gen = EmbeddingGenerator()
    
    test_queries = [
        "What are transformers in deep learning?",
        "How does attention mechanism work?",
        "What is transfer learning?"
    ]
    
    for query in test_queries:
        logger.info(f"\nQuery: {query}")
        
        query_embedding = embedding_gen.generate_embeddings([query])
        results = explorer.collection.query(
            query_embeddings=query_embedding.tolist(),
            n_results=3
        )
        
        logger.info("Top Results:")
        for i in range(min(3, len(results['ids'][0]))):
            title = results['metadatas'][0][i].get('title', 'Unknown')
            distance = results['distances'][0][i]
            similarity = 1 - distance
            logger.info(f"  {i+1}. [{similarity:.3f}] {title[:60]}...")
    
    logger.info("\n" + "="*60)
    logger.info("✓ Query test complete!")
    logger.info("="*60 + "\n")

# ============================================================================
# MAIN EXECUTION SCRIPT
# ============================================================================

if __name__ == "__main__":
    print("""
╔══════════════════════════════════════════════════════════════╗
║          RAG SYSTEM - RECOVERY & SETUP GUIDE                 ║
╚══════════════════════════════════════════════════════════════╝

This will:
1. Clean old data
2. Test ArXiv downloads (3 papers)
3. Run incremental pipeline (20 papers)
4. Validate everything
5. Test queries

Ready? Let's go!
""")
    
    input("Press Enter to start...")
    
    # Step 1: Clean
    print("\n[1/5] Cleaning old data...")
    import shutil
    for dir_path in ["data/raw", "data/embeddings"]:
        if Path(dir_path).exists():
            shutil.rmtree(dir_path)
    Path("data/raw").mkdir(parents=True, exist_ok=True)
    Path("data/embeddings/chromadb").mkdir(parents=True, exist_ok=True)
    print("✓ Cleaned")
    
    # Step 2: Test downloads
    print("\n[2/5] Testing ArXiv downloads...")
    if not test_arxiv_download():
        print("\n✗ Download test failed. Check your internet connection.")
        print("Try: ping arxiv.org")
        exit(1)
    
    # Step 3: Run pipeline
    print("\n[3/5] Running pipeline with 20 papers...")
    success = run_pipeline_incremental()
    
    if not success:
        print("\n✗ Pipeline failed. Check errors above.")
        exit(1)
    
    # Step 4: Test queries
    print("\n[4/5] Testing query system...")
    test_query_system()
    
    # Step 5: Success!
    print("""
╔══════════════════════════════════════════════════════════════╗
║                    ✓ SUCCESS!                                ║
╚══════════════════════════════════════════════════════════════╝

Your RAG system is now working with 20 papers!

NEXT STEPS:
1. Explore your data:
   streamlit run dashboard.py

2. Query the API:
   uvicorn src.api.app:app --reload
   
3. Scale to 1000 papers:
   python
   >>> from fix_and_run import scale_to_full
   >>> scale_to_full()

4. Run validation anytime:
   python quick_check.py
""")

# ============================================================================
# TROUBLESHOOTING TIPS
# ============================================================================

print("""
TROUBLESHOOTING:

Issue: "requests module not found"
Fix: pip install requests

Issue: "ChromaDB errors"
Fix: pip install --upgrade chromadb

Issue: PDFs still not downloading
Fix: 
  1. Check internet: ping arxiv.org
  2. Try different network (some block ArXiv)
  3. Use VPN if behind firewall

Issue: "Out of memory"
Fix: Reduce batch_size in config/settings.yaml to 16

Issue: Slow downloads
Fix: Normal! ArXiv limits ~1 request/sec. 1000 papers = ~20 mins

Issue: Want to add papers incrementally
Fix: The vector store appends new data, no need to delete old

Issue: Collection already exists error
Fix: It's fine! The code uses get_or_create_collection
""")