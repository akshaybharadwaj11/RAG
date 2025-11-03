import yaml
import logging
from pathlib import Path

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def run_pipeline():
    # Load configuration
    with open("config/settings.yaml") as f:
        config = yaml.safe_load(f)
    
    # 1. Fetch papers
    logger.info("Step 1: Fetching papers from ArXiv")
    from src.ingestion.arxiv_fetcher import ArxivFetcher
    
    fetcher = ArxivFetcher(
        categories=config['data']['categories'],
        max_results=config['data']['max_papers']
    )
    papers = fetcher.fetch_papers(Path(config['data']['raw_dir']))
    
    # 2. Process papers
    logger.info("Step 2: Processing papers")
    from src.ingestion.processor import DocumentProcessor
    
    processor = DocumentProcessor(
        chunk_size=config['data']['chunk_size'],
        chunk_overlap=config['data']['chunk_overlap']
    )
    chunks = processor.process_papers(papers)
    
    # 3. Generate embeddings and store
    logger.info("Step 3: Generating embeddings and storing in vector DB")
    from src.vectorstore.embeddings import EmbeddingGenerator
    from src.vectorstore.store import VectorStore
    
    embedding_gen = EmbeddingGenerator(config['embeddings']['model'])
    vector_store = VectorStore(
        collection_name=config['vectorstore']['collection_name'],
        persist_directory=config['vectorstore']['persist_directory'],
        embedding_function=embedding_gen
    )
    vector_store.add_documents(chunks, batch_size=config['embeddings']['batch_size'])
    
    logger.info("Pipeline completed successfully!")

if __name__ == "__main__":
    run_pipeline()