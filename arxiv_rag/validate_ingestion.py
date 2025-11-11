# ============================================================================
# 1. INGESTION VALIDATOR - validate_ingestion.py
# ============================================================================

import json
import logging
from pathlib import Path
from collections import Counter
import pandas as pd

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class IngestionValidator:
    """Validate that documents were ingested correctly"""
    
    def __init__(self, data_dir: str = "data"):
        self.data_dir = Path(data_dir)
        self.raw_dir = self.data_dir / "raw"
        self.processed_dir = self.data_dir / "processed"
        
    def validate_raw_data(self) -> dict:
        """Validate raw data download"""
        logger.info("Validating raw data...")
        
        metadata_path = self.raw_dir / "metadata.json"
        if not metadata_path.exists():
            return {"status": "FAILED", "error": "metadata.json not found"}
        
        with open(metadata_path) as f:
            papers = json.load(f)
        
        # Check PDFs
        pdf_count = len(list(self.raw_dir.glob("*.pdf")))
        papers_with_pdfs = sum(1 for p in papers if "pdf_path" in p)
        
        stats = {
            "status": "SUCCESS",
            "total_papers": len(papers),
            "papers_with_pdfs": papers_with_pdfs,
            "pdf_files": pdf_count,
            "categories": Counter([cat for p in papers for cat in p.get("categories", [])]),
            "date_range": {
                "earliest": min(p["published"] for p in papers),
                "latest": max(p["published"] for p in papers)
            }
        }
        
        logger.info(f"✓ Raw data validation complete: {stats['total_papers']} papers")
        return stats
    
    def validate_vector_store(self) -> dict:
        """Validate vector store ingestion"""
        logger.info("Validating vector store...")
        
        try:
            import chromadb
            from chromadb.config import Settings
            
            # client = chromadb.Client(Settings(
            #     persist_directory=str(self.data_dir / "embeddings" / "chromadb"),
            #     anonymized_telemetry=False
            # ))

            client = chromadb.PersistentClient(path=str(self.data_dir / "embeddings" / "chromadb")) 
            
            collection = client.get_collection("arxiv_papers")
            count = collection.count()
            
            # Sample some documents
            sample = collection.get(limit=5, include=["documents", "metadatas", "embeddings"])
            
            stats = {
                "status": "SUCCESS",
                "total_chunks": count,
                "embedding_dimension": len(sample["embeddings"][0]) if sample["embeddings"] else 0,
                "sample_papers": [m.get("title") for m in sample["metadatas"]],
                "sample_chunk_lengths": [len(doc) for doc in sample["documents"]]
            }
            
            logger.info(f"✓ Vector store validation complete: {count} chunks indexed")
            return stats
            
        except Exception as e:
            return {"status": "FAILED", "error": str(e)}
    
    def run_full_validation(self) -> dict:
        """Run complete validation pipeline"""
        logger.info("="*60)
        logger.info("Starting Full Ingestion Validation")
        logger.info("="*60)
        
        results = {
            "raw_data": self.validate_raw_data(),
            "vector_store": self.validate_vector_store()
        }
        
        # Overall health check
        all_success = all(r["status"] == "SUCCESS" for r in results.values())
        results["overall_status"] = "HEALTHY" if all_success else "ISSUES_FOUND"
        
        # Print summary
        self._print_summary(results)
        
        return results
    
    def _print_summary(self, results: dict):
        """Print validation summary"""
        print("\n" + "="*60)
        print("VALIDATION SUMMARY")
        print("="*60)
        
        if results["raw_data"]["status"] == "SUCCESS":
            print(f"✓ Raw Data: {results['raw_data']['total_papers']} papers downloaded")
            print(f"  - PDFs: {results['raw_data']['papers_with_pdfs']}")
            print(f"  - Categories: {dict(results['raw_data']['categories'])}")
        else:
            print(f"✗ Raw Data: {results['raw_data'].get('error')}")
        
        print()
        
        if results["vector_store"]["status"] == "SUCCESS":
            print(f"✓ Vector Store: {results['vector_store']['total_chunks']} chunks indexed")
            print(f"  - Embedding dim: {results['vector_store']['embedding_dimension']}")
            print(f"  - Sample papers:")
            for paper in results["vector_store"]["sample_papers"][:3]:
                print(f"    • {paper}")
        else:
            print(f"✗ Vector Store: {results['vector_store'].get('error')}")
        
        print()
        print(f"Overall Status: {results['overall_status']}")
        print("="*60 + "\n")
