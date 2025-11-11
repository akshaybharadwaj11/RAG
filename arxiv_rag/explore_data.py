# ============================================================================
# 2. DATA EXPLORER - explore_data.py
# ============================================================================

import chromadb
from chromadb.config import Settings
import pandas as pd
from collections import defaultdict

class DataExplorer:
    """Explore ingested data"""
    
    def __init__(self, persist_directory: str = "data/embeddings/chromadb"):
        self.client = chromadb.PersistentClient(path=persist_directory)
        self.collection = self.client.get_collection("arxiv_papers")
    
    def get_statistics(self) -> dict:
        """Get comprehensive statistics"""
        total_count = self.collection.count()
        
        # Get all metadata (in batches to avoid memory issues)
        batch_size = 1000
        all_metadata = []
        
        for offset in range(0, total_count, batch_size):
            batch = self.collection.get(
                limit=batch_size,
                offset=offset,
                include=["metadatas", "documents"]
            )
            all_metadata.extend(batch["metadatas"])
        
        # Analyze
        papers = defaultdict(int)
        categories = Counter()
        chunk_distribution = defaultdict(int)
        
        for meta in all_metadata:
            papers[meta["paper_id"]] += 1
            if "categories" in meta:
                for cat in meta["categories"].split(", "):
                    categories[cat] += 1
            chunk_distribution[meta.get("total_chunks", 0)] += 1
        
        return {
            "total_chunks": total_count,
            "unique_papers": len(papers),
            "avg_chunks_per_paper": sum(papers.values()) / len(papers),
            "categories": dict(categories.most_common(10)),
            "chunk_distribution": dict(sorted(chunk_distribution.items())[:10])
        }
    
    def search_papers(self, keyword: str, limit: int = 10) -> pd.DataFrame:
        """Search papers by keyword"""
        results = self.collection.get(
            where={"title": {"$contains": keyword}},
            limit=limit,
            include=["metadatas"]
        )
        
        if not results["metadatas"]:
            return pd.DataFrame()
        
        df = pd.DataFrame(results["metadatas"])
        return df[["title", "authors", "categories", "published"]].drop_duplicates()
    
    def get_sample_chunks(self, paper_id: str = None, n: int = 3) -> list:
        """Get sample chunks from a paper"""
        filters = {"paper_id": paper_id} if paper_id else None
        
        results = self.collection.get(
            where=filters,
            limit=n,
            include=["documents", "metadatas"]
        )
        
        samples = []
        for doc, meta in zip(results["documents"], results["metadatas"]):
            samples.append({
                "chunk_id": meta.get("chunk_id"),
                "paper_title": meta.get("title"),
                "text_preview": doc[:200] + "..." if len(doc) > 200 else doc,
                "length": len(doc)
            })
        
        return samples
