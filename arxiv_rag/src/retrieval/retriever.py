from typing import List, Dict
import logging

logger = logging.getLogger(__name__)

class Retriever:
    """Handle document retrieval with re-ranking"""
    
    def __init__(self, vector_store, top_k: int = 5, score_threshold: float = 0.7):
        self.vector_store = vector_store
        self.top_k = top_k
        self.score_threshold = score_threshold
    
    def retrieve(self, query: str, filters: Dict = None) -> List[Dict]:
        """Retrieve relevant documents"""
        results = self.vector_store.query(
            query_text=query,
            n_results=self.top_k,
            where=filters
        )
        
        # Format results
        documents = []
        for i in range(len(results['ids'][0])):
            doc = {
                "id": results['ids'][0][i],
                "text": results['documents'][0][i],
                "metadata": results['metadatas'][0][i],
                "distance": results['distances'][0][i] if 'distances' in results else None
            }
            documents.append(doc)
        
        # Filter by score threshold
        if self.score_threshold:
            documents = [doc for doc in documents 
                        if doc['distance'] and (1 - doc['distance']) >= self.score_threshold]
        
        logger.info(f"Retrieved {len(documents)} relevant documents")
        return documents
