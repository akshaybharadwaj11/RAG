import chromadb
from chromadb.config import Settings
from typing import List, Dict, Optional
import logging

logger = logging.getLogger(__name__)

class VectorStore:
    """Manage vector storage and retrieval"""
    
    def __init__(self, 
                 collection_name: str,
                 persist_directory: str,
                 embedding_function):
        # self.client = chromadb.Client(Settings(
        #     persist_directory=persist_directory,
        #     anonymized_telemetry=False
        # ))

        self.client = chromadb.PersistentClient(path=persist_directory) 
         
        self.collection = self.client.get_or_create_collection(
            name=collection_name,
            metadata={"hnsw:space": "cosine"}
        )
        self.embedding_function = embedding_function
        
    def add_documents(self, chunks: List[Dict], batch_size: int = 100):
        """Add document chunks to vector store"""
        logger.info(f"Adding {len(chunks)} chunks to vector store")
        
        for i in range(0, len(chunks), batch_size):
            batch = chunks[i:i + batch_size]
            
            texts = [chunk["text"] for chunk in batch]
            embeddings = self.embedding_function.generate_embeddings(texts)
            
            ids = [f"{chunk['metadata']['paper_id']}_chunk_{chunk['metadata']['chunk_id']}" 
                   for chunk in batch]
            metadatas = [chunk["metadata"] for chunk in batch]
            
            self.collection.add(
                ids=ids,
                embeddings=embeddings.tolist(),
                documents=texts,
                metadatas=metadatas
            )
            
            if (i + batch_size) % 500 == 0:
                logger.info(f"Added {i + batch_size} chunks")
        
        logger.info("All chunks added to vector store")
    
    def query(self, 
              query_text: str,
              n_results: int = 5,
              where: Optional[Dict] = None) -> Dict:
        """Query the vector store"""
        query_embedding = self.embedding_function.generate_embeddings([query_text])
        
        results = self.collection.query(
            query_embeddings=query_embedding.tolist(),
            n_results=n_results,
            where=where
        )
        
        return results
