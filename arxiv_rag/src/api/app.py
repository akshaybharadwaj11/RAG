from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import List, Optional, Dict
import yaml
from pathlib import Path
import logging

# Initialize logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(
    title="ArXiv RAG API",
    description="Production-scale RAG system for ArXiv research papers",
    version="1.0.0"
)

# Request/Response models
class QueryRequest(BaseModel):
    query: str
    top_k: Optional[int] = 5
    categories: Optional[List[str]] = None

class QueryResponse(BaseModel):
    query: str
    answer: str
    sources: List[Dict]
    retrieved_docs: int

# Global variables (initialized on startup)
retriever = None
llm_generator = None

@app.on_event("startup")
async def startup_event():
    """Initialize RAG components on startup"""
    global retriever, llm_generator
    
    logger.info("Initializing RAG system...")
    
    # Load config
    with open("config/settings.yaml") as f:
        config = yaml.safe_load(f)
    
    # Initialize components
    from src.vectorstore.embeddings import EmbeddingGenerator
    from src.vectorstore.store import VectorStore
    from src.retrieval.retriever import Retriever
    from src.generation.llm import LLMGenerator
    
    embedding_gen = EmbeddingGenerator(config['embeddings']['model'])
    vector_store = VectorStore(
        collection_name=config['vectorstore']['collection_name'],
        persist_directory=config['vectorstore']['persist_directory'],
        embedding_function=embedding_gen
    )
    
    retriever = Retriever(
        vector_store=vector_store,
        top_k=config['retrieval']['top_k'],
        score_threshold=config['retrieval']['score_threshold']
    )
    
    llm_generator = LLMGenerator(
        model=config['llm']['model'],
        temperature=config['llm']['temperature']
    )
    
    logger.info("RAG system initialized successfully")

@app.post("/query", response_model=QueryResponse)
async def query_papers(request: QueryRequest):
    """Query the RAG system"""
    try:
        # Build filters if categories specified
        filters = None
        if request.categories:
            filters = {"categories": {"$in": request.categories}}
        
        # Retrieve relevant documents
        docs = retriever.retrieve(request.query, filters=filters)
        
        if not docs:
            return QueryResponse(
                query=request.query,
                answer="No relevant papers found for your query.",
                sources=[],
                retrieved_docs=0
            )
        
        # Generate response
        result = llm_generator.generate_response(request.query, docs)
        
        return QueryResponse(
            query=request.query,
            answer=result['answer'],
            sources=result['sources'],
            retrieved_docs=len(docs)
        )
        
    except Exception as e:
        logger.error(f"Error processing query: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {"status": "healthy", "service": "arxiv-rag"}
