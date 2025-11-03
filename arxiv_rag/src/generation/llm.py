from openai import OpenAI
from typing import List, Dict
import logging

logger = logging.getLogger(__name__)

class LLMGenerator:
    """Generate responses using LLM"""
    
    def __init__(self, model: str = "gpt-4-turbo-preview", temperature: float = 0.1):
        self.client = OpenAI()
        self.model = model
        self.temperature = temperature
    
    def generate_response(self, query: str, context_docs: List[Dict]) -> Dict:
        """Generate response based on retrieved context"""
        
        # Build context from retrieved documents
        context = "\n\n---\n\n".join([
            f"Paper: {doc['metadata']['title']}\n"
            f"Authors: {doc['metadata']['authors']}\n"
            f"Content: {doc['text']}"
            for doc in context_docs
        ])
        
        system_prompt = """You are a helpful research assistant with access to ArXiv papers. 
        Answer questions based on the provided context from research papers. 
        If the context doesn't contain relevant information, say so.
        Always cite the papers you reference."""
        
        user_prompt = f"""Context from research papers:

{context}

Question: {query}

Please provide a detailed answer based on the context above. Include paper titles when citing sources."""
        
        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt}
                ],
                temperature=self.temperature
            )
            
            return {
                "answer": response.choices[0].message.content,
                "sources": [doc['metadata'] for doc in context_docs],
                "model": self.model
            }
        except Exception as e:
            logger.error(f"Error generating response: {e}")
            return {
                "answer": "Sorry, I encountered an error generating a response.",
                "sources": [],
                "error": str(e)
            }
