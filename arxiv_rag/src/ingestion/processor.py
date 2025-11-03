from PyPDF2 import PdfReader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from typing import List, Dict
import logging

logger = logging.getLogger(__name__)

class DocumentProcessor:
    """Process PDFs and chunk documents"""
    
    def __init__(self, chunk_size: int = 1000, chunk_overlap: int = 200):
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            separators=["\n\n", "\n", ". ", " ", ""]
        )
    
    def extract_text_from_pdf(self, pdf_path: str) -> str:
        """Extract text from PDF file"""
        try:
            reader = PdfReader(pdf_path)
            text = ""
            for page in reader.pages:
                text += page.extract_text() + "\n"
            return text
        except Exception as e:
            logger.error(f"Error extracting text from {pdf_path}: {e}")
            return ""
    
    def process_papers(self, papers: List[Dict]) -> List[Dict]:
        """Process all papers into chunks"""
        processed_chunks = []
        
        for paper in papers:
            if "pdf_path" not in paper:
                continue
                
            # Extract text
            full_text = self.extract_text_from_pdf(paper["pdf_path"])
            
            if not full_text:
                continue
            
            # Combine abstract and full text
            combined_text = f"Title: {paper['title']}\n\n"
            combined_text += f"Abstract: {paper['abstract']}\n\n"
            combined_text += f"Full Text:\n{full_text}"
            
            # Split into chunks
            chunks = self.text_splitter.split_text(combined_text)
            
            # Create chunk documents with metadata
            for i, chunk in enumerate(chunks):
                chunk_doc = {
                    "text": chunk,
                    "metadata": {
                        "paper_id": paper["id"],
                        "title": paper["title"],
                        "authors": ", ".join(paper["authors"]),
                        "categories": ", ".join(paper["categories"]),
                        "published": paper["published"],
                        "chunk_id": i,
                        "total_chunks": len(chunks)
                    }
                }
                processed_chunks.append(chunk_doc)
            
            if len(processed_chunks) % 100 == 0:
                logger.info(f"Processed {len(processed_chunks)} chunks")
        
        logger.info(f"Total chunks created: {len(processed_chunks)}")
        return processed_chunks
