
import arxiv
import logging
from typing import List, Dict
from pathlib import Path
import json
from datetime import datetime
import requests 

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class ArxivFetcher:
    """Fetch papers from ArXiv API"""
    
    def __init__(self, categories: List[str], max_results: int = 10):
        self.categories = categories
        self.max_results = max_results
        
    def fetch_papers(self, output_dir: Path) -> List[Dict]:
        """Fetch papers from ArXiv and save metadata"""
        output_dir.mkdir(parents=True, exist_ok=True)
        papers = []
        
        for category in self.categories:
            logger.info(f"Fetching papers for category: {category}")
            
            search = arxiv.Search(
                query=f"cat:{category}",
                max_results=self.max_results // len(self.categories),
                sort_by=arxiv.SortCriterion.SubmittedDate
            )
            
            for result in search.results():
                paper_data = {
                    "id": result.entry_id,
                    "title": result.title,
                    "authors": [str(author) for author in result.authors],
                    "abstract": result.summary,
                    "categories": result.categories,
                    "published": result.published.isoformat(),
                    "pdf_url": result.pdf_url,
                    "fetched_at": datetime.now().isoformat()
                }
                papers.append(paper_data)
                
                # Download PDF
                pdf_path = output_dir / f"{result.entry_id.split('/')[-1]}.pdf"
                try:
                    # result.download_pdf(str(pdf_path))
                    response = requests.get(result.pdf_url, timeout=30, stream=True)  # ✅ Proven to work
                    with open(pdf_path, 'wb') as f:
                        for chunk in response.iter_content(chunk_size=8192):
                            f.write(chunk)
                    paper_data["pdf_path"] = str(pdf_path)
                except Exception as e:
                    logger.error(f"Failed to download {result.entry_id}: {e}")
                    continue
                
                if len(papers) % 50 == 0:
                    logger.info(f"Fetched {len(papers)} papers")
        
        # Save metadata
        metadata_path = output_dir / "metadata.json"
        with open(metadata_path, 'w') as f:
            json.dump(papers, f, indent=2)
        
        logger.info(f"Total papers fetched: {len(papers)}")
        return papers
