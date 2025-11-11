# ArXiv Research Assistant - Production RAG System

A production-scale Retrieval-Augmented Generation (RAG) system for querying ArXiv research papers. 

## Features

- 🔍 **Semantic Search** - Vector-based similarity search with ChromaDB
- 📚 **Real Data** - ArXiv research papers (CS.AI, CS.LG, CS.CL)
- 🤖 **LLM Integration** - GPT-4 powered answer generation with citations
- ⚡ **FastAPI Backend** - RESTful API with async support
- 📊 **Interactive Dashboard** - Streamlit UI for exploration
- 🐳 **Docker Ready** - Easy deployment
- ✅ **Production Best Practices** - Proper error handling, logging, validation

## Quick Start

### Prerequisites
```bash
python 3.10+
pip
OpenAI API key
```

### Installation

```bash
# Clone repository
git clone https://github.com/akshaybharadwaj11/Arxiv_Assistant.git
cd Arxiv_Assistant

# Create virtual environment
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Set OpenAI API key
export OPENAI_API_KEY='your-key-here'

# Create directories
mkdir -p data/raw data/embeddings/chromadb config
```

### Configuration

Create `config/settings.yaml`:

```yaml
data:
  raw_dir: "data/raw"
  processed_dir: "data/processed"
  embeddings_dir: "data/embeddings"
  categories: ["cs.AI", "cs.LG", "cs.CL"]
  max_papers: 100
  chunk_size: 1000
  chunk_overlap: 200

embeddings:
  model: "sentence-transformers/all-mpnet-base-v2"
  batch_size: 32
  dimension: 768

vectorstore:
  collection_name: "arxiv_papers"
  persist_directory: "data/embeddings/chromadb"

retrieval:
  top_k: 5
  score_threshold: 0.7

llm:
  model: "gpt-4-turbo-preview"
  temperature: 0.1
  max_tokens: 1000

api:
  host: "0.0.0.0"
  port: 8000
```

## Usage

### 1. Test Installation (2 minutes)
```bash
# Verify downloads work with 5 papers
python test_fixed_pipeline.py
```

Expected output:
```
✓ Downloaded 5/5 PDFs
✓ Created 87 chunks
✓ Vector store created
✓✓✓ ALL TESTS PASSED!
```

### 2. Run Full Pipeline
```bash
# Fetch, process, and index papers
python main.py
```

This will:
- Fetch papers from ArXiv
- Download and process PDFs
- Generate embeddings
- Store in vector database

Time: ~10-30 minutes depending on number of papers

### 3. Validate Data
```bash
python quick_check.py
```

### 4. Start API Server
```bash
uvicorn src.api.app:app --reload
```

API runs at: `http://localhost:8000`

### 5. Query the System

**Via cURL:**
```bash
curl -X POST "http://localhost:8000/query" \
  -H "Content-Type: application/json" \
  -d '{"query": "What are transformer architectures?"}'
```

**Via Python:**
```python
import requests

response = requests.post(
    "http://localhost:8000/query",
    json={"query": "Explain attention mechanisms"}
)

print(response.json()["answer"])
```

**Via Dashboard:**
```bash
streamlit run dashboard.py
```

Dashboard at: `http://localhost:8501`

## System Architecture

```
┌─────────────────────────────────────────────────────────┐
│                    RAG PIPELINE                         │
└─────────────────────────────────────────────────────────┘

1. INGEST: ArXiv API → Fetch Papers → Download PDFs
2. PROCESS: Extract Text → Chunk (1000 chars) → Metadata
3. EMBED: sentence-transformers → 768-dim vectors
4. STORE: ChromaDB → Vector Index
5. RETRIEVE: Query → Search → Top-K Results
6. GENERATE: Context + Query → GPT-4 → Answer
```

## Project Structure

```
arxiv-rag-system/
├── config/
│   └── settings.yaml
├── data/
│   ├── raw/                    # PDFs and metadata
│   └── embeddings/chromadb/    # Vector database
├── src/
│   ├── ingestion/
│   │   ├── arxiv_fetcher.py   # Fetch papers
│   │   └── processor.py       # Text processing
│   ├── vectorstore/
│   │   ├── embeddings.py      # Generate embeddings
│   │   └── store.py           # Vector storage
│   ├── retrieval/
│   │   └── retriever.py       # Document retrieval
│   ├── generation/
│   │   └── llm.py             # LLM integration
│   └── api/
│       └── app.py             # FastAPI server
├── main.py                     # Main pipeline
├── test_fixed_pipeline.py      # Quick test
├── quick_check.py              # Validation
├── dashboard.py                # Streamlit UI
├── requirements.txt
└── README.md
```

## Tech Stack

- **Python 3.10+** - Core language
- **FastAPI** - REST API framework
- **ChromaDB** - Vector database
- **sentence-transformers** - Text embeddings (all-mpnet-base-v2)
- **OpenAI GPT-4** - Answer generation
- **Streamlit** - Interactive dashboard
- **ArXiv API** - Research papers dataset

## Troubleshooting

### PDF Downloads Failing
```bash
# Run diagnostic
python diagnose_download.py

# Or use abstract-only mode (no PDFs needed)
python abstract_only_mode.py
```

### ChromaDB Issues
```bash
pip install --upgrade chromadb
```

### Out of Memory
Reduce `batch_size` in `config/settings.yaml` to `16`

### Slow Queries
Reduce `top_k` in `config/settings.yaml` to `3`

## Validation

Check system health:
```bash
# Quick validation
python quick_check.py

# Health endpoint
curl http://localhost:8000/health

# Dashboard
streamlit run dashboard.py
```

## Docker Deployment

```bash
# Build and run
docker build -t arxiv-rag .
docker run -p 8000:8000 -e OPENAI_API_KEY=$OPENAI_API_KEY arxiv-rag

# Or use docker-compose
docker-compose up -d
```

## Performance

| Metric | Value |
|--------|-------|
| Papers | 100-1000 |
| Chunks per paper | ~18 |
| Query latency | <500ms |
| Storage | ~2-3 GB |
| Embedding time | 1-3 min (GPU) |

## Example Queries

```python
# Basic query
"What is a transformer in deep learning?"

# Technical deep dive
"Explain the attention mechanism with mathematical details"

# Comparison
"Compare BERT and GPT architectures"

# Recent research
"What are the latest advances in vision transformers?"
```

## API Endpoints

### Query Papers
```bash
POST /query
{
  "query": "your question here",
  "top_k": 5
}
```

### Health Check
```bash
GET /health
```

### API Documentation
Interactive docs at: `http://localhost:8000/docs`

## Development

```bash
# Install dev dependencies
pip install -r requirements-dev.txt

# Run tests
pytest tests/

# Format code
black src/ tests/
isort src/ tests/
```

## Roadmap

- [x] Core RAG pipeline
- [x] FastAPI backend
- [x] Streamlit dashboard
- [x] Docker support
- [ ] Evaluation metrics (RAGAS)
- [ ] Hybrid search (BM25 + vector)
- [ ] Query caching
- [ ] Authentication
- [ ] Prometheus metrics

## Common Issues

**Q: Why are no PDFs downloading?**  
A: Run `python diagnose_download.py` to identify the issue. Use `abstract_only_mode.py` as fallback.

**Q: How do I add more papers?**  
A: Increase `max_papers` in `config/settings.yaml` and re-run `python main.py`. The system appends new data.

**Q: Can I use a different LLM?**  
A: Yes, modify `src/generation/llm.py` to use any LLM provider (Anthropic, local models, etc.)

**Q: How do I reset the system?**  
A: Delete `data/` directory and re-run pipeline.

## Contributing

Contributions welcome! Please:
1. Fork the repository
2. Create a feature branch
3. Add tests for new features
4. Submit a pull request

## License

MIT License - see LICENSE file for details

## Acknowledgments

- ArXiv for providing open access to research papers
- Hugging Face for sentence-transformers
- ChromaDB team for the vector database

---

**Questions?** Open an issue on GitHub