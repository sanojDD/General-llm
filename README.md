# RAG Pipeline

A production-ready Retrieval-Augmented Generation (RAG) pipeline for building intelligent question-answering systems with document retrieval capabilities.

## 📋 Table of Contents

- [Overview](#overview)
- [Architecture](#architecture)
- [Features](#features)
- [Prerequisites](#prerequisites)
- [Installation](#installation)
- [Quick Start](#quick-start)
- [Configuration](#configuration)
- [Usage](#usage)
- [API Reference](#api-reference)
- [Advanced Topics](#advanced-topics)
- [Performance Optimization](#performance-optimization)
- [Troubleshooting](#troubleshooting)
- [Contributing](#contributing)
- [License](#license)

## 🎯 Overview

This RAG pipeline combines the power of document retrieval with large language models to provide accurate, context-aware responses grounded in your document corpus. It's designed for:

- Enterprise knowledge bases
- Document Q&A systems
- Research assistance tools
- Customer support automation
- Legal/medical document analysis

## 🏗 Architecture

```
┌─────────────┐
│  Documents  │
└──────┬──────┘
       │
       ▼
┌─────────────────┐
│  Text Splitter  │
└──────┬──────────┘
       │
       ▼
┌─────────────────┐
│   Embeddings    │
└──────┬──────────┘
       │
       ▼
┌─────────────────┐
│  Vector Store   │
└──────┬──────────┘
       │
       ▼
┌─────────────────┐      ┌─────────────┐
│    Retriever    │◄─────┤    Query    │
└──────┬──────────┘      └─────────────┘
       │
       ▼
┌─────────────────┐
│   LLM + Prompt  │
└──────┬──────────┘
       │
       ▼
┌─────────────────┐
│    Response     │
└─────────────────┘
```

### Components

1. **Document Loader**: Ingests various document formats (PDF, TXT, DOCX, HTML)
2. **Text Splitter**: Chunks documents into manageable pieces
3. **Embedding Model**: Converts text to vector representations
4. **Vector Store**: Stores and indexes embeddings (Pinecone, Weaviate, FAISS, Chroma)
5. **Retriever**: Finds relevant documents based on query similarity
6. **LLM**: Generates responses using retrieved context
7. **Orchestrator**: Manages the entire pipeline flow

## ✨ Features

- 📄 **Multi-format Support**: PDF, DOCX, TXT, Markdown, HTML, CSV
- 🔍 **Hybrid Search**: Combines semantic and keyword-based retrieval
- 🎯 **Smart Chunking**: Preserves context with overlapping chunks
- 💾 **Multiple Vector Stores**: Support for Pinecone, Weaviate, FAISS, Chroma
- 🔄 **Streaming Responses**: Real-time answer generation
- 📊 **Source Attribution**: Track which documents informed the answer
- 🔧 **Customizable Prompts**: Fine-tune response generation
- 🚀 **Production Ready**: Built with scalability and reliability in mind
- 📈 **Evaluation Metrics**: Built-in accuracy and relevance scoring
- 🔒 **Security**: Document-level access control

## 📦 Prerequisites

- Python 3.8+
- pip or conda
- API keys for:
  - OpenAI (or alternative LLM provider)
  - Vector database (Pinecone, Weaviate, etc.)

## 🚀 Installation

### Using pip

```bash
# Clone the repository
git clone https://github.com/yourusername/rag-pipeline.git
cd rag-pipeline

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

### Using Docker

```bash
docker build -t rag-pipeline .
docker run -p 8000:8000 rag-pipeline
```

### Requirements.txt

```
langchain>=0.1.0
openai>=1.0.0
tiktoken>=0.5.0
chromadb>=0.4.0
pinecone-client>=3.0.0
sentence-transformers>=2.2.0
PyPDF2>=3.0.0
python-docx>=1.0.0
fastapi>=0.104.0
uvicorn>=0.24.0
pydantic>=2.0.0
python-dotenv>=1.0.0
```

## ⚡ Quick Start

### 1. Environment Setup

Create a `.env` file:

```bash
OPENAI_API_KEY=your_openai_api_key
PINECONE_API_KEY=your_pinecone_api_key
PINECONE_ENV=your_pinecone_environment
```

### 2. Basic Usage

```python
from rag_pipeline import RAGPipeline

# Initialize pipeline
rag = RAGPipeline(
    embedding_model="text-embedding-3-small",
    llm_model="gpt-4-turbo-preview",
    vector_store="chroma"
)

# Load documents
rag.load_documents("./documents")

# Query
response = rag.query("What is the refund policy?")
print(response.answer)
print(response.sources)
```

### 3. Run API Server

```bash
python app.py
```

Access the API at `http://localhost:8000`

## ⚙️ Configuration

### config.yaml

```yaml
# Document Processing
chunk_size: 1000
chunk_overlap: 200
separators: ["\n\n", "\n", " ", ""]

# Embeddings
embedding_model: "text-embedding-3-small"
embedding_dimension: 1536

# Vector Store
vector_store: "chroma"
persist_directory: "./chroma_db"
collection_name: "documents"

# Retrieval
retrieval_method: "similarity"  # similarity, mmr, similarity_score_threshold
top_k: 5
score_threshold: 0.7

# LLM
llm_provider: "openai"
llm_model: "gpt-4-turbo-preview"
temperature: 0.2
max_tokens: 1000

# Prompts
system_prompt: |
  You are a helpful assistant that answers questions based on the provided context.
  Always cite your sources and admit when you don't know the answer.
```

## 📖 Usage

### Document Ingestion

```python
from rag_pipeline import DocumentLoader, TextSplitter

# Load documents
loader = DocumentLoader()
documents = loader.load_directory("./docs", glob_pattern="**/*.pdf")

# Split into chunks
splitter = TextSplitter(chunk_size=1000, chunk_overlap=200)
chunks = splitter.split_documents(documents)

# Add to vector store
rag.add_documents(chunks)
```

### Querying

```python
# Simple query
response = rag.query("What are the main features?")

# Query with filters
response = rag.query(
    "What is pricing?",
    filters={"category": "sales", "year": 2024}
)

# Streaming response
for chunk in rag.query_stream("Explain the architecture"):
    print(chunk, end="", flush=True)
```

### Advanced Retrieval

```python
# Hybrid search (semantic + keyword)
response = rag.query(
    "machine learning algorithms",
    retrieval_method="hybrid",
    alpha=0.7  # Weight between semantic (1.0) and keyword (0.0)
)

# MMR (Maximum Marginal Relevance) for diversity
response = rag.query(
    "renewable energy sources",
    retrieval_method="mmr",
    fetch_k=20,
    lambda_mult=0.5
)
```

### Custom Prompts

```python
custom_prompt = """
Use the following context to answer the question.
If you cannot answer based on the context, say so clearly.

Context: {context}

Question: {question}

Answer in a structured format with bullet points.
"""

rag.set_prompt_template(custom_prompt)
```

## 🔌 API Reference

### REST API Endpoints

#### POST /ingest
Upload and process documents

```bash
curl -X POST "http://localhost:8000/ingest" \
  -F "file=@document.pdf" \
  -F "metadata={\"category\":\"sales\"}"
```

#### POST /query
Query the RAG pipeline

```bash
curl -X POST "http://localhost:8000/query" \
  -H "Content-Type: application/json" \
  -d '{
    "question": "What is the pricing model?",
    "top_k": 5,
    "include_sources": true
  }'
```

#### GET /health
Health check endpoint

```bash
curl "http://localhost:8000/health"
```

### Python SDK

```python
from rag_pipeline import RAGPipeline, QueryConfig

# Initialize
pipeline = RAGPipeline.from_config("config.yaml")

# Configure query
config = QueryConfig(
    top_k=5,
    score_threshold=0.75,
    include_sources=True,
    stream=False
)

# Execute query
result = pipeline.query("your question", config=config)
```

## 🎓 Advanced Topics

### Custom Embedding Models

```python
from sentence_transformers import SentenceTransformer

# Use custom embedding model
embedding_model = SentenceTransformer('all-MiniLM-L6-v2')
rag = RAGPipeline(embedding_function=embedding_model.encode)
```

### Metadata Filtering

```python
# Add documents with metadata
rag.add_documents(
    chunks,
    metadatas=[
        {"source": "manual.pdf", "page": 1, "category": "technical"},
        {"source": "manual.pdf", "page": 2, "category": "technical"}
    ]
)

# Query with metadata filters
response = rag.query(
    "installation steps",
    filters={"category": "technical", "page": {"$gte": 1, "$lte": 10}}
)
```

### Re-ranking

```python
from rag_pipeline.rerankers import CohereReranker

# Add re-ranker
reranker = CohereReranker(api_key="your_key")
rag.set_reranker(reranker, top_n=3)
```

### Evaluation

```python
from rag_pipeline.evaluation import evaluate_pipeline

# Evaluate with test questions
test_data = [
    {"question": "What is RAG?", "expected_answer": "Retrieval-Augmented Generation..."},
    # ... more test cases
]

metrics = evaluate_pipeline(rag, test_data)
print(f"Accuracy: {metrics.accuracy}")
print(f"Avg Relevance: {metrics.avg_relevance}")
```

## 🚀 Performance Optimization

### Caching

```python
# Enable query caching
rag.enable_cache(backend="redis", ttl=3600)
```

### Batch Processing

```python
# Process multiple queries in parallel
questions = ["Q1", "Q2", "Q3"]
responses = rag.batch_query(questions, batch_size=10)
```

### Vector Store Optimization

```python
# Create indexes for faster retrieval
rag.create_index(index_type="IVF", nlist=100)

# Optimize chunk size based on your data
rag.optimize_chunk_size(sample_documents=docs, target_chunks=50)
```

## 🔧 Troubleshooting

### Common Issues

**Issue**: Slow retrieval times
```python
# Solution: Reduce chunk size or enable indexing
rag.config.chunk_size = 500
rag.create_index()
```

**Issue**: Irrelevant results
```python
# Solution: Adjust similarity threshold
rag.config.score_threshold = 0.8
rag.config.top_k = 3
```

**Issue**: Out of memory
```python
# Solution: Use batched ingestion
rag.add_documents_batch(chunks, batch_size=100)
```

### Debugging

```python
# Enable debug logging
import logging
logging.basicConfig(level=logging.DEBUG)

# Get retrieval diagnostics
response = rag.query("test", debug=True)
print(response.debug_info)
```

## 🤝 Contributing

We welcome contributions! Please see [CONTRIBUTING.md](CONTRIBUTING.md) for guidelines.

```bash
# Fork the repo and create a feature branch
git checkout -b feature/amazing-feature

# Make your changes and commit
git commit -m "Add amazing feature"

# Push and create a Pull Request
git push origin feature/amazing-feature
```

## 📄 License

This project is licensed under the MIT License - see [LICENSE](LICENSE) file for details.

## 📚 Additional Resources

- [RAG Best Practices](docs/best_practices.md)
- [Architecture Deep Dive](docs/architecture.md)
- [API Documentation](docs/api.md)
- [Examples](examples/)

## 🙏 Acknowledgments

- LangChain for the orchestration framework
- OpenAI for embedding and LLM models
- The open-source community

## 📞 Support

- 📧 Email: support@ragpipeline.com
- 💬 Discord: [Join our community](https://discord.gg/ragpipeline)
- 🐛 Issues: [GitHub Issues](https://github.com/yourusername/rag-pipeline/issues)

---

**Star ⭐ this repo if you find it helpful!**
