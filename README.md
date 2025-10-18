# RAG Pipeline API with Conversation Memory

[![CI/CD](https://github.com/yourusername/rag-pipeline/actions/workflows/CI.yml/badge.svg)](https://github.com/yourusername/rag-pipeline/actions)
[![Python 3.11](https://img.shields.io/badge/python-3.11-blue.svg)](https://www.python.org/downloads/)

A production-ready Retrieval-Augmented Generation (RAG) system with conversation memory, built with LangChain, FastAPI, and modern MLOps practices.

## 🌟 Features

### Core Features
- **Multi-Provider LLM Support**: OpenAI, Google Gemini, Groq, and Ollama
- **Advanced Retrieval**: Hybrid search (vector + BM25) with reranking
- **Conversation Memory**: SQLite-based persistent conversation history
- **Document Processing**: Support for TXT and PDF files
- **Semantic & Recursive Chunking**: Flexible text splitting strategies
- **RESTful API**: FastAPI with automatic OpenAPI documentation

### Production Features
- ✅ **Comprehensive Logging**: Structured logging with rotation
- ✅ **Error Handling**: Graceful error handling and recovery
- ✅ **Health Checks**: Kubernetes-ready health endpoints
- ✅ **Rate Limiting**: API rate limiting to prevent abuse
- ✅ **CORS Support**: Configurable CORS for web applications
- ✅ **Docker Support**: Multi-stage builds with health checks
- ✅ **CI/CD Pipeline**: Automated testing, security scanning, and deployment
- ✅ **Configuration Management**: Environment-based configuration

## 🏗️ Architecture

```
┌─────────────────┐
│   User Query    │
└────────┬────────┘
         │
         v
┌─────────────────┐
│  FastAPI Server │
│  - Rate Limit   │
│  - CORS         │
│  - Logging      │
└────────┬────────┘
         │
         v
┌─────────────────────────────┐
│   Conversation Memory       │
│   (SQLite)                  │
│   - Session Management      │
│   - History Retrieval       │
└─────────────┬───────────────┘
              │
              v
┌─────────────────────────────┐
│   Document Retrieval        │
│   ┌──────────────────────┐  │
│   │ Vector Search (Chroma)│  │
│   │ + BM25 (Keyword)      │  │
│   └──────────────────────┘  │
│   ┌──────────────────────┐  │
│   │ Ensemble + Reranking │  │
│   └──────────────────────┘  │
└─────────────┬───────────────┘
              │
              v
┌─────────────────────────────┐
│   LLM (Gemini/OpenAI/etc)   │
│   - Context + History       │
│   - Answer Generation       │
└─────────────┬───────────────┘
              │
              v
┌─────────────────────────────┐
│   Response + Memory Update  │
└─────────────────────────────┘
```

## 📦 Installation

### Prerequisites
- Python 3.11+
- Docker & Docker Compose (optional)
- API keys for your chosen LLM provider

### Local Setup

1. **Clone the repository**
```bash
git clone https://github.com/yourusername/rag-pipeline.git
cd rag-pipeline
```

2. **Create virtual environment**
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. **Install dependencies**
```bash
pip install -r requirements.txt
```

4. **Configure environment variables**
```bash
cp .env.example .env
# Edit .env with your API keys and configuration
```

5. **Create necessary directories**
```bash
mkdir -p data logs vector_store
```

### Docker Setup

```bash
# Build and run with Docker Compose
docker-compose up -d

# Check logs
docker-compose logs -f

# Stop
docker-compose down
```

## 🚀 Usage

### Starting the Server

**Local:**
```bash
python app.py
```

**Docker:**
```bash
docker-compose up
```

The API will be available at `http://localhost:8000`

### API Endpoints

#### 1. Health Check
```bash
curl http://localhost:8000/health
```

#### 2. Create a Session
```bash
curl -X POST http://localhost:8000/session/create
```

Response:
```json
{
  "session_id": "550e8400-e29b-41d4-a716-446655440000",
  "message": "Session created successfully"
}
```

#### 3. Query the RAG System
```bash
curl -X POST http://localhost:8000/query \
  -H "Content-Type: application/json" \
  -d '{
    "query": "What is the main objective?",
    "session_id": "550e8400-e29b-41d4-a716-446655440000"
  }'
```

Response:
```json
{
  "session_id": "550e8400-e29b-41d4-a716-446655440000",
  "input_query": "What is the main objective?",
  "answer": "The main objective is...",
  "context": [...],
  "message_count": 2
}
```

#### 4. Add Documents
```bash
curl -X POST http://localhost:8000/add-documents \
  -F "files=@document1.pdf" \
  -F "files=@document2.txt"
```

#### 5. Clear Session
```bash
curl -X DELETE http://localhost:8000/session/{session_id}
```

#### 6. Get Session Info
```bash
curl http://localhost:8000/session/{session_id}/info
```

#### 7. List All Sessions
```bash
curl http://localhost:8000/sessions
```

### Interactive API Documentation

Visit `http://localhost:8000/docs` for Swagger UI documentation.

## 🧪 Testing

### Run All Tests
```bash
pytest -v
```

### Run with Coverage
```bash
pytest --cov=src --cov-report=html
```

### Run Specific Test
```bash
pytest tests/test_integration.py::test_memory_system -v
```

### Run Integration Test Script
```bash
python tests/test.py
```

## ⚙️ Configuration

All configuration is managed through environment variables. See `.env.example` for all available options.

### Key Configuration Options

| Variable | Default | Description |
|----------|---------|-------------|
| `LLM_PROVIDER` | `gemini` | LLM provider (openai/gemini/groq/ollama) |
| `LLM_MODEL` | `gemini-2.5-flash` | Model name |
| `MAX_MEMORY_MESSAGES` | `10` | Max messages per session |
| `RETRIEVAL_TOP_K` | `5` | Documents to retrieve |
| `RERANKER_TOP_N` | `3` | Documents after reranking |
| `SPLITTING_METHOD` | `semantic` | Text splitting method |
| `LOG_LEVEL` | `INFO` | Logging level |

## 📊 MLOps & CI/CD

### CI/CD Pipeline

The project includes a comprehensive GitHub Actions pipeline:

1. **Code Quality**: Black, isort, flake8, pylint
2. **Security Scanning**: Bandit, Safety
3. **Testing**: pytest with coverage
4. **Docker Build**: Multi-stage builds
5. **Integration Tests**: Docker-based E2E tests
6. **Deployment**: Automated deployment on main branch

### Monitoring

- **Health Checks**: `/health` endpoint with detailed status
- **Logging**: Structured logging with rotation
- **Metrics**: Request count, latency, error rates (via logs)

### Local Development Workflow

```bash
# 1. Create feature branch
git checkout -b feature/new-feature

# 2. Make changes and test
pytest -v

# 3. Check code quality
black .
isort .
flake8 .

# 4. Commit and push
git add .
git commit -m "Add new feature"
git push origin feature/new-feature

# 5. Create PR (CI will run automatically)
```

## 🗂️ Project Structure

```
rag-pipeline/
├── src/
│   ├── chains.py              # LLM and RAG chain logic
│   ├── data_processing.py     # Document loading and splitting
│   ├── retrieval.py           # Retrieval and reranking
│   └── memory.py              # Conversation memory system
├── tests/
│   ├── test_integration.py    # Integration tests
│   └── test.py                # Manual test script
├── .github/
│   └── workflows/
│       └── CI.yml             # CI/CD pipeline
├── app.py                     # FastAPI application
├── config.py                  # Configuration management
├── logging_config.py          # Logging setup
├── main.py                    # CLI entry point
├── requirements.txt           # Python dependencies
├── Dockerfile                 # Docker image
├── docker-compose.yml         # Docker Compose config
├── .env.example               # Environment template
├── .gitignore                 # Git ignore rules
└── README.md                  # This file
```

## 🔒 Security

- API keys stored in environment variables
- Rate limiting on all endpoints
- Input validation with Pydantic
- Security scanning in CI/CD pipeline
- No sensitive data in logs

## 🐛 Troubleshooting

### Common Issues

**1. Import errors**
```bash
export PYTHONPATH=.
```

**2. Memory database locked**
```bash
rm conversation_memory.db
```

**3. Vector store corruption**
```bash
rm -rf vector_store/
# Restart server to rebuild
```

**4. Docker health check failing**
```bash
docker-compose logs rag-api
# Check for initialization errors
```

## 📚 Resources

- [LangChain Documentation](https://python.langchain.com/)
- [FastAPI Documentation](https://fastapi.tiangolo.com/)
- [Chroma Documentation](https://docs.trychroma.com/)
- [Sentence Transformers](https://www.sbert.net/)

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Run tests and code quality checks
5. Submit a pull request


## 🙏 Acknowledgments

- LangChain for the RAG framework
- Hugging Face for embedding models
- FastAPI for the web framework
- The open-source community

---

**Made with ❤️ for production-ready RAG systems**