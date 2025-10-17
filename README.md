# Advanced RAG Pipeline API

[![CI Pipeline](https://github.com/gokhaneraslan/advanced_rag/actions/workflows/CI.yml/badge.svg)](https://github.com/gokhaneraslan/advanced_rag/actions/workflows/CI.yml)
[![Python Version](https://img.shields.io/badge/python-3.11-blue.svg)](https://www.python.org/downloads/release/python-3110/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

This project provides a high-performance, containerized **Retrieval-Augmented Generation (RAG)** API built with FastAPI and Docker. It allows you to perform question-answering on your own documents (PDFs, TXT files) with state-of-the-art techniques that go beyond standard RAG implementations to maximize retrieval quality and accuracy.

## 🚀 Advanced Features

This RAG system offers more than just a simple vector search. It incorporates a sophisticated pipeline to ensure the most relevant context is provided to the Large Language Model (LLM).

1.  **Hybrid Search**:
    *   Combines the strengths of both keyword-based **BM25 Retriever** and semantic **Vector Retriever** (using ChromaDB). This dual-approach ensures that both exact keyword matches and contextually similar passages are found.

2.  **Reciprocal Rank Fusion (RRF)**:
    *   The results from the two different retrievers are intelligently merged using the RRF algorithm. This produces a single, more robust ranked list that prioritizes documents highly ranked by both search methods.

3.  **Cross-Encoder Reranking**:
    *   The fused list of documents is then passed to a powerful **Cross-Encoder model** (e.g., `BAAI/bge-reranker-large`). This model re-evaluates the relevance of each document against the original query, significantly improving the quality of the final context sent to the LLM.

4.  **Redundancy Filtering**:
    *   To provide a more diverse and information-dense context, semantically similar and redundant documents from the reranked list are filtered out.

5.  **Long Context Reordering**:
    *   This technique combats the "lost in the middle" problem where LLMs tend to ignore information placed in the middle of a long context. The pipeline reorders the documents to place the most relevant ones at the beginning and end of the context window.

6.  **API-First & Dockerized**:
    *   The entire system is exposed as a modern REST API using `FastAPI` and is containerized with `Docker Compose` for easy, one-command setup and deployment.

7.  **Continuous Integration (CI/CD)**:
    *   A CI pipeline using `GitHub Actions` automatically runs a full suite of integration tests on every push and pull request, ensuring code quality and reliability.

## 🏛️ Architecture Flow

```
User Query
     |
     v
FastAPI API Endpoint (/query)
     |
     v
[Ensemble Retriever] -> Hybrid Search (BM25 + Vector) -> Merge with RRF
     |
     v
[Compression Pipeline] -> Filter Redundancy -> Rerank with Cross-Encoder -> Reorder
     |
     v
Optimized & Relevant Context
     |
     v
Large Language Model (LLM - Gemini, OpenAI, etc.)
     |
     v
Generated Answer
     |
     v
API Response
```

## 🛠️ Setup and Installation

Docker and Docker Compose are required to run this project locally.

**1. Clone the Repository:**
```bash
git clone https://github.com/gokhaneraslan/advanced_rag.git
cd advanced_rag
```

**2. Configure Environment Variables:**
Create a `.env` file in the project root. You can use the provided template:
```env
# .env.example

# API Keys for LLM Providers (fill in at least one)
GOOGLE_API_KEY="AIzaSy...your_google_api_key"
OPENAI_API_KEY="sk-...your_openai_api_key"
GROQ_API_KEY="gsk_...your_groq_api_key"
```

**3. Add Your Documents:**
Place the `.pdf` and `.txt` files you want to query into the `data/` directory.

**4. Launch with Docker Compose:**
This command will build the Docker image, install dependencies, and start the API service.
```bash
docker-compose up --build -d
```
The server will be running at `http://127.0.0.1:8000`.

## 🚀 API Usage

You can access the interactive API documentation (Swagger UI) at `http://127.0.0.1:8000/docs`.

### 1. Add New Documents (`/add-documents`)

Use this endpoint to upload new documents to the knowledge base.

**`curl` Example:**
```bash
curl -X 'POST' \
  'http://127.0.0.1:8000/add-documents' \
  -F 'files=@/path/to/your/document1.pdf' \
  -F 'files=@/path/to/your/document2.txt'
```

### 2. Ask a Question (`/query`)

Use this endpoint to ask questions about your documents.

**`curl` Example:**
```bash
curl -X 'POST' \
  'http://127.0.0.1:8000/query' \
  -H 'Content-Type: application/json' \
  -d '{
    "query": "What was the primary objective of Project Starlight?"
  }'
```

## 🧪 Testing

The project includes an integration test suite using `pytest`.

1.  **Install Test Dependencies:**
    ```bash
    pip install pytest reportlab
    ```
2.  **Run Tests:**
    (Execute from the project's root directory)
    ```bash
    pytest -v
    ```

## ⚙️ Configuration

You can easily configure the models and paths used in the pipeline by editing the `config.py` file:
*   `EMBEDDING_MODEL`
*   `RERANKER_MODEL`
*   `LLM_PROVIDER` and `LLM_MODEL`
*   `DATA_DIR` and `VECTOR_STORE_DIR`
