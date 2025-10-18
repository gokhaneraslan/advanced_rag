import os
import tempfile
import uvicorn
import logging
from contextlib import asynccontextmanager
from fastapi import FastAPI, HTTPException, UploadFile, File, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field
from typing import List, Dict, Any, Optional
from slowapi import Limiter, _rate_limit_exceeded_handler
from slowapi.util import get_remote_address
from slowapi.errors import RateLimitExceeded
import config
from dotenv import load_dotenv
from logging_config import setup_logging

from langchain_huggingface import HuggingFaceEmbeddings
from langchain_core.runnables import Runnable

from src.data_processing import load_documents_from_directory, split_text, load_document
from src.retrieval import (
    create_or_load_chroma_retriever,
    create_bm25_retriever,
    create_ensemble_retriever,
    create_compression_retriever,
)
from src.chains import initialize_llm, create_rag_chain, format_chat_history
from src.memory import ConversationMemory

logger = logging.getLogger(__name__)

# Global state
rag_chain: Runnable = None
embedding_model = None
final_retriever = None
memory_system: ConversationMemory = None
llm = None

# Rate limiter
limiter = Limiter(key_func=get_remote_address)


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan manager for startup and shutdown."""
    # Startup
    logger.info("=" * 60)
    logger.info("RAG API Server Starting...")
    logger.info("=" * 60)
    
    try:
        await startup_event()
        logger.info("✅ Server startup completed successfully")
    except Exception as e:
        logger.critical(f"❌ Server startup failed: {e}")
        raise
    
    yield
    
    # Shutdown
    logger.info("=" * 60)
    logger.info("RAG API Server Shutting Down...")
    logger.info("=" * 60)
    await shutdown_event()
    logger.info("✅ Server shutdown completed")


app = FastAPI(
    title="RAG Pipeline API",
    description="An API for question-answering using a RAG pipeline with conversation memory.",
    version="2.0.0",
    lifespan=lifespan
)

# Rate limiting
app.state.limiter = limiter
app.add_exception_handler(RateLimitExceeded, _rate_limit_exceeded_handler)

# CORS middleware
if config.ENABLE_CORS:
    app.add_middleware(
        CORSMiddleware,
        allow_origins=config.CORS_ORIGINS,
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )
    logger.info(f"CORS enabled for origins: {config.CORS_ORIGINS}")


# Pydantic models
class QueryRequest(BaseModel):
    query: str = Field(..., description="The question to ask the RAG pipeline.", example="What is the main goal of Project Starlight?")
    session_id: Optional[str] = Field(None, description="Session ID for conversation continuity. If not provided, a new session will be created.")


class SessionCreateResponse(BaseModel):
    session_id: str
    message: str


class RetrievedDocument(BaseModel):
    page_content: str
    metadata: Dict[str, Any]


class QueryResponse(BaseModel):
    session_id: str
    input_query: str
    answer: str
    context: List[RetrievedDocument]
    message_count: int


class HealthResponse(BaseModel):
    status: str
    version: str
    config: Dict[str, Any]
    memory_sessions: int


async def startup_event():
    """Initialize all components on startup."""
    global rag_chain, embedding_model, final_retriever, memory_system, llm
    
    # Load environment variables
    load_dotenv()
    
    # Setup logging
    setup_logging(
        log_level=config.LOG_LEVEL,
        log_file=config.LOG_FILE,
        log_format=config.LOG_FORMAT
    )
    
    # Validate configuration
    logger.info("Step 1/6: Validating configuration...")
    try:
        config.validate_config()
    except ValueError as e:
        logger.error(f"Configuration validation failed: {e}")
        raise
    
    # Initialize memory system
    logger.info("Step 2/6: Initializing conversation memory system...")
    try:
        memory_system = ConversationMemory(
            db_path=str(config.MEMORY_DB_PATH),
            max_messages=config.MAX_MEMORY_MESSAGES
        )
        logger.info("✅ Memory system initialized")
    except Exception as e:
        logger.error(f"Failed to initialize memory system: {e}")
        raise
    
    # Initialize models
    logger.info("Step 3/6: Initializing embedding and LLM models...")
    try:
        embedding_model = HuggingFaceEmbeddings(
            model_name=config.EMBEDDING_MODEL,
            model_kwargs={'device': 'cpu'},  # Use 'cuda' if GPU available
            encode_kwargs={'normalize_embeddings': True}
        )
        logger.info(f"✅ Embedding model loaded: {config.EMBEDDING_MODEL}")
        
        llm = initialize_llm(
            provider=config.LLM_PROVIDER,
            model_name=config.LLM_MODEL,
            temperature=config.LLM_TEMPERATURE
        )
        logger.info(f"✅ LLM initialized: {config.LLM_PROVIDER}/{config.LLM_MODEL}")
    except Exception as e:
        logger.error(f"Failed to initialize models: {e}")
        raise
    
    # Load and process documents
    logger.info(f"Step 4/6: Loading documents from '{config.DATA_DIR}'...")
    documents = []
    chunks = []
    
    try:
        if os.path.exists(config.DATA_DIR) and os.listdir(config.DATA_DIR):
            documents = load_documents_from_directory(config.DATA_DIR)
            logger.info(f"✅ Loaded {len(documents)} documents")
            
            if documents:
                chunks = split_text(
                    documents, 
                    method=config.SPLITTING_METHOD, 
                    embeddings=embedding_model if config.SPLITTING_METHOD == "semantic" else None,
                    chunk_size=config.CHUNK_SIZE,
                    chunk_overlap=config.CHUNK_OVERLAP
                )
                logger.info(f"✅ Split into {len(chunks)} chunks")
        else:
            logger.warning("⚠️  No initial documents found. Knowledge base is empty.")
    except Exception as e:
        logger.error(f"Failed to load documents: {e}")
        raise
    
    # Build retrieval system
    logger.info("Step 5/6: Building retrieval system...")
    try:
        vector_retriever = create_or_load_chroma_retriever(
            str(config.VECTOR_STORE_DIR),
            config.COLLECTION_NAME,
            embedding_model,
            documents=chunks if chunks else None,
            search_type="mmr",
            search_kwargs={"k": config.RETRIEVAL_TOP_K}
        )
        
        keyword_retriever = create_bm25_retriever(chunks if chunks else [], k=config.RETRIEVAL_TOP_K)
        
        ensemble_retriever = create_ensemble_retriever([vector_retriever, keyword_retriever])
        
        final_retriever = create_compression_retriever(
            ensemble_retriever,
            embedding_model,
            reranker_model_name=config.RERANKER_MODEL,
            top_n=config.RERANKER_TOP_N,
            similarity_threshold=config.SIMILARITY_THRESHOLD
        )
        logger.info("✅ Retrieval system built successfully")
    except Exception as e:
        logger.error(f"Failed to build retrieval system: {e}")
        raise
    
    # Create RAG chain
    logger.info("Step 6/6: Creating RAG chain...")
    try:
        rag_chain = create_rag_chain(final_retriever, llm, include_chat_history=True)
        logger.info("✅ RAG chain created successfully")
    except Exception as e:
        logger.error(f"Failed to create RAG chain: {e}")
        raise
    
    logger.info("🚀 RAG API is ready to accept requests!")


async def shutdown_event():
    """Cleanup on shutdown."""
    global memory_system
    
    logger.info("Performing cleanup...")
    
    # Optional: Clean up old sessions on shutdown
    if memory_system:
        try:
            logger.info(f"Cleaning up sessions older than {config.MEMORY_CLEANUP_DAYS} days...")
            memory_system.delete_old_sessions(config.MEMORY_CLEANUP_DAYS)
        except Exception as e:
            logger.error(f"Failed to cleanup old sessions: {e}")
    
    logger.info("Cleanup completed")


# Routes

@app.get("/", summary="API Root")
def read_root():
    """Check if the API is running."""
    return {
        "status": "online",
        "message": "RAG API with Conversation Memory is running.",
        "version": "2.0.0"
    }


@app.get("/health", response_model=HealthResponse, summary="Health Check")
def health_check():
    """
    Comprehensive health check endpoint.
    Returns server status, configuration, and memory statistics.
    """
    if not rag_chain or not memory_system:
        raise HTTPException(status_code=503, detail="Server is not fully initialized")
    
    try:
        sessions = memory_system.get_all_sessions()
        
        return HealthResponse(
            status="healthy",
            version="2.0.0",
            config=config.get_config_summary(),
            memory_sessions=len(sessions)
        )
    except Exception as e:
        logger.error(f"Health check failed: {e}")
        raise HTTPException(status_code=500, detail=f"Health check failed: {str(e)}")


@app.post("/session/create", response_model=SessionCreateResponse, summary="Create New Session")
@limiter.limit(config.API_RATE_LIMIT)
def create_session(request: Request = None):
    """
    Create a new conversation session.
    Returns a unique session ID that should be used for subsequent queries.
    """
    if not memory_system:
        raise HTTPException(status_code=503, detail="Memory system not initialized")
    
    try:
        session_id = memory_system.create_session()
        logger.info(f"Created new session: {session_id}")
        
        return SessionCreateResponse(
            session_id=session_id,
            message="Session created successfully. Use this session_id for your queries."
        )
    except Exception as e:
        logger.error(f"Failed to create session: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to create session: {str(e)}")


@app.delete("/session/{session_id}", summary="Clear Session History")
@limiter.limit(config.API_RATE_LIMIT)
def clear_session(session_id: str, request: Request = None):
    """Clear all conversation history for a specific session."""
    if not memory_system:
        raise HTTPException(status_code=503, detail="Memory system not initialized")
    
    try:
        memory_system.clear_session(session_id)
        logger.info(f"Cleared session: {session_id}")
        
        return {"message": f"Session {session_id} cleared successfully"}
    except Exception as e:
        logger.error(f"Failed to clear session {session_id}: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to clear session: {str(e)}")


@app.get("/session/{session_id}/info", summary="Get Session Info")
def get_session_info(session_id: str):
    """Get information about a specific session."""
    if not memory_system:
        raise HTTPException(status_code=503, detail="Memory system not initialized")
    
    try:
        info = memory_system.get_session_info(session_id)
        return info
    except Exception as e:
        logger.error(f"Failed to get session info for {session_id}: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to get session info: {str(e)}")


@app.get("/sessions", summary="List All Sessions")
def list_sessions():
    """Get a list of all active session IDs."""
    if not memory_system:
        raise HTTPException(status_code=503, detail="Memory system not initialized")
    
    try:
        sessions = memory_system.get_all_sessions()
        return {"sessions": sessions, "count": len(sessions)}
    except Exception as e:
        logger.error(f"Failed to list sessions: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to list sessions: {str(e)}")


@app.post("/add-documents", summary="Add New Documents to Knowledge Base")
@limiter.limit("20/minute")
def add_documents(files: List[UploadFile] = File(...), request: Request = None):
    """
    Upload one or more documents (.txt, .pdf) to add to the knowledge base.
    Documents will be processed and indexed automatically.
    """
    if not final_retriever or not embedding_model:
        raise HTTPException(status_code=503, detail="Retriever is not initialized")

    saved_files = []
    try:
        logger.info(f"Received {len(files)} files for upload")
        
        for file in files:
            if not file.filename.endswith(('.txt', '.pdf')):
                raise HTTPException(
                    status_code=400, 
                    detail=f"Unsupported file type: {file.filename}. Only .txt and .pdf are supported."
                )
            
            with tempfile.NamedTemporaryFile(delete=False, suffix=file.filename) as tmp:
                tmp.write(file.file.read())
                saved_files.append(tmp.name)
                logger.debug(f"Saved temporary file: {tmp.name}")

        new_docs = []
        for file_path in saved_files:
            docs = load_document(file_path)
            new_docs.extend(docs)
            logger.debug(f"Loaded {len(docs)} documents from {file_path}")

        if not new_docs:
            return {"message": "No new documents were processed"}
        
        new_chunks = split_text(
            new_docs, 
            method=config.SPLITTING_METHOD, 
            embeddings=embedding_model if config.SPLITTING_METHOD == "semantic" else None,
            chunk_size=config.CHUNK_SIZE,
            chunk_overlap=config.CHUNK_OVERLAP
        )
        logger.info(f"Split new documents into {len(new_chunks)} chunks")

        vector_retriever = final_retriever.base_retriever.retrievers[0]
        vector_retriever.vectorstore.add_documents(new_chunks)
        logger.info(f"✅ Added {len(new_chunks)} chunks to vector store")

        return {
            "message": f"Successfully added {len(files)} document(s) to the knowledge base",
            "files_processed": len(files),
            "chunks_created": len(new_chunks)
        }

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to add documents: {e}")
        raise HTTPException(status_code=500, detail=f"An error occurred: {str(e)}")
    
    finally:
        for file_path in saved_files:
            try:
                os.remove(file_path)
                logger.debug(f"Cleaned up temporary file: {file_path}")
            except Exception as e:
                logger.warning(f"Failed to remove temp file {file_path}: {e}")


@app.post("/query", response_model=QueryResponse, summary="Query the RAG Pipeline")
@limiter.limit(config.API_RATE_LIMIT)
def process_query(request_data: QueryRequest, request: Request = None):
    """
    Submit a question to the RAG pipeline with conversation memory support.
    
    - If session_id is provided, the conversation history will be used for context.
    - If session_id is not provided, a new session will be created automatically.
    """
    if not rag_chain or not memory_system:
        raise HTTPException(
            status_code=503, 
            detail="RAG chain or memory system not ready. Please wait for initialization."
        )

    try:
        # Create or use existing session
        session_id = request_data.session_id
        if not session_id:
            session_id = memory_system.create_session()
            logger.info(f"Created new session for query: {session_id}")
        
        logger.info(f"Processing query for session {session_id}: '{request_data.query[:100]}'")
        
        # Get conversation history
        history = memory_system.get_history(session_id)
        chat_history = format_chat_history(history)
        
        # Invoke RAG chain
        response_dict = rag_chain.invoke({
            "input": request_data.query,
            "chat_history": chat_history
        })
        
        # Store user query and assistant response in memory
        memory_system.add_message(session_id, "user", request_data.query)
        memory_system.add_message(
            session_id, 
            "assistant", 
            response_dict["answer"],
            metadata={"context_docs": len(response_dict["context"])}
        )
        
        # Format response
        retrieved_docs = [
            RetrievedDocument(
                page_content=doc.page_content,
                metadata=doc.metadata
            )
            for doc in response_dict["context"]
        ]
        
        current_history = memory_system.get_history(session_id)
        
        logger.info(f"✅ Query processed successfully for session {session_id}")
        
        return QueryResponse(
            session_id=session_id,
            input_query=response_dict["input"],
            answer=response_dict["answer"],
            context=retrieved_docs,
            message_count=len(current_history)
        )
        
    except Exception as e:
        logger.error(f"Error processing query: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Query processing failed: {str(e)}")


# Import Request for rate limiting
from starlette.requests import Request

if __name__ == "__main__":
    uvicorn.run(
        app, 
        host=config.API_HOST, 
        port=config.API_PORT,
        log_config=None  # We use our own logging configuration
    )
    