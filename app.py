import os
import tempfile
import uvicorn
from fastapi import FastAPI, HTTPException, UploadFile, File
from pydantic import BaseModel, Field
from typing import List, Dict, Any

import config
from dotenv import load_dotenv

from langchain_huggingface import HuggingFaceEmbeddings
from langchain_core.runnables import Runnable

from src.data_processing import load_documents_from_directory
from src.data_processing import split_text, load_document
from src.retrieval import (
    create_or_load_chroma_retriever,
    create_bm25_retriever,
    create_ensemble_retriever,
    create_compression_retriever,
)
from src.chains import initialize_llm, create_rag_chain


app = FastAPI(
    title="RAG Pipeline API",
    description="An API for question-answering using a RAG pipeline.",
    version="1.0.0"
)


class QueryRequest(BaseModel):
    query: str = Field(..., description="The question to ask the RAG pipeline.", example="What is the main goal of Project Starlight?")


class DocumentMetadata(BaseModel):
    source: str

class RetrievedDocument(BaseModel):
    page_content: str
    metadata: Dict[str, Any]

class QueryResponse(BaseModel):
    input_query: str
    answer: str
    context: List[RetrievedDocument]


rag_chain: Runnable = None
embedding_model = None
final_retriever = None 



@app.on_event("startup")
def startup_event():
    """
    Initializes the RAG pipeline when the FastAPI application starts.
    This includes loading models, processing initial data, and building the retriever.
    """
    
    global rag_chain, embedding_model, final_retriever
    load_dotenv()

    print("--- Server Starting: Initializing RAG Pipeline ---")

    print("Step 1/5: Initializing models...")
    
    embedding_model = HuggingFaceEmbeddings(model_name=config.EMBEDDING_MODEL)
    
    llm = initialize_llm(
        provider=config.LLM_PROVIDER,
        model_name=config.LLM_MODEL,
        temperature=0
    )

    print(f"Step 2/5: Loading and processing initial documents from '{config.DATA_DIR}'...")
    
    documents = []
    
    if os.path.exists(config.DATA_DIR) and os.listdir(config.DATA_DIR):
        documents = load_documents_from_directory(config.DATA_DIR)
    
    if not documents:
        print("Warning: No initial documents found. The knowledge base is empty.")
        chunks = []
        
    else:
        chunks = split_text(documents, method="semantic", embeddings=embedding_model)


    print("Step 3/5: Building the retrieval system...")
    
    vector_retriever = create_or_load_chroma_retriever(
        str(config.VECTOR_STORE_DIR),
        config.COLLECTION_NAME,
        embedding_model,
        documents=chunks
    )
    
    keyword_retriever = create_bm25_retriever(chunks)
    
    ensemble_retriever = create_ensemble_retriever([vector_retriever, keyword_retriever])
    
    final_retriever = create_compression_retriever(
        ensemble_retriever,
        embedding_model,
        reranker_model_name=config.RERANKER_MODEL
    )

    print("Step 4/5: Creating the RAG chain...")
    
    rag_chain = create_rag_chain(final_retriever, llm)

    print("Step 5/5: Pipeline ready. The server is now accepting requests.")



@app.get("/", summary="Check API Status")

def read_root():
    """A simple endpoint to check if the API is running."""
    
    return {"status": "online", "message": "RAG API is running."}


@app.post("/add-documents", summary="Add New Documents to the Knowledge Base")
def add_documents(files: List[UploadFile] = File(...)):
    """
    Upload one or more documents (.txt, .pdf). The documents will be processed
    and added to the retriever's knowledge base.
    """
    
    if not final_retriever:
        raise HTTPException(status_code=503, detail="Retriever is not initialized.")

    saved_files = []
    try:
        
        for file in files:
            
            with tempfile.NamedTemporaryFile(delete=False, suffix=file.filename) as tmp:
                tmp.write(file.file.read())
                saved_files.append(tmp.name)

        new_docs = []
        for file_path in saved_files:
            new_docs.extend(load_document(file_path))

        if not new_docs:
            return {"message": "No new documents were processed."}
        
        new_chunks = split_text(new_docs, method="semantic", embeddings=embedding_model)

        vector_retriever = final_retriever.base_retriever.retrievers[0]
        vector_retriever.vectorstore.add_documents(new_chunks)
        print(f"Added {len(new_chunks)} new chunks to the vector store.")

        return {"message": f"Successfully added {len(files)} new document(s) to the knowledge base."}

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"An error occurred: {str(e)}")
    
    finally:
        for file_path in saved_files:
            os.remove(file_path)


@app.post("/query", response_model=QueryResponse, summary="Query the RAG Pipeline")
def process_query(request: QueryRequest):
    """
    Accepts a question, uses the RAG pipeline to find an answer based on the
    loaded documents, and returns the response.
    """
    
    if rag_chain is None:
        raise HTTPException(status_code=503, detail="RAG chain is not ready. Please wait for the server to initialize.")

    try:
        
        print(f"Received query: '{request.query}'")
        response_dict = rag_chain.invoke({"input": request.query})

        retrieved_docs = []
        for doc in response_dict["context"]:
            retrieved_docs.append(
                RetrievedDocument(
                    page_content=doc.page_content,
                    metadata=doc.metadata
                )
            )
            
        return QueryResponse(
            input_query=response_dict["input"],
            answer=response_dict["answer"],
            context=retrieved_docs
        )
        
    except Exception as e:
        print(f"An error occurred while processing the query: {e}")
        raise HTTPException(status_code=500, detail=str(e))



# To run the app, use the command: uvicorn main:app --reload
if __name__ == "__main__":
     uvicorn.run(app, host="0.0.0.0", port=8000)