import os
import shutil
from dotenv import load_dotenv

from reportlab.pdfgen import canvas
from reportlab.lib.pagesizes import letter

from langchain_huggingface import HuggingFaceEmbeddings

from src.chains import initialize_llm, create_rag_chain
from src.data_processing import load_documents_from_directory, split_text
from src.retrieval import (
    create_or_load_chroma_retriever, 
    create_bm25_retriever, 
    create_ensemble_retriever,
    create_compression_retriever
)


DATA_DIR = "./data"
DB_DIR = "./chroma_db_test"
COLLECTION_NAME = "integration_test_collection"

EMBEDDING_MODEL_NAME = "BAAI/bge-large-en-v1.5"
RERANKER_MODEL_NAME = "BAAI/bge-reranker-large"
LLM_PROVIDER = "gemini"  # Options: "openai", "gemini", "groq", "ollama"
simple_query = "Hello! Is there anybody there?"

def setup_environment():
    
    """Creates dummy data files and directories for testing."""
    
    print("--- 1. Setting up test environment ---")
    
    if os.path.exists(DATA_DIR):
        shutil.rmtree(DATA_DIR)
        
    os.makedirs(DATA_DIR)
    
    if os.path.exists(DB_DIR):
        shutil.rmtree(DB_DIR)
    

    txt_content = (
        "Project Starlight was a success. The primary objective was to analyze "
        "stellar radiation from Alpha Centauri. The data revealed a new energy signature. "
        "The project was led by Dr. Evelyn Reed."
    )
    
    with open(os.path.join(DATA_DIR, "test.txt"), "w") as f:
        f.write(txt_content)
    

    pdf_path = os.path.join(DATA_DIR, "test.pdf")
    
    c = canvas.Canvas(pdf_path, pagesize=letter)
    
    c.drawString(72, 800, "Project Starlight: Final Report")
    c.drawString(72, 780, "The secondary objective was quantum entanglement communication.")
    c.drawString(72, 760, "Experiments confirmed that messages could be sent faster than light.")
    c.save()
    
    print("Test environment created with test.txt and test.pdf.\n")


def main():
    
    """Runs the full integration test for the RAG pipeline."""
    
    load_dotenv()
    setup_environment()
    embedding_model = None
    llm = None
    
    print("--- 2. Initializing Core Models ---")
    
    try:

        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        print(f"Using '{device}' for the embedding model...")

        embedding_model = HuggingFaceEmbeddings(
            model_name="BAAI/bge-large-en-v1.5",
            model_kwargs={'device': device},
            encode_kwargs={'normalize_embeddings': True}
        )

        print("Embedding model loaded successfully.")

    except Exception as e:
        print(f"An error occurred while loading the embedding model: {e}")
        print("Semantic splitting test will be skipped.")

    
    if os.getenv("OPENAI_API_KEY"):
        
        try:
            
            print("\n--- Testing OpenAI ---")
            llm = initialize_llm(
                "openai", 
                model_name="gpt-4o", 
                temperature=0
            )
            
            response = llm.invoke(simple_query)
            
            print("OpenAI Response:\n", response.content)
        
        except Exception as e:
            print(f"Error testing OpenAI: {e}")
            
    else:
        print("\nSkipping OpenAI test: OPENAI_API_KEY not set.")
    
    if os.getenv("GOOGLE_API_KEY"):
        
        try:
            
            print("\n--- Testing Google Gemini ---")
            
            llm = initialize_llm(
                "gemini", 
                model_name="gemini-2.5-flash", 
                temperature=0
            )
            
            response = llm.invoke(simple_query)
            
            print("Gemini Response:\n", response.content)
        
        except Exception as e:
            print(f"Error testing Gemini: {e}")
    
    else:
        print("\nSkipping Gemini test: GOOGLE_API_KEY not set.")
        
    if os.getenv("GROQ_API_KEY"):
        
        try:
            
            print("\n--- Testing Groq ---")
            
            llm = initialize_llm(
                "groq", 
                temperature=0
            )
            
            response = llm.invoke(simple_query)
            print("Groq Response:\n", response.content)
        
        except Exception as e:
            print(f"Error testing Groq: {e}")
            
    else:
        print("\nSkipping Groq test: GROQ_API_KEY not set.")


    print("--- 3. Testing Directory Document Loading ---")
    
    all_documents = load_documents_from_directory(DATA_DIR)
    
    assert len(all_documents) >= 2, "Failed to load all documents from directory."
    
    print("Document loading successful.\n")
    
    print("--- 4. Testing Text Splitting ---")

    print("--- Recursive Method Test ---")
    recursive_chunks = split_text(
        all_documents,
        method="recursive",
        chunk_size=1000,
        chunk_overlap=100
    )

    assert len(recursive_chunks) > 0, "Recursive Text splitting produced no chunks."
    print("Recursive Text splitting successful.\n")

    print("\n--- Semantic Method Test ---")

    semantic_chunks = split_text(
        all_documents,
        method="semantic",
        embeddings=embedding_model
    )

    assert len(semantic_chunks) > 0, "Semantic Text splitting produced no chunks."
    print("Semantic Text splitting successful.\n")

    print("--- 5. Testing Retriever Creation ---")
    
    vector_retriever = create_or_load_chroma_retriever(
        DB_DIR, 
        COLLECTION_NAME, 
        embedding_model, 
        documents=semantic_chunks,
        search_type = "mmr",
        search_kwargs={"k": 5}
    )
    
    keyword_retriever = create_bm25_retriever(
        semantic_chunks,
        k=5
    )
    
    ensemble_retriever = create_ensemble_retriever(
        [vector_retriever, keyword_retriever]
    )
    
    print("Base retrievers created successfully.\n")

    print("--- 6. Testing Compression/Reranking Retriever ---")
    
    final_retriever = create_compression_retriever(
        ensemble_retriever, 
        embedding_model, 
        reranker_model_name=RERANKER_MODEL_NAME, 
        top_n=3
    )
    
    print("Final compression retriever created successfully.\n")

    print("--- 7. Testing RAG Chain Creation ---")
    
    rag_chain = create_rag_chain(final_retriever, llm)
    
    print("RAG Chain created successfully.\n")

    print("--- 8. Performing End-to-End RAG Test ---")
    
    test_question = """What was the primary objective of Project Starlight 
        and what did its communication experiments confirm?"""
        
    print(f"QUERY: {test_question}\n")
    
    response = rag_chain.invoke({"input": test_question})
    
    print("--- Retrieved Context ---")
    
    for i, doc in enumerate(response["context"]):
        docx = doc.page_content[:150].replace('\n', ' ')
        print(f"Doc {i+1}: {docx}...")
        
    print("-" * 25)
    
    print("\n--- Final LLM Answer ---")
    
    print(response["answer"])
    
    print("-" * 25)
    
    print("\n\n✅✅✅ Integration test completed successfully! ✅✅✅")


if __name__ == "__main__":
    import torch
    main()
    