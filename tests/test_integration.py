import os
import shutil
import pytest
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


DATA_DIR = "./tests/data"
DB_DIR = "./tests/chroma_db_test"
COLLECTION_NAME = "integration_test_collection"
EMBEDDING_MODEL_NAME = "BAAI/bge-base-en-v1.5" 
RERANKER_MODEL_NAME = "BAAI/bge-reranker-base"


@pytest.fixture(scope="module")
def setup_test_environment():
    """
        Pytest fixture: Creates fake data files and folders before tests begin, 
        and cleans them all up after tests are finished.
    """
    
    print("--- Setting up test environment ---")
    
    os.makedirs(DATA_DIR, exist_ok=True)
    
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
    
    print("Test environment created.")
    
    yield
    
    print("\n--- Tearing down test environment ---")
    
    shutil.rmtree(DATA_DIR)
    
    if os.path.exists(DB_DIR):
        shutil.rmtree(DB_DIR)
        
    print("Test environment cleaned up.")



def test_full_rag_pipeline(setup_test_environment):
    """
        RAG tests the entire pijplijn from start to finish.
    """
    
    load_dotenv()
    
    print("--- Initializing models ---")

    assert os.getenv("GOOGLE_API_KEY") is not None, "GOOGLE_API_KEY secret is not set!"
    
    llm = initialize_llm(
        provider="gemini", 
        model_name="gemini-2.5-flash", 
        temperature=0
    )
    
    embedding_model = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL_NAME)


    print("--- Loading and splitting documents ---")
    documents = load_documents_from_directory(DATA_DIR)
    assert len(documents) >= 2
    
    chunks = split_text(documents, method="semantic", embeddings=embedding_model)
    assert len(chunks) > 0


    print("--- Building retrievers ---")
    vector_retriever = create_or_load_chroma_retriever(
        DB_DIR, 
        COLLECTION_NAME, 
        embedding_model, 
        documents=chunks
    )
    
    keyword_retriever = create_bm25_retriever(chunks)
    
    ensemble_retriever = create_ensemble_retriever([vector_retriever, keyword_retriever])
    
    final_retriever = create_compression_retriever(
        ensemble_retriever,
        embedding_model, 
        reranker_model_name=RERANKER_MODEL_NAME
    )

    print("--- Creating RAG chain ---")
    rag_chain = create_rag_chain(final_retriever, llm)
    
    print("--- Performing end-to-end RAG query ---")
    test_question = "What was the primary objective of Project Starlight and what did its communication experiments confirm?"
    
    response = rag_chain.invoke({"input": test_question})
    
    answer = response["answer"].lower()
    print(f"Final Answer: {answer}")
    
    assert "starlight" in answer
    assert "primary objective" in answer or "alpha centauri" in answer
    assert "communication" in answer or "faster than light" in answer
    
    print("\n✅✅✅ Integration test completed successfully! ✅✅✅")