import os
import shutil
import pytest
from dotenv import load_dotenv

from reportlab.pdfgen import canvas
from reportlab.lib.pagesizes import letter

from langchain_huggingface import HuggingFaceEmbeddings
from src.chains import initialize_llm, create_rag_chain, format_chat_history
from src.data_processing import load_documents_from_directory, split_text
from src.memory import ConversationMemory

from src.retrieval import (
    create_or_load_chroma_retriever, 
    create_bm25_retriever, 
    create_ensemble_retriever,
    create_compression_retriever
)


DATA_DIR = "./tests/data"
DB_DIR = "./tests/chroma_db_test"
MEMORY_DB = "./tests/test_memory.db"
COLLECTION_NAME = "integration_test_collection"
EMBEDDING_MODEL_NAME = "BAAI/bge-base-en-v1.5" 
RERANKER_MODEL_NAME = "BAAI/bge-reranker-base"


@pytest.fixture(scope="module")
def setup_test_environment():
    """
    Pytest fixture: Creates fake data files and folders before tests begin, 
    and cleans them all up after tests are finished.
    """
    
    print("\n--- Setting up test environment ---")
    
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
    
    if os.path.exists(DATA_DIR):
        shutil.rmtree(DATA_DIR)
    
    if os.path.exists(DB_DIR):
        shutil.rmtree(DB_DIR)
    
    if os.path.exists(MEMORY_DB):
        os.remove(MEMORY_DB)
        
    print("Test environment cleaned up.")


def test_memory_system():
    """Test the conversation memory system."""
    print("\n--- Testing Memory System ---")
    
    memory = ConversationMemory(db_path=MEMORY_DB, max_messages=5)
    
    # Create session
    session_id = memory.create_session()
    assert session_id is not None
    print(f"✅ Session created: {session_id}")
    
    # Add messages
    memory.add_message(session_id, "user", "What is Project Starlight?")
    memory.add_message(session_id, "assistant", "It's a research project.")
    memory.add_message(session_id, "user", "Who led it?")
    memory.add_message(session_id, "assistant", "Dr. Evelyn Reed led it.")
    
    # Get history
    history = memory.get_history(session_id)
    assert len(history) == 4
    print(f"✅ History retrieved: {len(history)} messages")
    
    # Test max messages limit
    for i in range(10):
        memory.add_message(session_id, "user", f"Message {i}")
    
    history = memory.get_history(session_id)
    assert len(history) <= 5
    print(f"✅ Max message limit working: {len(history)} messages")
    
    # Test session info
    info = memory.get_session_info(session_id)
    assert "message_count" in info
    print(f"✅ Session info retrieved: {info}")
    
    # Test clear session
    memory.clear_session(session_id)
    history = memory.get_history(session_id)
    assert len(history) == 0
    print("✅ Session cleared successfully")


def test_full_rag_pipeline(setup_test_environment):
    """Test the entire RAG pipeline from start to finish."""
    
    load_dotenv()
    
    print("\n--- Testing Full RAG Pipeline ---")

    assert os.getenv("GOOGLE_API_KEY") is not None, "GOOGLE_API_KEY environment variable is not set!"
    
    print("Initializing models...")
    llm = initialize_llm(
        provider="gemini", 
        model_name="gemini-2.5-flash", 
        temperature=0
    )
    
    embedding_model = HuggingFaceEmbeddings(
        model_name=EMBEDDING_MODEL_NAME,
        model_kwargs={'device': 'cpu'},
        encode_kwargs={'normalize_embeddings': True}
    )
    print("✅ Models initialized")

    print("Loading and splitting documents...")
    documents = load_documents_from_directory(DATA_DIR)
    assert len(documents) >= 2
    
    chunks = split_text(documents, method="semantic", embeddings=embedding_model)
    assert len(chunks) > 0
    print(f"✅ Loaded {len(documents)} documents, split into {len(chunks)} chunks")

    print("Building retrievers...")
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
    print("✅ Retrievers built")

    print("Creating RAG chain...")
    rag_chain = create_rag_chain(final_retriever, llm, include_chat_history=True)
    print("✅ RAG chain created")
    
    print("Performing end-to-end RAG query...")
    test_question = "What was the primary objective of Project Starlight?"
    
    response = rag_chain.invoke({
        "input": test_question,
        "chat_history": []
    })
    
    answer = response["answer"].lower()
    print(f"Answer: {answer}")
    
    assert "starlight" in answer or "alpha centauri" in answer or "stellar radiation" in answer
    print("✅ Basic RAG query successful")


def test_rag_with_conversation_memory(setup_test_environment):
    """Test RAG pipeline with conversation memory."""
    
    load_dotenv()
    
    print("\n--- Testing RAG with Conversation Memory ---")

    # Initialize components
    llm = initialize_llm(provider="gemini", model_name="gemini-2.5-flash", temperature=0)
    embedding_model = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL_NAME)
    memory = ConversationMemory(db_path=MEMORY_DB, max_messages=10)
    
    # Load documents
    documents = load_documents_from_directory(DATA_DIR)
    chunks = split_text(documents, method="semantic", embeddings=embedding_model)
    
    # Build retrievers
    vector_retriever = create_or_load_chroma_retriever(
        DB_DIR, COLLECTION_NAME, embedding_model, documents=chunks
    )
    keyword_retriever = create_bm25_retriever(chunks)
    ensemble_retriever = create_ensemble_retriever([vector_retriever, keyword_retriever])
    final_retriever = create_compression_retriever(
        ensemble_retriever, embedding_model, reranker_model_name=RERANKER_MODEL_NAME
    )
    
    # Create RAG chain with history support
    rag_chain = create_rag_chain(final_retriever, llm, include_chat_history=True)
    
    # Create session
    session_id = memory.create_session()
    print(f"✅ Session created: {session_id}")
    
    # First query
    query1 = "What is Project Starlight?"
    history = memory.get_history(session_id)
    chat_history = format_chat_history(history)
    
    response1 = rag_chain.invoke({"input": query1, "chat_history": chat_history})
    
    memory.add_message(session_id, "user", query1)
    memory.add_message(session_id, "assistant", response1["answer"])
    
    print(f"Q1: {query1}")
    print(f"A1: {response1['answer'][:100]}...")
    
    # Follow-up query (testing context awareness)
    query2 = "Who led it?"
    history = memory.get_history(session_id)
    chat_history = format_chat_history(history)
    
    response2 = rag_chain.invoke({"input": query2, "chat_history": chat_history})
    
    memory.add_message(session_id, "user", query2)
    memory.add_message(session_id, "assistant", response2["answer"])
    
    print(f"Q2: {query2}")
    print(f"A2: {response2['answer']}")
    
    # Verify conversation continuity
    final_history = memory.get_history(session_id)
    assert len(final_history) == 4  # 2 user + 2 assistant messages
    
    answer2_lower = response2["answer"].lower()
    assert "reed" in answer2_lower or "evelyn" in answer2_lower
    
    print("✅ Conversation memory and context awareness working correctly")
    
    # Clean up
    memory.clear_session(session_id)


def test_error_handling():
    """Test error handling scenarios."""
    print("\n--- Testing Error Handling ---")
    
    # Test with non-existent directory
    with pytest.raises(FileNotFoundError):
        load_documents_from_directory("/non/existent/path")
    print("✅ FileNotFoundError handled correctly")
    
    # Test memory with invalid session
    memory = ConversationMemory(db_path=MEMORY_DB, max_messages=10)
    history = memory.get_history("non_existent_session")
    assert history == []
    print("✅ Invalid session handled correctly")
    
    # Test invalid split method
    with pytest.raises(ValueError):
        split_text([], method="invalid_method")
    print("✅ Invalid split method handled correctly")


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"])