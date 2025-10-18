import config
from dotenv import load_dotenv
from langchain_huggingface import HuggingFaceEmbeddings

from src.data_processing import load_documents_from_directory, split_text
from src.chains import initialize_llm, create_rag_chain
from src.retrieval import (
    create_or_load_chroma_retriever,
    create_bm25_retriever,
    create_ensemble_retriever,
    create_compression_retriever,
)



def run_pipeline():
    """Executes the complete RAG pipeline from loading data to answering a query."""
    
    load_dotenv()

    print("--- Initializing Models ---")
    
    embedding_model = HuggingFaceEmbeddings(model_name=config.EMBEDDING_MODEL)
    
    llm = initialize_llm(
        provider=config.LLM_PROVIDER, 
        model_name=config.LLM_MODEL, 
        temperature=0
    )


    print("\n--- Processing Data ---")
    
    documents = load_documents_from_directory(config.DATA_DIR)
    
    chunks = split_text(
        documents, 
        method="semantic", 
        embeddings=embedding_model
    )

    print("\n--- Building Retriever ---")
    
    vector_retriever = create_or_load_chroma_retriever(
        str(config.VECTOR_STORE_DIR), 
        config.COLLECTION_NAME, 
        embedding_model, 
        documents=chunks
    )
    
    keyword_retriever = create_bm25_retriever(chunks)
    
    ensemble_retriever = create_ensemble_retriever(
        [vector_retriever, keyword_retriever]
    )
    
    final_retriever = create_compression_retriever(
        ensemble_retriever, 
        embedding_model, 
        reranker_model_name=config.RERANKER_MODEL
    )


    print("\n--- Creating RAG Chain ---")
    
    rag_chain = create_rag_chain(final_retriever, llm)

    print("\n--- Ready to Query ---")
    query = "What is the primary objective of Project Starlight?"
    
    response = rag_chain.invoke({"input": query})
    
    print(f"\nQuery: {query}")
    print("\nAnswer:")
    print(response["answer"])
    

if __name__ == "__main__":
    run_pipeline()
