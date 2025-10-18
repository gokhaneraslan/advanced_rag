import logging
import chromadb
from typing import List, Optional

from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings

from langchain_core.documents import Document
from langchain_core.vectorstores import VectorStoreRetriever

from langchain.retrievers.ensemble import EnsembleRetriever
from langchain_community.retrievers import BM25Retriever
from langchain_core.retrievers import BaseRetriever

from langchain.retrievers import ContextualCompressionRetriever
from langchain_community.document_transformers import LongContextReorder
from langchain_community.document_transformers import EmbeddingsRedundantFilter
from langchain.retrievers.document_compressors import CrossEncoderReranker
from langchain_community.cross_encoders import HuggingFaceCrossEncoder
from langchain.retrievers.document_compressors import DocumentCompressorPipeline

logger = logging.getLogger(__name__)


def create_or_load_chroma_retriever(
    persist_directory: str,
    collection_name: str,
    embeddings: HuggingFaceEmbeddings, 
    documents: Optional[List[Document]] = None,
    search_type: str = "mmr",
    search_kwargs: dict = {"k": 5}
) -> VectorStoreRetriever:
    """
    Creates and persists a new Chroma vector store or loads an existing one.

    This function checks if a collection with the given name already exists in the
    persist_directory. If it does, it loads the store. If not, it creates a
    new one using the provided documents.

    Args:
        persist_directory (str): The directory to save to or load from.
        collection_name (str): The name for the collection within Chroma.
        embeddings (HuggingFaceEmbeddings): The embedding model to use.
        documents (Optional[List[Document]], optional): The list of split documents. 
            Required only if the collection does not already exist. Defaults to None.
        search_type (str, optional): The type of search for the retriever. Defaults to "mmr".
        search_kwargs (dict, optional): Keyword arguments for the search. Defaults to {"k": 5}.

    Returns:
        VectorStoreRetriever: A configured retriever for the Chroma vector store.
        
    Raises:
        ValueError: If the collection does not exist and no documents are provided.
    """

    try:
        client = chromadb.PersistentClient(path=persist_directory)
        existing_collections = [c.name for c in client.list_collections()]
        
        if collection_name in existing_collections:
            logger.info(f"Loading existing collection '{collection_name}' from '{persist_directory}'")
            
            vectorstore = Chroma(
                persist_directory=persist_directory,
                embedding_function=embeddings,
                collection_name=collection_name
            )
            
            # Get collection stats
            collection = client.get_collection(collection_name)
            doc_count = collection.count()
            logger.info(f"✅ Loaded collection with {doc_count} documents")
            
        else:
            logger.info(f"Collection '{collection_name}' not found. Creating new collection...")
            
            if not documents:
                logger.warning("No documents provided for new collection. Creating empty collection.")
                # Create empty collection
                vectorstore = Chroma(
                    persist_directory=persist_directory,
                    embedding_function=embeddings,
                    collection_name=collection_name
                )
            else:
                vectorstore = Chroma.from_documents(
                    documents=documents,
                    embedding=embeddings,
                    collection_name=collection_name,
                    persist_directory=persist_directory
                )
                logger.info(f"✅ New collection created with {len(documents)} documents")

        logger.debug(f"Configuring retriever: search_type='{search_type}', search_kwargs={search_kwargs}")
        
        retriever = vectorstore.as_retriever(
            search_type=search_type, 
            search_kwargs=search_kwargs
        )
        
        return retriever
        
    except Exception as e:
        logger.error(f"Failed to create/load Chroma retriever: {e}")
        raise


def create_bm25_retriever(
    documents: List[Document],
    k: int = 5
) -> BM25Retriever:
    """
    Creates a BM25Retriever for keyword-based search from a list of documents.

    BM25 is a ranking function that scores documents based on the query terms
    appearing in each document, without using semantic understanding.

    Args:
        documents (List[Document]): The list of documents to index for keyword search.
        k (int, optional): The number of documents to retrieve. Defaults to 5.

    Returns:
        BM25Retriever: A configured retriever for keyword search.
    """
    
    try:
        logger.info(f"Creating BM25Retriever with k={k}...")
        
        if not documents:
            logger.warning("No documents provided for BM25Retriever. Creating with empty corpus.")
            documents = [Document(page_content="", metadata={})]
        
        bm25_retriever = BM25Retriever.from_documents(
            documents=documents,
            k=k
        )
        
        logger.info(f"✅ BM25Retriever created with {len(documents)} documents")
        return bm25_retriever
        
    except Exception as e:
        logger.error(f"Failed to create BM25Retriever: {e}")
        raise


def create_ensemble_retriever(
    retrievers: List[BaseRetriever],
    weights: Optional[List[float]] = None
) -> EnsembleRetriever:
    """
    Creates an EnsembleRetriever to combine the results of multiple retrievers.
    It uses Reciprocal Rank Fusion (RRF) to re-rank the combined results,
    providing a more robust final ranking.

    Args:
        retrievers (List[BaseRetriever]): A list of retrievers to combine 
            (e.g., [chroma_retriever, bm25_retriever]).
        weights (Optional[List[float]]): Optional weights for each retriever.
            If None, equal weights are used.

    Returns:
        EnsembleRetriever: A retriever that combines and re-ranks results.
    """
    
    try:
        logger.info(f"Creating EnsembleRetriever with {len(retrievers)} retrievers...")
        
        if not retrievers:
            raise ValueError("At least one retriever must be provided")
        
        kwargs = {
            "retrievers": retrievers,
            "c": 60  # RRF constant
        }
        
        if weights:
            if len(weights) != len(retrievers):
                raise ValueError("Number of weights must match number of retrievers")
            kwargs["weights"] = weights
            logger.debug(f"Using custom weights: {weights}")
        
        ensemble_retriever = EnsembleRetriever(**kwargs)
        
        logger.info("✅ EnsembleRetriever created successfully")
        return ensemble_retriever
        
    except Exception as e:
        logger.error(f"Failed to create EnsembleRetriever: {e}")
        raise


def create_compression_retriever(
    base_retriever: BaseRetriever,
    embeddings: HuggingFaceEmbeddings,
    reranker_model_name: str = "BAAI/bge-reranker-large",
    top_n: int = 5,
    similarity_threshold: float = 0.95
) -> ContextualCompressionRetriever:
    """
    Wraps a base retriever with a compression and reranking pipeline.

    This pipeline enhances retrieval results by:
    1. Filtering out redundant documents (semantically similar ones).
    2. Reranking the remaining documents with a powerful Cross-Encoder model for relevance.
    3. Reordering the documents to place the most relevant ones at the beginning and end,
       combating the "lost in the middle" problem for Large Language Models.

    Args:
        base_retriever (BaseRetriever): The retriever to enhance (e.g., an EnsembleRetriever).
        embeddings (HuggingFaceEmbeddings): The embedding model, needed for the redundant filter.
        reranker_model_name (str, optional): The name of the Cross-Encoder model for reranking.
        top_n (int, optional): The number of top documents to return after reranking. Defaults to 5.
        similarity_threshold (float, optional): The threshold for filtering similar documents.
                                                 Defaults to 0.95.

    Returns:
        ContextualCompressionRetriever: The enhanced retriever.
    """
    
    try:
        logger.info("Creating compression and reranking pipeline...")

        logger.info(f"Loading reranker model: {reranker_model_name}...")
        reranker_model = HuggingFaceCrossEncoder(model_name=reranker_model_name)
        compressor = CrossEncoderReranker(model=reranker_model, top_n=top_n)
        logger.info("✅ Reranker model loaded")

        logger.debug(f"Configuring redundant filter with threshold={similarity_threshold}")
        redundant_filter = EmbeddingsRedundantFilter(
            embeddings=embeddings,
            similarity_threshold=similarity_threshold
        )

        reordering = LongContextReorder()

        # The order is important: filter -> rerank -> reorder
        pipeline_compressor = DocumentCompressorPipeline(
            transformers=[redundant_filter, compressor, reordering]
        )
        
        logger.info("✅ Compressor pipeline created")

        compression_retriever = ContextualCompressionRetriever(
            base_compressor=pipeline_compressor,
            base_retriever=base_retriever
        )
        
        logger.info("✅ ContextualCompressionRetriever created successfully")
        return compression_retriever
        
    except Exception as e:
        logger.error(f"Failed to create compression retriever: {e}")
        raise
    