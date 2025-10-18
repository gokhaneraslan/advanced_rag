import os
import logging
from typing import Optional, Literal, List, Dict

from langchain_core.retrievers import BaseRetriever
from langchain_groq import ChatGroq
from langchain_openai import ChatOpenAI
from langchain_ollama.chat_models import ChatOllama
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.language_models.chat_models import BaseChatModel

from langchain_core.runnables import Runnable
from langchain.chains import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.messages import HumanMessage, AIMessage

logger = logging.getLogger(__name__)

LLMProvider = Literal["openai", "gemini", "groq", "ollama"]


def initialize_llm(
    provider: LLMProvider,
    model_name: str = None,
    **kwargs
) -> BaseChatModel:
    """
    Initializes and returns a LangChain Chat Model from a specified provider.

    This function acts as a factory for different LLM providers, loading the
    necessary API keys from environment variables.

    Args:
        provider (Literal["openai", "gemini", "groq", "ollama"]):
            The LLM provider to use.
        model_name (str, optional): The specific model to use from the provider.
            If None, a sensible default will be used for each provider.
        **kwargs: Additional keyword arguments to pass to the model's constructor
                  (e.g., temperature=0.7, max_tokens=1024).

    Returns:
        BaseChatModel: An instance of the requested LangChain chat model.

    Raises:
        ValueError: If an unsupported provider is specified or if the required
                    API key environment variable is not set.
    """
    
    provider = provider.lower()
    
    logger.info(f"Initializing LLM from provider: '{provider}'...")

    try:
        if provider == "gemini":
            if not os.getenv("GOOGLE_API_KEY"):
                raise ValueError("GOOGLE_API_KEY environment variable not set.")
            
            model = model_name or "gemini-2.5-flash"
            llm = ChatGoogleGenerativeAI(model=model, **kwargs)
            
        elif provider == "openai":
            if not os.getenv("OPENAI_API_KEY"):
                raise ValueError("OPENAI_API_KEY environment variable not set.")
            
            model = model_name or "gpt-4o"
            llm = ChatOpenAI(model=model, **kwargs)

        elif provider == "groq":
            if not os.getenv("GROQ_API_KEY"):
                raise ValueError("GROQ_API_KEY environment variable not set.")
            
            model = model_name or "llama-3.3-70b-versatile"
            llm = ChatGroq(model_name=model, **kwargs)

        elif provider == "ollama":
            model = model_name or "llama3"
            logger.info(f"Note: Ensure the Ollama service is running and you have pulled the '{model}' model.")
            llm = ChatOllama(model=model, **kwargs)

        else:
            raise ValueError(
                f"Unsupported LLM provider: '{provider}'. "
                "Supported providers are 'openai', 'gemini', 'groq', 'ollama'."
            )
        
        logger.info(f"LLM initialized successfully: {provider}/{model}")
        return llm
        
    except Exception as e:
        logger.error(f"Failed to initialize LLM: {e}")
        raise


def create_rag_chain(
    retriever: BaseRetriever, 
    llm: BaseChatModel, 
    prompt_template: Optional[str] = None,
    include_chat_history: bool = True
) -> Runnable:
    """
    Creates a Retrieval-Augmented Generation (RAG) chain with optional chat history.

    This chain orchestrates the entire process:
    1. It takes a user's question (and optional chat history).
    2. It uses the provided retriever to fetch relevant documents.
    3. It stuffs the documents, chat history, and question into a prompt.
    4. It sends the prompt to the LLM to generate an answer.

    Args:
        retriever (BaseRetriever): The configured retriever instance 
            (e.g., the final compression retriever).
        llm (BaseChatModel): The initialized language model.
        prompt_template (Optional[str], optional): A custom system prompt template string.
            Must include a '{context}' placeholder. If None, a default prompt is used.
        include_chat_history (bool): Whether to include chat history in the prompt.

    Returns:
        Runnable: A LangChain runnable object that can be invoked with a query.
                  The output is a dictionary containing "input", "context", "chat_history", and "answer".
    """
    
    logger.info("Creating RAG chain...")

    if prompt_template is None:
        if include_chat_history:
            prompt_template = (
                "You are an assistant for question-answering tasks. "
                "Use the following pieces of retrieved context to answer the question. "
                "If you don't know the answer, just say that you don't know. "
                "Keep the answer concise and based ONLY on the provided context.\n\n"
                "Consider the conversation history to provide contextually relevant answers.\n\n"
                "CONTEXT:\n{context}"
            )
        else:
            prompt_template = (
                "You are an assistant for question-answering tasks. "
                "Use the following pieces of retrieved context to answer the question. "
                "If you don't know the answer, just say that you don't know. "
                "Keep the answer concise and based ONLY on the provided context.\n\n"
                "CONTEXT:\n{context}"
            )

    # Create prompt with optional chat history
    if include_chat_history:
        prompt = ChatPromptTemplate.from_messages([
            ("system", prompt_template),
            MessagesPlaceholder(variable_name="chat_history"),
            ("human", "{input}"),
        ])
    else:
        prompt = ChatPromptTemplate.from_messages([
            ("system", prompt_template),
            ("human", "{input}"),
        ])

    try:
        document_chain = create_stuff_documents_chain(llm, prompt)
        retrieval_chain = create_retrieval_chain(retriever, document_chain)
        
        logger.info("RAG chain created successfully")
        return retrieval_chain
        
    except Exception as e:
        logger.error(f"Failed to create RAG chain: {e}")
        raise


def format_chat_history(history: List[Dict[str, str]]) -> List:
    """
    Convert conversation history from dict format to LangChain message objects.
    
    Args:
        history: List of dicts with 'role' and 'content' keys
        
    Returns:
        List of HumanMessage and AIMessage objects
    """
    messages = []
    for msg in history:
        if msg["role"] == "user":
            messages.append(HumanMessage(content=msg["content"]))
        elif msg["role"] == "assistant":
            messages.append(AIMessage(content=msg["content"]))
    return messages
