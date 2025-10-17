import os
from typing import List

from langchain_community.document_loaders import TextLoader, PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_experimental.text_splitter import SemanticChunker
from langchain_core.documents import Document


def load_document(file_path: str) -> List[Document]:
    """
    Loads a document from the given file path based on its extension (.txt or .pdf).

    Args:
        file_path (str): The path to the document to be loaded.

    Returns:
        List[Document]: A list of loaded LangChain Document objects.
    
    Raises:
        ValueError: If the file extension is not .txt or .pdf.
        FileNotFoundError: If the file is not found at the specified path.
    """
    
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"File not found at: {file_path}")

    _, file_extension = os.path.splitext(file_path)

    if file_extension.lower() == '.txt':
        loader = TextLoader(file_path, encoding='utf-8')
    elif file_extension.lower() == '.pdf':
        loader = PyPDFLoader(file_path)
    else:
        raise ValueError(f"Unsupported file extension: '{file_extension}'. Only '.txt' and '.pdf' are supported.")
    
    print(f"Loading '{os.path.basename(file_path)}'...")
    
    documents = loader.load()
    
    print("Loading complete.")
    
    return documents


def load_documents_from_directory(directory_path: str) -> List[Document]:
    
    """
    Loads all supported documents (.txt and .pdf) from a specified directory.

    It iterates through all files in the given directory, identifies files
    with '.txt' or '.pdf' extensions, and loads them using the appropriate
    LangChain loader.

    Args:
        directory_path (str): The path to the directory to scan for documents.

    Returns:
        List[Document]: A single list containing all loaded documents from the directory.

    Raises:
        FileNotFoundError: If the specified directory does not exist.
    """
    
    if not os.path.isdir(directory_path):
        raise FileNotFoundError(f"Directory not found at: {directory_path}")

    all_documents = []
    supported_extensions = ['.txt', '.pdf']
    
    print(f"Scanning directory '{directory_path}' for supported files ({', '.join(supported_extensions)})...")

    for filename in os.listdir(directory_path):
        file_path = os.path.join(directory_path, filename)

        _, file_extension = os.path.splitext(filename)
        

        if os.path.isfile(file_path) and file_extension.lower() in supported_extensions:
            
            print(f"  -> Found and loading '{filename}'...")
            
            if file_extension.lower() == '.txt':
                loader = TextLoader(file_path, encoding='utf-8')
            elif file_extension.lower() == '.pdf':
                loader = PyPDFLoader(file_path)
            

            try:
                
                loaded_docs = loader.load()
                all_documents.extend(loaded_docs)
                
            except Exception as e:
                print(f"    [Warning] Failed to load or process {filename}: {e}")

    if not all_documents:
        print("Warning: No supported documents were found in the directory.")
        
    print(f"Directory scan complete. Total documents loaded: {len(all_documents)}")
    
    return all_documents



def split_text(documents: List[Document], method: str = "recursive", **kwargs) -> List[Document]:
    """
    Splits the loaded documents according to the specified method.

    Args:
        documents (List[Document]): A list of LangChain Document objects.
        method (str): The splitting method. Can be "recursive" or "semantic".
        **kwargs: Method-specific arguments.
            - For "recursive": `chunk_size` (int), `chunk_overlap` (int)
            - For "semantic": `embeddings` (an instance of HuggingFaceEmbeddings)

    Returns:
        List[Document]: A list of split text chunks as LangChain Document objects.
    """
    
    print(f"Starting splitting process with '{method}' method...")

    if method == "recursive":
        chunk_size = kwargs.get('chunk_size', 1000)
        chunk_overlap = kwargs.get('chunk_overlap', 100)
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap
        )
        
    elif method == "semantic":
        embeddings = kwargs.get('embeddings')
        
        if embeddings is None:
            raise ValueError("An 'embeddings' model is required for the semantic method.")
        
        text_splitter = SemanticChunker(embeddings)
        
    else:
        raise ValueError(f"Invalid method: '{method}'. Choose 'recursive' or 'semantic'.")

    texts = text_splitter.split_documents(documents)
    
    print(f"The text was split into {len(texts)} chunks.")
    
    return texts
