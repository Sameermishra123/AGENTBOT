# rag_agent_app/backend/vectorstore.py

import os
from langchain_pinecone import Pinecone
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter


# Import API key from config
from config import PINECONE_API_KEY

# Set environment variables for Pinecone
os.environ["PINECONE_API_KEY"] = PINECONE_API_KEY

# Define Hugging Face embedding model
embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

# Define Pinecone index name
INDEX_NAME = "langgraph-rag-index"  # must match your actual index name

# --- Retriever (Existing function) ---
def get_retriever():
    """Initializes and returns the Pinecone vector store retriever."""
    
    # Initialize Pinecone vector store
    vectorstore = Pinecone(
        index_name=INDEX_NAME,
        embedding=embeddings
    )
    
    return vectorstore.as_retriever()

# --- Function to add documents to the vector store ---
def add_document_to_vectorstore(text_content: str):
    """Adds a single text document to the Pinecone vector store."""
    
    if not text_content:
        raise ValueError("Document content cannot be empty.")

    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200,
        add_start_index=True,
    )
    
    documents = text_splitter.create_documents([text_content])
    
    print(f"Splitting document into {len(documents)} chunks for indexing...")
    
    vectorstore = Pinecone(
        index_name=INDEX_NAME,
        embedding=embeddings
    )
    
    vectorstore.add_documents(documents)
    print(f"Successfully added {len(documents)} chunks to Pinecone index '{INDEX_NAME}'.")
