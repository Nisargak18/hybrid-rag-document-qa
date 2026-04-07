import os
from langchain_core.embeddings import Embeddings
from langchain_community.vectorstores import Chroma
from fastembed import TextEmbedding

class DirectFastEmbeddings(Embeddings):
    def __init__(self, model_name="BAAI/bge-large-en-v1.5"):
        self.model = TextEmbedding(model_name=model_name)

    def embed_documents(self, texts):
        return [e.tolist() for e in self.model.embed(texts)]

    def embed_query(self, text):
        return next(self.model.query_embed(text)).tolist()

def create_vector_store(chunks, persist_directory="./chroma_db"):
    """
    Takes document chunks, creates embeddings using BAAI/bge-large-en-v1.5,
    and stores them in a Chroma DB vector store.
    """
    embeddings = DirectFastEmbeddings(model_name="BAAI/bge-base-en-v1.5")
    
    print(f"Creating ChromaDB vector store with {len(chunks)} chunks...")
    # Create the vector store
    vectorstore = Chroma.from_documents(
        documents=chunks,
        embedding=embeddings,
        persist_directory=persist_directory
    )
    
    print(f"Vector store successfully saved to {persist_directory}")
    return vectorstore

def load_vector_store(persist_directory="./chroma_db"):
    """
    Loads an existing vector store.
    """
    embeddings = DirectFastEmbeddings(model_name="BAAI/bge-base-en-v1.5")
    
    vectorstore = Chroma(persist_directory=persist_directory, embedding_function=embeddings)
    return vectorstore
