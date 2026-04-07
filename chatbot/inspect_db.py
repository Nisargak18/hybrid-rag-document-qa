import os
from langchain_community.vectorstores import Chroma
from vector_store import DirectFastEmbeddings

def inspect_database(persist_directory="./chroma_db_flask"):
    """
    Loads the ChromaDB and prints its content in a structured 'Row/Column' format.
    """
    if not os.path.exists(persist_directory):
        print(f"Error: Database directory '{persist_directory}' not found.")
        return

    # Load embeddings model
    embeddings = DirectFastEmbeddings(model_name="BAAI/bge-base-en-v1.5")
    
    # Load the vector store
    print(f"Loading database from {persist_directory}...")
    vectorstore = Chroma(persist_directory=persist_directory, embedding_function=embeddings)
    
    # Get all data from the collection
    results = vectorstore.get()
    
    ids = results['ids']
    docs = results['documents']
    metadatas = results['metadatas']
    
    total_records = len(ids)
    print(f"\n--- Database Inspection (Total Chunks: {total_records}) ---")
    
    if total_records == 0:
        print("Database is empty.")
        return

    # Print Header
    header = f"{'Chunk ID':<15} | {'Page':<5} | {'Snippet (First 60 chars)'}"
    print("-" * len(header))
    print(header)
    print("-" * len(header))
    
    # Print Rows (Show first 10 for demonstration)
    limit = min(10, total_records)
    for i in range(limit):
        chunk_id = ids[i][:12] + "..." # Shorten ID for display
        page = metadatas[i].get('page', 'N/A')
        snippet = docs[i].replace('\n', ' ')[:60] + "..."
        
        print(f"{chunk_id:<15} | {page:<5} | {snippet}")

    if total_records > limit:
        print(f"\n... and {total_records - limit} more records.")

if __name__ == "__main__":
    inspect_database()
