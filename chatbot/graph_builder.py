import os
from langchain_neo4j import Neo4jGraph
from langchain_groq import ChatGroq
from langchain_experimental.graph_transformers import LLMGraphTransformer

from pathlib import Path
from dotenv import load_dotenv

def populate_graph_database(chunks):
    """
    Takes document chunks, runs them through the Groq LLM to extract 
    nodes and relationships, and inserts them into Neo4j.
    """
    # Load .env from the parent directory of this script (the project root)
    env_path = Path(__file__).resolve().parent.parent / '.env'
    load_dotenv(dotenv_path=env_path)
    print("\n--- Initializing Graph Database Builder ---")
    print("Checking for Neo4j credentials...")
    
    # Retrieve credentials explicitly to ensure they are picked up
    uri = os.environ.get("NEO4J_URI")
    user = os.environ.get("NEO4J_USERNAME")
    password = os.environ.get("NEO4J_PASSWORD")

    if not uri or not user or not password:
        print("Error: Missing Neo4j credentials in environment variables.")
        return False
        
    try:
        # Connect to Neo4j explicitly
        print(f"Connecting to Neo4j at {uri} as {user}...")
        graph = Neo4jGraph(url=uri, username=user, password=password)
        # Verify connection with a test query
        graph.query("RETURN 1")
        print("Successfully connected and verified Neo4j!")
    except Exception as e:
        print(f"CRITICAL: Failed to connect to Neo4j.")
        print(f"Technical Error: {e}")
        if "Unauthorized" in str(e) or "authentication" in str(e).lower():
            print("Action Needed: Your NEO4J_PASSWORD or NEO4J_USERNAME in .env is incorrect.")
        elif "ServiceUnavailable" in str(e):
            print("Action Needed: Your Neo4j instance at AuraDB might be paused or the URI is wrong.")
        return False

    print("Initializing LLM for Entity Extraction (Using Llama-3)...")
    primary_llm = ChatGroq(temperature=0, model_name="llama-3.3-70b-versatile")
    fallback_llm = ChatGroq(temperature=0, model_name="llama-3.1-8b-instant")
    llm = primary_llm.with_fallbacks([fallback_llm])
    
    # The Transformer uses the LLM to read text and pull out Subject-Action-Object pairs
    llm_transformer = LLMGraphTransformer(llm=llm)
    
    print(f"Extracting Graph Entities from {len(chunks)} document chunks...")
    print("This may take some time depending on document size and Groq API limits.")
    
    try:
        # Convert text documents to graph documents
        graph_documents = llm_transformer.convert_to_graph_documents(chunks)
        
        print(f"Extracted {len(graph_documents)} graph components. Inserting into Neo4j...")
        # Insert into Neo4j database
        graph.add_graph_documents(
            graph_documents, 
            baseEntityLabel=True, # Helps keep the graph structured
            include_source=True   # Links nodes back to original chunk
        )
        print("Successfully built the Knowledge Graph in Neo4j!")
        return True
    except Exception as e:
        print(f"Error during graph extraction/insertion: {e}")
        return False

if __name__ == "__main__":
    from pathlib import Path
    from dotenv import load_dotenv
    # Load .env from the parent directory of this script (the project root)
    env_path = Path(__file__).resolve().parent.parent / '.env'
    load_dotenv(dotenv_path=env_path)
    print("This script is designed to be called from the main pipeline.")
