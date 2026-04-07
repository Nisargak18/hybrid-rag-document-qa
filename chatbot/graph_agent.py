import os
from langchain_groq import ChatGroq
from langchain_community.graphs import Neo4jGraph
from langchain.chains import GraphCypherQAChain
from langchain.tools.retriever import create_retriever_tool
from langchain.agents import initialize_agent, AgentType, Tool

from pathlib import Path
from dotenv import load_dotenv

def create_hybrid_agent(vectorstore):
    """
    Creates an AI Agent that combines a Vector Database (ChromaDB) 
    and a Graph Database (Neo4j) to solve complex questions.
    """
    # Load .env from the parent directory of this script (the project root)
    env_path = Path(__file__).resolve().parent.parent / '.env'
    load_dotenv(dotenv_path=env_path)
    # Use Llama-3 with fallbacks to avoid rate limits
    primary_llm = ChatGroq(model_name="llama-3.3-70b-versatile", temperature=0)
    fallback_llm = ChatGroq(model_name="llama-3.1-8b-instant", temperature=0)
    llm = primary_llm.with_fallbacks([fallback_llm])

    # ---------------------------------------------------------
    # TOOL 1: Vector Database (ChromaDB)
    # Solves semantic search and unstructured text retrieval
    # ---------------------------------------------------------
    retriever = vectorstore.as_retriever(search_kwargs={"k": 3})
    vector_tool = create_retriever_tool(
        retriever,
        "vector_search",
        "Use this tool to search for general information, unstructured text, or semantic context from the document."
    )

    tools = [vector_tool]

    # ---------------------------------------------------------
    # TOOL 2: Graph Database (Neo4j)
    # Solves multi-hop reasoning and explicit relationships
    # ---------------------------------------------------------
    print("Attempting to connect to Neo4j Graph Database...")
    try:
        # Relies on NEO4J_URI, NEO4J_USERNAME, and NEO4J_PASSWORD from .env
        graph = Neo4jGraph()
        
        # This chain translates english to Cypher (Neo4j's query language)
        graph_chain = GraphCypherQAChain.from_llm(
            graph=graph, 
            llm=llm, 
            verbose=True
        )
        
        graph_tool = Tool(
            name="graph_search",
            func=graph_chain.run,
            description="Use this tool to search for structured relationships, explicit connections, entities, and multi-hop reasoning (e.g. 'Who acquired the company that John works for?')."
        )
        
        tools.append(graph_tool)
        print("Success: Hybrid Agent will use BOTH Vector (Chroma) and Graph (Neo4j) databases!")
    except Exception as e:
        print("Notice: Neo4j connection not found or failed. Ensure Neo4j credentials are in .env.")
        print("The Agent will fall back to using ONLY the Vector Database for now.")

    # ---------------------------------------------------------
    # INITIALIZE AGENT
    # ---------------------------------------------------------
    agent = initialize_agent(
        tools=tools,
        llm=llm,
        agent=AgentType.CHAT_ZERO_SHOT_REACT_DESCRIPTION,
        verbose=True, # Show the agent's thought process in the terminal
        handle_parsing_errors=True
    )
    
    return agent
