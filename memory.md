# Task Memory

- 13:14:02 - Created `document_processor.py` to implement PDF loading using `PyPDFLoader` and text splitting using `RecursiveCharacterTextSplitter`. This prepares the pipeline for the vector database insertion which will be implemented later.
- 13:17:02 - Created `vector_store.py` to handle chunk embedding using `BAAI/bge-large-en-v1.5` and ingestion into `ChromaDB` for retrieval.
- 13:19:02 - Created `rag_chain.py` implementing the final steps of setting up `ChatGroq` (Llama-3-8B-Instruct), retrieving the top 3 specific document chunks, and using a RAG structure with graceful fallback semantics to indicate when information isn't in the provided text.
- 13:20:02 - Created `main.py` tying the components together into a functional command-line chatbot script.
- 13:22:02 - Moved all Python scripts (`document_processor.py`, `vector_store.py`, `rag_chain.py`, `main.py`) into a new `chatbot` folder per user instructions.
- 13:33:02 - Added `neo4j` to requirements and created `chatbot/graph_agent.py` to implement Hybrid RAG, combining Vector (ChromaDB) and Graph (Neo4j) into a smart AI Agent.
- 13:36:02 - Added `langchain-experimental` to requirements and created `chatbot/graph_builder.py` to automatically populate the Neo4j instance by extracting graph relationships directly from PDF chunks.
- 14:14:02 - Added `streamlit` support to build a responsive frontend (`chatbot/app.py`) natively handling PDF uploads and visual Web Chat interfacing, while fixing a legacy dependency issue.
- 14:39:02 - Pivoted architecture from Streamlit to a Flask REST API backend (`chatbot/flask_app.py`) with a custom HTML/CSS/JS frontend using rich aesthetics (`templates/index.html`).
- 11:44:07 - Refactored `graph_builder.py`, `rag_chain.py`, and `graph_agent.py` to use `llama-3.1-8b-instant` as a fallback LLM to mitigate 429 rate limit issues on the Groq free tier plan.
- 19:08:07 - Created `README.md` containing the project problem statement and requirements.
- 19:12:00 - Added tech stack information (Python, LangChain, ChromaDB, Neo4j, Flask) to `README.md`.
- 19:16:00 - Added the Hybrid RAG logic and the three Mermaid architecture diagrams (Hybrid Architecture, Document Ingestion, and Conversational Retrieval Flow) to `README.md`.
- 19:46:00 - Added `.env` to `.gitignore` to prevent secret leakage.
