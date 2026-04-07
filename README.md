# Problem Statement

Design and develop a Python-based chatbot that can answer user questions by extracting relevant information from a given PDF document. The system should use open-source Large Language Models (LLMs) to ensure accuracy, comprehensiveness, and document-grounded responses, while gracefully handling queries for which information is not available.

## Requirements

* **Factual Accuracy:** The chatbot should prioritize factual answers derived from the document.
* **Context Understanding:** The chatbot should demonstrate an ability to understand the context of user questions and utilize relevant information from the documents.
* **Knowledge of Unknowns:** The chatbot should gracefully indicate when information is not found within the document.

## Tech Stack

* **Programming Language:** Python
* **Large Language Models (LLMs):** llama-3.3-70b-versatile (primary) and llama-3.1-8b-instant via Groq
* **LLM Framework:** LangChain (for RAG pipeline)
* **Embedding Model:** BAAI/bge-large-en-v1.5
* **Vector Database:** ChromaDB
* **Graph Database:** Neo4j
* **Backend Framework:** Flask

## Hybrid RAG Architecture (Vector + Graph)

**Why Hybrid?** Vector search alone fails at multi-hop reasoning (relationships).
**The Core Approach:** Parallel retrieval tracks.
*   **Track 1 (Vector):** Handles semantic similarity and unstructured text (ChromaDB).
*   **Track 2 (Graph):** Handles explicit relationships and entity connections (Neo4j).
**Orchestration:** A ReAct Agent that intelligently decides which database "tool" to use based on the query.

### Phase 1: Data Ingestion
When a user uploads a PDF, the system performs 2 parallel actions:
*   **Vector Ingestion:** Text split (RecursiveCharacterTextSplitter) → Convert Embedding → Vector database
*   **Graph Ingestion:** The system uses LLMs to extract entities (name, companies) and relationships to build a Neo4j knowledge graph

### Phase 2: When user asks queries
Example: *"How much did he pay?"*
The system doesn't just search for those words. The **historyawareretriever** looks at the chat history, realizes he refers to a CEO mentioned earlier, and reformulates the question into:
*How much did [CEO name] pay for the [company name]?*

### Phase 3: Hybrid Retrieval (Agent step)
The langchain ReAct agent receives the reformulated question and decides which tool to use.
*   **Scenario A:** Vector Database (ChromaDB) for general query
*   **Scenario B:** Graph Database (Neo4j) for complex Relationship
The agent can even combine both results if needed.

### Phase 4: Final Result
The retrieved context + query is sent to LLM (brain). The LLM generates answers based on PDF context and sends the answers to frontend.

---

## Architecture Diagrams

### 1. Hybrid Architecture Diagram


### 2. Document Ingestion Pipeline and Conversational Retrieval Flow

