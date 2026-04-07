import os
from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_classic.chains.retrieval import create_retrieval_chain
from langchain_classic.chains.combine_documents.stuff import create_stuff_documents_chain
from langchain_classic.chains.history_aware_retriever import create_history_aware_retriever

def build_rag_chain(vectorstore):
    """
    Builds the Retrieval-Augmented Generation chain using Llama-3 via Groq.
    This version includes Conversational Memory so it remembers previous statements.
    """
    print("Setting up LLM: Llama-3.3-70B via Groq API (with 8B fallback)...")
    
    primary_llm = ChatGroq(
        temperature=0,
        model_name="llama-3.3-70b-versatile"
    )
    
    fallback_llm = ChatGroq(
        temperature=0,
        model_name="llama-3.1-8b-instant"
    )
    
    llm = primary_llm.with_fallbacks([fallback_llm])
    
    retriever = vectorstore.as_retriever(search_kwargs={"k": 10})

    # 1. Update Retriever to parse Chat History
    # This teaches the LLM to rewrite questions that rely on previous sentences.
    # (e.g. User: "Who is the CEO?", then User: "How old is he?". The LLM rewrites it to "How old is the CEO?")
    contextualize_q_system_prompt = (
        "Given a chat history and the latest user question "
        "which might reference context in the chat history, "
        "formulate a standalone question which can be understood "
        "without the chat history. Do NOT answer the question, "
        "just reformulate it if needed and otherwise return it as is."
    )
    contextualize_q_prompt = ChatPromptTemplate.from_messages(
        [
            ("system", contextualize_q_system_prompt),
            MessagesPlaceholder("chat_history"),
            ("human", "{input}"),
        ]
    )
    
    history_aware_retriever = create_history_aware_retriever(
        fallback_llm, retriever, contextualize_q_prompt
    )
    
    # 2. Main Question Answering Chain
    system_prompt = (
        "You are an expert factual assistant for document-based question answering. "
        "Use the following pieces of retrieved context to answer the specific question. "
        "Analyze the provided context carefully to find the answer. "
        "If the answer is not explicitly contained within the provided context, "
        "you must gracefully indicate that the information is not found within the document. "
        "Do not invent facts, rely on outside knowledge, or hallucinate answers.\n\n"
        "Formatting:\n"
        "- If the retrieved context contains tables or structured financial data (e.g., balance sheets, income statements), "
        "you MUST format your response using a Markdown Table for better readability.\n"
        "- **IMPORTANT**: Each row of the table must be on a SINGLE line. Do not insert newlines within cells, as this breaks the Markdown formatting.\n"
        "- Use clear headers and align columns properly.\n\n"
        "Context:\n{context}"
    )
    
    qa_prompt = ChatPromptTemplate.from_messages(
        [
            ("system", system_prompt),
            MessagesPlaceholder("chat_history"),
            ("human", "{input}"),
        ]
    )
    
    document_chain = create_stuff_documents_chain(llm, qa_prompt)
    
    # 3. Create the final retrieval chain
    print("Creating the conversational retrieval chain...")
    rag_chain = create_retrieval_chain(history_aware_retriever, document_chain)
    
    return rag_chain
