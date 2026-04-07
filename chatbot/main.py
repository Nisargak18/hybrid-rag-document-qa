import os
from document_processor import load_and_split_document
from vector_store import DirectFastEmbeddings, create_vector_store, load_vector_store
from rag_chain import build_rag_chain
from pathlib import Path
from dotenv import load_dotenv
from langchain_core.messages import HumanMessage, AIMessage

def main():
    # Load .env from the parent directory of this script (the project root)
    env_path = Path(__file__).resolve().parent.parent / '.env'
    load_dotenv(dotenv_path=env_path)
    
    # Configuration
    pdf_path = "document.pdf" # Replace with the actual PDF document name
    persist_directory = "./chroma_db"
    
    # 1. Process Document and Setup Vector Store
    if not os.path.exists(persist_directory):
        print("Vector store not found. Creating a new one...")
        if not os.path.exists(pdf_path):
            print(f"\nError: '{pdf_path}' not found.")
            print("Please place a PDF document in this project directory to continue.")
            return
        
        chunks = load_and_split_document(pdf_path)
        vectorstore = create_vector_store(chunks, persist_directory=persist_directory)
    else:
        print("Loading existing vector store...")
        vectorstore = load_vector_store(persist_directory=persist_directory)
        
    # 2. Build Conversational RAG Chain
    rag_chain = build_rag_chain(vectorstore)
    
    # 3. Chatbot Loop
    print("\n" + "="*50)
    print("Document QA Chatbot Initialized")
    print("Powered by LangChain, ChromaDB, FastEmbed, and Llama-3.3 via Groq")
    print("Type 'quit' or 'exit' to end the conversation.")
    print("="*50 + "\n")
    
    # This stores the conversation so the bot remembers context!
    chat_history = []
    
    while True:
        user_input = input("You: ")
        if user_input.lower() in ['quit', 'exit']:
            print("Chatbot: Goodbye!")
            break
            
        if not user_input.strip():
            continue
            
        try:
            # Query the RAG pipeline with our Chat History
            response = rag_chain.invoke({
                "input": user_input,
                "chat_history": chat_history
            })
            
            answer = response['answer']
            print(f"Chatbot: {answer}\n")
            
            # Save this back-and-forth into the memory variable
            chat_history.append(HumanMessage(content=user_input))
            chat_history.append(AIMessage(content=answer))
            
        except Exception as e:
            print(f"Chatbot Error: {e}\n")

if __name__ == "__main__":
    main()
