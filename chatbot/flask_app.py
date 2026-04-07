from flask import Flask, request, jsonify, render_template
import os
import tempfile
from document_processor import load_and_split_document
from vector_store import create_vector_store
from rag_chain import build_rag_chain
from dotenv import load_dotenv
from pathlib import Path

# Load .env from the parent directory of this script (the project root)
env_path = Path(__file__).resolve().parent.parent / '.env'
load_dotenv(dotenv_path=env_path)
app = Flask(__name__)

# Global memory for the server
app.config['RAG_CHAIN'] = None
chat_history = []
current_chunks = None # Store chunks for graph building

@app.route("/")
def index():
    return render_template("index.html")

@app.route("/upload", methods=["POST"])
def upload_pdf():
    global chat_history
    if 'file' not in request.files:
        return jsonify({"error": "No file part"}), 400
    file = request.files['file']
    if file.filename == '':
        return jsonify({"error": "No selected file"}), 400
    
    if file and file.filename.endswith('.pdf'):
        # Save to temp
        with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as temp_file:
            file.save(temp_file.name)
            temp_pdf_path = temp_file.name
            
        try:
            global current_chunks
            chunks = load_and_split_document(temp_pdf_path)
            current_chunks = chunks
            
            print("Step 1: Building Vector Database...")
            vectorstore = create_vector_store(chunks, persist_directory="./chroma_db_flask")
            app.config['RAG_CHAIN'] = build_rag_chain(vectorstore)
            chat_history.clear() # Reset memory when new PDF is uploaded
            
            print("Step 2: Attempting to build Knowledge Graph (Neo4j)...")
            from graph_builder import populate_graph_database
            # Attempt graph building (using first 50 chunks for speed)
            graph_success = populate_graph_database(chunks[:50])
            
            if graph_success:
                return jsonify({
                    "message": "Success! Vector and Graph databases are both ready.",
                    "graph_status": "built"
                })
            else:
                return jsonify({
                    "message": "Vector Database is ready, but Graph building failed. Check credentials in .env.",
                    "graph_status": "failed"
                })
        except Exception as e:
            return jsonify({"error": f"Build Process Failed: {str(e)}"}), 500
    
    
    return jsonify({"error": "Invalid file type. Please upload a PDF."}), 400

# The /build_graph route is now merged into /upload for a smoother user experience

@app.route("/chat", methods=["POST"])
def chat():
    global chat_history
    data = request.json
    user_input = data.get("message")
    
    rag_chain = app.config.get('RAG_CHAIN')
    if not rag_chain:
        return jsonify({"error": "Please upload and process a document first."}), 400
        
    try:
        response = rag_chain.invoke({
            "input": user_input,
            "chat_history": chat_history
        })
        answer = response['answer']
        
        # Manually import required core classes locally inside chat
        from langchain_core.messages import HumanMessage, AIMessage
        chat_history.append(HumanMessage(content=user_input))
        chat_history.append(AIMessage(content=answer))
        
        return jsonify({"answer": answer})
    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == "__main__":
    app.run(port=5001, debug=False)
