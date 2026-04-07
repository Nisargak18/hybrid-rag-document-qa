import os
import pypdf
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter

def load_and_split_document(pdf_path: str, chunk_size: int = 500, chunk_overlap: int = 150):
    """
    Loads a PDF document and splits it into chunks using LangChain.
    """
    print(f"Loading document from: {pdf_path}")
    if not os.path.exists(pdf_path):
        raise FileNotFoundError(f"The file {pdf_path} does not exist.")

    # 1. Load the PDF directly using pypdf
    documents = []
    with open(pdf_path, "rb") as f:
        reader = pypdf.PdfReader(f)
        for i, page in enumerate(reader.pages):
            text = page.extract_text()
            if text:
                documents.append(Document(page_content=text, metadata={"source": pdf_path, "page": i}))
    
    print(f"Loaded {len(documents)} pages from the PDF.")

    # 2. Split the Text
    # We use RecursiveCharacterTextSplitter as it tries to keep paragraphs/sentences together
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        add_start_index=True,
    )
    
    chunks = text_splitter.split_documents(documents)
    print(f"Split the document into {len(chunks)} chunks.")
    
    return chunks

if __name__ == "__main__":
    # Example usage:
    # pdf_path = "sample.pdf"  # Replace with your actual PDF path later
    # chunks = load_and_split_document(pdf_path)
    # print(f"First chunk preview: {chunks[0].page_content[:200]}...")
    print("Document processor module is ready.")


