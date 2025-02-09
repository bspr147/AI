import pdfplumber
from langchain.text_splitter import RecursiveCharacterTextSplitter
import chromadb

# Initialize ChromaDB
db = chromadb.PersistentClient(path="./pdf_data")
collection = db.get_or_create_collection("pdf_documents")

def extract_text_from_pdf(pdf_path):
    """Extract text from a PDF file using pdfplumber."""
    text = ""
    with pdfplumber.open(pdf_path) as pdf:
        for page in pdf.pages:
            text += page.extract_text() + "\n"
    return text

def split_text_into_chunks(text, chunk_size=1000, chunk_overlap=100):
    """Split text into manageable chunks."""
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size, chunk_overlap=chunk_overlap
    )
    return splitter.split_text(text)

def store_pdf_text(pdf_name, pdf_text):
    """Store extracted text chunks into the vector database."""
    chunks = split_text_into_chunks(pdf_text)
    collection.add(
        ids=[f"{pdf_name}_{i}" for i in range(len(chunks))],
        documents=chunks,
        metadatas=[{"name": pdf_name, "chunk_index": i} for i in range(len(chunks))]
    )

def search_pdf(query, max_chunks=5, max_chunk_length=500):
    """Retrieve and refine relevant chunks dynamically based on the user query."""
    results = collection.query(query_texts=[query], n_results=max_chunks)

    if results["documents"]:
        # Flatten documents into a single list
        flattened_documents = [
            item[:max_chunk_length]
            for sublist in results["documents"]
            for item in sublist
        ]

        # Combine chunks dynamically based on the query
        combined_text = "\n".join(flattened_documents[:max_chunks])
        if combined_text:
            return combined_text.strip()

    return "No relevant information found in the document."

