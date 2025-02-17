import pdfplumber
from langchain.text_splitter import RecursiveCharacterTextSplitter
import sqlite3
import sys
#sys.modules['sqlite3'] = sys.modules.pop('pysqlite3')
import chromadb
from langchain.embeddings import OpenAIEmbeddings


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

def store_pdf_text(pdf_name, pdf_text, embedding_model):
    """Store extracted text chunks into the vector database."""
    chunks = split_text_into_chunks(pdf_text)
    # Ensure chunks are cleaned
    cleaned_chunks = [''.join(char for char in chunk if ord(char) < 128) for chunk in chunks]
    embeddings = embedding_model.encode(cleaned_chunks)  # Use 'encode' instead of 'embed'
    collection.add(
        ids=[f"{pdf_name}_{i}" for i in range(len(cleaned_chunks))],
        documents=cleaned_chunks,
        embeddings=embeddings,  # Add embeddings to the collection
        metadatas=[{"name": pdf_name, "chunk_index": i} for i in range(len(cleaned_chunks))]
    )



def search_pdf(query, max_chunks=15, max_chunk_length=1500):
    """Retrieve and refine relevant chunks dynamically based on the user query."""
    print("Number of documents in collection:", collection.count())
    results = collection.query(query_texts=[query], n_results=max_chunks)

    print("results", results)
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

