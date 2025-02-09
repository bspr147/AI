import os
import pdfplumber
import streamlit as st
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
            page_text = page.extract_text()
            if page_text:
                text += page_text + "\n"
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
        metadatas=[{"name": pdf_name, "chunk_index": i} for i in range(len(chunks))],
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

# Streamlit UI
st.title("📚 PDF Query System (No OpenAI Key)")
st.subheader("Upload a PDF and query its content using ChromaDB.")

# File upload section
uploaded_file = st.file_uploader("Upload a PDF file", type=["pdf"])

if uploaded_file is not None:
    with st.spinner("Processing PDF..."):
        pdf_path = f"./uploads/{uploaded_file.name}"
        os.makedirs("uploads", exist_ok=True)

        # Save the uploaded file locally
        with open(pdf_path, "wb") as f:
            f.write(uploaded_file.getbuffer())

        # Extract and store PDF text
        pdf_text = extract_text_from_pdf(pdf_path)
        if pdf_text:
            store_pdf_text(uploaded_file.name, pdf_text)
            st.success(f"PDF '{uploaded_file.name}' has been uploaded and indexed!")
        else:
            st.error("Failed to extract text from the uploaded PDF.")

# Query section
user_query = st.text_input("Type your query here:")

if st.button("🔍 Search PDF", key="search_pdf_button"):
    if user_query.strip():
        with st.spinner("Searching in PDF..."):
            relevant_text = search_pdf(user_query)
            if relevant_text != "No relevant information found in the document.":
                st.success("Query Results:")
                st.markdown(relevant_text)
            else:
                st.warning(relevant_text)
    else:
        st.warning("Please enter a query before searching.")
