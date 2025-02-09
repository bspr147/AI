import os
import streamlit as st
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from sentence_transformers import SentenceTransformer

# Initialize FAISS store
if not os.path.exists("faiss_index"):
    os.makedirs("faiss_index")
vector_store = None

# Use SentenceTransformers for embeddings (No OpenAI key required)
embedding_model = SentenceTransformer("all-MiniLM-L6-v2")  # Lightweight, efficient model


## Function to process and store PDF text
def process_and_store_pdf(pdf_file):
    """
    Process the uploaded PDF and store the extracted text chunks into FAISS.
    """
    global vector_store
    try:
        # Load and process the PDF
        loader = PyPDFLoader(pdf_file)
        documents = loader.load()

        # Split text into manageable chunks
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
        docs = text_splitter.split_documents(documents)

        # Extract text and create embeddings
        texts = [doc.page_content for doc in docs]
        text_embeddings = embedding_model.encode(texts)

        # Store in FAISS
        vector_store = FAISS.from_embeddings(text_embeddings, embedding_model)
        vector_store.save_local("faiss_index")
        return f"PDF '{os.path.basename(pdf_file)}' successfully indexed!"
    except Exception as e:
        return f"Error processing PDF: {e}"


# Function to search PDFs using FAISS
def search_pdf(query, max_results=5):
    """
    Search for relevant chunks in the PDF vector store using FAISS.
    """
    global vector_store
    try:
        if vector_store is None:
            vector_store = FAISS.load_local("faiss_index", embedding_model)

        retriever = vector_store.as_retriever(search_kwargs={"k": max_results})
        docs = retriever.get_relevant_documents(query)

        if docs:
            return "\n\n".join([doc.page_content for doc in docs])
        return "No relevant information found."
    except Exception as e:
        return f"Error during search: {e}"


# Streamlit UI
st.title("📚 Enhanced PDF Chatbot (No OpenAI Key)")
st.subheader("Upload a PDF and ask questions or query directly.")

# File upload section
uploaded_file = st.file_uploader("Upload a PDF file", type=["pdf"])

if uploaded_file is not None:
    with st.spinner("Processing PDF..."):
        pdf_path = f"./uploads/{uploaded_file.name}"
        os.makedirs("uploads", exist_ok=True)

        # Save the uploaded file locally
        with open(pdf_path, "wb") as f:
            f.write(uploaded_file.getbuffer())

        # Process and store the PDF text
        result_message = process_and_store_pdf(pdf_path)
        if "successfully indexed" in result_message:
            st.success(result_message)
        else:
            st.error(result_message)

# Query input section
user_query = st.text_input("Type your query here:")

if st.button("🔍 Search PDF"):
    if user_query.strip():
        with st.spinner("Searching in PDFs..."):
            relevant_text = search_pdf(user_query)
            if relevant_text != "No relevant information found.":
                st.success("Relevant Information from PDF:")
                st.markdown(relevant_text)
            else:
                st.warning(relevant_text)
    else:
        st.warning("Please enter a query before searching.")
