import pdfplumber
import os
import streamlit as st
from dotenv import load_dotenv
from pdf_processor import store_pdf_text, search_pdf
from langchain_groq import ChatGroq
from tiktoken import get_encoding
from sentence_transformers import SentenceTransformer
import chromadb

# Initialize the embedding model
embedding_model = SentenceTransformer('all-MiniLM-L6-v2')

# Initialize ChromaDB
db = chromadb.PersistentClient(path="./pdf_data")
collection = db.get_or_create_collection("pdf_documents")

# Load environment variables
load_dotenv()

# Define the LlamaChain class
class LlamaChain:
    def __init__(self):
        self.llm = ChatGroq(
            temperature=0,
            groq_api_key=os.getenv("GROQ_API_KEY"),
            model_name="llama-3.1-8b-instant"  # Use a smaller, faster model
        )

    def get_response(self, query):
        try:
            response = self.llm.invoke(query)
            return response.content if hasattr(response, 'content') else "No response received."
        except Exception as e:
            return f"Failed to get response: {e}"

# Utility function to validate token count
def validate_token_count(text, max_tokens=6000):
    """Validate if the input size is within token limit."""
    encoder = get_encoding("cl100k_base")
    tokens = encoder.encode(text)
    return len(tokens) <= max_tokens

# Function to extract text from a PDF file
def extract_text_from_pdf(uploaded_file):
    """Extract text from a PDF file using pdfplumber."""
    text = ""
    with pdfplumber.open(uploaded_file) as pdf:
        for page in pdf.pages:
            text += page.extract_text() + "\n"
    return text

# Streamlit UI
st.title("📚 PDF Chatbot")
st.subheader("Upload a PDF and ask questions or query directly using LLaMA.")

# File upload section
uploaded_files = st.file_uploader("Upload a PDF file", type=["pdf"], accept_multiple_files=True)

if uploaded_files:
    for uploaded_file in uploaded_files:
        # Extract and store the PDF content
        pdf_text = extract_text_from_pdf(uploaded_file)
        store_pdf_text(uploaded_file.name, pdf_text, embedding_model)
        st.success(f"PDF '{uploaded_file.name}' uploaded and stored!")

# Query input section
user_query = st.text_input("Type your query here:")

# Buttons for querying PDF and LLaMA
col1, col2 = st.columns(2)

with col1:
    if st.button("🔍 Search in PDF", key="search_pdf_button"):
        if user_query.strip():
            with st.spinner("Searching in PDFs..."):
                relevant_text = search_pdf(user_query)

                if relevant_text != "No relevant information found.":
                    # Construct the prompt dynamically
                    prompt = f"The following text was retrieved from the PDF:\n\n{relevant_text}\n\nUser query: {user_query}\n\n"
                    if validate_token_count(prompt):
                        llama_chain = LlamaChain()
                        refined_response = llama_chain.get_response(prompt)
                        st.success("Refined Response from PDF:")
                        st.markdown(refined_response)
                    else:
                        st.error("Input exceeds token limit. Please refine your query or input.")
                else:
                    st.warning("No relevant information found in the PDF.")
        else:
            st.warning("Please enter a query before searching.")

with col2:
    if st.button("🤖 Ask LLaMA", key="ask_llama_button"):
        if user_query.strip():
            with st.spinner("Fetching response from LLaMA..."):
                llama_chain = LlamaChain()
                response = llama_chain.get_response(user_query)
                st.success("Response from LLaMA:")
                st.markdown(response)
        else:
            st.warning("Please enter a query before submitting.")

