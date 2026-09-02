import os
import pandas as pd
import streamlit as st
import google.generativeai as genai
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import GooglePalmEmbeddings

# Configure Streamlit page layout
st.set_page_config(page_title="Multi-Document QA Bot", layout="wide")
st.title("📄 Document & Data Q&A Bot")

# API Key Sidebar Configuration
st.sidebar.header("Configuration")
api_key = st.sidebar.text_input("Enter Google Gemini API Key", type="password")

if api_key:
    genai.configure(api_key=api_key)

# File Uploader component
uploaded_file = st.sidebar.file_uploader(
    "Upload a PDF or CSV file", type=["pdf", "csv"]
)

if uploaded_file and api_key:
    file_type = uploaded_file.name.split(".")[-1].lower()

    # Handling CSV Files
    if file_type == "csv":
        df = pd.read_csv(uploaded_file)
        st.subheader("Data Preview")
        st.dataframe(df.head())

        query = st.text_input("Ask a question about your CSV data:")
        if query:
            with st.spinner("Analyzing data..."):
                model = genai.GenerativeModel("gemini-1.5-flash")
                prompt = f"""
                You are a data analyst. Answer the user's question based strictly on the dataset context provided.
                
                Dataset Schema and First 50 Rows:
                {df.head(50).to_string()}
                
                User Query: {query}
                """
                response = model.generate_content(prompt)
                st.write("**Answer:**")
                st.write(response.text)

    # Handling PDF Files
    elif file_type == "pdf":
        # Save file temporarily to disk for PyPDFLoader processing
        temp_pdf_path = f"temp_{uploaded_file.name}"
        with open(temp_pdf_path, "wb") as f:
            f.write(uploaded_file.getbuffer())

        st.info("Processing PDF document...")

        # Load and chunk PDF content
        loader = PyPDFLoader(temp_pdf_path)
        docs = loader.load()
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000, chunk_overlap=200
        )
        splits = text_splitter.split_documents(docs)

        query = st.text_input("Ask a question about your PDF document:")
        if query:
            with st.spinner("Extracting relevant context and generating response..."):
                # Retrieve top relevant text chunks
                full_text = "\n\n".join([doc.page_content for doc in splits[:5]])

                model = genai.GenerativeModel("gemini-1.5-flash")
                prompt = f"""
                Answer the question based strictly on the following context from the document:
                
                Context:
                {full_text}
                
                Question: {query}
                """
                response = model.generate_content(prompt)
                st.write("**Answer:**")
                st.write(response.text)

        # Clean up temporary file
        if os.path.exists(temp_pdf_path):
            os.remove(temp_pdf_path)

elif not api_key:
    st.warning("Please enter your Gemini API Key in the sidebar to proceed.")
