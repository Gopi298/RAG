import os
import pandas as pd
import streamlit as st
from google import genai
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter

# 1. Page Configuration
st.set_page_config(
    page_title="Document & Data Q&A Bot", page_icon="🤖", layout="wide"
)
st.title("📄 Multi-Format Data Q&A Assistant")

# 2. Authentication Setup (Prioritize Streamlit Secrets, fallback to Sidebar Input)
api_key = st.secrets.get("GEMINI_API_KEY", "")

if not api_key:
    api_key = st.sidebar.text_input(
        "Enter Gemini API Key (AIzaSy...)", type="password"
    )

if not api_key:
    st.warning("⚠️ Please provide a Gemini API Key to proceed.")
    st.stop()

# Initialize Google GenAI Client
client = genai.Client(api_key=api_key)

# 3. File Upload Interface
uploaded_file = st.sidebar.file_uploader(
    "Upload a PDF or CSV file", type=["pdf", "csv"]
)

if uploaded_file:
    file_extension = uploaded_file.name.split(".")[-1].lower()

    # ==================== CSV ANALYSIS ====================
    if file_extension == "csv":
        st.subheader("📊 CSV Data Preview")
        try:
            df = pd.read_csv(uploaded_file)
            st.dataframe(df.head(10), use_container_width=True)

            user_query = st.text_input("Ask a question about your CSV data:")

            if user_query:
                with st.spinner("Analyzing spreadsheet..."):
                    prompt = f"""
                    You are an expert data analyst. Answer the user's question based strictly on the provided dataset preview and schema.

                    Dataset Context (First 50 Rows):
                    {df.head(50).to_string()}

                    User Question: {user_query}
                    """

                    response = client.models.generate_content(
                        model="gemini-2.5-flash", contents=prompt
                    )

                    st.markdown("### 💡 Answer:")
                    st.write(response.text)
        except Exception as e:
            st.error(f"Error reading CSV file: {e}")

    # ==================== PDF ANALYSIS ====================
    elif file_extension == "pdf":
        st.subheader("📑 PDF Document Processing")

        # Write uploaded file to a temporary location for PyPDFLoader
        temp_file_path = f"temp_{uploaded_file.name}"
        with open(temp_file_path, "wb") as f:
            f.write(uploaded_file.getbuffer())

        try:
            # Chunking document
            loader = PyPDFLoader(temp_file_path)
            documents = loader.load()

            text_splitter = RecursiveCharacterTextSplitter(
                chunk_size=1000, chunk_overlap=150
            )
            chunks = text_splitter.split_documents(documents)

            st.success(
                f"Successfully parsed document into {len(chunks)} text chunks."
            )

            user_query = st.text_input(
                "Ask a question about your PDF document:"
            )

            if user_query:
                with st.spinner("Searching document & generating answer..."):
                    # Extract top relevant context chunks
                    document_context = "\n\n".join(
                        [chunk.page_content for chunk in chunks[:5]]
                    )

                    prompt = f"""
                    You are a helpful document assistant. Answer the user's question strictly using the provided context from the PDF document.

                    Document Context:
                    {document_context}

                    User Question: {user_query}
                    """

                    response = client.models.generate_content(
                        model="gemini-2.5-flash", contents=prompt
                    )

                    st.markdown("### 💡 Answer:")
                    st.write(response.text)

        except Exception as e:
            st.error(f"Error processing PDF file: {e}")

        finally:
            # Clean up temporary file
            if os.path.exists(temp_file_path):
                os.remove(temp_file_path)
