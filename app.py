```python
import os
import streamlit as st

from PyPDF2 import PdfReader

from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_community.vectorstores import FAISS

from langchain_core.prompts import ChatPromptTemplate
from langchain.chains.combine_documents import create_stuff_documents_chain


# ============================================================
# PAGE CONFIGURATION
# ============================================================

st.set_page_config(
    page_title="PDF Chatbot",
    page_icon="📚",
    layout="wide"
)


# ============================================================
# OPENAI API KEY
# ============================================================

# IMPORTANT:
# Do NOT put your API key directly in this Python file.
#
# For Streamlit Cloud:
# Go to:
# App -> Settings -> Secrets
#
# Add:
#
# OPENAI_API_KEY = "your-new-api-key"
#
# Then the code below will read it.

OPENAI_API_KEY = st.secrets.get("OPENAI_API_KEY", os.getenv("OPENAI_API_KEY"))


if not OPENAI_API_KEY:
    st.error(
        "OpenAI API key is missing. "
        "Add OPENAI_API_KEY in Streamlit Cloud Secrets."
    )
    st.stop()


# ============================================================
# TITLE
# ============================================================

st.title("📚 My PDF Chatbot")
st.write("Upload a PDF and ask questions about its content.")


# ============================================================
# SIDEBAR
# ============================================================

with st.sidebar:

    st.title("📄 Your Documents")

    file = st.file_uploader(
        "Upload a PDF file and start asking questions",
        type=["pdf"]
    )


# ============================================================
# PROCESS PDF
# ============================================================

if file is not None:

    # --------------------------------------------------------
    # Read PDF
    # --------------------------------------------------------

    pdf_reader = PdfReader(file)

    text = ""

    for page in pdf_reader.pages:

        page_text = page.extract_text()

        if page_text:
            text += page_text + "\n"


    # --------------------------------------------------------
    # Check extracted text
    # --------------------------------------------------------

    if not text.strip():

        st.error(
            "No readable text was found in this PDF. "
            "If this is a scanned PDF, OCR may be required."
        )

        st.stop()


    # --------------------------------------------------------
    # Split text into chunks
    # --------------------------------------------------------

    text_splitter = RecursiveCharacterTextSplitter(
        separators=["\n\n", "\n", " ", ""],
        chunk_size=1000,
        chunk_overlap=150,
        length_function=len
    )

    chunks = text_splitter.split_text(text)


    # --------------------------------------------------------
    # Display PDF information
    # --------------------------------------------------------

    st.success("PDF uploaded successfully.")

    col1, col2 = st.columns(2)

    with col1:
        st.metric(
            "Pages",
            len(pdf_reader.pages)
        )

    with col2:
        st.metric(
            "Text Chunks",
            len(chunks)
        )


    # ========================================================
    # CREATE OPENAI EMBEDDINGS
    # ========================================================

    try:

        embeddings = OpenAIEmbeddings(
            api_key=OPENAI_API_KEY,
            model="text-embedding-3-small"
        )

    except Exception as e:

        st.error("Error creating OpenAI embeddings.")
        st.exception(e)
        st.stop()


    # ========================================================
    # CREATE FAISS VECTOR STORE
    # ========================================================

    try:

        vector_store = FAISS.from_texts(
            chunks,
            embedding=embeddings
        )

    except Exception as e:

        st.error("Error creating FAISS vector database.")
        st.exception(e)
        st.stop()


    # ========================================================
    # USER QUESTION
    # ========================================================

    user_question = st.text_input(
        "💬 Type your question here"
    )


    # ========================================================
    # QUESTION ANSWERING
    # ========================================================

    if user_question:

        # ----------------------------------------------------
        # Similarity Search
        # ----------------------------------------------------

        try:

            documents = vector_store.similarity_search(
                user_question,
                k=4
            )

        except Exception as e:

            st.error("Error while searching the document.")
            st.exception(e)
            st.stop()


        # ----------------------------------------------------
        # Create ChatGPT model
        # ----------------------------------------------------

        try:

            llm = ChatOpenAI(
                api_key=OPENAI_API_KEY,
                model="gpt-4o-mini",
                temperature=0,
                max_tokens=1000
            )

        except Exception as e:

            st.error("Error creating OpenAI model.")
            st.exception(e)
            st.stop()


        # ====================================================
        # PROMPT
        # ====================================================

        prompt = ChatPromptTemplate.from_template(
            """
You are a helpful PDF question-answering assistant.

Answer the user's question using ONLY the information
contained in the provided PDF context.

If the answer cannot be found in the PDF, say:

"I could not find the answer in the uploaded PDF."

Do not invent information.

Keep the answer clear and concise.

Context:
{context}

Question:
{input}

Answer:
"""
        )


        # ====================================================
        # CREATE DOCUMENT CHAIN
        # ====================================================

        try:

            document_chain = create_stuff_documents_chain(
                llm,
                prompt
            )

        except Exception as e:

            st.error("Error creating document chain.")
            st.exception(e)
            st.stop()


        # ====================================================
        # GENERATE ANSWER
        # ====================================================

        try:

            response = document_chain.invoke(
                {
                    "input": user_question,
                    "context": documents
                }
            )

            st.subheader("🤖 Answer")

            st.write(response)

        except Exception as e:

            st.error("Error generating answer.")
            st.exception(e)


        # ====================================================
        # SHOW SOURCES
        # ====================================================

        with st.expander("📖 View Retrieved PDF Content"):

            for i, document in enumerate(documents):

                st.markdown(
                    f"### Source {i + 1}"
                )

                st.write(document.page_content)
```
