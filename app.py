```python
import os
import streamlit as st

from PyPDF2 import PdfReader

from langchain_text_splitters import RecursiveCharacterTextSplitter

from langchain_openai import OpenAIEmbeddings, ChatOpenAI

from langchain_community.vectorstores import FAISS

from langchain_core.prompts import ChatPromptTemplate

from langchain.chains.combine_documents import (
    create_stuff_documents_chain
)


# ============================================================
# PAGE CONFIGURATION
# ============================================================

st.set_page_config(
    page_title="PDF Chatbot",
    page_icon="📚",
    layout="wide"
)


# ============================================================
# CUSTOM CSS
# ============================================================

st.markdown(
    """
    <style>

    .main-title {
        font-size: 42px;
        font-weight: 700;
        text-align: center;
        margin-bottom: 10px;
    }

    .subtitle {
        text-align: center;
        font-size: 18px;
        color: #666666;
        margin-bottom: 30px;
    }

    .answer-box {
        padding: 20px;
        border-radius: 15px;
        background-color: #f5f7fb;
        border: 1px solid #dddddd;
        margin-top: 20px;
    }

    </style>
    """,
    unsafe_allow_html=True
)


# ============================================================
# HEADER
# ============================================================

st.markdown(
    '<div class="main-title">📚 PDF Chatbot</div>',
    unsafe_allow_html=True
)

st.markdown(
    '<div class="subtitle">Upload your PDF and ask questions from the document</div>',
    unsafe_allow_html=True
)


# ============================================================
# OPENAI API KEY
# ============================================================

# IMPORTANT:
#
# DO NOT put your OpenAI API key directly in this file.
#
# Streamlit Cloud:
#
# Manage App
#     ↓
# Settings
#     ↓
# Secrets
#
# Add:
#
# OPENAI_API_KEY = "your-new-api-key"
#
# ============================================================

OPENAI_API_KEY = st.secrets.get(
    "OPENAI_API_KEY",
    os.getenv("OPENAI_API_KEY")
)


# ============================================================
# CHECK API KEY
# ============================================================

if not OPENAI_API_KEY:

    st.error(
        "❌ OpenAI API key is missing."
    )

    st.info(
        "Go to Streamlit Cloud → Manage App → Settings → Secrets "
        "and add OPENAI_API_KEY."
    )

    st.stop()


# ============================================================
# SIDEBAR
# ============================================================

with st.sidebar:

    st.title("📄 Your Documents")

    st.write(
        "Upload a PDF document to start chatting."
    )

    uploaded_file = st.file_uploader(
        "Upload PDF",
        type=["pdf"]
    )

    st.divider()

    st.info(
        "Supported format: PDF"
    )


# ============================================================
# PROCESS PDF
# ============================================================

if uploaded_file is not None:

    # ========================================================
    # READ PDF
    # ========================================================

    try:

        pdf_reader = PdfReader(
            uploaded_file
        )

    except Exception as e:

        st.error(
            "❌ Unable to read the PDF."
        )

        st.exception(e)

        st.stop()


    # ========================================================
    # EXTRACT TEXT
    # ========================================================

    text = ""

    for page_number, page in enumerate(
        pdf_reader.pages,
        start=1
    ):

        try:

            page_text = page.extract_text()

            if page_text:

                text += page_text + "\n"

        except Exception:

            continue


    # ========================================================
    # CHECK TEXT
    # ========================================================

    if not text.strip():

        st.error(
            "❌ No readable text was found in this PDF."
        )

        st.warning(
            "This may be a scanned/image-only PDF. "
            "OCR is required for scanned documents."
        )

        st.stop()


    # ========================================================
    # PDF INFORMATION
    # ========================================================

    st.success(
        "✅ PDF uploaded successfully!"
    )

    col1, col2, col3 = st.columns(3)

    with col1:

        st.metric(
            "📄 Pages",
            len(pdf_reader.pages)
        )

    with col2:

        st.metric(
            "📝 Characters",
            len(text)
        )

    # ========================================================
    # TEXT SPLITTER
    # ========================================================

    text_splitter = RecursiveCharacterTextSplitter(

        separators=[
            "\n\n",
            "\n",
            " ",
            ""
        ],

        chunk_size=1000,

        chunk_overlap=150,

        length_function=len
    )


    chunks = text_splitter.split_text(
        text
    )


    with col3:

        st.metric(
            "🧩 Chunks",
            len(chunks)
        )


    # ========================================================
    # CREATE EMBEDDINGS
    # ========================================================

    try:

        with st.spinner(
            "🔄 Creating document embeddings..."
        ):

            embeddings = OpenAIEmbeddings(

                api_key=OPENAI_API_KEY,

                model="text-embedding-3-small"
            )

    except Exception as e:

        st.error(
            "❌ Error creating OpenAI embeddings."
        )

        st.exception(e)

        st.stop()


    # ========================================================
    # CREATE FAISS VECTOR DATABASE
    # ========================================================

    try:

        with st.spinner(
            "🔄 Creating FAISS vector database..."
        ):

            vector_store = FAISS.from_texts(

                chunks,

                embedding=embeddings
            )

    except Exception as e:

        st.error(
            "❌ Error creating FAISS vector database."
        )

        st.exception(e)

        st.stop()


    # ========================================================
    # QUESTION INPUT
    # ========================================================

    st.divider()

    st.subheader(
        "💬 Ask a Question"
    )

    user_question = st.text_input(
        "Type your question here:",
        placeholder="Example: What is this document about?"
    )


    # ========================================================
    # QUESTION PROCESSING
    # ========================================================

    if user_question:

        # ====================================================
        # SEARCH DOCUMENT
        # ====================================================

        try:

            with st.spinner(
                "🔍 Searching the PDF..."
            ):

                documents = vector_store.similarity_search(

                    user_question,

                    k=4
                )

        except Exception as e:

            st.error(
                "❌ Error searching the document."
            )

            st.exception(e)

            st.stop()


        # ====================================================
        # CREATE CHAT MODEL
        # ====================================================

        try:

            llm = ChatOpenAI(

                api_key=OPENAI_API_KEY,

                model="gpt-4o-mini",

                temperature=0,

                max_tokens=1000
            )

        except Exception as e:

            st.error(
                "❌ Error creating OpenAI Chat model."
            )

            st.exception(e)

            st.stop()


        # ====================================================
        # PROMPT
        # ====================================================

        prompt = ChatPromptTemplate.from_template(
            """
You are an intelligent PDF question-answering assistant.

Your job is to answer the user's question using ONLY
the information contained in the provided PDF context.

Rules:

1. Do not invent information.
2. Do not use outside knowledge.
3. If the answer is not available in the PDF, say:
   "I could not find the answer in the uploaded PDF."
4. Give a clear and useful answer.
5. If possible, explain the answer in simple language.

PDF CONTEXT:

{context}

USER QUESTION:

{input}

ANSWER:
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

            st.error(
                "❌ Error creating LangChain document chain."
            )

            st.exception(e)

            st.stop()


        # ====================================================
        # GENERATE ANSWER
        # ====================================================

        try:

            with st.spinner(
                "🤖 Generating answer..."
            ):

                response = document_chain.invoke(
                    {
                        "input": user_question,
                        "context": documents
                    }
                )

        except Exception as e:

            st.error(
                "❌ Error generating answer."
            )

            st.exception(e)

            st.stop()


        # ====================================================
        # DISPLAY ANSWER
        # ====================================================

        st.subheader(
            "🤖 Answer"
        )

        st.markdown(
            f"""
            <div class="answer-box">

            {response}

            </div>
            """,
            unsafe_allow_html=True
        )


        # ====================================================
        # SHOW SOURCES
        # ====================================================

        st.divider()

        with st.expander(
            "📖 View Retrieved PDF Sources"
        ):

            for index, document in enumerate(
                documents,
                start=1
            ):

                st.markdown(
                    f"### 📄 Source {index}"
                )

                st.write(
                    document.page_content
                )

                st.divider()


# ============================================================
# NO PDF MESSAGE
# ============================================================

else:

    st.info(
        "👈 Upload a PDF from the sidebar to start."
    )

    st.markdown(
        """
        ### How to use

        1. Upload a PDF.
        2. Wait for the document to be processed.
        3. Enter your question.
        4. The chatbot searches the PDF.
        5. The AI generates an answer from the retrieved content.
        """
    )
```
