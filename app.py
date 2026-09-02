# ============================================================
# PDF + CSV RAG QUESTION ANSWERING APPLICATION
# Streamlit + LangChain + Chroma + DeepSeek
# ============================================================

import os
import tempfile

import streamlit as st

from langchain_community.document_loaders import (
    PyPDFLoader,
    CSVLoader
)

from langchain_text_splitters import (
    RecursiveCharacterTextSplitter
)

from langchain_chroma import Chroma

from langchain_openai import ChatOpenAI

from langchain_core.messages import HumanMessage


# ============================================================
# PAGE CONFIGURATION
# ============================================================

st.set_page_config(
    page_title="PDF & CSV RAG Assistant",
    page_icon="📚",
    layout="wide"
)


# ============================================================
# PAGE TITLE
# ============================================================

st.title("📚 PDF & CSV Question-Answering RAG")

st.write(
    "Upload a PDF or CSV file and ask questions based on "
    "the uploaded document."
)


# ============================================================
# DEEPSEEK CONFIGURATION
# ============================================================

DEEPSEEK_API_KEY = st.secrets["DEEPSEEK_API_KEY"]

DEEPSEEK_BASE_URL = st.secrets.get(
    "DEEPSEEK_BASE_URL",
    "https://api.deepseek.com"
)

DEEPSEEK_MODEL = "DeepSeek-V4-Flash-0731-bucket"


# ============================================================
# EMBEDDING MODEL
# ============================================================

EMBEDDING_MODEL = (
    "sentence-transformers/all-MiniLM-L6-v2"
)


# ============================================================
# LOAD EMBEDDINGS
# ============================================================

@st.cache_resource
def load_embeddings():

    from langchain_huggingface import (
        HuggingFaceEmbeddings
    )

    embeddings = HuggingFaceEmbeddings(
        model_name=EMBEDDING_MODEL
    )

    return embeddings


embeddings = load_embeddings()


# ============================================================
# LOAD DEEPSEEK
# ============================================================

@st.cache_resource
def load_llm():

    llm = ChatOpenAI(
        model=DEEPSEEK_MODEL,
        api_key=DEEPSEEK_API_KEY,
        base_url=DEEPSEEK_BASE_URL,
        temperature=0
    )

    return llm


llm = load_llm()


# ============================================================
# PDF LOADER
# ============================================================

def extract_text_from_pdf(file_path):

    loader = PyPDFLoader(
        file_path
    )

    documents = loader.load()

    return documents


# ============================================================
# CSV LOADER
# ============================================================

def extract_text_from_csv(file_path):

    loader = CSVLoader(
        file_path=file_path,
        encoding="utf-8"
    )

    documents = loader.load()

    return documents


# ============================================================
# LOAD DOCUMENT
# ============================================================

def load_document(
    file_path,
    file_type
):

    if file_type == "pdf":

        return extract_text_from_pdf(
            file_path
        )

    elif file_type == "csv":

        return extract_text_from_csv(
            file_path
        )

    return []


# ============================================================
# TEXT CHUNKING
# ============================================================

def split_documents(
    documents,
    chunk_size=1000,
    chunk_overlap=200
):

    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap
    )

    chunks = text_splitter.split_documents(
        documents
    )

    return chunks


# ============================================================
# CREATE VECTOR STORE
# ============================================================

def create_vector_store(chunks):

    vector_store = Chroma.from_documents(
        documents=chunks,
        embedding=embeddings
    )

    return vector_store


# ============================================================
# GENERATE DEEPSEEK RESPONSE
# ============================================================

def generate_response(
    vector_store,
    query
):

    # --------------------------------------------------------
    # SEARCH VECTOR DATABASE
    # --------------------------------------------------------

    matching_docs = vector_store.similarity_search(
        query,
        k=5
    )


    # --------------------------------------------------------
    # CREATE CONTEXT
    # --------------------------------------------------------

    context_parts = []

    for index, doc in enumerate(
        matching_docs,
        start=1
    ):

        context_parts.append(
            f"""
SOURCE {index}:

{doc.page_content}
"""
        )


    context = "\n".join(
        context_parts
    )


    # --------------------------------------------------------
    # PROMPT
    # --------------------------------------------------------

    prompt = f"""
You are an intelligent document question-answering assistant.

Answer the user's question ONLY using the information
contained in the provided context.

IMPORTANT RULES:

1. Do not invent information.
2. Do not use outside knowledge.
3. If the answer is not available in the context, say:
   "I could not find the answer in the uploaded document."
4. Give a clear and accurate answer.
5. Use bullet points when appropriate.
6. Preserve numbers, dates, names and values accurately.

============================================================
DOCUMENT CONTEXT
============================================================

{context}

============================================================
USER QUESTION
============================================================

{query}

============================================================
ANSWER
============================================================
"""


    # --------------------------------------------------------
    # SEND TO DEEPSEEK
    # --------------------------------------------------------

    response = llm.invoke(
        [
            HumanMessage(
                content=prompt
            )
        ]
    )


    # --------------------------------------------------------
    # RETURN ANSWER
    # --------------------------------------------------------

    return response.content, matching_docs


# ============================================================
# FILE UPLOADER
# ============================================================

uploaded_file = st.file_uploader(
    "📁 Upload PDF or CSV",
    type=[
        "pdf",
        "csv"
    ]
)


# ============================================================
# PROCESS FILE
# ============================================================

if uploaded_file is not None:

    file_name = uploaded_file.name

    file_extension = os.path.splitext(
        file_name
    )[1].lower()


    # --------------------------------------------------------
    # DISPLAY FILE
    # --------------------------------------------------------

    st.success(
        f"Uploaded file: {file_name}"
    )


    # --------------------------------------------------------
    # DETERMINE FILE TYPE
    # --------------------------------------------------------

    if file_extension == ".pdf":

        file_type = "pdf"

    elif file_extension == ".csv":

        file_type = "csv"

    else:

        st.error(
            "Unsupported file type."
        )

        st.stop()


    # --------------------------------------------------------
    # TEMPORARY FILE
    # --------------------------------------------------------

    temp_file_path = None


    try:

        # ----------------------------------------------------
        # SAVE FILE
        # ----------------------------------------------------

        with tempfile.NamedTemporaryFile(
            delete=False,
            suffix=file_extension
        ) as temp_file:

            temp_file.write(
                uploaded_file.getbuffer()
            )

            temp_file_path = temp_file.name


        # ----------------------------------------------------
        # LOAD DOCUMENT
        # ----------------------------------------------------

        with st.spinner(
            "📖 Reading document..."
        ):

            documents = load_document(
                temp_file_path,
                file_type
            )


        # ----------------------------------------------------
        # CHECK DOCUMENT
        # ----------------------------------------------------

        if not documents:

            st.error(
                "No readable content was found."
            )

            st.stop()


        st.success(
            f"Successfully loaded "
            f"{len(documents)} document sections."
        )


        # ----------------------------------------------------
        # SPLIT DOCUMENT
        # ----------------------------------------------------

        with st.spinner(
            "✂️ Splitting document into chunks..."
        ):

            chunks = split_documents(
                documents
            )


        st.info(
            f"Created {len(chunks)} text chunks."
        )


        # ----------------------------------------------------
        # VECTOR DATABASE
        # ----------------------------------------------------

        with st.spinner(
            "🧠 Creating vector database..."
        ):

            vector_store = create_vector_store(
                chunks
            )


        st.success(
            "✅ Vector database created successfully."
        )


        # ====================================================
        # QUESTION
        # ====================================================

        st.subheader(
            "💬 Ask a Question"
        )


        query = st.text_input(
            "Enter your question:",
            placeholder=(
                "Example: What is this document about?"
            )
        )


        # ====================================================
        # ASK DEEPSEEK
        # ====================================================

        if query:

            with st.spinner(
                "🤖 DeepSeek is analyzing the document..."
            ):

                answer, matching_docs = generate_response(
                    vector_store,
                    query
                )


            # ------------------------------------------------
            # ANSWER
            # ------------------------------------------------

            st.subheader(
                "🤖 DeepSeek Answer"
            )

            st.write(
                answer
            )


            # ------------------------------------------------
            # SOURCES
            # ------------------------------------------------

            with st.expander(
                "📄 View Retrieved Sources"
            ):

                for index, doc in enumerate(
                    matching_docs,
                    start=1
                ):

                    st.markdown(
                        f"### Source {index}"
                    )

                    st.write(
                        doc.page_content
                    )


                    if "page" in doc.metadata:

                        st.caption(
                            "PDF Page: "
                            + str(
                                doc.metadata["page"] + 1
                            )
                        )


    except Exception as e:

        st.error(
            "❌ An error occurred while processing the file."
        )

        st.exception(e)


    finally:

        # ----------------------------------------------------
        # DELETE TEMP FILE
        # ----------------------------------------------------

        if (
            temp_file_path
            and os.path.exists(temp_file_path)
        ):

            os.unlink(
                temp_file_path
            )


# ============================================================
# NO FILE
# ============================================================

else:

    st.info(
        "👆 Please upload a PDF or CSV file to start."
    )


# ============================================================
# SIDEBAR
# ============================================================

with st.sidebar:

    st.header(
        "⚙️ Configuration"
    )

    st.write(
        "### AI Model"
    )

    st.code(
        DEEPSEEK_MODEL
    )

    st.write(
        "### Embedding Model"
    )

    st.code(
        EMBEDDING_MODEL
    )

    st.divider()

    st.write(
        "### Supported Files"
    )

    st.write(
        "📄 PDF"
    )

    st.write(
        "📊 CSV"
    )

    st.divider()

    st.caption(
        "LangChain + Chroma + DeepSeek"
    )
