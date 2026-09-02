# ============================================================
# PDF + CSV RAG QUESTION ANSWERING APPLICATION
# Streamlit + LangChain + Chroma + Ollama
# ============================================================

import os
import tempfile
import shutil

import streamlit as st

from langchain_community.document_loaders import (
    PyPDFLoader,
    CSVLoader
)

from langchain_text_splitters import RecursiveCharacterTextSplitter

from langchain_chroma import Chroma

from langchain_ollama import (
    OllamaEmbeddings,
    OllamaLLM
)

from langchain_core.prompts import ChatPromptTemplate

from langchain.chains.combine_documents import (
    create_stuff_documents_chain
)


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

st.title("📚 PDF & CSV Question-Answering RAG App")

st.write(
    "Upload a PDF or CSV file and ask questions based on its content."
)


# ============================================================
# OLLAMA CONFIGURATION
# ============================================================

EMBEDDING_MODEL = "nomic-embed-text"
LLM_MODEL = "llama3.1"


# ============================================================
# CREATE EMBEDDINGS
# ============================================================

@st.cache_resource
def load_embeddings():

    embeddings = OllamaEmbeddings(
        model=EMBEDDING_MODEL
    )

    return embeddings


embeddings = load_embeddings()


# ============================================================
# LOAD LLM
# ============================================================

@st.cache_resource
def load_llm():

    llm = OllamaLLM(
        model=LLM_MODEL
    )

    return llm


llm = load_llm()


# ============================================================
# LOAD PDF FILE
# ============================================================

def extract_text_from_pdf(file_path):

    loader = PyPDFLoader(file_path)

    documents = loader.load()

    return documents


# ============================================================
# LOAD CSV FILE
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

def load_document(file_path, file_type):

    if file_type == "pdf":

        documents = extract_text_from_pdf(
            file_path
        )

    elif file_type == "csv":

        documents = extract_text_from_csv(
            file_path
        )

    else:

        documents = []

    return documents


# ============================================================
# SPLIT DOCUMENTS INTO CHUNKS
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
# GENERATE RESPONSE
# ============================================================

def generate_response(
    vector_store,
    query
):

    # --------------------------------------------------------
    # PROMPT
    # --------------------------------------------------------

    prompt = ChatPromptTemplate.from_template(
        """
You are a helpful document question-answering assistant.

Answer the user's question ONLY using the information
contained in the provided context.

If the answer cannot be found in the context,
say:

"I could not find the answer in the uploaded document."

Do not make up information.

Keep the answer clear and easy to understand.

<context>
{context}
</context>

Question:
{input}

Answer:
"""
    )


    # --------------------------------------------------------
    # CREATE DOCUMENT CHAIN
    # --------------------------------------------------------

    document_chain = create_stuff_documents_chain(
        llm,
        prompt
    )


    # --------------------------------------------------------
    # SIMILARITY SEARCH
    # --------------------------------------------------------

    matching_docs = vector_store.similarity_search(
        query,
        k=5
    )


    # --------------------------------------------------------
    # GENERATE ANSWER
    # --------------------------------------------------------

    response = document_chain.invoke(
        {
            "input": query,
            "context": matching_docs
        }
    )


    return response, matching_docs


# ============================================================
# FILE UPLOADER
# ============================================================

uploaded_file = st.file_uploader(
    "📁 Upload PDF or CSV file",
    type=["pdf", "csv"]
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
    # SHOW FILE INFORMATION
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
            "Unsupported file format."
        )

        st.stop()


    # --------------------------------------------------------
    # SAVE FILE TEMPORARILY
    # --------------------------------------------------------

    temp_file_path = None

    try:

        with tempfile.NamedTemporaryFile(
            delete=False,
            suffix=file_extension
        ) as temp_file:

            temp_file.write(
                uploaded_file.getbuffer()
            )

            temp_file_path = temp_file.name


        # ----------------------------------------------------
        # PROCESSING MESSAGE
        # ----------------------------------------------------

        with st.spinner(
            "📖 Reading uploaded document..."
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
                "No readable content was found in the file."
            )

            st.stop()


        st.success(
            f"Successfully loaded {len(documents)} document sections."
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
        # CREATE VECTOR DATABASE
        # ----------------------------------------------------

        with st.spinner(
            "🧠 Creating vector database..."
        ):

            vector_store = create_vector_store(
                chunks
            )


        st.success(
            "Vector database created successfully."
        )


        # ====================================================
        # QUESTION SECTION
        # ====================================================

        st.subheader(
            "💬 Ask a Question"
        )


        query = st.text_input(
            "Enter your question:",
            placeholder="Example: What is this document about?"
        )


        # ----------------------------------------------------
        # ASK QUESTION
        # ----------------------------------------------------

        if query:

            with st.spinner(
                "🤖 Searching document and generating answer..."
            ):

                answer, matching_docs = generate_response(
                    vector_store,
                    query
                )


            # ------------------------------------------------
            # DISPLAY ANSWER
            # ------------------------------------------------

            st.subheader(
                "🤖 Answer"
            )

            st.write(
                answer
            )


            # ------------------------------------------------
            # SHOW SOURCES
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


                    # ----------------------------------------
                    # PDF PAGE NUMBER
                    # ----------------------------------------

                    if "page" in doc.metadata:

                        st.caption(
                            f"Page: {doc.metadata['page'] + 1}"
                        )


    except Exception as e:

        st.error(
            "An error occurred while processing the file."
        )

        st.exception(e)


    finally:

        # ----------------------------------------------------
        # DELETE TEMPORARY FILE
        # ----------------------------------------------------

        if (
            temp_file_path is not None
            and os.path.exists(temp_file_path)
        ):

            os.unlink(
                temp_file_path
            )


# ============================================================
# NO FILE MESSAGE
# ============================================================

else:

    st.warning(
        "Please upload a PDF or CSV file to start."
    )


# ============================================================
# SIDEBAR
# ============================================================

with st.sidebar:

    st.header(
        "⚙️ Configuration"
    )

    st.write(
        f"Embedding Model: `{EMBEDDING_MODEL}`"
    )

    st.write(
        f"LLM Model: `{LLM_MODEL}`"
    )

    st.divider()

    st.write(
        "Supported files:"
    )

    st.write(
        "📄 PDF"
    )

    st.write(
        "📊 CSV"
    )

    st.divider()

    st.write(
        "Powered by LangChain + Chroma + Ollama"
    )
