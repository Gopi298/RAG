```python
import os
import streamlit as st
from PyPDF2 import PdfReader
from openai import OpenAI
import numpy as np
import faiss


# ============================================================
# PAGE CONFIGURATION
# ============================================================

st.set_page_config(
    page_title="PDF RAG Chatbot",
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
        text-align: center;
        font-size: 42px;
        font-weight: bold;
        margin-bottom: 5px;
    }

    .subtitle {
        text-align: center;
        font-size: 18px;
        margin-bottom: 30px;
    }

    .answer-box {
        padding: 20px;
        border-radius: 15px;
        border: 1px solid #dddddd;
        margin-top: 15px;
    }

    </style>
    """,
    unsafe_allow_html=True
)


# ============================================================
# TITLE
# ============================================================

st.markdown(
    '<div class="main-title">📚 PDF RAG Chatbot</div>',
    unsafe_allow_html=True
)

st.markdown(
    '<div class="subtitle">Upload a PDF and ask questions about it</div>',
    unsafe_allow_html=True
)


# ============================================================
# OPENAI API KEY
# ============================================================

OPENAI_API_KEY = st.secrets.get(
    "OPENAI_API_KEY",
    os.getenv("OPENAI_API_KEY")
)


# ============================================================
# CHECK API KEY
# ============================================================

if not OPENAI_API_KEY:

    st.error("❌ OPENAI_API_KEY is missing.")

    st.info(
        """
        Add your API key in:

        Streamlit Cloud
        → Manage App
        → Settings
        → Secrets

        Add:

        OPENAI_API_KEY = "your-new-api-key"
        """
    )

    st.stop()


# ============================================================
# OPENAI CLIENT
# ============================================================

client = OpenAI(
    api_key=OPENAI_API_KEY
)


# ============================================================
# SIDEBAR
# ============================================================

with st.sidebar:

    st.title("📄 Your Documents")

    uploaded_file = st.file_uploader(
        "Upload a PDF file",
        type=["pdf"]
    )

    st.divider()

    st.write(
        "Upload your PDF and then ask questions "
        "about the document."
    )


# ============================================================
# TEXT CHUNKING FUNCTION
# ============================================================

def create_chunks(
    text,
    chunk_size=1000,
    chunk_overlap=150
):

    chunks = []

    start = 0

    text_length = len(text)

    while start < text_length:

        end = start + chunk_size

        chunk = text[start:end]

        if chunk.strip():

            chunks.append(chunk)

        start = end - chunk_overlap

    return chunks


# ============================================================
# CREATE EMBEDDINGS
# ============================================================

def create_embeddings(texts):

    response = client.embeddings.create(
        model="text-embedding-3-small",
        input=texts
    )

    embeddings = [
        item.embedding
        for item in response.data
    ]

    return np.array(
        embeddings,
        dtype="float32"
    )


# ============================================================
# CREATE FAISS INDEX
# ============================================================

def create_faiss_index(embeddings):

    dimension = embeddings.shape[1]

    index = faiss.IndexFlatL2(
        dimension
    )

    index.add(
        embeddings
    )

    return index


# ============================================================
# EXTRACT PDF TEXT
# ============================================================

def extract_pdf_text(uploaded_file):

    reader = PdfReader(
        uploaded_file
    )

    text = ""

    for page in reader.pages:

        page_text = page.extract_text()

        if page_text:

            text += page_text
            text += "\n"

    return text, len(reader.pages)


# ============================================================
# PROCESS PDF
# ============================================================

if uploaded_file is not None:

    try:

        with st.spinner(
            "📖 Reading PDF..."
        ):

            text, page_count = extract_pdf_text(
                uploaded_file
            )


        # ----------------------------------------------------
        # CHECK TEXT
        # ----------------------------------------------------

        if not text.strip():

            st.error(
                "❌ No readable text was found in this PDF."
            )

            st.warning(
                "This appears to be a scanned/image-only PDF. "
                "OCR is required."
            )

            st.stop()


        # ----------------------------------------------------
        # CREATE CHUNKS
        # ----------------------------------------------------

        chunks = create_chunks(
            text,
            chunk_size=1000,
            chunk_overlap=150
        )


        # ----------------------------------------------------
        # PDF INFORMATION
        # ----------------------------------------------------

        st.success(
            "✅ PDF processed successfully!"
        )

        col1, col2, col3 = st.columns(3)

        with col1:

            st.metric(
                "📄 Pages",
                page_count
            )

        with col2:

            st.metric(
                "📝 Characters",
                len(text)
            )

        with col3:

            st.metric(
                "🧩 Chunks",
                len(chunks)
            )


        # ----------------------------------------------------
        # CREATE EMBEDDINGS
        # ----------------------------------------------------

        with st.spinner(
            "🔄 Creating embeddings..."
        ):

            embeddings = create_embeddings(
                chunks
            )


        # ----------------------------------------------------
        # CREATE FAISS INDEX
        # ----------------------------------------------------

        with st.spinner(
            "🔄 Creating FAISS vector database..."
        ):

            index = create_faiss_index(
                embeddings
            )


        st.success(
            "✅ PDF is ready for questions!"
        )


        # ====================================================
        # QUESTION
        # ====================================================

        st.divider()

        user_question = st.text_input(
            "💬 Ask a question about your PDF",
            placeholder="Example: What is this document about?"
        )


        # ====================================================
        # QUESTION PROCESSING
        # ====================================================

        if user_question:

            # ------------------------------------------------
            # QUESTION EMBEDDING
            # ------------------------------------------------

            with st.spinner(
                "🔍 Searching document..."
            ):

                question_embedding = create_embeddings(
                    [user_question]
                )


            # ------------------------------------------------
            # FAISS SEARCH
            # ------------------------------------------------

            distances, indices = index.search(
                question_embedding,
                4
            )


            # ------------------------------------------------
            # GET MATCHING CHUNKS
            # ------------------------------------------------

            retrieved_chunks = []

            for idx in indices[0]:

                if idx >= 0 and idx < len(chunks):

                    retrieved_chunks.append(
                        chunks[idx]
                    )


            # ------------------------------------------------
            # BUILD CONTEXT
            # ------------------------------------------------

            context = "\n\n".join(
                retrieved_chunks
            )


            # ------------------------------------------------
            # CREATE PROMPT
            # ------------------------------------------------

            prompt = f"""
You are a PDF question-answering assistant.

Answer the user's question using ONLY the information
provided in the PDF context below.

Rules:

1. Use only the PDF context.
2. Do not invent information.
3. Do not use outside knowledge.
4. If the answer is not present in the context,
   say exactly:

"I could not find the answer in the uploaded PDF."

5. Give a clear and concise answer.

PDF CONTEXT:
------------------------

{context}

------------------------

USER QUESTION:
{user_question}

ANSWER:
"""


            # ------------------------------------------------
            # ASK OPENAI
            # ------------------------------------------------

            try:

                with st.spinner(
                    "🤖 Generating answer..."
                ):

                    response = client.chat.completions.create(

                        model="gpt-4o-mini",

                        messages=[
                            {
                                "role": "system",
                                "content": (
                                    "You answer questions "
                                    "from PDF documents."
                                )
                            },
                            {
                                "role": "user",
                                "content": prompt
                            }
                        ],

                        temperature=0,

                        max_tokens=1000
                    )


                answer = response.choices[
                    0
                ].message.content


                # ------------------------------------------------
                # DISPLAY ANSWER
                # ------------------------------------------------

                st.subheader(
                    "🤖 Answer"
                )

                st.markdown(
                    f"""
                    <div class="answer-box">

                    {answer}

                    </div>
                    """,
                    unsafe_allow_html=True
                )


                # ------------------------------------------------
                # SHOW SOURCES
                # ------------------------------------------------

                st.divider()

                with st.expander(
                    "📖 View Retrieved PDF Sources"
                ):

                    for i, chunk in enumerate(
                        retrieved_chunks,
                        start=1
                    ):

                        st.markdown(
                            f"### Source {i}"
                        )

                        st.write(
                            chunk
                        )

                        st.divider()


            except Exception as e:

                st.error(
                    "❌ OpenAI API error."
                )

                st.exception(e)


    except Exception as e:

        st.error(
            "❌ Error processing PDF."
        )

        st.exception(e)


# ============================================================
# INITIAL MESSAGE
# ============================================================

else:

    st.info(
        "👈 Upload a PDF from the sidebar to get started."
    )

    st.markdown(
        """
        ### 🚀 How it works

        **1. Upload PDF**

        Upload your PDF document.

        **2. Extract Text**

        The application extracts text from the PDF.

        **3. Create Embeddings**

        OpenAI converts the PDF text into vectors.

        **4. FAISS Search**

        FAISS finds the most relevant parts of the PDF.

        **5. Ask Question**

        Enter your question.

        **6. AI Answer**

        GPT generates an answer using the retrieved
        PDF content.
        """
    )
```
