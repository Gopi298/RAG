import os
import streamlit as st

# Document Loading & Text Splitting
from langchain_community.document_loaders import TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter

# Vector Stores & Embeddings (Modern Updated Imports)
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings, HuggingFacePipeline

# Transformers Pipeline
from transformers import pipeline

# --- Page Config ---
st.set_page_config(page_title="RAG AI Assistant", page_icon="🤖", layout="wide")
st.title("🤖 Knowledge Base Assistant")
st.caption("Ask questions powered by your custom document context and Qwen2.5.")

DATA_PATH = "data/About_Ai.txt"

# --- Cached Resource Loading ---
@st.cache_resource
def load_vector_store():
    """Load text, chunk it, generate embeddings, and build the FAISS vector index."""
    # Ensure data directory and file exist to prevent cold-start crashes
    if not os.path.exists(DATA_PATH):
        os.makedirs("data", exist_ok=True)
        with open(DATA_PATH, "w", encoding="utf-8") as f:
            f.write("Welcome to Karthik's Show! This is a placeholder context document about AI.")
            
    loader = TextLoader(DATA_PATH, encoding="utf-8")
    docs = loader.load()

    splitter = RecursiveCharacterTextSplitter(chunk_size=300, chunk_overlap=50)
    chunks = splitter.split_documents(docs)

    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    db = FAISS.from_documents(chunks, embeddings)
    return db.as_retriever()


@st.cache_resource
def load_llm():
    """Initialize HuggingFace text-generation pipeline."""
    pipe = pipeline(
        "text-generation",
        model="Qwen/Qwen2.5-1.5B-Instruct",
        max_new_tokens=256,
        temperature=0.2,
        do_sample=True
    )
    return HuggingFacePipeline(pipeline=pipe)

# Initialize resources once
with st.spinner("Initializing embeddings and loading model into memory..."):
    retriever = load_vector_store()
    llm = load_llm()

# --- RAG Chain Handler ---
def get_rag_response(question: str):
    docs = retriever.invoke(question)
    context = "\n\n".join([d.page_content for d in docs])

    prompt_text = f"""You are an AI assistant. Use ONLY the provided context to answer.
If the answer is not in the context, say "I don't know".

Context:
{context}

Question: {question}

Answer:"""

    raw = llm.invoke(prompt_text)
    answer = raw.replace(prompt_text, "").strip()
    return answer, docs

# --- Sidebar ---
with st.sidebar:
    st.header("⚙️ App Info")
    st.markdown("**LLM:** Qwen/Qwen2.5-1.5B-Instruct")
    st.markdown("**Embeddings:** all-MiniLM-L6-v2")
    st.markdown("**Vector Store:** FAISS")
    st.divider()
    if st.button("Clear Chat History"):
        st.session_state.messages = []
        st.rerun()

# --- Chat Interface ---
if "messages" not in st.session_state:
    st.session_state.messages = []

# Display history
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])
        if "retrieved_docs" in msg:
            with st.expander("📄 View Retrieved Context"):
                for idx, doc in enumerate(msg["retrieved_docs"]):
                    st.markdown(f"**Chunk {idx+1}:**")
                    st.caption(doc.page_content)

# Process User Input
if user_query := st.chat_input("Ask a question about your context document..."):
    st.session_state.messages.append({"role": "user", "content": user_query})
    with st.chat_message("user"):
        st.markdown(user_query)

    with st.chat_message("assistant"):
        with st.spinner("Searching knowledge base..."):
            answer, retrieved_docs = get_rag_response(user_query)
            st.markdown(answer)
            with st.expander("📄 View Retrieved Context"):
                for idx, doc in enumerate(retrieved_docs):
                    st.markdown(f"**Chunk {idx+1}:**")
                    st.caption(doc.page_content)

    st.session_state.messages.append({
        "role": "assistant",
        "content": answer,
        "retrieved_docs": retrieved_docs
    })
