import os
import streamlit as st
from langchain_community.document_loaders import TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings
from transformers import pipeline
from langchain_community.llms import HuggingFacePipeline

# --- Page Setup ---
st.set_page_config(page_title="RAG AI Assistant", page_icon="🤖", layout="wide")
st.title("🤖 Knowledge Base Assistant")
st.caption("Ask questions powered by your custom document context and Qwen2.5.")

DATA_PATH = "data/About_Ai.txt"

# --- Resource Caching ---
@st.cache_resource
def load_vector_store():
    """Load text, chunk it, generate embeddings, and store in FAISS index."""
    if not os.path.exists(DATA_PATH):
        # Create a dummy file if missing to prevent deployment startup crash
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

# Load heavy resources with a spinner UI feedback
with st.spinner("Initializing models and indexing documents..."):
    retriever = load_vector_store()
    llm = load_llm()

# --- RAG Logic ---
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

# --- Sidebar UI ---
with st.sidebar:
    st.header("⚙️ App Info")
    st.markdown("**LLM:** Qwen/Qwen2.5-1.5B-Instruct")
    st.markdown("**Embeddings:** all-MiniLM-L6-v2")
    st.markdown("**Vector DB:** FAISS")
    st.divider()
    if st.button("Clear Chat History"):
        st.session_state.messages = []
        st.rerun()

# --- Chat Interface ---
if "messages" not in st.session_state:
    st.session_state.messages = []

# Display previous messages
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])
        if "retrieved_docs" in msg:
            with st.expander("📄 View Retrieved Context"):
                for idx, doc in enumerate(msg["retrieved_docs"]):
                    st.markdown(f"**Chunk {idx+1}:**")
                    st.caption(doc.page_content)

# Handle user query
if user_query := st.chat_input("Ask a question about your context document..."):
    # Render user message
    st.session_state.messages.append({"role": "user", "content": user_query})
    with st.chat_message("user"):
        st.markdown(user_query)

    # Generate and render AI response
    with st.chat_message("assistant"):
        with st.spinner("Searching context & generating response..."):
            answer, retrieved_docs = get_rag_response(user_query)
            st.markdown(answer)
            with st.expander("📄 View Retrieved Context"):
                for idx, doc in enumerate(retrieved_docs):
                    st.markdown(f"**Chunk {idx+1}:**")
                    st.caption(doc.page_content)

    # Save to history
    st.session_state.messages.append({
        "role": "assistant",
        "content": answer,
        "retrieved_docs": retrieved_docs
    })
