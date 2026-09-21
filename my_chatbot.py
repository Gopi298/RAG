import streamlit as st
from PyPDF2 import PdfReader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_openai import ChatOpenAI
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings

# DeepSeek API Key from Streamlit Secrets
DEEPSEEK_API_KEY = st.secrets["DEEPSEEK_API_KEY"]

st.set_page_config(page_title="My PDF Chatbot (DeepSeek)", page_icon="📄")
st.header("📄 My PDF Chatbot (DeepSeek)")

with st.sidebar:
    st.title("Your Documents")
    file = st.file_uploader("Upload a PDF and start asking questions", type="pdf")

if file is not None:
    # Extract text
    pdf_reader = PdfReader(file)
    text = ""
    for page in pdf_reader.pages:
        extracted = page.extract_text()
        if extracted:
            text += extracted

    if not text.strip():
        st.error("Could not extract any text from this PDF.")
        st.stop()

    # Split into chunks
    text_splitter = RecursiveCharacterTextSplitter(
        separators=["\n\n", "\n", " ", ""],
        chunk_size=1000,
        chunk_overlap=150,
        length_function=len
    )
    chunks = text_splitter.split_text(text)

    # Use free local embeddings (no OpenAI needed)
    @st.cache_resource
    def get_embeddings():
        return HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

    embeddings = get_embeddings()

    # Create vector store (cached)
    @st.cache_resource
    def create_vector_store(_chunks):
        return FAISS.from_texts(_chunks, embeddings)

    vector_store = create_vector_store(tuple(chunks))

    # User question
    user_question = st.text_input("Type your question here")

    if user_question:
        with st.spinner("Thinking with DeepSeek..."):
            try:
                # Retrieve relevant chunks
                docs = vector_store.similarity_search(user_question, k=4)
                context = "\n\n".join([doc.page_content for doc in docs])

                # DeepSeek Chat model (OpenAI-compatible)
                llm = ChatOpenAI(
                    api_key=DEEPSEEK_API_KEY,
                    base_url="https://api.deepseek.com",
                    model="deepseek-chat",          # or "deepseek-flash" / "deepseek-v4-pro"
                    temperature=0,
                    max_tokens=1500
                )

                prompt = f"""Answer the question based only on the following context.
If the answer is not in the context, say "I don't know based on the provided document."

Context:
{context}

Question: {user_question}

Answer:"""

                response = llm.invoke(prompt)
                st.write(response.content)

            except Exception as e:
                st.error(f"Error: {str(e)}")
