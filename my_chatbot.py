import streamlit as st
from PyPDF2 import PdfReader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document

# Get API key from Streamlit secrets
OPENAI_API_KEY = st.secrets["OPENAI_API_KEY"]

st.set_page_config(page_title="My PDF Chatbot", page_icon="📄")
st.header("📄 My PDF Chatbot")

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

    # Create embeddings + vector store
    embeddings = OpenAIEmbeddings(api_key=OPENAI_API_KEY)
    vector_store = FAISS.from_texts(chunks, embeddings)

    # User question
    user_question = st.text_input("Type your question here")

    if user_question:
        with st.spinner("Thinking..."):
            # 1. Retrieve relevant chunks
            docs = vector_store.similarity_search(user_question, k=4)

            # 2. Build context
            context = "\n\n".join([doc.page_content for doc in docs])

            # 3. Call the LLM directly (no chains → no Python 3.14 error)
            llm = ChatOpenAI(
                api_key=OPENAI_API_KEY,
                temperature=0,
                max_tokens=1000,
                model="gpt-3.5-turbo"   # or "gpt-4o-mini"
            )

            prompt = f"""Answer the question based only on the following context. 
If the answer is not in the context, say "I don't know based on the provided document."

Context:
{context}

Question: {user_question}

Answer:"""

            response = llm.invoke(prompt)
            st.write(response.content)
