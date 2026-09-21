import streamlit as st
from PyPDF2 import PdfReader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_community.vectorstores import FAISS
from langchain_core.prompts import ChatPromptTemplate
from langchain.chains.combine_documents import create_stuff_documents_chain

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

    # Embeddings + Vector store
    embeddings = OpenAIEmbeddings(api_key=OPENAI_API_KEY)
    vector_store = FAISS.from_texts(chunks, embeddings)

    # User question
    user_question = st.text_input("Type your question here")

    if user_question:
        with st.spinner("Thinking..."):
            # Retrieve relevant chunks
            docs = vector_store.similarity_search(user_question)

            # Modern replacement for load_qa_chain
            llm = ChatOpenAI(
                api_key=OPENAI_API_KEY,
                temperature=0,
                max_tokens=1000,
                model="gpt-3.5-turbo"   # or "gpt-4o-mini"
            )

            prompt = ChatPromptTemplate.from_template(
                """Answer the question based only on the following context:

{context}

Question: {input}
"""
            )

            chain = create_stuff_documents_chain(llm, prompt)
            response = chain.invoke({"context": docs, "input": user_question})

            st.write(response)
