import streamlit as st
from dotenv import load_dotenv
import os
from pinecone import Pinecone
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_openai.embeddings import OpenAIEmbeddings
from langchain_pinecone.vectorstores import Pinecone as PineconeVectorStore
from langchain_openai import ChatOpenAI
from langchain.chains import RetrievalQA
from langchain.document_loaders import PyPDFLoader

# Load environment variables (if running locally)
load_dotenv()

# Streamlit UI
st.title("AI Helpdesk Bot with Pinecone & Streamlit")

# Load API Keys from Streamlit Secrets
PINECONE_API_KEY = st.secrets["PINECONE_API_KEY"]
PINECONE_ENV = st.secrets["PINECONE_ENV"]
OPENAI_API_KEY = st.secrets["OPENAI_API_KEY"]
INDEX_NAME = "helpdesk-index"  # Change as per your Pinecone setup

# Initialize Pinecone
pc = Pinecone(api_key=PINECONE_API_KEY, environment=PINECONE_ENV)

# Ensure the index exists
if INDEX_NAME not in pc.list_indexes():
    st.error(f"Pinecone index '{INDEX_NAME}' does not exist. Create it first.")
    st.stop()

# Initialize Pinecone VectorStore
embeddings_model = OpenAIEmbeddings(openai_api_key=OPENAI_API_KEY)
vectordb = PineconeVectorStore.from_existing_index(index_name=INDEX_NAME, embedding=embeddings_model)

# File uploader for PDFs
uploaded_file = st.file_uploader("Upload a PDF", type=["pdf"])

if uploaded_file:
    loader = PyPDFLoader(uploaded_file)
    pages = loader.load()

    # Split text for embedding
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
    docs = text_splitter.split_documents(pages)

    # Store embeddings in Pinecone
    vectordb = PineconeVectorStore.from_texts(
        [doc.page_content for doc in docs],
        embeddings_model,
        index_name=INDEX_NAME
    )
    st.success("PDF uploaded and indexed successfully!")

# Chat input
query = st.text_input("Ask a question:")

if query:
    # Load LLM for answering queries
    llm = ChatOpenAI(openai_api_key=OPENAI_API_KEY, model_name="gpt-4", temperature=0.2)
    qa = RetrievalQA.from_chain_type(llm, retriever=vectordb.as_retriever())

    # Get answer
    answer = qa.run(query)
    st.write("### Answer:")
    st.write(answer)
