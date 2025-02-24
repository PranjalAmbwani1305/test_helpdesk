import streamlit as st
from langchain.chat_models import ChatOpenAI
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores import Pinecone as PineconeVectorStore
from pinecone import Pinecone
from dotenv import load_dotenv

# ✅ Load API Keys from Streamlit Secrets
OPENAI_API_KEY = st.secrets["OPENAI_API_KEY"]
PINECONE_API_KEY = st.secrets["PINECONE_API_KEY"]
PINECONE_ENV = st.secrets["PINECONE_ENVIRONMENT"]
PINECONE_INDEX = st.secrets["PINECONE_INDEX"]

# ✅ Initialize OpenAI Embeddings
embeddings_model = OpenAIEmbeddings(openai_api_key=OPENAI_API_KEY)

# ✅ Initialize Pinecone
pc = Pinecone(api_key=PINECONE_API_KEY, environment=PINECONE_ENV)
index = pc.Index(PINECONE_INDEX)

# ✅ Function to Embed and Store Data
def embed_and_store(texts):
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
    split_texts = text_splitter.split_text(texts)
    
    # Store in Pinecone Vector Database
    vectordb = PineconeVectorStore.from_texts(split_texts, embeddings_model, index_name=PINECONE_INDEX)
    return vectordb

# ✅ Streamlit App UI
st.title("📌 AI Helpdesk with Pinecone & OpenAI")

# User Input
user_input = st.text_area("Enter your query:")

if st.button("Submit"):
    if user_input:
        vectordb = embed_and_store(user_input)
        st.success("Query embedded and stored successfully!")
    else:
        st.warning("Please enter some text.")

