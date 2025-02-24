import streamlit as st
import pinecone
import os
from langchain_community.text_splitter import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings
from langchain.vectorstores import Pinecone as PineconeVectorStore
from langchain.chat_models import ChatOpenAI
from dotenv import load_dotenv

# Load environment variables from Streamlit secrets
load_dotenv()

OPENAI_API_KEY = st.secrets["OPENAI_API_KEY"]
PINECONE_API_KEY = st.secrets["PINECONE_API_KEY"]
PINECONE_ENV = st.secrets["PINECONE_ENV"]
PINECONE_INDEX = st.secrets["PINECONE_INDEX"]

# Initialize Pinecone
pinecone.init(api_key=PINECONE_API_KEY, environment=PINECONE_ENV)

# Initialize OpenAI embeddings
embeddings_model = OpenAIEmbeddings(api_key=OPENAI_API_KEY)

# Streamlit UI
st.title("📌 AI Helpdesk Bot")

# User Input
user_input = st.text_input("Ask me something:", "")

# Function to Embed and Store Data in Pinecone
def embed_and_store(texts):
    index = pinecone.Index(PINECONE_INDEX)
    
    # Split text into chunks
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=100)
    chunks = text_splitter.split_text("\n".join(texts))
    
    # Convert chunks to embeddings
    vectordb = PineconeVectorStore.from_texts(chunks, embeddings_model, index_name=PINECONE_INDEX)
    
    return vectordb

# Chat Model Initialization
chat_model = ChatOpenAI(openai_api_key=OPENAI_API_KEY, temperature=0.5)

# Function to Get AI Response
def get_response(query):
    if not query:
        return "Please enter a valid question."
    
    response = chat_model.predict(query)
    return response

# Handle User Query
if user_input:
    response = get_response(user_input)
    st.write(f"🤖 AI: {response}")
