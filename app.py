import streamlit as st
from langchain.chat_models import ChatOpenAI
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores import Pinecone as PineconeVectorStore
from pinecone import Pinecone
from dotenv import load_dotenv

# Load secrets (Replace .env with st.secrets in Streamlit Cloud)
OPENAI_API_KEY = st.secrets["OPENAI_API_KEY"]
PINECONE_API_KEY = st.secrets["PINECONE_API_KEY"]
PINECONE_ENV = st.secrets["PINECONE_ENVIRONMENT"]
PINECONE_INDEX = st.secrets["PINECONE_INDEX"]

# Initialize OpenAI embedding model
embeddings_model = OpenAIEmbeddings(openai_api_key=OPENAI_API_KEY)

# Initialize Pinecone
pc = Pinecone(api_key=PINECONE_API_KEY, environment=PINECONE_ENV)
index = pc.Index(PINECONE_INDEX)

# Function to split text into chunks
def split_text(text):
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=100)
    return text_splitter.split_text(text)

# Function to embed and store text in Pinecone
def embed_and_store(texts):
    chunks = split_text(texts)
    vectordb = PineconeVectorStore.from_texts(chunks, embeddings_model, index_name=PINECONE_INDEX)
    return vectordb

# Function to perform a similarity search
def query_pinecone(query, top_k=5):
    embedding = embeddings_model.embed_query(query)
    results = index.query(embedding, top_k=top_k, include_metadata=True)
    return results

# Streamlit UI
def main():
    st.title("🔍 AI-Powered Helpdesk")
    
    # Text input for user query
    user_query = st.text_input("Ask a question:")

    if user_query:
        results = query_pinecone(user_query)
        st.subheader("📌 Relevant Results:")
        for match in results['matches']:
            st.write(f"- **Score:** {match['score']}")
            st.write(f"**Text:** {match['metadata']['text']}\n")

if __name__ == "__main__":
    main()
