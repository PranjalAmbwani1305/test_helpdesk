import streamlit as st
import pdfplumber
import os
from pinecone import Pinecone
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.vectorstores import Pinecone as PineconeVectorStore
from langchain.chat_models import ChatOpenAI
from langchain.chains import RetrievalQA
from streamlit_chat import message
from dotenv import load_dotenv

# Load environment variables
load_dotenv()
openai_api_key = os.getenv("OPENAI_API_KEY")
pinecone_api_key = os.getenv("PINECONE_API_KEY")
pinecone_env = os.getenv("PINECONE_ENV")

# Initialize Pinecone client
pc = Pinecone(api_key=pinecone_api_key)

# Connect to the Pinecone index
index_name = "helpdesk"
index = pc.Index(index_name)

# Initialize LangChain LLM
llm_model = "gpt-3.5-turbo-0301"
llm = ChatOpenAI(temperature=0.0, model=llm_model, openai_api_key=openai_api_key)

# Function to extract text from PDF
def extract_text_from_pdf(uploaded_file):
    with pdfplumber.open(uploaded_file) as pdf:
        pages = [page.extract_text() for page in pdf.pages if page.extract_text()]
    return pages

# Function to embed and store data in Pinecone
def embed_and_store(pages):
    embeddings_model = OpenAIEmbeddings(openai_api_key=openai_api_key)
    vectordb = PineconeVectorStore.from_texts(pages, embeddings_model, index_name=index_name)
    return vectordb

# Function to check if a PDF has been processed
def has_been_processed(file_name):
    processed_files = set()
    if os.path.exists("processed_files.txt"):
        with open("processed_files.txt", "r") as file:
            processed_files = set(file.read().splitlines())
    return file_name in processed_files

# Function to mark a PDF as processed
def mark_as_processed(file_name):
    with open("processed_files.txt", "a") as file:
        file.write(file_name + "\n")

# Handle user input
def handle_enter():
    if 'retriever' in st.session_state:
        user_input = st.session_state.user_input
        if user_input:
            st.session_state.chat_history.append(("You", user_input))
            with st.spinner("Processing..."):
                try:
                    qa = RetrievalQA.from_chain_type(llm, chain_type="stuff", retriever=st.session_state.retriever)
                    bot_response = qa.run(user_input)
                    st.session_state.chat_history.append(("Bot", bot_response))
                except Exception as e:
                    st.session_state.chat_history.append(("Bot", f"Error: {e}"))
            st.session_state.user_input = ""

# Streamlit UI
def main():
    st.title("PDF Q&A Bot with Pinecone")

    # Session states
    if 'chat_history' not in st.session_state:
        st.session_state.chat_history = []
    if 'pdf_processed' not in st.session_state:
        st.session_state.pdf_processed = False
    if 'user_input' not in st.session_state:
        st.session_state.user_input = ""

    uploaded_file = st.file_uploader("Upload your PDF here", type="pdf")

    if uploaded_file:
        file_name = uploaded_file.name
        if not has_been_processed(file_name):
            with st.spinner("Processing PDF..."):
                pages = extract_text_from_pdf(uploaded_file)
                vectordb = embed_and_store(pages)
                st.session_state.retriever = vectordb.as_retriever()
                mark_as_processed(file_name)
                st.success("PDF Processed and Stored!")
                st.session_state.pdf_processed = True
        else:
            if 'retriever' not in st.session_state:
                with st.spinner("Loading existing data..."):
                    embeddings = OpenAIEmbeddings(openai_api_key=openai_api_key)
                    docsearch = PineconeVectorStore.from_existing_index(index_name, embeddings)
                    st.session_state.retriever = docsearch.as_retriever()
                st.info("PDF already processed. Using existing data.")
                st.session_state.pdf_processed = True

    if st.session_state.pdf_processed:
        for idx, (speaker, text) in enumerate(st.session_state.chat_history):
            if speaker == "Bot":
                message(text, key=f"msg-{idx}")
            else:
                message(text, is_user=True, key=f"msg-{idx}")

        st.text_input("Enter your question:", key="user_input", on_change=handle_enter)

        if st.session_state.user_input:
            handle_enter()

if __name__ == "__main__":
    main()
