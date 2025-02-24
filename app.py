import streamlit as st
import pdfplumber
import os
import pinecone
from Q_generator import generate_questions
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.vectorstores import Pinecone as PineconeStore
from langchain.chat_models import ChatOpenAI
from langchain.chains import RetrievalQA
from streamlit_chat import message
import openai
from dotenv import load_dotenv, find_dotenv

# Load environment variables
_ = load_dotenv(find_dotenv())  
openai.api_key = os.getenv('OPENAI_API_KEY')

# Initialize Pinecone
pinecone.init(api_key=os.getenv("PINECONE_API_KEY"), environment=os.getenv("PINECONE_ENV"))
index_name = "helpdesk"

# Initialize LangChain LLM
llm_model = "gpt-3.5-turbo-0301"
llm = ChatOpenAI(temperature=0.0, model=llm_model)

def extract_text_from_pdf(uploaded_file):
    with pdfplumber.open(uploaded_file) as pdf:
        pages = [page.extract_text() for page in pdf.pages if page.extract_text()]
    return pages

def embed_and_store(pages, embeddings_model):
    # Split text for better embeddings
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    split_texts = text_splitter.split_text("\n".join(pages))

    # Store embeddings in Pinecone
    docsearch = PineconeStore.from_texts(split_texts, embeddings_model, index_name=index_name)
    return docsearch

def has_been_processed(file_name):
    """Check if the PDF has already been processed."""
    processed_files = set()
    if os.path.exists("processed_files.txt"):
        with open("processed_files.txt", "r") as file:
            processed_files = set(file.read().splitlines())
    return file_name in processed_files

def mark_as_processed(file_name):
    """Mark the PDF as processed."""
    with open("processed_files.txt", "a") as file:
        file.write(file_name + "\n")

def handle_enter():
    if 'retriever' in st.session_state:
        user_input = st.session_state.user_input.strip()
        if user_input:
            st.session_state.chat_history.append(("You", user_input))
            with st.spinner("Please wait..."):
                try:
                    qa = RetrievalQA.from_chain_type(llm, chain_type="stuff", retriever=st.session_state.retriever)
                    bot_response = qa.run(user_input)
                    st.session_state.chat_history.append(("Bot", bot_response))
                except Exception as e:
                    st.session_state.chat_history.append(("Bot", f"Error - {e}"))
            st.session_state.user_input = ""  # Clear the input box after processing

def main():
    st.title("Ask a PDF Questions")

    # Initialize session state variables
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
                if pages:
                    embeddings_model = OpenAIEmbeddings(openai_api_key=openai.api_key)
                    vectordb = embed_and_store(pages, embeddings_model)
                    st.session_state.retriever = vectordb.as_retriever()
                    mark_as_processed(file_name)
                    st.success("PDF Processed and Stored!")
                    st.session_state.pdf_processed = True
                else:
                    st.error("No extractable text found in the PDF.")
        else:
            if 'retriever' not in st.session_state:
                with st.spinner("Loading existing data..."):
                    embeddings = OpenAIEmbeddings(openai_api_key=openai.api_key)
                    docsearch = PineconeStore.from_existing_index(index_name, embeddings)
                    st.session_state.retriever = docsearch.as_retriever()
                st.info("PDF already processed. Using existing data.")
                st.session_state.pdf_processed = True
    
    if st.session_state.pdf_processed:
        for idx, (speaker, text) in enumerate(st.session_state.chat_history):
            message(text, is_user=(speaker == "You"), key=f"msg-{idx}")

        st.text_input("Enter your question here:", key="user_input", on_change=handle_enter)

if __name__ == "__main__":
    main()
