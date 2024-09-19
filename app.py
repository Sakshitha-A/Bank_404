import streamlit as st
import os
import shutil
import time
from langchain.chains import LLMChain
from langchain_cohere import ChatCohere
from langchain.prompts import PromptTemplate
from langchain.prompts import ChatPromptTemplate
from langchain.memory import ConversationBufferMemory
from trulens_eval import TruChain
from textblob import TextBlob
from PyPDF2 import PdfReader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores import Chroma
from langchain.embeddings import OllamaEmbeddings
from langchain_community.chat_models import ChatOllama
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from langchain_google_genai.embeddings import GoogleGenerativeAIEmbeddings
from langchain_google_genai.chat_models import ChatGoogleGenerativeAI
from dotenv import load_dotenv

load_dotenv()

# CONSTANTS
DB_PATH = "db/"
MAX_RETRIES = 3
RETRY_DELAY = 1

# Load environment variables from .streamlit/secrets.toml
cohere_api_key = st.secrets["COHERE_API_KEY"]

# Function to get Cohere API for the chatbot
def get_chat_llm():
    return ChatCohere(cohere_api_key=cohere_api_key)

# Function to get embeddings for PDF document retrieval
def get_embeddings():
    return GoogleGenerativeAIEmbeddings(model="models/embedding-001")

# Function to split text into chunks
def get_text_chunks(text):
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=512, chunk_overlap=64)
    return text_splitter.split_text(text)

# Function to process PDF and extract text
def get_pdf_text(pdf_docs):
    text = ""
    for pdf in pdf_docs:
        pdf_reader = PdfReader(pdf)
        for page in pdf_reader.pages:
            text += page.extract_text()
    return text

# Function to check spelling suggestions
def suggest_spelling(input_text):
    blob = TextBlob(input_text)
    suggestions = []
    for word in blob.words:
        corrected_word = str(TextBlob(word).correct())
        if corrected_word != word:
            suggestions.append((word, corrected_word))
    return suggestions

# Create chatbot conversation chain
def create_chatbot_chain():
    template = """You are a chatbot having a conversation with a human.
    {chat_history}
    Human: {human_input}
    Chatbot:"""
    
    prompt = PromptTemplate(
        input_variables=["chat_history", "human_input"], template=template
    )
    memory = ConversationBufferMemory(memory_key="chat_history")
    llm = get_chat_llm()
    chain = LLMChain(llm=llm, prompt=prompt, memory=memory, verbose=True)
    return chain

# Function to create a conversational chain for PDF Q&A
def get_pdf_conversational_chain(retriever):
    llm = ChatGoogleGenerativeAI(model="gemini-pro")

    prompt_template = """You are a helpful assistant providing detailed, accurate, and informative responses. 
    CONTEXT: {context} QUESTION: {question} ANSWER:"""
    
    chain = (
        {"context": retriever, "question": RunnablePassthrough()}
        | ChatPromptTemplate.from_template(prompt_template)
        | llm
        | StrOutputParser()
    )
    return chain

# Create a retriever based on the uploaded PDFs
def get_pdf_retriever(text_chunks):
    embeddings = get_embeddings()
    vector_store = Chroma.from_texts(text_chunks, embedding=embeddings, persist_directory=DB_PATH)
    retriever = vector_store.as_retriever(search_kwargs={"k": 2})
    return retriever

# Function to clean and recreate the database
def delete_and_recreate_db_directory():
    try:
        shutil.rmtree(DB_PATH)
    except OSError as e:
        return False
    try:
        os.makedirs(DB_PATH)
        return True
    except OSError as e:
        return False

# Retry logic for cleaning the database
def retries():
    for attempt in range(MAX_RETRIES):
        if delete_and_recreate_db_directory():
            break
        time.sleep(RETRY_DELAY)

# Main user input handling for PDF document queries
def process_pdf_query(user_question):
    embeddings = get_embeddings()
    db = Chroma(persist_directory=DB_PATH, embedding_function=embeddings)
    retriever = db.as_retriever(search_kwargs={"k": 2})

    chain = get_pdf_conversational_chain(retriever)

    response_container = st.empty()
    response = ""
    for chunk in chain.stream(user_question):
        response += chunk
        response_container.markdown(response)

# Streamlit app layout
st.set_page_config(page_title="Multifunctional Chatbot", layout="wide")

st.sidebar.title("Navigation")
option = st.sidebar.selectbox("Choose a function", ["Chatbot", "PDF Query"])

if option == "Chatbot":
    st.title("BANK 404 BOT")
    
    chatbot_chain = create_chatbot_chain()

    if "messages" not in st.session_state:
        st.session_state.messages = []

    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    if prompt := st.chat_input("Enter your message"):
        spelling_suggestions = suggest_spelling(prompt)
        
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)

        with st.chat_message("assistant"):
            full_response = chatbot_chain.run(prompt)
            st.markdown(full_response)
        st.session_state.messages.append({"role": "assistant", "content": full_response})

elif option == "PDF Query":
    st.title("Query PDFs")

    user_question = st.text_input("Ask a Question from the PDF Files")
    if user_question:
        process_pdf_query(user_question)

    with st.sidebar:
        pdf_docs = st.file_uploader("Upload your PDF Files", accept_multiple_files=True)
        submit_button = st.button("Submit & Process")
        if submit_button:
            if pdf_docs:
                with st.spinner("Processing PDFs..."):
                    retries()
                    raw_text = get_pdf_text(pdf_docs)
                    text_chunks = get_text_chunks(raw_text)
                    retriever = get_pdf_retriever(text_chunks)
                    st.success("PDF files processed and ready for querying!")
            else:
                st.warning("Please upload PDF files first.")