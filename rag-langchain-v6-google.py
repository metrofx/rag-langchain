# python -m venv venv
# source venv/bin/activate  # On Windows, use `venv\Scripts\activate`
# streamlit run filename.py
# pip install python-dotenv streamlit langchain langchain_google_genai langchain-community faiss-cpu

import os
from dotenv import load_dotenv
import streamlit as st
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_google_genai import GoogleGenerativeAIEmbeddings, ChatGoogleGenerativeAI
from langchain_community.vectorstores import FAISS
from langchain.chains import RetrievalQA
from langchain_community.document_loaders import TextLoader
from langchain.prompts import PromptTemplate
import google.generativeai as genai

# Load environment variables
load_dotenv()

# Set up Google API key
os.environ["GOOGLE_API_KEY"] = os.getenv("GOOGLE_API_KEY")

# Configure Google Generative AI
genai.configure(api_key=os.environ["GOOGLE_API_KEY"])

# Get models and system prompt from environment variables
GEMINI_MODEL = "gemini-1.5-pro"
EMBEDDING_MODEL = "models/text-embedding-004"
SYSTEM_PROMPT = os.getenv("SYSTEM_PROMPT")

# Authentication setup remains the same
AUTH_USERNAME = os.getenv("AUTH_USERNAME")
AUTH_PASSWORD = os.getenv("AUTH_PASSWORD")

def check_password():
    """Returns `True` if the user had the correct password."""

    def password_entered():
        """Checks whether a password entered by the user is correct."""
        if st.session_state["username"] == AUTH_USERNAME and st.session_state["password"] == AUTH_PASSWORD:
            st.session_state["password_correct"] = True
            del st.session_state["password"]  # don't store password
        else:
            st.session_state["password_correct"] = False

    if "password_correct" not in st.session_state:
        # First run, show inputs for username + password.
        st.text_input("Username", on_change=password_entered, key="username")
        st.text_input("Password", type="password", on_change=password_entered, key="password")
        return False
    elif not st.session_state["password_correct"]:
        # Password incorrect, show input + error.
        st.text_input("Username", on_change=password_entered, key="username")
        st.text_input("Password", type="password", on_change=password_entered, key="password")
        st.error("😕 User not known or password incorrect")
        return False
    else:
        # Password correct.
        return True

def load_documents(directory):
    documents = []
    for filename in os.listdir(directory):
        if filename.endswith('.txt'):
            file_path = os.path.join(directory, filename)
            loader = TextLoader(file_path, encoding='utf-8')
            documents.extend(loader.load())
    return documents

def split_documents(documents):
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=5200,
        chunk_overlap=520,
        length_function=len
    )
    splits = text_splitter.split_documents(documents)
    return splits

def create_vector_store(splits):
    embeddings = GoogleGenerativeAIEmbeddings(model=EMBEDDING_MODEL)
    vectorstore = FAISS.from_documents(splits, embeddings)
    return vectorstore

def setup_qa_chain(vectorstore):
    llm = ChatGoogleGenerativeAI(model=GEMINI_MODEL, temperature=0)
    
    prompt_template = f"""
    {SYSTEM_PROMPT}
    
    Human: {{question}}
    
    Assistant: Let's approach this step-by-step:
    
    1) First, I'll review the relevant information from the IDI transcripts.
    2) Then, I'll provide a clear and concise answer based on that information.
    3) If there's any ambiguity or lack of information, I'll make note of it.
    
    Here's my response:
    
    {{context}}
    
    Given this context, here's my answer:
    """
    
    PROMPT = PromptTemplate(
        template=prompt_template, input_variables=["question", "context"]
    )
    
    qa_chain = RetrievalQA.from_chain_type(
        llm=llm,
        chain_type="stuff",
        retriever=vectorstore.as_retriever(),
        return_source_documents=True,
        chain_type_kwargs={
            "prompt": PROMPT,
        }
    )
    return qa_chain

def main():
    # Set the title of the tab
    st.set_page_config(page_title="TAQS - Transcript Analysis QA System")
    st.title("Transcript Analysis Q&A System")
    
    if check_password():
        # Sidebar
        st.sidebar.image("logo.png", use_column_width=True)
        st.sidebar.header("System")
        st.sidebar.info(f"Using Google Gemini model: {GEMINI_MODEL}")
        
        # Load documents
        documents = load_documents("rag-storage/")
        
        if documents:
            # Display file names in sidebar
            st.sidebar.header("Files in rag-storage")
            for doc in documents:
                st.sidebar.write(f"- {os.path.basename(doc.metadata['source'])}")
            
            # Process documents
            if 'vectorstore' not in st.session_state:
                with st.spinner("Processing documents and creating vector store..."):
                    splits = split_documents(documents)
                    st.session_state.vectorstore = create_vector_store(splits)
                    st.session_state.qa_chain = setup_qa_chain(st.session_state.vectorstore)
                st.success("Documents processed and vector store created!")
            
            # Initialize chat history
            if 'messages' not in st.session_state:
                st.session_state.messages = []

            # Display chat messages from history on app rerun
            for message in st.session_state.messages:
                with st.chat_message(message["role"]):
                    st.markdown(message["content"])

            # Accept user input
            if prompt := st.chat_input("What would you like to know about the transcripts?"):
                # Add user message to chat history
                st.session_state.messages.append({"role": "user", "content": prompt})
                # Display user message in chat message container
                with st.chat_message("user"):
                    st.markdown(prompt)
                
                # Generate assistant response
                with st.chat_message("assistant"):
                    message_placeholder = st.empty()
                    with st.spinner("Generating answer..."):
                        result = st.session_state.qa_chain({"query": prompt})
                        response = result["result"]
                        message_placeholder.markdown(response)
                        
                    # Display sources
                    st.write("***Sources:***")
                    for doc in result["source_documents"]:
                        st.write(f"- {os.path.basename(doc.metadata['source'])}")

                # Add assistant response to chat history
                st.session_state.messages.append({"role": "assistant", "content": response})

        else:
            st.write("No documents found in rag-storage/. Please ensure your files are uploaded to this directory.")

if __name__ == "__main__":
    main()