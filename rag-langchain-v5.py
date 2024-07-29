# python -m venv venv
# source venv/bin/activate  # On Windows, use `venv\Scripts\activate`
# streamlit run filename.py
# pip install python-dotenv streamlit langchain langchain-openai langchain-community faiss-cpu

import os
from dotenv import load_dotenv
import streamlit as st
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.embeddings import OpenAIEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_openai import ChatOpenAI
from langchain.chains import RetrievalQA
from langchain_community.document_loaders import TextLoader
from langchain.prompts import PromptTemplate
from langchain.callbacks import get_openai_callback

# Load environment variables
load_dotenv()

# Set up OpenAI API key
os.environ["OPENAI_API_KEY"] = os.getenv("OPENAI_API_KEY")

# Get models and system prompt from environment variables
COMPLETION_MODEL = os.getenv("COMPLETION_MODEL")
EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL")
SYSTEM_PROMPT = os.getenv("SYSTEM_PROMPT")


# Get authentication credentials from environment variables
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
        chunk_size=1000,
        chunk_overlap=200,
        length_function=len
    )
    splits = text_splitter.split_documents(documents)
    return splits

def create_vector_store(splits):
    embeddings = OpenAIEmbeddings(model=EMBEDDING_MODEL)
    vectorstore = FAISS.from_documents(splits, embeddings)
    return vectorstore

def setup_qa_chain(vectorstore):
    llm = ChatOpenAI(model_name=COMPLETION_MODEL, temperature=0)
    
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
    st.title("IDI Transcript Analysis QA System")
    
    if check_password():
        # Sidebar
        st.sidebar.header("Information")
        st.sidebar.info(f"Using OpenAI models:  \nCompletion: {COMPLETION_MODEL}  \nEmbedding: {EMBEDDING_MODEL}")
        
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
            
            # Initialize chat history and token usage
            if 'messages' not in st.session_state:
                st.session_state.messages = []
            if 'total_tokens' not in st.session_state:
                st.session_state.total_tokens = 0
            if 'total_cost' not in st.session_state:
                st.session_state.total_cost = 0

            # Display chat messages from history on app rerun
            for message in st.session_state.messages:
                with st.chat_message(message["role"]):
                    st.markdown(message["content"])

            # Accept user input
            if prompt := st.chat_input("What would you like to know about the IDI transcripts?"):
                # Add user message to chat history
                st.session_state.messages.append({"role": "user", "content": prompt})
                # Display user message in chat message container
                with st.chat_message("user"):
                    st.markdown(prompt)
                
                # Generate assistant response
                with st.chat_message("assistant"):
                    message_placeholder = st.empty()
                    with st.spinner("Generating answer..."):
                        with get_openai_callback() as cb:
                            result = st.session_state.qa_chain({"query": prompt})
                            response = result["result"]
                            message_placeholder.markdown(response)
                        
                        # Update and display token usage and cost
                        st.session_state.total_tokens += cb.total_tokens
                        st.session_state.total_cost += cb.total_cost
                        
                    # Display sources
                    st.write("***Sources:***")
                    for doc in result["source_documents"]:
                        st.write(f"- {os.path.basename(doc.metadata['source'])}")

                    # Display token usage and cost after sources with Markdown styling
                    st.markdown(
                        f"""
                        ```
                        Tokens used: {cb.total_tokens} (Session: {st.session_state.total_tokens})
                        Cost: ${cb.total_cost:.4f} (Session: ${st.session_state.total_cost:.4f})
                        ```
                        """,
                        unsafe_allow_html=True
                    )
                
                # Add assistant response to chat history
                st.session_state.messages.append({"role": "assistant", "content": response})

        else:
            st.write("No documents found in rag-storage/. Please ensure your files are uploaded to this directory.")

if __name__ == "__main__":
    main()