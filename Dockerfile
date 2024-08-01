# Dockerfile
FROM python:3.9-slim

# Set the working directory
WORKDIR /app

# Copy the current directory contents into the container

# Install the required packages
RUN pip install --upgrade pip setuptools wheel \
    && pip install streamlit python-dotenv langchain langchain-openai langchain_google_genai langchain-community faiss-cpu

# Expose the Streamlit port
EXPOSE 8501
