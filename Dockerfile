# Dockerfile
FROM python:3.9-slim

# Set the working directory
WORKDIR /app

# Copy the current directory contents into the container
#COPY . /app

# Install the required packages
RUN pip install --upgrade pip setuptools wheel \
    && pip install streamlit python-dotenv langchain langchain-openai langchain-community faiss-cpu

# Expose the Streamlit port
EXPOSE 8501

# Command to run the Streamlit app
# CMD ["streamlit", "run", "app.py"]
