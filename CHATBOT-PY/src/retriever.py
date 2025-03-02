import os
from dotenv import load_dotenv
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_chroma import Chroma

# Load environment variables from .env file
load_dotenv()

try:
    # Initialize HuggingFace embeddings
    # Using all-MiniLM-L6-v2 which has 384 dimensions
    embeddings = HuggingFaceEmbeddings(
        model_name="all-MiniLM-L6-v2"
    )

    # Initialize Chroma vector store
    persist_directory = "./chroma_db"
    vectorstore = Chroma(
        embedding_function=embeddings,
        persist_directory=persist_directory
    )

    retriever = vectorstore.as_retriever()
except Exception as e:
    print(f"An error occurred: {e}")