import os
from dotenv import load_dotenv
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_chroma import Chroma

# Load environment variables from .env file
load_dotenv()

try:
    # Initialize HuggingFace embeddings
    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
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