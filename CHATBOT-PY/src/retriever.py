#  src/retriever.py
import os
from dotenv import load_dotenv
from langchain_chroma.vectorstores import Chroma
from langchain_openai import OpenAIEmbeddings

# Load environment variables from .env file
load_dotenv()

openai_api_key = os.getenv("OPENAI_API_KEY")

if not openai_api_key:
    raise ValueError("OpenAI API key not set. Please set it in the .env file")

try:
    # Initialize OpenAI embeddings
    embeddings = OpenAIEmbeddings(openai_api_key=openai_api_key)

    # Initialize Chroma vector store
    persist_directory = "./chroma_db"
    vectorstore = Chroma(embedding_function=embeddings, persist_directory=persist_directory)

    retriever = vectorstore.as_retriever()
except Exception as e:
    print(f"An error occurred: {e}")

# Debugging function
def debug_retrieved_documents(response):
    context_docs = response.get('context', [])
    if not context_docs:
        print("No documents retrieved.")
        return

    for doc in context_docs:
        print(f"Retrieved document: {doc.page_content}")

    print(f"Answer based on context: {response.get('answer', 'No answer found')}")