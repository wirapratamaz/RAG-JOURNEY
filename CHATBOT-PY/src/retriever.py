import os
import sys
import logging
from dotenv import load_dotenv
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_chroma import Chroma

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Load environment variables from .env file
load_dotenv()

# Set the persist_directory for Chroma
if os.path.exists("./chroma_db"):
    persist_directory = "./chroma_db"
elif os.path.exists("../chroma_db"):
    persist_directory = "../chroma_db"
elif os.path.exists("/mount/src/rag-journey/CHATBOT-PY/chroma_db"):
    persist_directory = "/mount/src/rag-journey/CHATBOT-PY/chroma_db"
else:
    # Create the directory if it doesn't exist
    persist_directory = "./chroma_db"
    os.makedirs(persist_directory, exist_ok=True)
    logger.warning(f"Created new chroma_db directory at {persist_directory}")

# Initialize a dummy retriever as fallback
class DummyRetriever:
    def get_relevant_documents(self, query):
        logger.warning("Using DummyRetriever - no actual retrieval happening!")
        return [type('obj', (object,), {
            'page_content': 'This is a dummy document. The vector store could not be initialized.',
            'metadata': {'source': 'dummy'}
        })]

# Initialize the real retriever
try:
    # Initialize HuggingFace embeddings
    # Using all-MiniLM-L6-v2 which has 384 dimensions
    logger.info(f"Initializing embeddings and vector store from {persist_directory}")
    embeddings = HuggingFaceEmbeddings(
        model_name="all-MiniLM-L6-v2"
    )

    # Initialize Chroma vector store
    vectorstore = Chroma(
        embedding_function=embeddings,
        persist_directory=persist_directory
    )

    retriever = vectorstore.as_retriever()
    logger.info("Successfully initialized retriever from Chroma")
except Exception as e:
    logger.error(f"Error initializing retriever: {e}")
    retriever = DummyRetriever()
    logger.warning("Using DummyRetriever as fallback")

# Export retriever for easy import
__all__ = ['retriever']