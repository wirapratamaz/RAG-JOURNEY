import os
import logging
from dotenv import load_dotenv
import chromadb
from langchain_community.document_loaders import DirectoryLoader, TextLoader, PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import Chroma

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Load environment variables
load_dotenv()

# Create a dedicated ChromaDB directory for the standalone API
STANDALONE_CHROMA_DIR = "./standalone_chroma_db"
os.makedirs(STANDALONE_CHROMA_DIR, exist_ok=True)

# Set the path to the dataset directory
DATASET_DIR = "../dataset"

def get_chroma_client():
    """Get a ChromaDB client instance"""
    client = chromadb.PersistentClient(path=STANDALONE_CHROMA_DIR)
    logger.info("ChromaDB client initialized")
    return client

def load_documents():
    """Load documents from the dataset directory"""
    documents = []
    
    # Load PDF documents from the dataset directory
    try:
        # Check if dataset directory exists using absolute path
        base_dir = os.path.dirname(os.path.abspath(__file__))
        dataset_dir = os.path.join(os.path.dirname(base_dir), "dataset")
        
        if os.path.exists(dataset_dir):
            logger.info(f"Loading PDF documents from {dataset_dir}")
            pdf_loader = DirectoryLoader(dataset_dir, glob="**/*.pdf", loader_cls=PyPDFLoader)
            pdf_docs = pdf_loader.load()
            logger.info(f"Loaded {len(pdf_docs)} PDF documents")
            for doc in pdf_docs:
                # Print the document metadata to debug
                logger.info(f"Loaded document: {doc.metadata.get('source', 'Unknown source')}")
            documents.extend(pdf_docs)
        else:
            logger.error(f"Dataset directory not found: {dataset_dir}")
    except Exception as e:
        logger.error(f"Error loading PDF documents: {e}")
    
    return documents

def process_documents(documents):
    """Split documents into chunks and process them for vector storage"""
    if not documents:
        logger.warning("No documents to process")
        return []
    
    logger.info(f"Processing {len(documents)} documents")
    
    # Define text splitter for chunking
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200,
        length_function=len,
    )
    
    # Split documents into chunks
    chunks = text_splitter.split_documents(documents)
    logger.info(f"Created {len(chunks)} chunks")
    
    return chunks

def create_vector_store(chunks):
    """Create a vector store with document chunks"""
    if not chunks:
        logger.warning("No chunks to add to vector store")
        return
    
    try:
        # Initialize embeddings
        embeddings = OpenAIEmbeddings()
        
        # Get ChromaDB client
        client = get_chroma_client()
        collection_name = "standalone_api"
        
        # Check if collection exists - updated for ChromaDB v0.6.0+
        collections = client.list_collections()
        collection_exists = collection_name in collections
        
        if collection_exists:
            logger.info(f"Deleting existing collection '{collection_name}' to ensure clean data")
            client.delete_collection(collection_name)
            logger.info(f"Collection '{collection_name}' deleted")
        
        # Create new collection with fresh data
        logger.info(f"Creating new collection: {collection_name}")
        vector_store = Chroma.from_documents(
            documents=chunks,
            embedding=embeddings,
            client=client,
            collection_name=collection_name,
            persist_directory=STANDALONE_CHROMA_DIR
        )
        
        # Note on persistence
        logger.info(f"Created vector store with {len(chunks)} chunks (auto-persisted)")
    except Exception as e:
        logger.error(f"Error creating vector store: {e}")
        raise

if __name__ == "__main__":
    logger.info("Starting document loading process")
    
    # Load documents
    documents = load_documents()
    
    if documents:
        # Process documents
        chunks = process_documents(documents)
        
        # Create vector store
        create_vector_store(chunks)
        
        logger.info("Document loading process completed")
    else:
        logger.error("No documents loaded, check the dataset directory path") 