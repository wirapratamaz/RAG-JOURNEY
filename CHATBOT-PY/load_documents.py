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

# Railway ChromaDB Configuration
CHROMA_URL = os.getenv("CHROMA_URL", "https://chroma-production-0925.up.railway.app")
CHROMA_TOKEN = os.getenv("CHROMA_TOKEN", "qkj1zn522ppyn02i8wk5athfz98a1f4i")

# Set the path to the dataset directory
DATASET_DIR = "../dataset"

def get_chroma_client():
    """Get a ChromaDB client instance"""
    try:
        # Connect to Railway ChromaDB - updated for ChromaDB 0.6.3
        chroma_url = CHROMA_URL
        host = chroma_url.replace("https://", "").replace("http://", "")
        client = chromadb.HttpClient(
            host=host,
            port=443,
            ssl=True,
            headers={"Authorization": f"Bearer {CHROMA_TOKEN}"}
        )
        logger.info(f"Connected to Railway ChromaDB at {CHROMA_URL}")
        return client
    except Exception as e:
        logger.error(f"Error connecting to Railway ChromaDB: {e}")
        # Fallback to local ChromaDB if available
        try:
            local_dir = "./standalone_chroma_db"
            os.makedirs(local_dir, exist_ok=True)
            client = chromadb.PersistentClient(path=local_dir)
            logger.info("Fallback to local ChromaDB client")
            return client
        except Exception as inner_e:
            logger.error(f"Failed to initialize any ChromaDB client: {inner_e}")
            raise

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
        
        # Check if collection exists - updated for ChromaDB 0.6.3
        try:
            collections = client.list_collections()
            # In ChromaDB 0.6.0+, list_collections returns collection names
            collection_exists = collection_name in collections
            logger.info(f"Available collections: {collections}")
        except Exception as e:
            logger.error(f"Error listing collections: {e}")
            collection_exists = False
        
        if collection_exists:
            logger.info(f"Deleting existing collection '{collection_name}' to ensure clean data")
            client.delete_collection(collection_name)
            logger.info(f"Collection '{collection_name}' deleted")
        
        # Create collection if it doesn't exist
        try:
            client.create_collection(collection_name)
            logger.info(f"Created new collection: {collection_name}")
        except Exception as e:
            logger.error(f"Error creating collection: {e}")
        
        # Create new collection with fresh data
        logger.info(f"Adding documents to collection: {collection_name}")
        vector_store = Chroma.from_documents(
            documents=chunks,
            embedding=embeddings,
            client=client,
            collection_name=collection_name
        )
        
        # Note on persistence
        logger.info(f"Created vector store with {len(chunks)} chunks")
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