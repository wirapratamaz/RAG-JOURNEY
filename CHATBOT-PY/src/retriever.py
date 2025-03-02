import os
import sys
import logging
from dotenv import load_dotenv

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Try to fix sqlite3 version issues by using pysqlite3 if available
try:
    import sqlite3
    sqlite_version = sqlite3.sqlite_version_info
    if sqlite_version < (3, 35, 0):
        logger.warning(f"SQLite version {sqlite3.sqlite_version} is too old for ChromaDB (needs 3.35.0+)")
        
        # Use a conditional import pattern that avoids linter errors
        pysqlite3 = None  # type: ignore
        try:
            import pysqlite3  # type: ignore
            sys.modules['sqlite3'] = pysqlite3
            # Import sqlite3 again to verify it's been replaced
            import sqlite3
            logger.info(f"Successfully replaced sqlite3 with pysqlite3 (version: {sqlite3.sqlite_version})")
        except ImportError:
            logger.warning("pysqlite3 is not installed. You may need to:")
            logger.warning("1. Install a newer version of SQLite")
            logger.warning("2. Or manually install pysqlite3 from source")
            logger.warning("Attempting to continue with the existing SQLite version...")
except Exception as e:
    logger.error(f"Error handling sqlite3: {e}")
    raise ImportError(f"Failed to initialize database dependencies: {e}")

# Now try to import langchain_chroma
try:
    from langchain_community.embeddings import HuggingFaceEmbeddings
    from langchain_chroma import Chroma
    import chromadb
    logger.info("Successfully imported langchain_chroma and chromadb")
except Exception as e:
    logger.error(f"Error importing langchain_chroma: {e}")
    raise ImportError(f"Failed to import langchain_chroma: {e}")

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

    # Handle potential schema mismatch by creating client settings
    client_settings = chromadb.Settings(
        anonymized_telemetry=False,
        allow_reset=True,  # This allows schema reset if there's a mismatch
    )

    # Always start with a fresh database in deployment
    if os.environ.get('STREAMLIT_SHARING_MODE') == 'streamlit_app' or 'STREAMLIT_RUNTIME' in os.environ:
        logger.info("Detected Streamlit deployment environment - using clean database approach")
        try:
            # In deployment, prioritize a clean database to avoid schema issues
            if os.path.exists(persist_directory):
                import shutil
                # Keep backup just in case
                backup_dir = f"{persist_directory}_backup"
                if os.path.exists(backup_dir):
                    shutil.rmtree(backup_dir)
                shutil.copytree(persist_directory, backup_dir)
                # Create fresh directory
                shutil.rmtree(persist_directory)
                os.makedirs(persist_directory, exist_ok=True)
                logger.info("Created fresh database directory in Streamlit environment")
        except Exception as e:
            logger.warning(f"Error handling directory in Streamlit: {e}")
    
    try:
        # First try with allow_reset=True to handle schema migrations
        vectorstore = Chroma(
            embedding_function=embeddings,
            persist_directory=persist_directory,
            client_settings=client_settings
        )
        logger.info("Successfully initialized Chroma with client settings")
    except Exception as e:
        logger.warning(f"Failed to initialize Chroma with existing DB, trying to recreate: {e}")
        
        # If that fails, try to create a clean database by deleting and recreating
        try:
            import shutil
            # Backup the old directory first
            if os.path.exists(persist_directory):
                backup_dir = f"{persist_directory}_backup"
                logger.info(f"Backing up existing chroma_db to {backup_dir}")
                if os.path.exists(backup_dir):
                    shutil.rmtree(backup_dir)
                shutil.copytree(persist_directory, backup_dir)
                
                # Remove the problematic directory
                shutil.rmtree(persist_directory)
                os.makedirs(persist_directory, exist_ok=True)
            
            # Initialize a fresh Chroma instance
            vectorstore = Chroma(
                embedding_function=embeddings,
                persist_directory=persist_directory
            )
            logger.info("Successfully initialized Chroma with a fresh database")
        except Exception as e2:
            logger.error(f"Could not recreate Chroma database: {e2}")
            # Last resort - try memory-only database
            try:
                logger.warning("Attempting to create in-memory ChromaDB as last resort")
                vectorstore = Chroma(
                    embedding_function=embeddings
                )
                logger.info("Created in-memory ChromaDB instance")
            except Exception as e3:
                logger.error(f"Even in-memory DB failed: {e3}")
                raise

    # Configure the retriever with more comprehensive settings
    retriever = vectorstore.as_retriever(
        search_type="mmr",  # Use Maximum Marginal Relevance for better diversity
        search_kwargs={
            "k": 12,  # Increase number of retrieved documents further
            "fetch_k": 20,  # Fetch more documents initially before filtering
            "lambda_mult": 0.5,  # Better balance between relevance and diversity (lower value = more diversity)
        }
    )
    logger.info("Successfully initialized retriever from Chroma with enhanced retrieval settings")
except Exception as e:
    logger.error(f"Error initializing retriever: {e}")
    # Use the dummy retriever instead of re-raising
    logger.warning("Falling back to DummyRetriever")
    retriever = DummyRetriever()

# Export retriever for easy import
__all__ = ['retriever']