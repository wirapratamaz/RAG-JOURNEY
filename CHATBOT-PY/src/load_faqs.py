import os
import logging
from dotenv import load_dotenv
import chromadb
from langchain.docstore.document import Document
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

# Path to the FAQs file
FAQ_FILE = "../si_faqs.txt"

def get_chroma_client():
    """Get a ChromaDB client instance"""
    client = chromadb.PersistentClient(path=STANDALONE_CHROMA_DIR)
    logger.info("ChromaDB client initialized")
    return client

def load_faqs():
    """Load FAQs from the si_faqs.txt file"""
    documents = []
    
    try:
        # Check if FAQs file exists
        if not os.path.exists(FAQ_FILE):
            base_dir = os.path.dirname(os.path.abspath(__file__))
            faq_path = os.path.join(os.path.dirname(base_dir), "si_faqs.txt")
            if os.path.exists(faq_path):
                logger.info(f"Found FAQs file at {faq_path}")
                faq_file = faq_path
            else:
                logger.error(f"FAQs file not found at {FAQ_FILE} or {faq_path}")
                return []
        else:
            faq_file = FAQ_FILE
            
        # Read the FAQs file
        with open(faq_file, 'r', encoding='utf-8') as f:
            content = f.read()
            
        # Parse FAQs: Each FAQ consists of a question (line starting with number and tab),
        # followed by an answer (lines until the next question or end of file)
        lines = content.split('\n')
        current_question = None
        current_answer = ""
        faq_id = None
        
        # Skip the header line
        if lines and "SI FAQNO" in lines[0]:
            lines = lines[1:]
            
        for line in lines:
            # Check if line starts with a number (potential new question)
            if line.strip() and line[0].isdigit() and '\t' in line:
                # If we have a previous question/answer pair, save it
                if current_question and current_answer:
                    doc = Document(
                        page_content=f"Question: {current_question}\nAnswer: {current_answer.strip()}",
                        metadata={
                            "source": "si_faqs.txt",
                            "type": "faq",
                            "id": faq_id,
                            "question": current_question
                        }
                    )
                    documents.append(doc)
                
                # Extract question ID and text
                parts = line.split('\t')
                if len(parts) >= 2:
                    faq_id = parts[0].strip()
                    current_question = parts[1].strip().strip('"')
                    current_answer = ""
            elif current_question:  # We're in an answer section
                if line.strip():  # Only add non-empty lines
                    current_answer += line.strip() + " "
        
        # Add the last FAQ
        if current_question and current_answer:
            doc = Document(
                page_content=f"Question: {current_question}\nAnswer: {current_answer.strip()}",
                metadata={
                    "source": "si_faqs.txt",
                    "type": "faq",
                    "id": faq_id,
                    "question": current_question
                }
            )
            documents.append(doc)
            
        logger.info(f"Loaded {len(documents)} FAQs from {faq_file}")
        
    except Exception as e:
        logger.error(f"Error loading FAQs: {e}")
    
    return documents

def add_to_vector_store(documents):
    """Add documents to the vector store"""
    if not documents:
        logger.warning("No documents to add to vector store")
        return
    
    try:
        # Initialize embeddings
        embeddings = OpenAIEmbeddings()
        
        # Get ChromaDB client
        client = get_chroma_client()
        collection_name = "faqs_collection"
        
        # Check if collection exists - updated for ChromaDB v0.6.0+
        collections = client.list_collections()
        collection_exists = collection_name in collections
        
        if collection_exists:
            logger.info(f"Using existing collection: {collection_name}")
            vector_store = Chroma(
                client=client,
                collection_name=collection_name,
                embedding_function=embeddings,
                persist_directory=STANDALONE_CHROMA_DIR
            )
            
            # Add documents to existing collection
            vector_store.add_documents(documents)
        else:
            # Collection doesn't exist, create it
            logger.info(f"Creating new collection: {collection_name}")
            vector_store = Chroma.from_documents(
                documents=documents,
                embedding=embeddings,
                client=client,
                collection_name=collection_name,
                persist_directory=STANDALONE_CHROMA_DIR
            )
        
        logger.info(f"Vector store updated with {len(documents)} FAQs")
    except Exception as e:
        logger.error(f"Error adding FAQs to vector store: {e}")
        raise

if __name__ == "__main__":
    logger.info("Starting FAQs loading process")
    
    # Load FAQs
    faq_documents = load_faqs()
    
    if faq_documents:
        # Add to vector store
        add_to_vector_store(faq_documents)
        
        logger.info("FAQs loading process completed")
    else:
        logger.error("No FAQs loaded, check the input file") 