import os
import logging
from dotenv import load_dotenv
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores import Chroma
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.docstore.document import Document
from web_crawler import get_crawled_content
import PyPDF2

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Load environment variables from .env file
load_dotenv()

openai_api_key = os.getenv("OPENAI_API_KEY")

if not openai_api_key:
    logger.error("OpenAI API key not set. Please set it in the .env file")
    raise ValueError("OpenAI API key not set. Please set it in the .env file")

try:
    # Get all PDF files from the dataset directory
    dataset_dir = os.path.join(os.path.dirname(__file__), '..', 'dataset')
    if not os.path.exists(dataset_dir):
        logger.error(f"Dataset directory not found at path: {dataset_dir}")
        raise FileNotFoundError(f"Dataset directory not found at path: {dataset_dir}")

    # Initialize combined text
    combined_text = ""

    # Process each PDF file in the dataset directory
    for filename in os.listdir(dataset_dir):
        if filename.endswith('.pdf'):
            pdf_path = os.path.join(dataset_dir, filename)
            logger.info(f"Processing PDF file: {filename}")
            
            with open(pdf_path, "rb") as file:
                reader = PyPDF2.PdfReader(file)
                for page_num in range(len(reader.pages)):
                    page = reader.pages[page_num]
                    extracted_text = page.extract_text()
                    if extracted_text:
                        combined_text += f"\n=== Start of {filename} ===\n"
                        combined_text += extracted_text
                        combined_text += f"\n=== End of {filename} ===\n"

    logger.info("Successfully extracted text from all PDF files.")

    # Crawl the website
    logger.info("Starting to crawl the website...")
    web_content = get_crawled_content()
    
    if not web_content:
        logger.warning("No content fetched from the website.")
    else:
        logger.info("Successfully crawled the website.")
        combined_text += "\n=== Start of Web Content ===\n"
        combined_text += web_content
        combined_text += "\n=== End of Web Content ===\n"

    # Split the combined text
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1500,         # Increased chunk size to keep more context
        chunk_overlap=500,       # Increased overlap to maintain bullet point context
        separators=["\n\n", "\n• ", "\n", "• ", " ", ""],  # Added more bullet point separators
        length_function=len,
        keep_separator=True      # Keep the separators to maintain formatting
    )
    split_texts = text_splitter.split_text(combined_text)
    logger.info(f"Split the combined text into {len(split_texts)} chunks.")

    # Create document objects with metadata
    docs = []
    for chunk in split_texts:
        # Try to determine the source file from the chunk content
        source = "unknown"
        for filename in os.listdir(dataset_dir):
            if filename.endswith('.pdf') and f"=== Start of {filename} ===" in chunk:
                source = filename
                break
        if "=== Start of Web Content ===" in chunk:
            source = "web_content"
            
        docs.append(Document(
            page_content=chunk,
            metadata={"source": source}
        ))
    logger.info("Created document objects from split texts.")

    # Initialize embeddings
    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
    )
    logger.info("Initialized HuggingFace embeddings.")

    # Initialize Chroma vector store
    persist_directory = "./chroma_db"
    vectorstore = Chroma.from_documents(
        documents=docs,
        embedding=embeddings,
        persist_directory=persist_directory
    )
    logger.info(f"Initialized Chroma vector store at '{persist_directory}'.")

    print("Success!")
except Exception as e:
    logger.error(f"An error occurred: {e}")
    print(f"An error occurred: {e}")

def update_vectorstore(new_content):
    """
    Updates the vector store with new content
    """
    try:
        # Create new document objects from the content
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=300,
            separators=["\n\n", "\n", "•", " ", ""]
        )
        split_texts = text_splitter.split_text(new_content)
        
        # Create document objects
        new_docs = [Document(
            page_content=chunk,
            metadata={"source": "update"}
        ) for chunk in split_texts]
        
        # Add new documents to the existing vector store
        vectorstore.add_documents(new_docs)
        logger.info(f"Added {len(new_docs)} new documents to vector store")
        
    except Exception as e:
        logger.error(f"Error updating vector store: {e}")
        raise