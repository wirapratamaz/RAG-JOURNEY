import os
import logging
from dotenv import load_dotenv
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores import Chroma
from langchain.embeddings import OpenAIEmbeddings
from langchain.docstore.document import Document
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

    combined_text = ""
    pdf_files = [f for f in os.listdir(dataset_dir) if f.endswith('.pdf')]
    
    if not pdf_files:
        logger.error("No PDF files found in the dataset directory")
        raise FileNotFoundError("No PDF files found in the dataset directory")

    # Process each PDF file
    for pdf_file in pdf_files:
        pdf_path = os.path.join(dataset_dir, pdf_file)
        logger.info(f"Processing PDF file: {pdf_file}")
        
        with open(pdf_path, "rb") as file:
            reader = PyPDF2.PdfReader(file)
            for page_num in range(len(reader.pages)):
                page = reader.pages[page_num]
                extracted_text = page.extract_text()
                if extracted_text:
                    combined_text += extracted_text + "\n\n"
        
        logger.info(f"Successfully extracted text from {pdf_file}")

    # Split the combined text
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=2000,         # Increased chunk size for larger dataset
        chunk_overlap=200,       # Increased overlap to maintain context
        separators=["\n\n", "\n", " ", ""]
    )
    split_texts = text_splitter.split_text(combined_text)
    logger.info(f"Split the combined text into {len(split_texts)} chunks.")

    # Create document objects
    docs = [Document(page_content=chunk) for chunk in split_texts]
    logger.info("Created document objects from split texts.")

    # Initialize OpenAI embeddings
    embeddings = OpenAIEmbeddings(openai_api_key=openai_api_key)
    logger.info("Initialized OpenAI embeddings.")

    # Initialize Chroma vector store
    persist_directory = "./chroma_db"
    vectorstore = Chroma.from_documents(
        documents=docs,
        embedding=embeddings,
        persist_directory=persist_directory
    )
    logger.info(f"Initialized Chroma vector store at '{persist_directory}'.")

    print("Success! All PDF files have been processed and stored in the vector database.")
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
            chunk_size=2000,
            chunk_overlap=200,
            separators=["\n\n", "\n", " ", ""]
        )
        split_texts = text_splitter.split_text(new_content)
        
        # Create document objects
        new_docs = [Document(page_content=chunk) for chunk in split_texts]
        
        # Add new documents to the existing vector store
        vectorstore.add_documents(new_docs)
        logger.info(f"Added {len(new_docs)} new documents to vector store")
        
    except Exception as e:
        logger.error(f"Error updating vector store: {e}")
        raise