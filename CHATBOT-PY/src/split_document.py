import os
import logging
from dotenv import load_dotenv
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores import Chroma
from langchain.embeddings import OpenAIEmbeddings
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
    # Read the PDF dataset
    pdf_file_path = os.path.join(os.path.dirname(__file__), '..', 'dataset', 'dataset.pdf')
    if not os.path.exists(pdf_file_path):
        logger.error(f"PDF dataset not found at path: {pdf_file_path}")
        raise FileNotFoundError(f"PDF dataset not found at path: {pdf_file_path}")

    with open(pdf_file_path, "rb") as file:
        reader = PyPDF2.PdfReader(file)
        text = ""
        for page_num in range(len(reader.pages)):
            page = reader.pages[page_num]
            extracted_text = page.extract_text()
            if extracted_text:
                text += extracted_text + "\n"

    logger.info("Successfully extracted text from the PDF dataset.")

    # Crawl the website
    logger.info("Starting to crawl the website...")
    web_content = get_crawled_content()
    
    if not web_content:
        logger.warning("No content fetched from the website.")
    else:
        logger.info("Successfully crawled the website.")

    # Combine both sources
    combined_text = text + "\n" + web_content

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

    # Persist the vector store
    # No need to call vectorstore.persist() as Chroma handles persistence during initialization

    print("Success!")
except Exception as e:
    logger.error(f"An error occurred: {e}")
    print(f"An error occurred: {e}")