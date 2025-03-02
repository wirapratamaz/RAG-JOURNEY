"""
Populate ChromaDB for deployment

This script loads documents into ChromaDB, which will then be committed to Git
and used in the Streamlit deployment.

Run this script locally before pushing to GitHub.
"""

import os
import logging
from dotenv import load_dotenv
import sys
import glob

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("populate_db")

# Make sure we're in the project root
if os.path.basename(os.getcwd()) != "CHATBOT-PY":
    logger.warning("This script should be run from the project root directory")

# Import the fetch_posts module
try:
    sys.path.append('.')
    from src.fetch_posts import get_latest_posts, process_and_embed_posts
    from src.retriever import retriever  # This will initialize the DB
    from langchain.text_splitter import RecursiveCharacterTextSplitter
    from PyPDF2 import PdfReader
    import chromadb
    
    # Access the vectorstore directly from the retriever
    if hasattr(retriever, 'vectorstore'):
        vectorstore = retriever.vectorstore
    elif hasattr(retriever, '_vectorstore'):
        vectorstore = retriever._vectorstore
    else:
        raise ValueError("Could not access vectorstore from retriever")
        
    logger.info("Imported required modules")
except Exception as e:
    logger.error(f"Error importing modules: {e}")
    sys.exit(1)

def process_pdf(pdf_path):
    """Process a PDF file and return its chunks for embedding."""
    logger.info(f"Processing PDF: {pdf_path}")
    
    try:
        # Read PDF
        pdf_reader = PdfReader(pdf_path)
        text = ""
        
        # Extract text from each page
        for page in pdf_reader.pages:
            page_text = page.extract_text()
            if page_text:
                text += page_text + "\n\n"
        
        if not text.strip():
            logger.warning(f"No text extracted from {pdf_path}")
            return []
            
        # Split text into chunks
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=200,
            length_function=len
        )
        
        chunks = text_splitter.split_text(text)
        logger.info(f"Split {pdf_path} into {len(chunks)} chunks")
        
        # Return chunks with source information
        return chunks, os.path.basename(pdf_path)
    except Exception as e:
        logger.error(f"Error processing PDF {pdf_path}: {e}")
        return [], None

def add_pdf_documents(docs_dir="./docs"):
    """Add all PDF documents from the docs directory to the vector store."""
    logger.info(f"Adding PDF documents from {docs_dir}")
    
    # Find all PDFs in the directory
    pdf_files = glob.glob(os.path.join(docs_dir, "*.pdf"))
    
    if not pdf_files:
        logger.warning(f"No PDF files found in {docs_dir}")
        return
    
    logger.info(f"Found {len(pdf_files)} PDF files")
    
    # Process each PDF and add to the vectorstore
    total_chunks = 0
    
    for pdf_file in pdf_files:
        chunks, source = process_pdf(pdf_file)
        if chunks and source:
            try:
                # Add text chunks to the vectorstore using the add_texts method
                metadata = [{"source": source} for _ in chunks]
                vectorstore.add_texts(
                    texts=chunks,
                    metadatas=metadata
                )
                total_chunks += len(chunks)
                logger.info(f"Added {len(chunks)} chunks from {source} to the vectorstore")
            except Exception as e:
                logger.error(f"Error adding {pdf_file} to vectorstore: {e}")
                # Try an alternative approach for error cases
                try:
                    # Create Document objects and try add_documents
                    from langchain.schema.document import Document
                    documents = [
                        Document(page_content=chunk, metadata={"source": source})
                        for chunk in chunks
                    ]
                    vectorstore.add_documents(documents)
                    logger.info(f"Added {len(chunks)} chunks from {source} using alternative method")
                except Exception as e2:
                    logger.error(f"Alternative method also failed: {e2}")
    
    logger.info(f"Added a total of {total_chunks} chunks from {len(pdf_files)} PDF files")

if __name__ == "__main__":
    load_dotenv()
    
    # Check if the database directory exists
    persist_directory = "./chroma_db"
    if not os.path.exists(persist_directory):
        os.makedirs(persist_directory, exist_ok=True)
        logger.info(f"Created chroma_db directory at {persist_directory}")
    
    # Fetch and add posts to the database
    try:
        # Get the maximum number of posts to fetch from command line or default to 20
        max_posts = int(sys.argv[1]) if len(sys.argv) > 1 else 20
        logger.info(f"Fetching {max_posts} posts...")
        
        # Add posts to the database
        posts = get_latest_posts(formatted=False, max_posts=max_posts)
        logger.info(f"Successfully fetched {len(posts)} posts")
        
        # Process all PDF documents in the docs directory
        add_pdf_documents("./docs")
        
        logger.info("Database populated successfully!")
        logger.info("Now you can commit the chroma_db directory to Git and deploy to Streamlit")
    except Exception as e:
        logger.error(f"Error populating database: {e}")
        sys.exit(1) 