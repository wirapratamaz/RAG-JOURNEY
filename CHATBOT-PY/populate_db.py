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
    logger.info("Imported required modules")
except Exception as e:
    logger.error(f"Error importing modules: {e}")
    sys.exit(1)

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
        
        # Add any other data you need to load into the database
        # For example:
        # add_pdf_documents()
        # add_website_content()
        
        logger.info("Database populated successfully!")
        logger.info("Now you can commit the chroma_db directory to Git and deploy to Streamlit")
    except Exception as e:
        logger.error(f"Error populating database: {e}")
        sys.exit(1) 