import os
import logging
import argparse
from dotenv import load_dotenv
import chromadb
import uvicorn
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List, Dict

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Load environment variables
load_dotenv()

def create_railway_env_file():
    """Create a .env file for Railway deployment"""
    try:
        # Check if credentials are available
        chroma_url = os.getenv("CHROMA_URL", "https://chroma-production-0925.up.railway.app")
        chroma_token = os.getenv("CHROMA_TOKEN", "qkj1zn522ppyn02i8wk5athfz98a1f4i")
        openai_api_key = os.getenv("OPENAI_API_KEY")
        
        if not openai_api_key:
            logger.error("OPENAI_API_KEY not found. Please set it in your environment.")
            return False
            
        # Create .env file for Railway
        with open(".env.railway", "w") as f:
            f.write(f"CHROMA_URL={chroma_url}\n")
            f.write(f"CHROMA_TOKEN={chroma_token}\n")
            f.write(f"OPENAI_API_KEY={openai_api_key}\n")
            f.write("PORT=8000\n")
        
        logger.info("Created .env.railway file for deployment")
        return True
    except Exception as e:
        logger.error(f"Error creating Railway .env file: {e}")
        return False

def test_chroma_connection():
    """Test the connection to ChromaDB"""
    try:
        chroma_url = os.getenv("CHROMA_URL", "https://chroma-production-0925.up.railway.app")
        chroma_token = os.getenv("CHROMA_TOKEN", "qkj1zn522ppyn02i8wk5athfz98a1f4i")
        
        # Updated client initialization for ChromaDB 0.6.3
        client = chromadb.HttpClient(
            host=chroma_url.replace("https://", "").replace("http://", ""),
            port=443,
            ssl=True,
            headers={"Authorization": f"Bearer {chroma_token}"}
        )
        
        # List collections to test connection
        collections = client.list_collections()
        collection_names = [c.name for c in collections]
        
        logger.info(f"Successfully connected to ChromaDB at {chroma_url}")
        logger.info(f"Available collections: {collection_names}")
        
        return True
    except Exception as e:
        logger.error(f"Failed to connect to ChromaDB: {e}")
        return False

def create_railway_toml():
    """Create a railway.toml configuration file"""
    try:
        with open("railway.toml", "w") as f:
            f.write("[build]\n")
            f.write("builder = \"nixpacks\"\n")
            f.write("buildCommand = \"pip install -r requirements.txt\"\n\n")
            f.write("[deploy]\n")
            f.write("startCommand = \"python src/standalone_api.py\"\n")
            f.write("healthcheckPath = \"/health\"\n")
            f.write("healthcheckTimeout = 100\n")
            f.write("restartPolicyType = \"on-failure\"\n")
            f.write("restartPolicyMaxRetries = 10\n")
        
        logger.info("Created railway.toml configuration file")
        return True
    except Exception as e:
        logger.error(f"Error creating railway.toml: {e}")
        return False

def create_procfile():
    """Create a Procfile for Railway deployment"""
    try:
        with open("Procfile", "w") as f:
            f.write("web: python src/standalone_api.py\n")
        
        logger.info("Created Procfile for deployment")
        return True
    except Exception as e:
        logger.error(f"Error creating Procfile: {e}")
        return False

def create_requirements_file():
    """Create a requirements.txt file with all necessary dependencies"""
    try:
        with open("requirements.txt", "w") as f:
            f.write("fastapi>=0.95.0\n")
            f.write("uvicorn>=0.22.0\n")
            f.write("python-dotenv>=1.0.0\n")
            f.write("langchain>=0.1.5\n")
            f.write("langchain-community>=0.0.15\n")
            f.write("langchain-openai>=0.0.5\n")
            f.write("openai>=1.10.0\n")
            f.write("chromadb>=0.4.22\n")
            f.write("pydantic>=2.5.0\n")
            f.write("tiktoken>=0.5.0\n")
        
        logger.info("Created requirements.txt file")
        return True
    except Exception as e:
        logger.error(f"Error creating requirements.txt: {e}")
        return False

def main():
    parser = argparse.ArgumentParser(description="Prepare for Railway deployment")
    parser.add_argument("--test-only", action="store_true", help="Only test ChromaDB connection without creating files")
    args = parser.parse_args()
    
    if args.test_only:
        test_chroma_connection()
        return
    
    # Create necessary files for deployment
    env_success = create_railway_env_file()
    toml_success = create_railway_toml()
    procfile_success = create_procfile()
    req_success = create_requirements_file()
    
    # Test ChromaDB connection
    connection_success = test_chroma_connection()
    
    if env_success and toml_success and procfile_success and req_success and connection_success:
        logger.info("All deployment files created successfully!")
        logger.info("To deploy to Railway:")
        logger.info("1. Push your code to GitHub")
        logger.info("2. Create a new project in Railway from your GitHub repo")
        logger.info("3. Set up the environment variables from .env.railway")
        logger.info("4. Deploy your project")
    else:
        logger.error("Failed to create some deployment files or test connection.")

if __name__ == "__main__":
    main() 