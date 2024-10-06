import os
import sys
import logging
from fastapi import FastAPI, HTTPException
from dotenv import load_dotenv
from langchain_openai import OpenAIEmbeddings
from langchain_community.chat_models import ChatOpenAI

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Ensure the correct PYTHONPATH
project_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if project_dir not in sys.path:
    sys.path.append(project_dir)

# Load environment variables
load_dotenv()

# Configure OpenAI
openai_api_key = os.getenv("OPENAI_API_KEY")
if not openai_api_key:
    raise ValueError("OPENAI_API_KEY not found in environment variables")

try:
    embeddings = OpenAIEmbeddings()
    llm = ChatOpenAI()
    logger.info("OpenAI components initialized successfully")
except Exception as e:
    logger.error(f"Error initializing OpenAI components: {e}")
    raise

from src.api.api import router as api_router

app = FastAPI()
app.include_router(api_router, prefix="/api")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("src.main_api:app", host="0.0.0.0", port=8000, reload=True)