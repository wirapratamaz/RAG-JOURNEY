import os
import sys
import logging
from fastapi import FastAPI, HTTPException
from dotenv import load_dotenv
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from scheduler import init_scheduler
import atexit
from fastapi.middleware.cors import CORSMiddleware
import uvicorn

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

# Initialize FastAPI app
app = FastAPI(
    title="Undiksha RAG Chatbot API",
    description="API for the RAG-based chatbot for Undiksha University",
    version="1.0.0"
)

# Configure CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # In production, replace with specific origins
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Import and include routers
from src.api.api import router as main_router
from src.api.chat_api import router as chat_router

app.include_router(main_router, prefix="/api")
app.include_router(chat_router, prefix="/api")

@app.get("/")
async def root():
    return {"message": "Welcome to the Undiksha RAG Chatbot API. Use /api/chat for chat functionality."}

# Initialize scheduler when the API starts
scheduler = init_scheduler()

# Shut down scheduler gracefully when the application stops
atexit.register(lambda: scheduler.shutdown())

if __name__ == "__main__":
    uvicorn.run("src.main_api:app", host="0.0.0.0", port=8000, reload=True)