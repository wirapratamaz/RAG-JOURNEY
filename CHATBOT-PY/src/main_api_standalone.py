from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
import uvicorn
import logging
import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

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
from src.api.standalone_api import router as api_router

app.include_router(api_router, prefix="/api")

@app.get("/")
async def root():
    return {"message": "Welcome to the Undiksha RAG Chatbot API. Use /api/chat for chat functionality or /api/query for simple queries."}

if __name__ == "__main__":
    uvicorn.run("src.main_api_standalone:app", host="0.0.0.0", port=8000, reload=True) 