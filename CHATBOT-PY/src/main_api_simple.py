from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List, Dict
import logging
import os
import uvicorn
from dotenv import load_dotenv
import sys

# Set up correct paths
src_path = os.path.dirname(os.path.abspath(__file__))
project_path = os.path.dirname(src_path)
sys.path.insert(0, project_path)

# Load environment variables
load_dotenv()

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize FastAPI app
app = FastAPI(
    title="Undiksha RAG Chatbot API",
    description="Simple API for RAG-based chatbot",
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

# Import RAG Pipeline
try:
    from src.rag_pipeline_standalone import RAGPipeline
    rag = RAGPipeline()
    logger.info("RAG Pipeline initialized successfully")
except Exception as e:
    logger.error(f"Error initializing RAG Pipeline: {e}")
    raise

class QueryRequest(BaseModel):
    query: str

class QueryResponse(BaseModel):
    answer: str
    sources: List[str]

@app.post("/api/query", response_model=QueryResponse)
async def query(request: QueryRequest):
    """Process a simple query using the RAG pipeline"""
    try:
        logger.info(f"Received query: {request.query}")
        result = rag.query(request.query)
        return QueryResponse(
            answer=result["answer"],
            sources=result["sources"]
        )
    except Exception as e:
        logger.error(f"Error processing query: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {"status": "healthy"}

@app.get("/")
async def root():
    return {"message": "Welcome to the Undiksha RAG Chatbot API. Use /api/query for simple queries."}

if __name__ == "__main__":
    import argparse
    
    # Parse command line arguments
    parser = argparse.ArgumentParser(description="Run the Undiksha RAG Chatbot API")
    parser.add_argument("--host", type=str, default="0.0.0.0", help="Host to run the server on")
    parser.add_argument("--port", type=int, default=8000, help="Port to run the server on")
    args = parser.parse_args()
    
    # Run the server
    logger.info(f"Starting server on {args.host}:{args.port}")
    uvicorn.run("main_api_simple:app", host=args.host, port=args.port, reload=True) 