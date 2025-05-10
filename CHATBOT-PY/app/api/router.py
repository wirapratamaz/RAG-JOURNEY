from fastapi import APIRouter, HTTPException
import logging

from app.api.models import QueryRequest, QueryResponse
from app.core.rag_service import RAGService

# Configure logging
logger = logging.getLogger(__name__)

# Initialize API router
api_router = APIRouter()

# Initialize RAG service with better error handling
try:
    rag_service = RAGService()
    logger.info("RAG Service initialized successfully")
except Exception as e:
    logger.error(f"Error initializing RAG Service: {e}")
    # We'll create a dummy service that returns error messages
    class DummyRAGService:
        def query(self, question):
            return {
                "answer": "Sorry, the RAG service is currently unavailable. Please try again later.",
                "sources": []
            }
    rag_service = DummyRAGService()
    logger.warning("Using DummyRAGService as fallback")

@api_router.post("/query", response_model=QueryResponse)
async def query(request: QueryRequest):
    """Process a query using the RAG service"""
    try:
        logger.info(f"Received query: {request.query}")
        result = rag_service.query(request.query)
        return QueryResponse(
            answer=result["answer"],
            sources=result["sources"]
        )
    except Exception as e:
        logger.error(f"Error processing query: {str(e)}")
        # Return a friendly error message instead of raising an exception
        return QueryResponse(
            answer="I'm sorry, I encountered an error while processing your query. Please try again later.",
            sources=[]
        ) 