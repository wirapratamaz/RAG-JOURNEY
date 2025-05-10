from fastapi import APIRouter, HTTPException
import logging

from app.api.models import QueryRequest, QueryResponse
from app.core.rag_service import RAGService

# Configure logging
logger = logging.getLogger(__name__)

# Initialize API router
api_router = APIRouter()

# Initialize RAG service
try:
    rag_service = RAGService()
    logger.info("RAG Service initialized successfully")
except Exception as e:
    logger.error(f"Error initializing RAG Service: {e}")
    raise

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
        raise HTTPException(status_code=500, detail=str(e)) 