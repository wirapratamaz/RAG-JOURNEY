from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from typing import List, Dict, Optional
import logging
from ..rag_pipeline_standalone import RAGPipeline

logger = logging.getLogger(__name__)
router = APIRouter()

# Initialize RAG pipeline
rag = RAGPipeline()

class ChatMessage(BaseModel):
    role: str  # "user" or "assistant"
    content: str

class ChatRequest(BaseModel):
    messages: List[ChatMessage]
    stream: bool = False

class ChatResponse(BaseModel):
    answer: str
    sources: List[str]

class QueryRequest(BaseModel):
    query: str

@router.post("/chat", response_model=ChatResponse)
async def chat(request: ChatRequest):
    """Process a chat request using the RAG pipeline"""
    try:
        # Extract the latest user message
        user_messages = [msg.content for msg in request.messages if msg.role == "user"]
        if not user_messages:
            raise HTTPException(status_code=400, detail="No user message found")
        
        latest_query = user_messages[-1]
        
        # Get chat history (excluding the latest message)
        chat_history = []
        for i in range(0, len(request.messages) - 1, 2):
            if i+1 < len(request.messages):
                if request.messages[i].role == "user" and request.messages[i+1].role == "assistant":
                    chat_history.append((request.messages[i].content, request.messages[i+1].content))
        
        # Process query with RAG pipeline
        result = rag.query(latest_query, chat_history)
        
        return ChatResponse(
            answer=result["answer"],
            sources=result["sources"]
        )
    except Exception as e:
        logger.error(f"Error processing chat: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/query", response_model=ChatResponse)
async def query(request: QueryRequest):
    """Process a simple query using the RAG pipeline"""
    try:
        # Process query with RAG pipeline
        result = rag.query(request.query)
        
        return ChatResponse(
            answer=result["answer"],
            sources=result["sources"]
        )
    except Exception as e:
        logger.error(f"Error processing query: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/health")
async def health_check():
    """Health check endpoint"""
    return {"status": "healthy"} 