#src/api/models.py
from pydantic import BaseModel
from typing import List, Dict

class ChatMessage(BaseModel):
    type: str
    content: str

class QueryRequest(BaseModel):
    query: str
    chat_history: List[ChatMessage] = []