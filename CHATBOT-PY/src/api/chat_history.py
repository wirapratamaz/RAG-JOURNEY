#src/api/chat_history.py
from typing import List
from langchain_community.chat_message_histories import ChatMessageHistory
from langchain_core.chat_history import BaseChatMessageHistory
from .models import ChatMessage

def convert_to_chat_message_history(session_history: List[ChatMessage]) -> BaseChatMessageHistory:
    chat_history = ChatMessageHistory()
    for message in session_history:
        if message.type == "human":
            chat_history.add_user_message(message.content)
        else:
            chat_history.add_ai_message(message.content)
    return chat_history