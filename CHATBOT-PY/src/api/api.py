from fastapi import APIRouter, HTTPException
from .models import QueryRequest
from .chat_history import convert_to_chat_message_history
from langchain_core.runnables.history import RunnableWithMessageHistory
from main import rag_chain

router = APIRouter()

@router.get("/")
def read_root():
    return {"message": "Welcome to the RAG Chatbot API"}

@router.post("/query/")
def answer_query(request: QueryRequest):
    try:
        # Convert session state chat history to ChatMessageHistory
        chat_history = convert_to_chat_message_history(request.chat_history)
        
        # Perform question answering
        conversational_rag_chain = RunnableWithMessageHistory(
            rag_chain,
            lambda _: chat_history,
            input_messages_key="input",
            history_messages_key="chat_history",
            output_messages_key="answer",
        )
        
        response = conversational_rag_chain.invoke(
            {"input": request.query},
            config={"configurable": {"session_id": "abc123"}}
        )
        
        return {"answer": response["answer"]}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))