from fastapi import APIRouter, HTTPException
from .models import QueryRequest
from .chat_history import convert_to_chat_message_history
from langchain_core.runnables.history import RunnableWithMessageHistory
from ..main import rag_chain
from ..retriever import debug_retrieved_documents

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
        
        # Add debug logs to confirm data flow
        print(f"Received query: {request.query}")
        print(f"Chat history: {request.chat_history}")
        
        response = conversational_rag_chain.invoke(
            {"input": request.query},
            config={"configurable": {"session_id": "abc123"}}
        )
        
        # Debugging the response
        print(f"Response from RAG chain: {response}")
        
        # Debug retrieved documents and answer
        debug_retrieved_documents(response)
        
        if "answer" in response:
            return {"answer": response["answer"]}
        else:
            raise ValueError("RAG chain did not return an answer")
    except Exception as e:
        print(f"An error occurred while answering the query: {e}")
        raise HTTPException(status_code=500, detail=str(e))