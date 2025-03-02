import os
import logging
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain.chains import ConversationalRetrievalChain
from langchain.memory import ConversationBufferMemory
from retriever import retriever

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Load environment variables
load_dotenv()
openai_api_key = os.getenv("OPENAI_API_KEY")

# Check if the API key is set
if not openai_api_key:
    logger.error("OpenAI API key not set. Please set it in the .env file")
    exit(1)

# Initialize memory
memory = ConversationBufferMemory(
    memory_key="chat_history",
    output_key="answer",
    return_messages=True
)

# Initialize the OpenAI chat model
llm = ChatOpenAI(
    model_name="gpt-3.5-turbo",
    openai_api_key=openai_api_key
)

# Create the RAG chain
rag_chain = ConversationalRetrievalChain.from_llm(
    llm=llm,
    retriever=retriever,
    memory=memory,
    return_source_documents=True,
    verbose=True
)

# Test query
query = "Apa saja tahapan utama dalam proses penyelesaian skripsi di Program Studi Sistem Informasi?"

# Get response
try:
    response = rag_chain.invoke({
        "question": query,
        "chat_history": []
    })
    
    print("\n--- RAG Chain Response ---")
    print(f"Answer: {response['answer']}")
    print("\n--- Source Documents ---")
    for i, doc in enumerate(response['source_documents']):
        print(f"\nDocument {i+1}:")
        print(f"Content: {doc.page_content[:300]}...")
        print(f"Source: {doc.metadata.get('source', 'unknown')}")
except Exception as e:
    logger.error(f"Error generating response: {e}") 