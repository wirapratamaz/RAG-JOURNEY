import os
import logging
from typing import List, Dict, Optional
from dotenv import load_dotenv
from langchain_community.vectorstores import Chroma
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate
import re
import chromadb

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Load environment variables
load_dotenv()

# Global ChromaDB client to ensure consistent settings
_CHROMA_CLIENT = None

def get_chroma_client():
    global _CHROMA_CLIENT
    if _CHROMA_CLIENT is None:
        persist_directory = "./chroma_db"
        _CHROMA_CLIENT = chromadb.PersistentClient(path=persist_directory)
        logger.info("ChromaDB client initialized")
    return _CHROMA_CLIENT

class RAGPipeline:
    def __init__(self):
        self.openai_api_key = os.getenv("OPENAI_API_KEY")
        if not self.openai_api_key:
            raise ValueError("OpenAI API key not found in environment variables")
        
        self.embeddings = OpenAIEmbeddings()
        self.vector_store = None
        self.llm = ChatOpenAI(temperature=0, model="gpt-4")
        
    def initialize_vector_store(self):
        """Initialize or load the existing vector store with consistent settings"""
        # Use the global client instead of creating a new one
        persist_directory = "./chroma_db"
        chroma_client = get_chroma_client()
        
        try:
            # First check if collection exists
            collection_name = "langchain"
            collections = chroma_client.list_collections()
            collection_exists = any(col.name == collection_name for col in collections)
            
            if collection_exists:
                logger.info(f"Using existing ChromaDB collection: {collection_name}")
            else:
                logger.info(f"Creating new ChromaDB collection: {collection_name}")
            
            self.vector_store = Chroma(
                persist_directory=persist_directory,
                embedding_function=self.embeddings,
                client=chroma_client,
                collection_name=collection_name
            )
            logger.info("Vector store initialized successfully")
        except Exception as e:
            logger.error(f"Error initializing vector store: {e}")
            # Fallback to direct initialization if the client approach fails
            self.vector_store = Chroma(
                persist_directory=persist_directory,
                embedding_function=self.embeddings
            )
            logger.info("Vector store initialized with fallback method")
    
    def setup_retriever(self):
        """Set up an advanced retriever with MMR search"""
        base_retriever = self.vector_store.as_retriever(
            search_type="mmr",
            search_kwargs={
                "k": 8,          # Number of documents to return
                "fetch_k": 20,   # Number of documents to fetch before filtering
                "lambda_mult": 0.7,  # Balance between relevance and diversity
                "score_threshold": 0.3  # Minimum relevance score
            }
        )
        
        return base_retriever

    def extract_drive_links(self, text):
        """Extract Google Drive links from text"""
        drive_pattern = r'https://drive\.google\.com/\S+'
        return re.findall(drive_pattern, text)

    def setup_qa_chain(self):
        """Set up the QA chain with custom prompt"""
        prompt_template = """You are an AI assistant for Undiksha University, designed to be helpful, friendly, and adaptive to users' communication styles. 

        Core Principles:
        1. Match the user's language style (formal/informal Indonesian or English)
        2. Be concise but comprehensive
        3. Be friendly and engaging
        4. Use clear formatting for readability
        5. Provide contextual suggestions and examples
        6. Use emojis appropriately for informal communication
        7. Always preserve and include relevant Google Drive links
        8. Maintain consistency with previous answers
        9. Provide specific details about dates, semesters, and requirements

        Context Handling Guidelines:
        1. Always reference the most specific and up-to-date information
        2. When discussing curriculum years, clearly state which batch of students it applies to
        3. For program timelines (like internships), always specify:
           - Exact semester numbers
           - Any conditions or prerequisites
           - Whether it applies to specific curriculum years
        4. If information seems to conflict, prioritize the most detailed source
        5. Always maintain consistency with previous answers about the same topic

        Response Structure:
        1. Direct Answer
           • Start with the most specific answer to the question
           • Include exact semester numbers, dates, or requirements
           • Highlight any curriculum-specific details

        2. Additional Context
           • Provide relevant prerequisites or conditions
           • Explain any variations based on curriculum year
           • Include implementation details if available

        3. Supporting Information
           • Add relevant links or references
           • Mention where to find more detailed information
           • Include any important deadlines or dates

        Current question: {question}
        Relevant context: {context}

        Answer: Let me help you with that."""

        PROMPT = PromptTemplate(
            template=prompt_template,
            input_variables=["context", "question"]
        )

        chain = RetrievalQA.from_chain_type(
            llm=self.llm,
            chain_type="stuff",
            retriever=self.setup_retriever(),
            return_source_documents=True,
            chain_type_kwargs={
                "prompt": PROMPT,
                "verbose": True
            }
        )
        
        return chain
    
    def query(self, question: str, chat_history: Optional[List] = None) -> Dict:
        """
        Query the RAG system with a question
        Returns both the answer and source documents
        """
        if not self.vector_store:
            self.initialize_vector_store()
        
        logger.info(f"Processing query: {question}")
        
        chain = self.setup_qa_chain()
        response = chain({"query": question})
        
        # Extract drive links from source documents
        drive_links = set()
        for doc in response["source_documents"]:
            links = self.extract_drive_links(doc.page_content)
            drive_links.update(links)
        
        # If there are drive links, ensure they're included in the response
        if drive_links:
            result = response["result"]
            links_in_response = any(link in result for link in drive_links)
            
            if not links_in_response:
                result += "\n\nInformasi lengkap dapat diakses melalui link:\n"
                for link in drive_links:
                    result += f"{link}\n"
            response["result"] = result
        
        return {
            "answer": response["result"],
            "sources": [doc.page_content for doc in response["source_documents"]]
        } 