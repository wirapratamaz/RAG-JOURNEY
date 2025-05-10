import os
import sys
import logging
import uvicorn
import argparse
from dotenv import load_dotenv
import chromadb
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List, Dict, Optional
from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_community.vectorstores import Chroma
import re

# Add src directory to path if running from root
current_dir = os.path.dirname(os.path.abspath(__file__))
if os.path.basename(current_dir) == 'CHATBOT-PY':
    # We're in the root directory
    sys.path.append(os.path.join(current_dir, 'src'))
elif os.path.basename(current_dir) == 'src':
    # We're in the src directory
    sys.path.append(os.path.dirname(current_dir))

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Load environment variables
load_dotenv()

# Initialize FastAPI app
app = FastAPI(
    title="Undiksha RAG Chatbot API",
    description="Standalone API for RAG-based chatbot",
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

# Railway ChromaDB Configuration
CHROMA_URL = os.getenv("CHROMA_URL", "https://chroma-production-0925.up.railway.app")
CHROMA_TOKEN = os.getenv("CHROMA_TOKEN", "qkj1zn522ppyn02i8wk5athfz98a1f4i")

# Global client to ensure consistent settings
_CHROMA_CLIENT = None

def get_chroma_client():
    global _CHROMA_CLIENT
    if _CHROMA_CLIENT is None:
        try:
            # Connect to Railway ChromaDB - updated for ChromaDB 0.6.3
            chroma_url = CHROMA_URL
            host = chroma_url.replace("https://", "").replace("http://", "")
            _CHROMA_CLIENT = chromadb.HttpClient(
                host=host,
                port=443,
                ssl=True,
                headers={"Authorization": f"Bearer {CHROMA_TOKEN}"}
            )
            logger.info(f"Connected to Railway ChromaDB at {CHROMA_URL}")
        except Exception as e:
            logger.error(f"Error connecting to Railway ChromaDB: {e}")
            # Fallback to local ChromaDB if available
            try:
                _CHROMA_CLIENT = chromadb.PersistentClient(path="./standalone_chroma_db")
                logger.info("Fallback to local ChromaDB client")
            except Exception as inner_e:
                logger.error(f"Failed to initialize any ChromaDB client: {inner_e}")
                raise
    return _CHROMA_CLIENT

class RAGPipeline:
    def __init__(self):
        self.openai_api_key = os.getenv("OPENAI_API_KEY")
        if not self.openai_api_key:
            raise ValueError("OpenAI API key not found in environment variables")
        
        self.embeddings = OpenAIEmbeddings()
        self.main_vector_store = None
        self.faq_vector_store = None
        self.llm = ChatOpenAI(temperature=0, model="gpt-4")
        self.initialize_vector_stores()
        
    def initialize_vector_stores(self):
        """Initialize all vector stores with consistent settings"""
        try:
            # Use the client and a specific collection name
            chroma_client = get_chroma_client()
            
            # Main collection for general knowledge
            main_collection_name = "standalone_api"
            
            # FAQ collection for exact answers
            faq_collection_name = "faqs_collection"
            
            # Check if collections exist - updated for ChromaDB 0.6.3
            try:
                collections = chroma_client.list_collections()
                # In ChromaDB 0.6.0+, list_collections returns collection names
                logger.info(f"Available collections: {collections}")
            except Exception as e:
                logger.error(f"Error listing collections: {e}")
                collections = []
            
            # Initialize main collection
            main_collection_exists = main_collection_name in collections
            if main_collection_exists:
                logger.info(f"Using existing ChromaDB collection: {main_collection_name}")
                self.main_vector_store = Chroma(
                    client=chroma_client,
                    collection_name=main_collection_name,
                    embedding_function=self.embeddings
                )
            else:
                logger.warning(f"Main collection {main_collection_name} not found. Creating it now.")
                # Create collection
                chroma_client.create_collection(main_collection_name)
                self.main_vector_store = Chroma(
                    client=chroma_client,
                    collection_name=main_collection_name,
                    embedding_function=self.embeddings
                )
                logger.info(f"Created new collection: {main_collection_name}")
                
            # Initialize FAQ collection
            faq_collection_exists = faq_collection_name in collections
            if faq_collection_exists:
                logger.info(f"Using existing FAQ collection: {faq_collection_name}")
                self.faq_vector_store = Chroma(
                    client=chroma_client,
                    collection_name=faq_collection_name,
                    embedding_function=self.embeddings
                )
            else:
                logger.warning(f"FAQ collection {faq_collection_name} not found. Creating it now.")
                # Create collection
                chroma_client.create_collection(faq_collection_name)
                self.faq_vector_store = Chroma(
                    client=chroma_client,
                    collection_name=faq_collection_name,
                    embedding_function=self.embeddings
                )
                logger.info(f"Created new collection: {faq_collection_name}")
                
            logger.info("Vector stores initialized successfully")
        except Exception as e:
            logger.error(f"Error initializing vector stores: {e}")
            raise
    
    def setup_main_retriever(self):
        """Set up a retriever for the main vector store"""
        if not self.main_vector_store:
            logger.warning("Main vector store not initialized")
            return None
            
        return self.main_vector_store.as_retriever(
            search_type="mmr",
            search_kwargs={
                "k": 5,
                "fetch_k": 10,
                "lambda_mult": 0.7,
                "score_threshold": 0.3
            }
        )
        
    def setup_faq_retriever(self):
        """Set up a retriever for the FAQ vector store"""
        if not self.faq_vector_store:
            logger.warning("FAQ vector store not initialized")
            return None
            
        return self.faq_vector_store.as_retriever(
            search_kwargs={
                "k": 1,  # Only get the best match
                "score_threshold": 0.7  # Higher threshold for FAQ matching
            }
        )

    def extract_drive_links(self, text):
        """Extract Google Drive links from text"""
        drive_pattern = r'https://drive\.google\.com/\S+'
        return re.findall(drive_pattern, text)
        
    def extract_web_links(self, text):
        """Extract web links from text"""
        web_pattern = r'https://[^\s<>"\']+|http://[^\s<>"\']+'
        return re.findall(web_pattern, text)
        
    def is_faq_question(self, question):
        """Check if this is likely an FAQ question"""
        # Common patterns for FAQ questions
        faq_patterns = [
            r"apa itu",
            r"bagaimana cara",
            r"apa saja",
            r"jelaskan tentang",
            r"siapa",
            r"dimana",
            r"berapa",
            r"kapan",
            r"mengapa",
            r"where",
            r"what",
            r"who",
            r"how",
            r"when",
            r"why"
        ]
        
        question_lower = question.lower()
        return any(re.search(pattern, question_lower) for pattern in faq_patterns)

    def query(self, question: str) -> Dict:
        """Process a query and return the answer"""
        logger.info(f"Processing query: {question}")
        
        # Check if this might be an FAQ question
        is_faq = self.is_faq_question(question)
        logger.info(f"Is FAQ question: {is_faq}")
        
        # Try to find an exact FAQ match first
        faq_match = None
        if is_faq and self.faq_vector_store:
            try:
                faq_retriever = self.setup_faq_retriever()
                if faq_retriever:
                    faq_docs = faq_retriever.get_relevant_documents(question)
                    logger.info(f"Found {len(faq_docs)} FAQ matches")
                    
                    # Check if we have a good match
                    if faq_docs:
                        top_doc = faq_docs[0]
                        if "Answer:" in top_doc.page_content:
                            # Extract the answer from the document
                            faq_match = top_doc.page_content.split("Answer:", 1)[1].strip()
                            logger.info(f"Using exact FAQ answer")
                            
                            # Process links in the FAQ answer
                            return self.process_final_answer(faq_match, faq_docs)
            except Exception as e:
                logger.error(f"Error searching FAQs: {e}")
        
        # If no FAQ match or not an FAQ question, use the main knowledge base
        if not faq_match and self.main_vector_store:
            # Set up enhanced prompt template for better answers
            prompt_template = """You are an AI assistant for Undiksha University. Please provide accurate, detailed, and helpful answers based on the provided context. 

When answering questions about curricula, programs, procedures, or any official information, make sure your answer:
1. Presents complete information from the context without omitting important details
2. Maintains the same level of detail and specificity as in the original context
3. Preserves any numbered or bulleted lists in their original format
4. Includes all relevant dates, requirements, procedures, and contextual information
5. Maintains the same tone and style as official university communications

When a question matches an FAQ, provide the COMPLETE and EXACT answer from the FAQ without summarizing or simplifying.

Question: {question}
Context: {context}

Answer:"""

            # Create QA chain using main retriever
            main_retriever = self.setup_main_retriever()
            if main_retriever:
                PROMPT = PromptTemplate(
                    template=prompt_template,
                    input_variables=["context", "question"]
                )

                qa_chain = RetrievalQA.from_chain_type(
                    llm=self.llm,
                    chain_type="stuff",
                    retriever=main_retriever,
                    return_source_documents=True,
                    chain_type_kwargs={
                        "prompt": PROMPT
                    }
                )
                
                # Process the query
                try:
                    response = qa_chain({"query": question})
                    result = response["result"]
                    source_docs = response["source_documents"]
                    
                    return self.process_final_answer(result, source_docs)
                except Exception as e:
                    logger.error(f"Error in query processing with main knowledge base: {e}")
                    raise
            else:
                logger.error("No main retriever available")
                raise ValueError("Main retriever not available")
        elif not faq_match:
            logger.error("No vector stores available")
            raise ValueError("Vector stores not initialized properly")
    
    def process_final_answer(self, result, source_docs):
        """Process the final answer, adding links if necessary"""
        # Extract drive links from source documents
        drive_links = set()
        web_links = set()
        
        # Track unique source files
        source_files = set()
        
        for doc in source_docs:
            links = self.extract_drive_links(doc.page_content)
            drive_links.update(links)
            
            # Also extract web links from the content
            web_links.update(self.extract_web_links(doc.page_content))
            
            # Get source file name from metadata if available
            if hasattr(doc, 'metadata') and 'source' in doc.metadata:
                source_name = doc.metadata['source']
                # Remove path prefix and file extension if present
                if isinstance(source_name, str):
                    source_name = os.path.basename(source_name)
                    source_name = os.path.splitext(source_name)[0]
                    source_files.add(source_name)
        
        # Remove drive links from web links (to avoid duplicates)
        web_links = {link for link in web_links if not link.startswith("https://drive.google.com")}
        
        # Check if links are already in the result
        links_in_result = any(link in result for link in drive_links.union(web_links))
        
        # Add missing links to the response if not already included
        if not links_in_result and (drive_links or web_links):
            # If the result doesn't end with a period, add one
            if result and not result.endswith(('.', '!', '?')):
                result += '.'
                
            # Add a newline if there isn't one already
            if not result.endswith('\n'):
                result += '\n\n'
            else:
                result += '\n'
                
            result += "Informasi lengkap dapat diakses melalui link:\n"
            
            # Add drive links first
            for link in drive_links:
                result += f"{link}\n"
                
            # Add other web links
            for link in web_links:
                result += f"{link}\n"
        
        # Add source citation at the end
        if source_files:
            # Ensure proper spacing before adding source citation
            if not result.endswith('\n'):
                result += '\n\n'
            elif not result.endswith('\n\n'):
                result += '\n'
                
            # Format source names
            sources_text = ", ".join(source_files)
            result += f"Sumber data: {sources_text}"
        
        # Return the final answer and sources
        return {
            "answer": result,
            "sources": [doc.page_content for doc in source_docs]
        }

# Initialize RAG Pipeline as a global instance
try:
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
    # Parse command line arguments
    parser = argparse.ArgumentParser(description="Run the Undiksha RAG Chatbot API")
    parser.add_argument("--host", type=str, default="0.0.0.0", help="Host to run the server on")
    parser.add_argument("--port", type=int, default=8000, help="Port to run the server on")
    args = parser.parse_args()
    
    # Run the server
    logger.info(f"Starting server on {args.host}:{args.port}")
    uvicorn.run("standalone_api:app", host=args.host, port=args.port, reload=True) 