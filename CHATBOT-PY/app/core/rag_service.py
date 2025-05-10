import os
import logging
import re
from typing import Dict, List
from dotenv import load_dotenv
import chromadb
from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_community.vectorstores import Chroma

# Configure logging
logger = logging.getLogger(__name__)

# Load environment variables
load_dotenv()

# ChromaDB Configuration
CHROMA_URL = os.getenv("CHROMA_URL", "https://chroma-production-0925.up.railway.app")
CHROMA_TOKEN = os.getenv("CHROMA_TOKEN", "qkj1zn522ppyn02i8wk5athfz98a1f4i")

# Global client to ensure consistent settings
_CHROMA_CLIENT = None

def get_chroma_client():
    """Get or initialize the ChromaDB client"""
    global _CHROMA_CLIENT
    if _CHROMA_CLIENT is None:
        try:
            # Connect to Railway ChromaDB
            chroma_url = CHROMA_URL
            host = chroma_url.replace("https://", "").replace("http://", "")
            _CHROMA_CLIENT = chromadb.HttpClient(
                host=host,
                port=443,
                ssl=True,
                headers={"Authorization": f"Bearer {CHROMA_TOKEN}"}
            )
            logger.info(f"Connected to Remote ChromaDB at {CHROMA_URL}")
        except Exception as e:
            logger.error(f"Error connecting to Remote ChromaDB: {e}")
            # Fallback to local ChromaDB
            try:
                _CHROMA_CLIENT = chromadb.PersistentClient(path="./chroma_db")
                logger.info("Fallback to local ChromaDB client")
            except Exception as inner_e:
                logger.error(f"Failed to initialize any ChromaDB client: {inner_e}")
                raise
    return _CHROMA_CLIENT

class RAGService:
    """RAG Service for processing queries"""
    
    def __init__(self):
        """Initialize the RAG service"""
        self.openai_api_key = os.getenv("OPENAI_API_KEY")
        if not self.openai_api_key:
            raise ValueError("OpenAI API key not found in environment variables")
        
        self.embeddings = OpenAIEmbeddings()
        self.main_vector_store = None
        self.llm = ChatOpenAI(temperature=0, model="gpt-4")
        self.initialize_vector_store()
        
    def initialize_vector_store(self):
        """Initialize the vector store"""
        try:
            # Use the client and a specific collection name
            chroma_client = get_chroma_client()
            
            # Main collection for general knowledge
            collection_name = "standalone_api"
            
            # Check if collection exists
            try:
                collections = chroma_client.list_collections()
                logger.info(f"Available collections: {collections}")
            except Exception as e:
                logger.error(f"Error listing collections: {e}")
                collections = []
            
            # Try to use an existing collection first
            try:
                # Instead of checking if the collection exists, try to get it directly
                logger.info(f"Attempting to use collection: {collection_name}")
                self.main_vector_store = Chroma(
                    client=chroma_client,
                    collection_name=collection_name,
                    embedding_function=self.embeddings
                )
                logger.info(f"Successfully connected to collection: {collection_name}")
            except Exception as e:
                logger.warning(f"Error connecting to collection {collection_name}: {e}")
                
                # Check if we should try to use another existing collection
                if collections:
                    # Try to use the first available collection
                    alt_collection = collections[0]
                    logger.info(f"Trying alternative collection: {alt_collection}")
                    try:
                        self.main_vector_store = Chroma(
                            client=chroma_client,
                            collection_name=alt_collection,
                            embedding_function=self.embeddings
                        )
                        logger.info(f"Using alternative collection: {alt_collection}")
                    except Exception as alt_e:
                        logger.error(f"Error using alternative collection: {alt_e}")
                        raise ValueError(f"Could not initialize vector store with any collection: {e}")
                else:
                    # Fallback to local ChromaDB
                    logger.warning(f"No collections available. Using local ChromaDB.")
                    try:
                        local_client = chromadb.PersistentClient(path="./chroma_db")
                        # Don't try to create a collection, just initialize an empty one
                        self.main_vector_store = Chroma(
                            client=local_client,
                            collection_name="local_fallback",
                            embedding_function=self.embeddings
                        )
                        logger.info("Initialized local ChromaDB fallback")
                    except Exception as local_e:
                        logger.error(f"Error initializing local fallback: {local_e}")
                        raise
            
            logger.info("Vector store initialized successfully")
        except Exception as e:
            logger.error(f"Error initializing vector store: {e}")
            raise
    
    def setup_retriever(self):
        """Set up a retriever for the vector store"""
        if not self.main_vector_store:
            logger.warning("Vector store not initialized")
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

    def extract_links(self, text):
        """Extract web links from text"""
        web_pattern = r'https://[^\s<>"\']+|http://[^\s<>"\']+'
        return re.findall(web_pattern, text)
        
    def query(self, question: str) -> Dict:
        """Process a query and return the answer"""
        logger.info(f"Processing query: {question}")
        
        if not self.main_vector_store:
            logger.error("Vector store not initialized")
            return {
                "answer": "Sorry, the knowledge base is not available right now. Please try again later.",
                "sources": []
            }
        
        # Set up prompt template
        prompt_template = """You are an AI assistant for Undiksha University. Please provide accurate, detailed, and helpful answers based on the provided context. 

When answering questions about curricula, programs, procedures, or any official information, make sure your answer:
1. Presents complete information from the context without omitting important details
2. Maintains the same level of detail and specificity as in the original context
3. Preserves any numbered or bulleted lists in their original format
4. Includes all relevant dates, requirements, procedures, and contextual information
5. Maintains the same tone and style as official university communications

Question: {question}
Context: {context}

Answer:"""

        # Create QA chain using retriever
        retriever = self.setup_retriever()
        if retriever:
            PROMPT = PromptTemplate(
                template=prompt_template,
                input_variables=["context", "question"]
            )

            qa_chain = RetrievalQA.from_chain_type(
                llm=self.llm,
                chain_type="stuff",
                retriever=retriever,
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
                
                return self.process_answer(result, source_docs)
            except Exception as e:
                logger.error(f"Error in query processing: {e}")
                return {
                    "answer": f"I encountered an error while processing your query. Please try again or rephrase your question.",
                    "sources": []
                }
        else:
            logger.error("No retriever available")
            return {
                "answer": "Sorry, I cannot retrieve information right now. Please try again later.",
                "sources": []
            }
    
    def process_answer(self, result, source_docs):
        """Process the answer, adding links if necessary"""
        # Extract links from source documents
        all_links = set()
        
        # Track unique source files
        source_files = set()
        
        for doc in source_docs:
            # Extract links from the content
            all_links.update(self.extract_links(doc.page_content))
            
            # Get source file name from metadata if available
            if hasattr(doc, 'metadata') and 'source' in doc.metadata:
                source_name = doc.metadata['source']
                # Remove path prefix and file extension if present
                if isinstance(source_name, str):
                    source_name = os.path.basename(source_name)
                    source_name = os.path.splitext(source_name)[0]
                    source_files.add(source_name)
        
        # Check if links are already in the result
        links_in_result = any(link in result for link in all_links)
        
        # Add missing links to the response if not already included
        if not links_in_result and all_links:
            # If the result doesn't end with a period, add one
            if result and not result.endswith(('.', '!', '?')):
                result += '.'
                
            # Add a newline if there isn't one already
            if not result.endswith('\n'):
                result += '\n\n'
            else:
                result += '\n'
                
            result += "Informasi lengkap dapat diakses melalui link:\n"
            
            # Add links
            for link in all_links:
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