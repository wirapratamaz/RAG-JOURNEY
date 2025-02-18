import os
import logging
from typing import List, Dict
from dotenv import load_dotenv
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import OpenAIEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.chains import RetrievalQA
from langchain_openai import ChatOpenAI
from langchain.prompts import PromptTemplate
from langchain.retrievers import ContextualCompressionRetriever
from langchain.retrievers.document_compressors import LLMChainExtractor

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Load environment variables
load_dotenv()

class RAGPipeline:
    def __init__(self):
        self.openai_api_key = os.getenv("OPENAI_API_KEY")
        if not self.openai_api_key:
            raise ValueError("OpenAI API key not found in environment variables")
        
        self.embeddings = OpenAIEmbeddings()
        self.vector_store = None
        self.llm = ChatOpenAI(temperature=0, model="gpt-4")
        
    def initialize_vector_store(self):
        """Initialize or load the existing vector store"""
        persist_directory = "./chroma_db"
        self.vector_store = Chroma(
            persist_directory=persist_directory,
            embedding_function=self.embeddings
        )
        logger.info("Vector store initialized successfully")
    
    def setup_retriever(self):
        """Set up an advanced retriever with contextual compression"""
        base_retriever = self.vector_store.as_retriever(
            search_type="mmr",  # Changed to MMR for better diversity in results
            search_kwargs={
                "k": 12,        # Increased from 8 to 12 to get more chunks
                "fetch_k": 30,  # Increased from 20 to 30
                "lambda_mult": 0.7,  # Controls diversity vs relevance trade-off
                "score_threshold": 0.3  # Lowered threshold to get more potential matches
            }
        )
        
        # Create a compressor that will help extract most relevant parts of documents
        compressor = LLMChainExtractor.from_llm(self.llm)
        
        # Create a contextual compression retriever with custom kwargs
        compression_retriever = ContextualCompressionRetriever(
            base_compressor=compressor,
            base_retriever=base_retriever,
            compression_kwargs={"min_chunk_length": 50}  # Ensure we don't compress too aggressively
        )
        
        return compression_retriever

    def setup_qa_chain(self):
        """Set up the QA chain with custom prompt"""
        prompt_template = """You are an AI assistant for Undiksha University. Use the following pieces of context to answer the question at the end. 
        If you don't know the answer, just say that you don't know, don't try to make up an answer.
        
        When formatting responses with bullet points:
        1. Start with an introductory sentence followed by a line break
        2. Each bullet point should be on a new line
        3. Use bullet points (•) for each item
        4. Add proper spacing between bullet points
        
        Context: {context}
        
        Question: {question}
        
        Answer: Let me help you with that."""

        PROMPT = PromptTemplate(
            template=prompt_template, input_variables=["context", "question"]
        )

        chain = RetrievalQA.from_chain_type(
            llm=self.llm,
            chain_type="stuff",
            retriever=self.setup_retriever(),
            return_source_documents=True,
            chain_type_kwargs={"prompt": PROMPT}
        )
        
        return chain
    
    def query(self, question: str) -> Dict:
        """
        Query the RAG system with a question
        Returns both the answer and source documents
        """
        if not self.vector_store:
            self.initialize_vector_store()
            
        chain = self.setup_qa_chain()
        response = chain({"query": question})
        
        return {
            "answer": response["result"],
            "sources": [doc.page_content for doc in response["source_documents"]]
        }

# Example usage
if __name__ == "__main__":
    rag = RAGPipeline()
    
    # Test query
    question = "What are the requirements for final year project (Karya Akhir)?"
    result = rag.query(question)
    
    print("\nQuestion:", question)
    print("\nAnswer:", result["answer"])
    print("\nSources used:")
    for idx, source in enumerate(result["sources"], 1):
        print(f"\nSource {idx}:")
        print(source) 