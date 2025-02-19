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
from spell_checker import SpellChecker

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
        self.spell_checker = SpellChecker()
        
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
            search_type="mmr",
            search_kwargs={
                "k": 8,          # Reduced for more focused results
                "fetch_k": 20,   # Adjusted fetch_k for better diversity
                "lambda_mult": 0.7,  # Increased to favor relevance over diversity
                "score_threshold": 0.3  # Increased threshold for higher quality matches
            }
        )
        
        # Enhanced prompt for the compressor
        enhanced_prompt = """Given the following context and question, identify and extract the most relevant information that directly answers the question or provides essential context. Pay special attention to:
1. Specific dates, semesters, or time periods mentioned
2. Requirements or conditions
3. Related policies or guidelines
4. Any additional context that helps provide a complete answer

Question: {question}
Context: {context}

Relevant Information:"""

        # Create a compressor with enhanced prompt
        compressor = LLMChainExtractor.from_llm(
            self.llm,
            prompt=enhanced_prompt
        )
        
        # Create a contextual compression retriever
        compression_retriever = ContextualCompressionRetriever(
            base_compressor=compressor,
            base_retriever=base_retriever,
            compression_kwargs={"min_chunk_length": 200}  # Increased to preserve more context
        )
        
        return compression_retriever

    def extract_drive_links(self, text):
        """Extract Google Drive links from text"""
        import re
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
        5. Help with spelling corrections when needed
        6. Provide contextual suggestions and examples
        7. Use emojis appropriately for informal communication
        8. Always preserve and include relevant Google Drive links
        9. Maintain consistency with previous answers
        10. Provide specific details about dates, semesters, and requirements

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

        Previous conversation context: {chat_history}
        Current question: {question}
        Relevant context: {context}

        Answer: Let me help you with that."""

        PROMPT = PromptTemplate(
            template=prompt_template,
            input_variables=["context", "question", "chat_history"]
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
    
    def query(self, question: str) -> Dict:
        """
        Query the RAG system with a question
        Returns both the answer and source documents
        """
        if not self.vector_store:
            self.initialize_vector_store()
        
        print("\n=== DEBUG: Starting query processing ===")
        print(f"Question: {question}")
        
        # First, correct any spelling mistakes
        corrected_question, corrections = self.spell_checker.correct_text(question)
        print(f"\nCorrected question: {corrected_question}")
        
        chain = self.setup_qa_chain()
        print("\n=== DEBUG: Retrieving documents ===")
        response = chain({"query": corrected_question})
        
        print("\n=== DEBUG: Retrieved Documents ===")
        for idx, doc in enumerate(response["source_documents"]):
            print(f"\nDocument {idx + 1}:")
            print("Content:", doc.page_content[:200] + "..." if len(doc.page_content) > 200 else doc.page_content)
            if hasattr(doc, 'metadata'):
                print("Metadata:", doc.metadata)
        
        # Extract drive links from source documents
        print("\n=== DEBUG: Extracting Drive Links ===")
        drive_links = set()
        for doc in response["source_documents"]:
            links = self.extract_drive_links(doc.page_content)
            if links:
                print(f"Found links in document: {links}")
            drive_links.update(links)
        
        print(f"\nTotal unique drive links found: {len(drive_links)}")
        if drive_links:
            print("Links:", drive_links)
        
        # If there are drive links, ensure they're included in the response
        if drive_links:
            result = response["result"]
            print("\n=== DEBUG: Checking if links are in response ===")
            links_in_response = any(link in result for link in drive_links)
            print(f"Links already in response: {links_in_response}")
            
            if not links_in_response:
                print("Adding links to response")
                result += "\n\nInformasi lengkap dapat diakses melalui link:\n"
                for link in drive_links:
                    result += f"{link}\n"
            response["result"] = result
        
        # If there were corrections, add them to the response
        if corrections:
            print("\n=== DEBUG: Processing corrections ===")
            # Determine the style based on the language and formality of the question
            style = "formal"
            if any(word in question.lower() for word in ["aku", "gue", "gw", "lu", "km"]):
                style = "casual"
            elif all(ord(c) < 128 for c in question):  # If all characters are ASCII (likely English)
                style = "english"
            
            print(f"Detected style: {style}")
            correction_text = self.spell_checker.format_corrections(corrections, style)
            response["result"] = correction_text + "\n" + response["result"]
        
        print("\n=== DEBUG: Final Response ===")
        print(response["result"])
        print("\n=== DEBUG: End of processing ===\n")
        
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