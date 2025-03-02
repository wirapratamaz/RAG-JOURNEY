"""
Retriever module - provides access to the retriever instance.
This is a compatibility layer that imports the retriever from either
src.retriever or simple_retriever depending on availability.
"""

import logging
import sys

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

try:
    # First try to import from src.retriever
    logger.info("Attempting to import retriever from src.retriever")
    from src.retriever import retriever
    logger.info("Successfully imported retriever from src.retriever")
except ImportError:
    # If that fails, try to import from simple_retriever
    logger.info("Failed to import from src.retriever, trying simple_retriever")
    try:
        from simple_retriever import retriever
        logger.info("Successfully imported retriever from simple_retriever")
    except ImportError:
        # Last resort - create a dummy retriever
        logger.error("Failed to import retriever from any source. Creating dummy retriever.")
        
        class DummyRetriever:
            def get_relevant_documents(self, query):
                logger.warning(f"DummyRetriever used for query: {query}")
                return [type('obj', (object,), {
                    'page_content': 'This is a dummy document from a fallback retriever.',
                    'metadata': {'source': 'dummy fallback'}
                })]
        
        # Create a dummy retriever instance
        retriever = DummyRetriever()

# Export the retriever
__all__ = ['retriever'] 