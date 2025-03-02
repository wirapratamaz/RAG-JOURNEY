import logging
from retriever import retriever

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Test query
query = "Apa saja tahapan utama dalam proses penyelesaian skripsi di Program Studi Sistem Informasi?"

# Get relevant documents
try:
    docs = retriever.get_relevant_documents(query)
    
    print(f"Found {len(docs)} relevant documents:")
    for i, doc in enumerate(docs):
        print(f"\n--- Document {i+1} ---")
        print(f"Content: {doc.page_content[:500]}...")  # Print first 500 chars
        print(f"Source: {doc.metadata.get('source', 'unknown')}")
except Exception as e:
    logger.error(f"Error retrieving documents: {e}") 