import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Simple Dummy Retriever for testing deployment
class SimpleRetriever:
    def __init__(self):
        logger.info("Initializing SimpleRetriever")
        self.dummy_documents = [
            {
                'page_content': 'Program Studi Sistem Informasi merupakan salah satu program studi di Universitas Pendidikan Ganesha.',
                'metadata': {'source': 'dummy data'}
            },
            {
                'page_content': 'Visi Program Studi Sistem Informasi adalah menjadi program studi yang unggul dalam bidang Sistem Informasi berlandaskan falsafah Tri Hita Karana di Asia pada tahun 2045.',
                'metadata': {'source': 'dummy data'}
            },
            {
                'page_content': 'Kurikulum Program Studi Sistem Informasi dirancang untuk memenuhi kebutuhan industri dengan fokus pada pengembangan aplikasi, analisis data, dan manajemen sistem informasi.',
                'metadata': {'source': 'dummy data'}
            }
        ]
    
    def get_relevant_documents(self, query):
        logger.info(f"SimpleRetriever received query: {query}")
        # Return all dummy documents regardless of query
        return [type('obj', (object,), {
            'page_content': doc['page_content'],
            'metadata': doc['metadata']
        }) for doc in self.dummy_documents]

# Initialize a simple retriever instance
retriever = SimpleRetriever()

# Export the retriever
__all__ = ['retriever'] 