import os
import logging
from dotenv import load_dotenv

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Load environment variables
load_dotenv()
CRAWL_URL = os.getenv("CRAWL_URL", "https://is.undiksha.ac.id/")

# Try to import necessary modules
try:
    import asyncio
    from bs4 import BeautifulSoup
    
    try:
        from crawl4ai import AsyncWebCrawler
        CRAWLER_AVAILABLE = True
    except ImportError:
        logger.warning("crawl4ai not available, will use fallback mechanism")
        CRAWLER_AVAILABLE = False
except ImportError as e:
    logger.error(f"Error importing required modules: {e}")
    CRAWLER_AVAILABLE = False

async def scraping_undiksha_website(url=CRAWL_URL):
    """
    Scrapes content from the Undiksha website.
    
    Args:
        url (str): The URL to scrape.
        
    Returns:
        str: The scraped content.
    """
    if not CRAWLER_AVAILABLE:
        logger.warning("Crawler not available, returning empty string")
        return ""
        
    try:
        async with AsyncWebCrawler(verbose=True) as crawler:
            result = await crawler.arun(url=url, bypass_cache=True)
            if result.success:
                soup = BeautifulSoup(result.cleaned_html, 'html.parser')
                # Extract relevant data, e.g., all paragraphs
                paragraphs = soup.find_all('p')
                content = "\n".join([p.get_text() for p in paragraphs])
                logger.info(f"Successfully scraped content from {url}")
                return content
            else:
                logger.warning(f"Failed to retrieve content from {url}")
                return ""
    except Exception as e:
        logger.error(f"An error occurred while scraping {url}: {e}")
        return ""

def get_crawled_content(url=CRAWL_URL, selector=None):
    """
    Gets crawled content from a website with fallback to mock data if needed.
    
    Args:
        url (str): The URL to crawl.
        selector (str): Optional CSS selector to target specific content.
        
    Returns:
        dict: A dictionary containing the title and content.
    """
    logger.info(f"Getting crawled content for {url}")
    
    try:
        # Try to use the async scraper if available
        if CRAWLER_AVAILABLE:
            try:
                content = asyncio.run(scraping_undiksha_website(url))
                if content:
                    return {
                        "title": "Scraped Content",
                        "content": content
                    }
            except Exception as e:
                logger.error(f"Error running async scraper: {e}")
        
        # Fall back to mock data if scraping failed or crawler not available
        return _get_mock_content(url)
    except Exception as e:
        logger.error(f"Error in get_crawled_content: {e}")
        return _get_mock_content(url)

def _get_mock_content(url=None):
    """
    Returns mock data when real content can't be scraped.
    """
    logger.info(f"Using mock content for {url}")
    
    # Return dummy content based on URL or default
    if url and "undiksha.ac.id" in url:
        return {
            "title": "Program Studi Sistem Informasi Undiksha",
            "content": """
            Program Studi Sistem Informasi Universitas Pendidikan Ganesha (Undiksha)
            merupakan program studi yang berfokus pada pengembangan solusi teknologi 
            informasi untuk mendukung proses bisnis.
            
            Visi:
            Menjadi program studi yang unggul dalam bidang Sistem Informasi 
            berlandaskan falsafah Tri Hita Karana di Asia pada tahun 2045.
            
            Misi:
            1. Menyelenggarakan pendidikan dan pengajaran yang bermutu dalam bidang
               Sistem Informasi dengan berlandaskan falsafah Tri Hita Karana.
            2. Menyelenggarakan penelitian yang inovatif dan kompetitif dalam 
               bidang Sistem Informasi untuk pengembangan ilmu dan kesejahteraan masyarakat.
            3. Menyelenggarakan pengabdian kepada masyarakat sebagai implementasi
               hasil pendidikan dan penelitian bidang Sistem Informasi.
            """
        }
    else:
        # Default content
        return {
            "title": "Informasi Web (Mock)",
            "content": "Ini adalah konten simulasi untuk keperluan pengujian aplikasi."
        }