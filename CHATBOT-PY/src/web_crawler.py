import os
import asyncio
import logging
from crawl4ai import AsyncWebCrawler
from bs4 import BeautifulSoup
from dotenv import load_dotenv

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Load environment variables
load_dotenv()
CRAWL_URL = os.getenv("CRAWL_URL", "https://is.undiksha.ac.id/")

async def scraping_undiksha_website(url=CRAWL_URL):
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
                logger.warning(f"failed to retrieve content from {url}")
                return ""
    except Exception as e:
        logger.error(f"An error occurred while scraping {url}: {e}")
        return ""

def get_scraped_content(url=CRAWL_URL):
    return asyncio.run(scraping_undiksha_website(url))