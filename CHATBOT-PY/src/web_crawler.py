import requests
from bs4 import BeautifulSoup

def crawl_undiksha_website(url="https://is.undiksha.ac.id/"):
    try:
        response = requests.get(url)
        response.raise_for_status()  # Raise an error for bad responses
        soup = BeautifulSoup(response.text, 'html.parser')

        # Extract relevant data, e.g., all paragraphs
        paragraphs = soup.find_all('p')
        content = "\n".join([p.get_text() for p in paragraphs])

        # You can also extract other elements like headings, links, etc.
        return content
    except Exception as e:
        print(f"An error occurred while crawling: {e}")
        return ""