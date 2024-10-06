import requests
import logging
import feedparser
from retriever import vectorstore
from langchain_openai import OpenAIEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter

logging.basicConfig(level=logging.INFO)

def fetch_rss_posts(feed_url="https://is.undiksha.ac.id/feed/", max_posts=5):
    """
    Fetches the latest posts from the RSS feed.

    Args:
        feed_url (str): The RSS feed URL.
        max_posts (int): Maximum number of posts to fetch.

    Returns:
        list: A list of dictionaries containing post data.
    """
    try:
        feed = feedparser.parse(feed_url)
        if feed.bozo:
            raise ValueError("Failed to parse RSS feed.")
        
        posts = []
        for entry in feed.entries[:max_posts]:
            title = entry.get('title', 'No Title')
            link = entry.get('link', '#')
            description = entry.get('description', 'No Content')
            posts.append({
                'title': title,
                'link': link,
                'content': description
            })
        
        logging.info(f"Fetched {len(posts)} posts from RSS feed.")
        return posts
    except Exception as e:
        logging.error(f"Error fetching RSS posts: {e}")
        return []

def process_and_embed_posts(posts):
    """
    Processes fetched posts and embeds them into the vector store.

    Args:
        posts (list): List of post dictionaries fetched from RSS feed.
    """
    try:
        texts = []
        metadatas = []
        for post in posts:
            title = post.get('title', 'No Title')
            content = post.get('content', 'No Content')
            full_text = f"{title}\n{content}"
            texts.append(full_text)
            metadatas.append({"source": "RSS Feed", "title": title, "link": post.get('link', '#')})

        # Split texts into chunks
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
        chunks = text_splitter.split_texts(texts)

        # Initialize embeddings
        embeddings = OpenAIEmbeddings()

        # Add documents to the vector store
        vectorstore.add_texts(texts=chunks, metadatas=metadatas, embedding=embeddings)
        logging.info("Successfully embedded and added posts to the vector store.")
    except Exception as e:
        logging.error(f"Error processing and embedding posts: {e}")

def get_latest_posts(formatted=False):
    """
    Fetches and processes the latest RSS feed posts.

    Args:
        formatted (bool): If True, returns the posts in a formatted manner for display.

    Returns:
        list or None: Returns processed posts or None if fetching fails.
    """
    posts = fetch_rss_posts()
    if posts:
        process_and_embed_posts(posts)
        if formatted:
            formatted_posts = [{"title": post.get('title', 'No Title'),
                                "link": post.get('link', '#')} for post in posts]
            return formatted_posts
    return None

if __name__ == "__main__":
    posts = fetch_rss_posts()
    if posts:
        process_and_embed_posts(posts)