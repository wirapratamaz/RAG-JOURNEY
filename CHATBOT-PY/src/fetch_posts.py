import logging
import sys
import time
import os
import json
from datetime import datetime, timedelta

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Cache directory
CACHE_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "cache")
os.makedirs(CACHE_DIR, exist_ok=True)
RSS_CACHE_FILE = os.path.join(CACHE_DIR, "rss_cache.json")
RSS_CACHE_EXPIRY = 3600  # Cache expiry in seconds (1 hour)

# Try to import necessary modules
try:
    import feedparser
    import requests
    from langchain.text_splitter import RecursiveCharacterTextSplitter
    
    # Try to import vector store - but don't fail if not available
    try:
        from src.retriever import retriever
        # Check if we have a real vectorstore
        vectorstore = None
        if hasattr(retriever, 'vectorstore'):
            vectorstore = retriever.vectorstore
        elif hasattr(retriever, '_vectorstore'):
            vectorstore = retriever._vectorstore
    except ImportError:
        logger.warning("Could not import retriever for embedding posts")
        vectorstore = None
        
    try:
        from langchain_community.embeddings import HuggingFaceEmbeddings
        embeddings_available = True
    except ImportError:
        logger.warning("HuggingFaceEmbeddings not available")
        embeddings_available = False
        
    FEEDPARSER_AVAILABLE = True
except ImportError as e:
    logger.error(f"Error importing required modules: {e}")
    FEEDPARSER_AVAILABLE = False

def fetch_rss_posts(feed_url="https://is.undiksha.ac.id/feed/", max_posts=5):
    """
    Fetches the latest posts from the RSS feed.
    
    Args:
        feed_url (str): The RSS feed URL.
        max_posts (int): Maximum number of posts to fetch.
        
    Returns:
        list: A list of dictionaries containing post data.
    """
    if not FEEDPARSER_AVAILABLE:
        logger.warning("Feedparser not available, returning empty list")
        return []
        
    try:
        feed = feedparser.parse(feed_url)
        if feed.bozo and not feed.entries:
            logger.error("Failed to parse RSS feed.")
            return []
        
        posts = []
        for entry in feed.entries[:max_posts]:
            title = entry.get('title', 'No Title')
            link = entry.get('link', '#')
            description = entry.get('description', 'No Content')
            published = entry.get('published', 'Unknown date')
            
            posts.append({
                'title': title,
                'link': link,
                'content': description,
                'published': published
            })
        
        logger.info(f"Fetched {len(posts)} posts from RSS feed.")
        return posts
    except Exception as e:
        logger.error(f"Error fetching RSS posts: {e}")
        return []

def process_and_embed_posts(posts):
    """
    Processes fetched posts and embeds them into the vector store if available.
    
    Args:
        posts (list): List of post dictionaries fetched from RSS feed.
    """
    # If no vectorstore is available, just return the posts
    if vectorstore is None or not embeddings_available:
        logger.warning("Vector store not available, skipping embeddings")
        return posts
        
    try:
        # Combine all post content into a single text
        combined_text = ""
        for post in posts:
            title = post.get('title', 'No Title')
            content = post.get('content', 'No Content')
            combined_text += f"Title: {title}\nContent: {content}\n\n"

        # Split texts into chunks
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=300,
            separators=["\n\n", "\n", "•", " ", ""]
        )
        chunks = text_splitter.split_text(combined_text)

        # Initialize embeddings
        embeddings = HuggingFaceEmbeddings(
            model_name="all-MiniLM-L6-v2"
        )

        # Add documents to the vector store if possible
        try:
            metadatas = [{"source": "RSS Feed"} for _ in chunks]
            
            # Try different methods of adding documents based on the ChromaDB version
            try:
                # First try add_texts with embedding parameter (older versions)
                vectorstore.add_texts(texts=chunks, metadatas=metadatas, embedding=embeddings)
                logger.info("Successfully added documents using add_texts with embedding parameter")
            except (TypeError, ValueError) as e:
                logger.warning(f"First add method failed: {e}, trying alternative methods...")
                
                try:
                    # Try add_texts without embedding parameter (newer versions)
                    vectorstore.add_texts(texts=chunks, metadatas=metadatas)
                    logger.info("Successfully added documents using add_texts without embedding parameter")
                except Exception as e2:
                    logger.warning(f"Second add method failed: {e2}, trying direct client access...")
                    
                    try:
                        # Try accessing the collection directly if possible
                        if hasattr(vectorstore, "_collection"):
                            collection = vectorstore._collection
                            # Generate embeddings manually
                            text_embeddings = embeddings.embed_documents(chunks)
                            # Add via the collection
                            collection.add(
                                embeddings=text_embeddings,
                                documents=chunks,
                                metadatas=metadatas,
                                ids=[f"rss-{i}" for i in range(len(chunks))]
                            )
                            logger.info("Successfully added documents using direct collection access")
                        else:
                            raise ValueError("No _collection attribute found")
                    except Exception as e3:
                        logger.error(f"All embedding methods failed: {e3}")
                        raise
                        
        except Exception as e:
            logger.error(f"Error adding to vector store: {e}")
            
    except Exception as e:
        logger.error(f"Error processing and embedding posts: {e}")
    
    return posts

def get_latest_posts(formatted=False, max_posts=3, use_cache=True, embed_posts=False):
    """
    Fetches and optionally processes the latest RSS feed posts.
    
    Args:
        formatted (bool): If True, returns the posts in a formatted manner for display.
        max_posts (int): Maximum number of posts to return.
        use_cache (bool): If True, tries to use cached posts if available and not expired.
        embed_posts (bool): If True, embeds the posts in the vector store. This is an expensive operation.
        
    Returns:
        list: Returns processed posts.
    """
    logger.info(f"Getting latest posts, formatted={formatted}, max_posts={max_posts}, use_cache={use_cache}, embed_posts={embed_posts}")
    
    # Try to use cache if requested
    if use_cache:
        cached_posts = _get_cached_posts()
        if cached_posts:
            logger.info("Using cached RSS posts")
            posts = cached_posts
        else:
            # Fetch posts from feed
            posts = fetch_rss_posts(max_posts=max_posts)
            
            # Cache the posts
            if posts:
                _cache_posts(posts)
    else:
        # Fetch posts from feed without using cache
        posts = fetch_rss_posts(max_posts=max_posts)
    
    # Try to process and embed if we have posts and embedding is requested
    if posts and embed_posts:
        logger.info("Embedding posts in vector store")
        process_and_embed_posts(posts)
    
    if formatted:
        # Return in a format ready for display
        return [
            {
                "title": post.get('title', 'No Title'),
                "link": post.get('link', '#'),
                "date": post.get('published', 'Unknown date')
            }
            for post in posts
        ]
    else:
        return posts

def _get_cached_posts():
    """
    Retrieves cached posts if they exist and are not expired.
    
    Returns:
        list: Cached posts or None if cache is invalid or expired.
    """
    try:
        if not os.path.exists(RSS_CACHE_FILE):
            return None
            
        # Check file modification time
        mod_time = os.path.getmtime(RSS_CACHE_FILE)
        if time.time() - mod_time > RSS_CACHE_EXPIRY:
            logger.info("RSS cache expired")
            return None
            
        with open(RSS_CACHE_FILE, 'r', encoding='utf-8') as f:
            cache_data = json.load(f)
            
        # Validate cache structure
        if not isinstance(cache_data, dict) or 'posts' not in cache_data:
            logger.warning("Invalid cache structure")
            return None
            
        # Check timestamp if available
        if 'timestamp' in cache_data:
            cache_time = datetime.fromisoformat(cache_data['timestamp'])
            if datetime.now() - cache_time > timedelta(seconds=RSS_CACHE_EXPIRY):
                logger.info("RSS cache timestamp expired")
                return None
                
        return cache_data['posts']
    except Exception as e:
        logger.error(f"Error reading RSS cache: {e}")
        return None

def _cache_posts(posts):
    """
    Caches the posts to a file.
    
    Args:
        posts (list): The posts to cache.
    """
    try:
        cache_data = {
            'timestamp': datetime.now().isoformat(),
            'posts': posts
        }
        
        with open(RSS_CACHE_FILE, 'w', encoding='utf-8') as f:
            json.dump(cache_data, f, ensure_ascii=False, indent=2)
            
        logger.info(f"Cached {len(posts)} RSS posts")
    except Exception as e:
        logger.error(f"Error caching RSS posts: {e}")

def embed_latest_posts(max_posts=10):
    """
    Fetches the latest RSS posts and embeds them in the vector store.
    This function is intended to be run periodically as a maintenance task.
    
    Args:
        max_posts (int): Maximum number of posts to fetch and embed.
        
    Returns:
        bool: True if successful, False otherwise.
    """
    try:
        logger.info(f"Fetching and embedding latest {max_posts} RSS posts")
        
        # Fetch posts from feed without using cache
        posts = fetch_rss_posts(max_posts=max_posts)
        
        if not posts:
            logger.warning("No posts fetched for embedding")
            return False
            
        # Process and embed the posts
        process_and_embed_posts(posts)
        
        # Update the cache with the latest posts
        _cache_posts(posts)
        
        logger.info(f"Successfully embedded {len(posts)} RSS posts")
        return True
    except Exception as e:
        logger.error(f"Error embedding latest posts: {e}")
        return False