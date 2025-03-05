#!/usr/bin/env python3
"""
Script to fetch and embed the latest RSS posts in the vector store.
This script is intended to be run periodically as a scheduled task.

Usage:
    python embed_rss_posts.py [--max-posts MAX_POSTS]

Options:
    --max-posts MAX_POSTS  Maximum number of posts to fetch and embed [default: 10]
"""

import os
import sys
import logging
import argparse
from datetime import datetime

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler(os.path.join(os.path.dirname(__file__), 'embed_rss.log'))
    ]
)
logger = logging.getLogger(__name__)

# Add the current directory to the Python path
sys.path.insert(0, os.path.abspath("."))

def main():
    """Main function to fetch and embed RSS posts."""
    parser = argparse.ArgumentParser(description='Fetch and embed RSS posts')
    parser.add_argument('--max-posts', type=int, default=10, help='Maximum number of posts to fetch and embed')
    args = parser.parse_args()
    
    try:
        from src.fetch_posts import embed_latest_posts
        
        logger.info(f"Starting RSS post embedding at {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        
        success = embed_latest_posts(max_posts=args.max_posts)
        
        if success:
            logger.info("RSS post embedding completed successfully")
        else:
            logger.error("RSS post embedding failed")
            sys.exit(1)
            
    except ImportError as e:
        logger.error(f"Error importing required modules: {e}")
        sys.exit(1)
    except Exception as e:
        logger.error(f"Unexpected error: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main() 