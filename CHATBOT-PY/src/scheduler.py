from apscheduler.schedulers.background import BackgroundScheduler
from apscheduler.triggers.cron import CronTrigger
from datetime import datetime
import logging
from web_crawler import get_crawled_content
from fetch_posts import fetch_rss_posts, process_and_embed_posts
from split_document import update_vectorstore

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def scheduled_scraping_task():
    """
    Performs the scheduled scraping and updating of the vector store
    """
    try:
        logger.info(f"Starting scheduled scraping task at {datetime.now()}")
        
        # Crawl website content
        web_content = get_crawled_content()
        if web_content:
            logger.info("Successfully crawled website content")
        
        # Fetch RSS posts
        posts = fetch_rss_posts()
        if posts:
            process_and_embed_posts(posts)
            logger.info("Successfully processed RSS feed posts")
            
        # Update vector store with new content
        update_vectorstore(web_content)
        logger.info("Vector store updated successfully")
        
    except Exception as e:
        logger.error(f"Error in scheduled scraping task: {e}")

def init_scheduler():
    """
    Initializes and starts the scheduler
    """
    try:
        scheduler = BackgroundScheduler()
        
        # Schedule the task to run monthly
        scheduler.add_job(
            scheduled_scraping_task,
            trigger=CronTrigger(
                day="1",  # Run on the 1st day of every month
                hour="0",  # At midnight
                minute="0"
            ),
            id='monthly_scraping_task',
            name='Monthly website and RSS feed scraping'
        )
        
        scheduler.start()
        logger.info("Scheduler initialized successfully")
        return scheduler
        
    except Exception as e:
        logger.error(f"Error initializing scheduler: {e}")
        raise