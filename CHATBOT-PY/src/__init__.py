# This file makes the src directory a Python package
# This allows relative imports to work correctly

import os
import sys
import logging

logger = logging.getLogger(__name__)
logger.info("Initializing src package")

# Make sure submodules can be imported directly
__all__ = ['main', 'retriever', 'fetch_posts', 'web_crawler']

# Make sure simple_retriever is available as retriever if needed
try:
    import src.retriever
except ImportError:
    logger.warning("Could not import src.retriever, trying to use fallback")
    try:
        import simple_retriever
        sys.modules['src.retriever'] = simple_retriever
        logger.info("Using simple_retriever as fallback")
    except ImportError:
        logger.error("Failed to set up retriever fallback")

__version__ = "1.0.0"
