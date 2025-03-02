import os
import sys
import streamlit as st
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Add the current directory to the Python path
sys.path.insert(0, os.path.abspath("."))

# Function to run the main app
def run_main_app():
    try:
        # Try to import the main function from src/main.py
        logger.info("Attempting to import main app")
        from src.main import main
        return main
    except ImportError as e:
        logger.error(f"Failed to import main app: {e}")
        return None

# Function to run the simple app
def run_simple_app():
    try:
        # Use a simpler module with fewer dependencies
        logger.info("Attempting to import simple app")
        # Use import from file approach
        import simple_app
        return simple_app.main
    except ImportError as e:
        logger.error(f"Failed to import simple app: {e}")
        return None

# Function to run debug app
def run_debug_app():
    # Show debug information
    logger.info("Running debug mode")
    
    st.title("Debug Mode - Deployment Troubleshooting")
    st.warning("The main app failed to load. This is a diagnostic page.")
    
    # Display environment information
    st.subheader("Environment Information")
    st.write("Python version:", sys.version)
    st.write("Current directory:", os.path.abspath("."))
    st.write("Python path:", sys.path)
    
    # Display directory contents
    st.subheader("Directory Contents")
    try:
        st.write("Root directory:", os.listdir("."))
        if os.path.exists("src"):
            st.write("src directory:", os.listdir("src"))
        else:
            st.error("src directory not found!")
    except Exception as e:
        st.error(f"Error listing directories: {e}")
    
    # Try importing modules
    st.subheader("Import Tests")
    modules_to_test = [
        "streamlit", "langchain", "langchain_openai", 
        "sentence_transformers", "chromadb", "openai"
    ]
    
    for module_name in modules_to_test:
        try:
            __import__(module_name)
            st.write(f"✅ {module_name} import successful")
        except ImportError as e:
            st.error(f"❌ {module_name} import failed: {e}")
    
    # Provide troubleshooting guidance
    st.subheader("Troubleshooting Steps")
    st.markdown("""
    1. Check that all dependencies are installed correctly:
       - Verify requirements.txt has all necessary packages
       - Make sure the packages are installed in the correct version
    
    2. Check your import paths:
       - Use absolute imports (e.g., `from src.module import X`)
       - Or use relative imports (e.g., `from .module import X`)
    
    3. File structure issues:
       - Ensure __init__.py exists in all module directories
       - Check file permissions
    
    4. Try simplified apps:
       - Deploy simple_app.py which has fewer dependencies
       - Or create a minimal version of your app to isolate the issue
    """)

if __name__ == "__main__":
    # Try to run the main app first
    main_func = run_main_app()
    
    if main_func:
        # If main app loaded successfully, run it
        logger.info("Main app loaded successfully")
        main_func()
    else:
        # Try the simple app
        logger.info("Trying simple app")
        simple_func = run_simple_app()
        
        if simple_func:
            # If simple app loaded successfully, run it
            logger.info("Simple app loaded successfully")
            simple_func()
        else:
            # Fall back to debug mode
            logger.info("Falling back to debug mode")
            run_debug_app() 