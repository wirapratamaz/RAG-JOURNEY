import os
import sys
import streamlit as st
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Add the current directory to the Python path
sys.path.insert(0, os.path.abspath("."))

# Check for API key first
def check_api_key():
    api_key = None
    try:
        # Try to get from Streamlit secrets
        api_key = st.secrets.get("OPENAI_API_KEY")
        if api_key:
            logger.info("Found API key in Streamlit secrets")
            return True
    except Exception as e:
        logger.error(f"Error accessing Streamlit secrets: {e}")
    
    # Try to get from .env file
    try:
        from dotenv import load_dotenv
        load_dotenv()
        api_key = os.getenv("OPENAI_API_KEY")
        if api_key:
            logger.info("Found API key in .env file")
            return True
    except Exception as e:
        logger.error(f"Error loading .env file: {e}")
    
    # If we get here, no API key was found
    logger.error("No OpenAI API key found")
    st.error("OpenAI API key not found. Please add it to your .streamlit/secrets.toml file or .env file.")
    
    # Show instructions for adding API key to Streamlit Cloud
    st.subheader("How to add your API key to Streamlit Cloud:")
    st.markdown("""
    1. Go to your Streamlit Cloud dashboard
    2. Find your app and click on it
    3. Go to "Settings" > "Secrets"
    4. Add your OpenAI API key in the following format:
       ```
       OPENAI_API_KEY = "your-actual-api-key-here"
       ```
    5. Save the changes and rerun your app
    """)
    
    return False

# Function to run the main app
def run_main_app():
    try:
        # First check if langchain_chroma is available
        try:
            import langchain_chroma
            logger.info("langchain_chroma is installed")
        except ImportError:
            logger.error("langchain_chroma is not installed, using alternative implementation")
            # Create a symbolic link to our alternative implementation
            import simple_retriever
            sys.modules['src.retriever'] = simple_retriever
        
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
    
    # Check requirements.txt
    if os.path.exists("requirements.txt"):
        with open("requirements.txt", "r") as f:
            requirements = f.read()
            st.subheader("Requirements.txt Contents")
            st.text(requirements)
            
            # Check for important packages
            important_packages = ["langchain-chroma", "openai", "streamlit", "langchain", "chromadb"]
            missing_packages = []
            for package in important_packages:
                if package not in requirements:
                    missing_packages.append(package)
            
            if missing_packages:
                st.error(f"Missing important packages in requirements.txt: {', '.join(missing_packages)}")
                st.markdown("""
                Update your requirements.txt file to include these packages:
                ```
                langchain-chroma>=0.0.6
                openai>=1.10.0
                streamlit>=1.31.0
                langchain>=0.1.5
                chromadb>=0.4.22
                ```
                """)
    
    # Try importing modules
    st.subheader("Import Tests")
    modules_to_test = [
        "streamlit", "langchain", "langchain_openai", 
        "sentence_transformers", "chromadb", "openai", "langchain_chroma"
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
    1. Add missing dependencies to requirements.txt:
       ```
       langchain-chroma>=0.0.6
       ```
    
    2. Set up your OpenAI API key in Streamlit Cloud:
       - Go to your app's settings
       - Click on "Secrets"
       - Add your API key in this format:
       ```
       OPENAI_API_KEY = "your-api-key-here"
       ```
    
    3. Try the simple app version:
       - In your app settings, change the main file path to `simple_app.py`
       
    4. Check the logs for more detailed error messages
    """)

if __name__ == "__main__":
    # First check if we have an API key
    if not check_api_key():
        st.stop()
    
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