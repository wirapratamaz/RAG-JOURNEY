import os
import sys
import streamlit as st
import logging

# Patch to prevent torch._classes.__path__._path error in Streamlit's file watcher
import importlib.abc
import importlib.machinery

class CustomImportFixer(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path, target=None):
        # Intercept attempts to access problematic module paths
        if fullname == 'torch._classes.__path__._path' or fullname.endswith('.__path__._path'):
            return None
        return None

# Add the import hook to sys.meta_path
sys.meta_path.insert(0, CustomImportFixer())

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Add the current directory to the Python path
sys.path.insert(0, os.path.abspath("."))

# Function to check if OpenAI API key is available
def check_api_key():
    openai_api_key = st.secrets.get("OPENAI_API_KEY", os.getenv("OPENAI_API_KEY"))
    
    # Check if the key exists
    if not openai_api_key:
        logger.error("OpenAI API key not found.")
        st.error("OpenAI API key not found. Please set OPENAI_API_KEY in your environment or secrets.")
        return False
    
    # Check if it's a placeholder value
    placeholder_values = ["your-api-key-here", "your-openai-api-key", "sk-..."]
    if any(placeholder in openai_api_key.lower() for placeholder in placeholder_values):
        logger.error("OpenAI API key appears to be a placeholder value.")
        st.error("The OpenAI API key appears to be a placeholder. Please set your actual API key in Streamlit secrets or .env file.")
        return False
        
    # Check key format - most OpenAI keys start with 'sk-'
    if not (openai_api_key.startswith('sk-') or openai_api_key.startswith('sk-proj-')):
        logger.warning("OpenAI API key format appears unusual. It should typically start with 'sk-'")
        # Continue anyway, as API key formats can change
    
    logger.info("Found valid API key in Streamlit secrets or environment variables")
    return True

# Function to run the main app
def run_main_app():
    try:
        # First check if we can patch sqlite3 for ChromaDB
        try:
            # Try to check sqlite3 version first
            import sqlite3
            sqlite_version = sqlite3.sqlite_version_info
            if sqlite_version < (3, 35, 0):
                logger.warning(f"SQLite version {sqlite3.sqlite_version} is too old for ChromaDB (needs 3.35.0+)")
                logger.info("Attempting to patch sqlite3 with pysqlite3...")
                
                # Try to use pysqlite3 instead
                try:
                    # First try to import without installing
                    import pysqlite3
                    sys.modules['sqlite3'] = pysqlite3
                    logger.info("Successfully patched sqlite3 with existing pysqlite3")
                except ImportError:
                    # If not available, try to install it
                    logger.info("pysqlite3 not found, attempting to install...")
                    import subprocess
                    try:
                        subprocess.check_call([sys.executable, "-m", "pip", "install", "pysqlite3-binary"])
                        import pysqlite3
                        sys.modules['sqlite3'] = pysqlite3
                        logger.info("Successfully installed and patched sqlite3 with pysqlite3")
                    except Exception as e:
                        logger.error(f"Failed to install pysqlite3: {e}")
                        raise ImportError(f"Cannot use ChromaDB due to SQLite version: {sqlite_version}")
            
            # Now try importing langchain_chroma
            import langchain_chroma
            import chromadb
            logger.info("langchain_chroma and chromadb are installed and working properly")
        except (ImportError, RuntimeError, AttributeError) as e:
            logger.error(f"Error with vector database dependencies: {e}")
            raise ImportError(f"ChromaDB cannot be initialized: {e}")
        
        # Try to import the main function from src/main.py
        logger.info("Attempting to import main app")
        try:
            from src.main import main
            return main
        except ImportError as e:
            # Check if it's specifically a retriever import issue
            if "retriever" in str(e):
                logger.error("Retriever import issue detected. Applying additional fix...")
                
                # Make sure mock modules are available
                if not os.path.exists('src'):
                    os.makedirs('src', exist_ok=True)
                
                # Create __init__.py in src directory if it doesn't exist
                if not os.path.exists('src/__init__.py'):
                    with open('src/__init__.py', 'w') as f:
                        f.write('# Package initialization\n')
                
                # Try again with the fixes
                from src.main import main
                return main
            else:
                # Re-raise if it's a different issue
                raise
    except Exception as e:
        logger.error(f"Error importing main app: {e}")
        st.error(f"Error loading the main application: {e}")
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

# Function to create a very simple UI if everything else fails
def create_fallback_ui():
    st.set_page_config(
        page_title="SI Undiksha Assistant (Fallback Mode)",
        page_icon="🔄",
        layout="wide"
    )
    
    st.title("SI Undiksha Assistant (Fallback Mode)")
    st.warning("⚠️ The full application could not be loaded due to technical issues.")
    
    st.markdown("""
    ### What can I help you with?
    
    I can still answer basic questions about Sistem Informasi Undiksha based on my general knowledge.
    
    However, specific document searching capabilities are currently unavailable.
    """)
    
    # Initialize session state
    if "messages" not in st.session_state:
        st.session_state.messages = []
    
    # Display chat history
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])
    
    # Get user input
    prompt = st.chat_input("What would you like to know about SI Undiksha?")
    
    # Process user input
    if prompt:
        # Add user message to chat history
        st.session_state.messages.append({"role": "user", "content": prompt})
        
        # Display user message
        with st.chat_message("user"):
            st.markdown(prompt)
        
        # Prepare API call
        openai_api_key = st.secrets.get("OPENAI_API_KEY", os.getenv("OPENAI_API_KEY"))
        
        if not openai_api_key:
            st.error("API key not available. Cannot generate response.")
            return

        # Check if it's a placeholder value
        placeholder_values = ["your-api-key-here", "your-openai-api-key", "sk-..."]
        if any(placeholder in openai_api_key.lower() for placeholder in placeholder_values):
            st.error("The OpenAI API key appears to be a placeholder. Please set your actual API key.")
            return
        
        # Display assistant response
        with st.chat_message("assistant"):
            try:
                from langchain_openai import ChatOpenAI
                llm = ChatOpenAI(
                    model_name="gpt-3.5-turbo",
                    openai_api_key=openai_api_key
                )
                
                message_placeholder = st.empty()
                full_response = ""
                
                # Create a simple system message
                system_message = """You are a virtual assistant for Sistem Informasi Undiksha (Universitas Pendidikan Ganesha).
                Answer user questions about the program, curriculum, and university based on your general knowledge.
                Be polite, helpful, and concise. If you don't know something specific, be honest about it."""
                
                # Generate response
                for chunk in llm.stream({
                    "system": system_message,
                    "human": prompt
                }):
                    full_response += chunk.content
                    message_placeholder.markdown(full_response + "▌")
                
                message_placeholder.markdown(full_response)
                
                # Add assistant response to chat history
                st.session_state.messages.append({"role": "assistant", "content": full_response})
                
            except Exception as e:
                st.error(f"Error generating response: {e}")
                st.info("The assistant is currently unavailable. Please try again later.")

# Main entry point of the application
if __name__ == "__main__":
    # Try to run the main app
    try:
        # Check if key is available first
        if not check_api_key():
            st.stop()

        # Add crawl4ai to the path if it errors out
        try:
            import crawl4ai
        except ImportError as e:
            logger.warning(f"crawl4ai import error: {e}. Patching...")
            # Creating a mock crawl4ai module to prevent errors
            mock_crawl4ai = type('', (), {})
            mock_crawler = type('', (), {'arun': lambda *args, **kwargs: None})
            mock_crawl4ai.AsyncWebCrawler = lambda *args, **kwargs: mock_crawler
            sys.modules['crawl4ai'] = mock_crawl4ai
            logger.info("Added mock crawl4ai module")

        # Try to run the main app first
        main_func = run_main_app()
        
        if main_func:
            # If main app loaded successfully, run it
            main_func()
        else:
            # If main app failed, try to run the simpler app
            logger.warning("Main app failed, attempting to create a fallback UI")
            try:
                create_fallback_ui()
            except Exception as e:
                logger.error(f"Fallback UI also failed: {e}")
                st.error("Unfortunately, the application could not be loaded due to technical issues.")
                st.info("Please try again later or contact support.")
                
                # Emergency minimal chat interface
                logger.warning("Attempting emergency minimal chat interface")
                create_fallback_ui()
    except Exception as e:
        logger.error(f"Error in main entry point: {e}")
        st.error("Unfortunately, the application could not be loaded due to technical issues.")
        st.info("Please try again later or contact support.") 