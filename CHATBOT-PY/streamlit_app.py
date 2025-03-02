import os
import sys
import streamlit as st

# Add the current directory to the Python path
sys.path.insert(0, os.path.abspath("."))

try:
    # Import the main function from src/main.py
    from src.main import main
    
    # Call the main function
    if __name__ == "__main__":
        main()
except ImportError as e:
    st.error(f"Import error: {e}")
    st.info("This is likely due to a module not being found. Check that all dependencies are installed and module paths are correct.")
    
    # Debug information
    st.write("Current directory:", os.path.abspath("."))
    st.write("Python path:", sys.path)
    
    # Check for src directory
    if os.path.exists("src"):
        files_in_src = os.listdir("src")
        st.write("Files in src directory:", files_in_src)
    else:
        st.warning("src directory not found!")
    
    # Provide a link to the debug app
    st.warning("If problems persist, try using the debug app to diagnose the issues.")
    
    if st.button("Run Debug App"):
        # Since we can't directly import the debug app here (due to the import error),
        # we'll manually display similar debug information
        st.subheader("Debug Mode")
        
        # Try to import some key modules
        try:
            from langchain_openai import ChatOpenAI
            st.write("✅ langchain_openai module imported successfully")
        except ImportError as e:
            st.error(f"❌ langchain_openai import failed: {e}")
            
        try:
            from sentence_transformers import SentenceTransformer
            st.write("✅ sentence_transformers module imported successfully")
        except ImportError as e:
            st.error(f"❌ sentence_transformers import failed: {e}")
        
        # Provide guidance on fixing the issue
        st.subheader("Suggested fixes:")
        st.markdown("""
        1. Make sure all dependencies are installed:
           - Check `requirements.txt` includes all necessary packages
           - For Streamlit Cloud, requirements.txt should be at the root of the repository
           
        2. Fix import paths in your code:
           - For imports from the same directory, use relative imports with dot notation
           - Example: Change `from retriever import retriever` to `from .retriever import retriever`
           
        3. Update the app structure:
           - Make sure __init__.py files are present in all directories that need to be imported as packages
           - Consider restructuring your code to avoid complex import paths
        """)
        
        # Provide a link to Streamlit help
        st.markdown("[Streamlit Deployment Documentation](https://docs.streamlit.io/deploy/streamlit-community-cloud/deploy-your-app)") 