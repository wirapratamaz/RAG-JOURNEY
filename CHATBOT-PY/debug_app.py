import os
import sys
import streamlit as st

# Add the current directory to the Python path
sys.path.insert(0, os.path.abspath("."))

st.title("Debug Application")
st.write("This is a debug application to diagnose import issues.")

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
except Exception as e:
    st.error(f"Error listing directories: {e}")

# Try importing modules
st.subheader("Import Tests")
try:
    import streamlit
    st.write("✅ streamlit import successful")
except ImportError as e:
    st.error(f"❌ streamlit import failed: {e}")

try:
    import langchain
    st.write("✅ langchain import successful")
except ImportError as e:
    st.error(f"❌ langchain import failed: {e}")

try:
    from langchain_openai import ChatOpenAI
    st.write("✅ langchain_openai import successful")
except ImportError as e:
    st.error(f"❌ langchain_openai import failed: {e}")

try:
    from sentence_transformers import SentenceTransformer
    st.write("✅ sentence_transformers import successful")
except ImportError as e:
    st.error(f"❌ sentence_transformers import failed: {e}")

# Test src module imports
st.subheader("src Module Import Tests")
try:
    import src
    st.write("✅ src package import successful")
except ImportError as e:
    st.error(f"❌ src package import failed: {e}")

try:
    # Test direct import path
    sys.path.append("src")
    import retriever
    st.write("✅ retriever module direct import successful")
except ImportError as e:
    st.error(f"❌ retriever module direct import failed: {e}")

try:
    # Test package import path
    from src import retriever
    st.write("✅ retriever module package import successful")
except ImportError as e:
    st.error(f"❌ retriever module package import failed: {e}")

# Instructions for next steps
st.subheader("Next Steps")
st.write("""
Based on the results above, you can:
1. Check if all required packages are installed
2. Verify the directory structure is correct
3. Update import statements in your code accordingly
""") 