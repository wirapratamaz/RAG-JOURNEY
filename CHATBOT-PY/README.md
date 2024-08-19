# RAG Chatbot

This is an LLM Chatbot powered by RAG. The tech stack includes Python, Langchain, OpenAI, and Chroma vector store.

LLM - Large Language Model  
RAG - Retrieval Augmented Generation  

## Step-by-Step Guide

1. **Clone the repository and navigate to the project directory:**
    ```bash
    git clone <repository-url>
    cd path/to/repo
    ```

2. **Create and activate a virtual environment:**
    ```bash
    python -m venv myvenv
    myvenv\Scripts\activate  # On Windows
    # source myvenv/bin/activate  # On macOS/Linux
    ```

3. **Install libraries and dependencies:**
    ```bash
    pip install -r requirements.txt
    ```

4. **Get [OpenAI API key](https://platform.openai.com/account/api-keys)**

5. **Split documents and save to Supabase Vector database (Run once or only when you need to store a document):**
    ```bash
    cd src
    python split_document.py
    ```

    If the operation is successful, you should see a `Success!` message.

6. **Run the Streamlit app:**
    ```bash
    streamlit run main.py
    ```

    After running this command, you can view your Streamlit app in your browser at:
    - Local URL: `http://localhost:8501`
    - Network URL: `http://192.168.18.16:8501` (or your local network IP)
    
## DOCUMENTATION
![alt text](image.png)

### More Docs and Links
[Streamlit Docs](https://docs.streamlit.io/get-started)  
[Langchain Python Docs](https://python.langchain.com/v0.2/docs/introduction/)  
[Langchain Conversational RAG Docs](https://python.langchain.com/v0.2/docs/tutorials/qa_chat_history/)  
