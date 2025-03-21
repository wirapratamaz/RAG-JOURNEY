# RAG Chatbot

## Overview

This project is an LLM Chatbot powered by Retrieval-Augmented Generation (RAG). The tech stack includes Python, Langchain, OpenAI, and Chroma vector store. The chatbot is designed to provide accurate and contextually relevant responses by leveraging external knowledge sources.

## Table of Contents

1. [Installation](#installation)
2. [Running the Application](#running-the-application)
3. [Technology Stack](#technology-stack)
4. [Logic and Implementation](#logic-and-implementation)
5. [Documentation](#documentation)

## Installation

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

## Running the Application

1. **Split documents and save to Supabase Vector database (Run once or only when you need to store a document):**
   ```bash
   cd src
   python split_document.py
   ```

   If the operation is successful, you should see a `Success!` message.

2. **Run the Streamlit app:**
   ```bash
   streamlit run main.py
   ```

   After running this command, you can view your Streamlit app in your browser at:
   - Local URL: `http://localhost:8501`
   - Network URL: `http://192.168.18.16:8501` (or your local network IP)

## Technology Stack

- **Langchain**: A framework for building applications with large language models. It provides tools for managing prompts, chains, and memory.
- **Chroma**: A vector store used to store and retrieve document embeddings efficiently.
- **Streamlit**: A web application framework for creating interactive applications with Python.
- **OpenAI**: Provides the language model API used for generating responses.

## Logic and Implementation

### Retrieval-Augmented Generation (RAG)

RAG is a technique that enhances the performance of LLMs by retrieving relevant documents from a knowledge base and using them as context for generating responses. This approach reduces the need for fine-tuning and allows for more agile updates to the knowledge base.

### Langchain

Langchain is used to manage the conversational flow and retrieval logic. It integrates with OpenAI's API to generate responses based on the retrieved context.

### Chroma

Chroma is used to store document embeddings. It allows for efficient retrieval of relevant documents based on user queries.

### Streamlit

Streamlit is used to create the user interface for the chatbot. It provides an interactive platform for users to input queries and receive responses.

### OpenAI

OpenAI's API is used to generate responses. The model is configured to use the `gpt-3.5-turbo` variant, which provides high-quality conversational capabilities.

## Documentation

For more detailed documentation, refer to the following resources:

- [Streamlit Docs](https://docs.streamlit.io/get-started)
- [Langchain Python Docs](https://python.langchain.com/v0.2/docs/introduction/)
- [Langchain Conversational RAG Docs](https://python.langchain.com/v0.2/docs/tutorials/qa_chat_history/)

## Deployment to Streamlit Community Cloud

To deploy this application to Streamlit Community Cloud, follow these steps:

1. **Create a GitHub repository** for your project if you haven't already.

2. **Push your code to GitHub**:
   ```bash
   git add .
   git commit -m "Prepare for Streamlit deployment"
   git push origin main
   ```

3. **Sign in to Streamlit Community Cloud**:
   - Go to [Streamlit Community Cloud](https://share.streamlit.io/)
   - Sign in with your GitHub account

4. **Deploy your app**:
   - Click "New app"
   - Select your GitHub repository
   - Set the main file path to one of:
     - `streamlit_app.py` (Main app with full functionality)
     - `simple_app.py` (Simplified version if you encounter issues)
     - `debug_app.py` (Diagnostic app if you need to troubleshoot)
   - Click "Deploy"

5. **Set up secrets**:
   - In the Streamlit Cloud dashboard, find your app
   - Click on "Settings" > "Secrets"
   - Add your OpenAI API key and any other required secrets:
     ```
     OPENAI_API_KEY = "your-openai-api-key"
     ```

6. **Advanced settings** (if needed):
   - You can specify Python version, packages, or other requirements in the Advanced settings section

7. **Troubleshooting Deployment Issues**:
   - If you encounter import errors, try deploying `simple_app.py` or `debug_app.py` first
   - Check that all dependencies are in `requirements.txt`
   - Make sure all import paths are correct (use `from src.module import something` instead of `from module import something`)
   - Ensure your Chroma database is properly initialized
   - Check the Streamlit Cloud logs for detailed error messages

Your app will now be deployed and publicly accessible via the URL provided by Streamlit!

![alt text](image.png)

## RSS Feed Integration

The application includes RSS feed integration to display the latest posts from the Undiksha Information Systems Program website. This feature has been optimized for performance:

1. **Cached RSS Feeds**: RSS feeds are fetched once and cached to improve performance. The cache is stored in the `cache/` directory and has a default expiry of 1 hour.

2. **Separation of Fetching and Embedding**: 
   - By default, RSS posts are fetched but not embedded in the vector store during normal operation
   - This significantly improves performance by avoiding expensive embedding operations

3. **Manual Embedding**: 
   - A separate script is provided to manually embed RSS posts in the vector store
   - This can be run as a scheduled task to keep the vector store updated

4. **Usage**:
   - To manually embed RSS posts, run:
     ```bash
     python embed_rss_posts.py --max-posts 10
     ```
   - This script can be scheduled to run periodically (e.g., using cron or Task Scheduler)

5. **Refresh Button**:
   - The UI includes a refresh button (🔄) to manually refresh the RSS feed
   - This only updates the displayed posts and does not perform embedding

This approach ensures that the application remains responsive while still providing up-to-date information from the RSS feed.