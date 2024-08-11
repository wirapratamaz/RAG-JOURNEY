import streamlit as st
import ollama
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import WebBaseLoader
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import OllamaEmbeddings

st.title("Chat with Webpage 🌐")
st.caption("This app allows you to chat with a webpage using local Llama 2 and RAG")

# Get the webpage URL from the user
webpage_url = "https://is.undiksha.ac.id/"

if webpage_url:
    try:
        # 1. Load the data
        loader = WebBaseLoader(webpage_url)
        docs = loader.load()
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=10)
        splits = text_splitter.split_documents(docs)

        # 2. Create Ollama embeddings and vector store
        embeddings = OllamaEmbeddings(model="llama2")
        vectorstore = Chroma.from_documents(documents=splits, embedding=embeddings)

        # 3. Call Ollama Llama2 model
        def ollama_llm(question, context):
            formatted_prompt = f"Question: {question}\n\nContext: {context}"
            response = ollama.chat(model='llama2', messages=[{'role': 'user', 'content': formatted_prompt}])
            return response['message']['content']

        # 4. RAG Setup
        retriever = vectorstore.as_retriever()

        def combine_docs(docs):
            return "\n\n".join(doc.page_content for doc in docs)

        def rag_chain(question):
            retrieved_docs = retriever.invoke(question)
            formatted_context = combine_docs(retrieved_docs)
            return ollama_llm(question, formatted_context)

        st.success(f"Loaded {webpage_url} successfully!")

        # Ask a question about the webpage
        prompt = st.text_input("Ask any question about the webpage, give detail information about the webpage, bring the answer not formal")

        # Chat with the webpage
        if prompt:
            result = rag_chain(prompt)
            st.write(result)

    except Exception as e:
        st.error(f"An error occurred: {str(e)}")
        st.error("Please make sure you have installed and pulled the Llama 2 model using Ollama.")