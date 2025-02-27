import os
import ssl
import streamlit as st
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain.chains import ConversationalRetrievalChain
from langchain.memory import ConversationBufferMemory
from retriever import retriever
import pandas as pd
import time
from fetch_posts import fetch_rss_posts, process_and_embed_posts, get_latest_posts
from sklearn.metrics.pairwise import cosine_similarity
from sentence_transformers import SentenceTransformer
from web_crawler import get_crawled_content
import logging

# Configure logging
logger = logging.getLogger(__name__)

# Bypass SSL verification if needed
ssl._create_default_https_context = ssl._create_unverified_context

# Load environment variables
load_dotenv()
openai_api_key = os.getenv("OPENAI_API_KEY")

# Check if the API key is set
if not openai_api_key:
    st.error("OpenAI API key not set. Please set it in the .env file")
    st.stop()

# Load a pre-trained model for embeddings
model = SentenceTransformer('all-MiniLM-L6-v2')

def calculate_relevance_scores(chunk, query):
    # Generate embeddings for the chunk and the query
    chunk_embedding = model.encode([chunk])
    query_embedding = model.encode([query])
    
    # Calculate cosine similarity
    similarity_score = cosine_similarity(chunk_embedding, query_embedding)
    
    # Return the similarity score as a list
    return similarity_score.flatten().tolist()

def chunking_and_retrieval(user_input, show_process=True):
    if show_process:
        st.subheader("1. Chunking and Retrieval")
    
    try:
        retrieved_docs = retriever.get_relevant_documents(user_input)
        embedded_data = []
        
        for doc in retrieved_docs:
            chunk = doc.page_content
            relevance_scores = calculate_relevance_scores(chunk, user_input)
            embedded_data.append((chunk, relevance_scores))
        
        if show_process:
            display_embedding_process(embedded_data)
            
        return embedded_data
    except Exception as e:
        if show_process:
            st.warning(f"An error occurred during retrieval: {e}")
        return []

def display_embedding_process(embedded_data):
    st.subheader("Embedding Process")
    # Ensure this expander is not nested
    with st.expander("Detail Embedding", expanded=True):
        # Prepare data for display
        data = {
            "No": range(1, len(embedded_data) + 1),
            "Konten": [chunk for chunk, _ in embedded_data],
            "Relevansi (3 nilai teratas)": [scores[:3] for _, scores in embedded_data]
        }
        
        df = pd.DataFrame(data)
        st.dataframe(df)
    
    st.success("Analisis informasi selesai!")

def generation(user_input, show_process=True):
    if show_process:
        st.subheader("2. Generation")
        with st.spinner("Sedang menyusun jawaban terbaik untuk Anda sabar dulu yaah..."):
            progress_bar = st.progress(0)
            for i in range(100):
                time.sleep(0.01)
                progress_bar.progress(i + 1)
    else:
        with st.spinner("Generating response..."):
            time.sleep(1)  # Minimal delay for user feedback
    
    try:
        response = rag_chain.invoke({
            "question": user_input,
            "chat_history": st.session_state.get('chat_history', [])
        })
        
        if isinstance(response, dict) and "answer" in response:
            answer = response["answer"]
        else:
            answer = "Maaf, saya tidak dapat memproses pertanyaan Anda saat ini."
            
    except Exception as e:
        logger.error(f"Error generating response: {e}")
        answer = "Maaf, terjadi kesalahan dalam memproses pertanyaan Anda."
    
    if show_process:
        st.success("Jawaban siap! 🚀")
        
    return answer

def main():
    st.set_page_config(
        page_title="Virtual Assistant SI Undiksha",
        page_icon="https://encrypted-tbn0.gstatic.com/images?q=tbn:ANd9GcSxGdjI58B_NuUd8eAWfRBlVms7f-2e2oI_SA&s",
        layout="wide"
    )
    
    # Initialize session state for chat history if it doesn't exist
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = []
    
    # Sidebar
    with st.sidebar:
        st.image("https://encrypted-tbn0.gstatic.com/images?q=tbn:ANd9GcSxGdjI58B_NuUd8eAWfRBlVms7f-2e2oI_SA&s", width=150)
        st.title("Virtual Assistant SI Undiksha")
        mode = st.selectbox("Mode", ["User Mode", "Developer Mode"])
        st.text("Universitas Pendidikan Ganesha")
        st.markdown("---")
        with st.expander("Lihat Info Terkini"):
            latest_posts = get_latest_posts(formatted=True)
            if latest_posts:
                for post in latest_posts:
                    st.markdown(f"- [{post['title']}]({post['link']})")
            else:
                st.info("Tidak ada informasi terkini saat ini.")

    # Main content area
    if not st.session_state.chat_history:
        st.header("Selamat Datang di Virtual Assistant SI Undiksha! 🤖💻")
        st.info("Halo, Saya adalah asisten virtual khusus untuk mahasiswa Sistem Informasi Undiksha. Tanyakan apa saja seputar program studi, kurikulum, atau informasi yang berkaitan!")

    # Chat container
    chat_container = st.container()
    with chat_container:
        for message in st.session_state.chat_history:
            with st.chat_message(message["role"]):
                st.markdown(message["content"])

    # Chat input
    user_input = st.chat_input("Ada yang ingin Anda tanyakan tentang Sistem Informasi Undiksha?")
    
    if user_input:
        # Add user message to chat history
        st.session_state.chat_history.append({"role": "user", "content": user_input})
        
        # Determine if we should show processing details based on mode
        show_process = mode == "Developer Mode"
        
        # Process the query
        if show_process:
            chunking_and_retrieval(user_input, show_process)
        
        answer = generation(user_input, show_process)
        
        # Add assistant response to chat history
        st.session_state.chat_history.append({"role": "assistant", "content": answer})
        
        # Use st.rerun() instead of st.experimental_rerun()
        st.rerun()

if __name__ == "__main__":
    # Initialize memory
    if 'memory' not in st.session_state:
        st.session_state.memory = ConversationBufferMemory(
            memory_key="chat_history",
            output_key="answer",  # Specify which output to store
            return_messages=True
        )

    # Initialize the OpenAI chat model
    llm = ChatOpenAI(
        model_name="gpt-3.5-turbo",
        openai_api_key=openai_api_key
    )

    # Create the RAG chain
    rag_chain = ConversationalRetrievalChain.from_llm(
        llm=llm,
        retriever=retriever,
        memory=st.session_state.memory,
        return_source_documents=True,
        verbose=True
    )

    main()