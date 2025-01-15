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

def chunking_and_retrieval(user_input):
    st.subheader("1. Chunking and Retrieval")
    # Use the retriever to fetch relevant documents from the Chroma vector store
    try:
        # Retrieve documents related to the user input
        retrieved_docs = retriever.get_relevant_documents(user_input)
        
        # Prepare data for display
        embedded_data = []
        for doc in retrieved_docs:
            chunk = doc.page_content  # Extract content from the document
            # Calculate relevance scores using the user input as the query
            relevance_scores = calculate_relevance_scores(chunk, user_input)
            embedded_data.append((chunk, relevance_scores))
        
        display_embedding_process(embedded_data)
    except Exception as e:
        st.warning(f"An error occurred during retrieval: {e}")

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

def generation(user_input):
    st.subheader("2. Generation")
    with st.spinner("Sedang menyusun jawaban terbaik untuk Anda sabar dulu yaah..."):
        # Simulate processing with a progress bar
        progress_bar = st.progress(0)
        for i in range(100):
            time.sleep(0.01)
            progress_bar.progress(i + 1)
        
        try:
            # Get response from the chain
            response = rag_chain.invoke({
                "question": user_input,
                "chat_history": st.session_state.get('chat_history', [])
            })
            
            # Extract answer from response
            if isinstance(response, dict) and "answer" in response:
                answer = response["answer"]
            else:
                answer = "Maaf, saya tidak dapat memproses pertanyaan Anda saat ini."
                
        except Exception as e:
            logger.error(f"Error generating response: {e}")
            answer = "Maaf, terjadi kesalahan dalam memproses pertanyaan Anda."
    
    st.success("Jawaban siap! 🚀")
    return answer

def main():
    st.set_page_config(
        page_title="SisInfoBot Undiksha",
        page_icon="https://encrypted-tbn0.gstatic.com/images?q=tbn:ANd9GcSxGdjI58B_NuUd8eAWfRBlVms7f-2e2oI_SA&s",
        layout="wide"
    )
    
    # Sidebar
    with st.sidebar:
        st.image("https://encrypted-tbn0.gstatic.com/images?q=tbn:ANd9GcSxGdjI58B_NuUd8eAWfRBlVms7f-2e2oI_SA&s", width=150)
        st.title("SisInfo Undiksha")
        st.selectbox("Model AI", ["GPT-3.5 Turbo"])
        st.text("Universitas Pendidikan Ganesha")
        st.markdown("---")
        st.markdown("### 🌟 Fitur Unggulan:")
        st.markdown("• Informasi Terkini Prodi")
        st.markdown("• Panduan Akademik")
        st.markdown("• Asisten Virtual Anda")
    
    # Main content
    st.header("Selamat Datang di SisInfo Undiksha! 🤖💻")
    st.info("Halo, Saya adalah asisten virtual khusus untuk mahasiswa Sistem Informasi Undiksha. Tanyakan apa saja seputar program studi, kurikulum, atau informasi yang berkaitan!")
    
    # Display chat history
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = []
    
    for i, message in enumerate(st.session_state.chat_history):
        if message["role"] == "user":
            st.text_area("Anda:", value=message["content"], height=50, disabled=True, key=f"user_{i}")
        else:
            st.text_area("SisInfo:", value=message["content"], height=100, disabled=True, key=f"bot_{i}")
    
    # User input
    user_input = st.text_input("Ada yang ingin Anda tanyakan tentang Sistem Informasi Undiksha?")
    
    if user_input:
        st.session_state.chat_history.append({"role": "user", "content": user_input})
        
        chunking_and_retrieval(user_input)
        answer = generation(user_input)
        
        st.session_state.chat_history.append({"role": "assistant", "content": answer})
        
        st.subheader("Jawaban SisInfo:")
        with st.container():
            st.markdown(f"""
           {answer}
            """, unsafe_allow_html=True)

    st.sidebar.markdown("---")
    st.sidebar.markdown("### 📢 Info Terkini:")
    with st.sidebar.expander("Lihat Info Terkini"):
        latest_posts = get_latest_posts(formatted=True)
        if latest_posts:
            for post in latest_posts:
                st.markdown(f"- [{post['title']}]({post['link']})")
        else:
            st.info("Tidak ada informasi terkini saat ini.")

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