import os
import ssl
import streamlit as st
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain.chains import ConversationalRetrievalChain
from langchain.memory import ConversationBufferMemory
from src.retriever import retriever
import pandas as pd
import time
from src.fetch_posts import fetch_rss_posts, process_and_embed_posts, get_latest_posts
from sklearn.metrics.pairwise import cosine_similarity
from sentence_transformers import SentenceTransformer
from src.web_crawler import get_crawled_content
import logging
from langchain.prompts import ChatPromptTemplate, PromptTemplate

# Configure logging
logger = logging.getLogger(__name__)

# Bypass SSL verification if needed
ssl._create_default_https_context = ssl._create_unverified_context

# Load environment variables
load_dotenv()

# Get API key: first check Streamlit secrets, then .env file
openai_api_key = st.secrets.get("OPENAI_API_KEY", os.getenv("OPENAI_API_KEY"))

# Check if the API key is set
if not openai_api_key:
    st.error("OpenAI API key not set. Please set it in the .env file or Streamlit secrets.")
    st.stop()

# Load a pre-trained model for embeddings
model = SentenceTransformer('all-MiniLM-L6-v2')

# Initialize global variables
rag_chain = None

def initialize_rag_chain():
    """Initialize the RAG chain for conversational retrieval"""
    global rag_chain
    
    try:
        # Initialize memory if not already done
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
        
        # Create a custom prompt template
        custom_prompt = PromptTemplate(
            template="""Use the following pieces of context to answer the user's question concisely.
If you don't know the answer, just say that you don't know, don't try to make up an answer.

IMPORTANT INSTRUCTIONS FOR FORMATTING YOUR RESPONSE:
1. For questions about processes or stages (like thesis completion), follow this format EXACTLY:
   - Start with "Terdapat [number] tahap utama dalam [process], yaitu:"
   - Present each stage as a SEPARATE PARAGRAPH (not as bullet points)
   - For each stage paragraph, start with the stage name in bold, followed by a description
   - Make sure to use the EXACT stage names as mentioned in the context
   - End with a paragraph about who is involved in the process

2. For the thesis completion process specifically:
   - Ensure the four stages are: Pengajuan Topik, Ujian Proposal Skripsi, Penyusunan Skripsi, and Pengesahan Laporan Skripsi
   - Format each stage as a complete paragraph, not a bullet point
   - Add a final concluding sentence about who is involved in the entire process

Context: {context}

Question: {question}

Answer:""",
            input_variables=["context", "question"]
        )
        
        # Verify that retriever is available
        if retriever is None:
            logger.error("Retriever is None - cannot initialize RAG chain")
            raise ValueError("Retriever is not available")
            
        # Create the RAG chain
        logger.info("Creating RAG chain with retriever and memory")
        chain = ConversationalRetrievalChain.from_llm(
            llm=llm,
            retriever=retriever,
            memory=st.session_state.memory,
            return_source_documents=True,
            verbose=True,
            combine_docs_chain_kwargs={"prompt": custom_prompt}
        )
        
        # Set the global variable
        rag_chain = chain
        logger.info("Successfully initialized RAG chain")
        
        return chain
    except Exception as e:
        logger.error(f"Error initializing RAG chain: {e}", exc_info=True)
        # Don't set the global variable if there was an error
        return None

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
        
        # Add a section to display the full raw text of all chunks for better debugging
        st.subheader("Full Retrieved Content")
        full_content = "\n\n---\n\n".join([chunk for chunk, _ in embedded_data])
        st.text_area("All retrieved chunks (raw text):", full_content, height=300)
    
    st.success("Analisis informasi selesai!")

def generation(user_input, show_process=True):
    global rag_chain
    
    # Ensure rag_chain is initialized
    try:
        if rag_chain is None:
            logger.info("Initializing RAG chain for the first time")
            rag_chain = initialize_rag_chain()
            if rag_chain is None:
                raise ValueError("Failed to initialize rag_chain properly")
    except Exception as e:
        logger.error(f"Failed to initialize RAG chain: {e}")
        return "Maaf, sistem tidak dapat memproses pertanyaan Anda saat ini. Terjadi masalah dalam inisialisasi komponen pencarian."
    
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
        # Convert chat history to the format expected by the RAG chain
        formatted_history = []
        for msg in st.session_state.get('chat_history', []):
            if len(formatted_history) > 0 and msg["role"] == "user":
                # Only include the most recent conversation pairs to avoid context overflow
                if len(formatted_history) > 4:
                    formatted_history = formatted_history[-4:]
                formatted_history.append((msg["content"], ""))
            elif len(formatted_history) > 0 and msg["role"] == "assistant":
                last_pair = formatted_history[-1]
                formatted_history[-1] = (last_pair[0], msg["content"])
        
        # Add additional logging to track what's happening
        logger.debug(f"Formatted history: {formatted_history}")
        logger.debug(f"Current question: {user_input}")
        logger.debug(f"RAG chain type: {type(rag_chain)}")
        
        # Call the RAG chain with the formatted history
        response = rag_chain.invoke({
            "question": user_input,
            "chat_history": formatted_history
        })
        
        if isinstance(response, dict) and "answer" in response:
            answer = response["answer"]
        else:
            logger.warning(f"Unexpected response format: {response}")
            answer = "Maaf, saya tidak dapat memproses pertanyaan Anda saat ini."
            
    except Exception as e:
        logger.error(f"Error generating response: {e}", exc_info=True)
        # Provide more informative error message
        if "name 'rag_chain' is not defined" in str(e):
            answer = "Maaf, terjadi kesalahan dalam komponen pencarian. Tim teknis kami sedang menyelesaikan masalah ini."
        else:
            answer = f"Maaf, terjadi kesalahan dalam memproses pertanyaan Anda. Detail: {str(e)}"
    
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
    # Initialize the RAG chain at startup
    try:
        logger.info("Initializing RAG chain at application startup")
        initialized_chain = initialize_rag_chain()
        if initialized_chain is None:
            logger.error("Failed to initialize RAG chain at startup - app may have limited functionality")
        else:
            logger.info("RAG chain successfully initialized at startup")
    except Exception as e:
        logger.error(f"Error during startup initialization: {e}", exc_info=True)
        
    # Run the main application
    main()