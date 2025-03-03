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
import random

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

3. For conversation endings (when user says thank you or similar):
   - If the user says "terima kasih", "makasih", "thank you", or similar expressions of gratitude:
     - Respond with a friendly closing like "Sama-sama! Senang bisa membantu. Jika ada pertanyaan lain, silakan tanyakan kembali."
   - Never respond with just a short phrase like "Prosedur pendaftaran sidang skripsi dilakukan secara online."
   - Always provide a complete, friendly closing that acknowledges the user's gratitude

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

def is_gratitude_expression(text):
    """Detect if the text contains expressions of gratitude in Indonesian or English"""
    # Convert to lowercase for case-insensitive matching
    text_lower = text.lower()
    
    # Common gratitude expressions in Indonesian and English
    gratitude_expressions = [
        "terima kasih", "makasih", "thank you", "thanks", "thx", 
        "terimakasih", "trims", "thank", "makasi", "mksh", "tq", 
        "baik terimakasih", "baik terima kasih", "ok terimakasih",
        "ok terima kasih", "okay terimakasih", "okay terima kasih"
    ]
    
    # Check if any gratitude expression is in the text
    return any(expr in text_lower for expr in gratitude_expressions)

def get_gratitude_response():
    """Generate a varied response to expressions of gratitude"""
    responses = [
        "Sama-sama! Senang bisa membantu. Jika ada pertanyaan lain, silakan tanyakan kembali.",
        "Dengan senang hati! Semoga informasi yang saya berikan bermanfaat. Jangan ragu untuk bertanya lagi jika diperlukan.",
        "Terima kasih kembali! Saya siap membantu jika Anda memiliki pertanyaan lainnya tentang Sistem Informasi Undiksha.",
        "Sama-sama! Semoga sukses dengan studi Anda di Sistem Informasi Undiksha. Jika butuh bantuan lagi, silakan hubungi saya.",
        "Senang bisa membantu Anda! Jika ada hal lain yang ingin ditanyakan, saya siap membantu kapan saja.",
        "Tidak masalah! Semoga informasi yang saya berikan membantu. Silakan bertanya lagi jika ada yang kurang jelas."
    ]
    
    return random.choice(responses)

def is_procedure_question(text):
    """Detect if the user is asking about a procedure or process"""
    # Convert to lowercase for case-insensitive matching
    text_lower = text.lower()
    
    # Keywords that indicate questions about procedures or processes
    procedure_keywords = [
        "bagaimana", "langkah", "tahap", "proses", "cara", "prosedur", 
        "mekanisme", "alur", "how to", "how do", "steps", "procedure",
        "apa yang harus", "what should", "what must", "apa saja yang",
        "setelah", "after", "selanjutnya", "next", "kemudian", "then"
    ]
    
    # Check if any procedure keyword is in the text
    return any(keyword in text_lower for keyword in procedure_keywords)

def is_asking_for_details(text):
    """Detect if the user is asking for more details"""
    # Convert to lowercase for case-insensitive matching
    text_lower = text.lower()
    
    # Keywords that indicate requests for more details
    detail_keywords = [
        "jelaskan lebih", "detail", "rinci", "elaborate", "explain more",
        "lebih lanjut", "further", "more information", "informasi lebih",
        "bisa dijelaskan", "can you explain", "tolong jelaskan", "please explain",
        "mohon jelaskan", "kindly explain", "tell me more", "ceritakan lebih"
    ]
    
    # Check if any detail keyword is in the text
    return any(keyword in text_lower for keyword in detail_keywords)

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
    
    # Check if the user is expressing gratitude
    if is_gratitude_expression(user_input):
        return get_gratitude_response()
    
    # Flag to indicate if this is a request for more details
    is_detail_request = is_asking_for_details(user_input)
    
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
            
            # Check if this is a procedure question but the answer is too short
            if is_procedure_question(user_input) and len(answer.split()) < 30 and not is_detail_request:
                # If the answer is too short for a procedure question, ask for more details
                logger.warning(f"Answer too short for procedure question: {answer}")
                answer += "\n\nApakah Anda ingin saya menjelaskan lebih detail tentang prosedur ini? Silakan beri tahu saya bagian mana yang ingin Anda ketahui lebih lanjut."
            
            # If this is a request for more details, make sure the answer is comprehensive
            if is_detail_request and len(answer.split()) < 100:
                logger.warning(f"Answer too short for detail request: {answer}")
                answer += "\n\nSaya harap penjelasan ini membantu. Jika Anda memerlukan informasi lebih spesifik, silakan tanyakan bagian tertentu yang ingin Anda ketahui lebih dalam."
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