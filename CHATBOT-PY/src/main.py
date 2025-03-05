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
import numpy as np
import csv
import datetime
import json
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
If you don't know the answer, don't try to make up an answer. Instead, respond politely and helpfully with:
1. An acknowledgment that you don't have specific information about that topic
2. An offer to help with other related topics that you might know about

IMPORTANT INSTRUCTIONS ABOUT KNOWLEDGE SCOPE:
- You ONLY have knowledge about Program Studi Sistem Informasi Undiksha.
- If the user asks about ANY other university (besides Undiksha) or ANY other program (besides Sistem Informasi), explicitly state that you do not have information about other programs or universities.
- NEVER provide specific information about programs other than Sistem Informasi or universities other than Undiksha.

IMPORTANT INSTRUCTIONS ABOUT LANGUAGE:
- ALWAYS respond in the SAME LANGUAGE that the user used in their question.
- If the user asks in Bahasa Indonesia, respond in Bahasa Indonesia.
- If the user asks in English, respond in English.
- Do not mix languages in your response.
- LANGUAGE MATCHING IS EXTREMELY IMPORTANT! Always check the language of the original question.
- For Indonesian questions, respond with: "Mohon maaf, saya tidak memiliki informasi spesifik tentang..."
- For English questions, respond with: "I'm sorry, I don't have specific information about..."

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
    chunk_embedding = model.encode(chunk)
    query_embedding = model.encode(query)
    
    # Calculate cosine similarity
    similarity = cosine_similarity([chunk_embedding], [query_embedding])[0][0]
    return similarity

def export_retrieval_to_csv(user_query, query_embedding, retrieved_data, filename=None):
    """
    Export retrieval data to CSV file
    
    Args:
        user_query (str): The user's query
        query_embedding (list): The embedding vector of the user's query
        retrieved_data (list): List of tuples (chunk, score)
        filename (str, optional): Custom filename for the CSV. Defaults to None.
    
    Returns:
        str: Path to the saved CSV file
    """
    if filename is None:
        # Create a timestamped filename
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"output/retrieval_data_{timestamp}.csv"
    
    # Ensure output directory exists
    os.makedirs("output", exist_ok=True)
    
    # Create CSV file
    with open(filename, 'w', newline='', encoding='utf-8') as csvfile:
        writer = csv.writer(csvfile)
        
        # Write header section - Question and its embedding
        writer.writerow(["Question"])
        writer.writerow([user_query])
        writer.writerow(["Question Embedding (Vector)"])
        writer.writerow([json.dumps(query_embedding.tolist())])
        writer.writerow([])  # Empty row as separator
        
        # Write retrieved chunks section
        writer.writerow(["Retrieved Chunks"])
        writer.writerow(["No", "Chunk", "Vector", "Score"])
        
        # Write each chunk with its data
        for i, (chunk, score) in enumerate(retrieved_data, 1):
            # Clean up excessive whitespace and newlines in the chunk
            cleaned_chunk = ' '.join(chunk.split())
            
            # Generate the embedding for this chunk
            chunk_embedding = model.encode(chunk).tolist()
            
            writer.writerow([
                i,
                cleaned_chunk,
                json.dumps(chunk_embedding),
                f"{score:.6f}"
            ])
    
    return filename

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

def chunking_and_retrieval(user_input, show_process=True, export_to_csv=False):
    if show_process:
        st.subheader("1. Chunking and Retrieval")
    
    try:
        retrieved_docs = retriever.get_relevant_documents(user_input)
        embedded_data = []
        
        # Generate query embedding for later use
        query_embedding = model.encode(user_input)
        
        for doc in retrieved_docs:
            chunk = doc.page_content
            relevance_score = calculate_relevance_scores(chunk, user_input)
            embedded_data.append((chunk, relevance_score))
        
        # Sort by relevance score in descending order
        embedded_data.sort(key=lambda x: x[1], reverse=True)
        
        if show_process and st.session_state.processing_new_question:
            # Display the embedding process only when processing a new question
            display_embedding_process(embedded_data, user_input, query_embedding)
            
            # Export to CSV if requested
            if export_to_csv:
                csv_path = export_retrieval_to_csv(user_input, query_embedding, embedded_data)
                st.success(f"Retrieval data exported to CSV: {csv_path}")
                
                # Provide download button for the CSV
                with open(csv_path, 'r', encoding='utf-8') as file:
                    csv_content = file.read()
                st.download_button(
                    label="Download CSV",
                    data=csv_content,
                    file_name=os.path.basename(csv_path),
                    mime="text/csv"
                )
                return embedded_data, query_embedding, csv_path
            
        return embedded_data, query_embedding
    except Exception as e:
        if show_process:
            st.warning(f"An error occurred during retrieval: {e}")
        return [], None

def display_embedding_process(embedded_data, query=None, query_embedding=None):
    st.subheader("Embedding Process")
    
    # Display original question and its embedding if available
    if query is not None and query_embedding is not None:
        st.subheader("Question Embedding")
        
        # Show the original question
        st.markdown("**Pertanyaan Asli**")
        st.text(query)
        
        # Show the full question embedding without truncation
        st.markdown("**Embedding Pertanyaan**")
        full_embedding = str(query_embedding.tolist())
        st.text_area("Full embedding vector:", full_embedding, height=200)
        
        # Add a separator
        st.markdown("---")
        
        # Display the retrieved chunks with full vector representations
        st.subheader("Retrieved Chunks")
        
        # Create data for the chunks table with full vectors
        chunks_data = []
        for i, (chunk, score) in enumerate(embedded_data, 1):
            # Generate vector for the chunk
            chunk_embedding = model.encode(chunk).tolist()
            # Format the chunk as a paragraph by replacing newlines with spaces
            formatted_chunk = ' '.join(chunk.split())
            chunks_data.append({
                "No": i,
                "Chunk": formatted_chunk,
                "Vektor": str(chunk_embedding),
                "Score": f"{score:.6f}"
            })
        
        # Create DataFrame
        chunks_df = pd.DataFrame(chunks_data)
        
        # Display the table with adjusted column widths
        st.dataframe(
            chunks_df,
            column_config={
                "No": st.column_config.NumberColumn(width="small"),
                "Chunk": st.column_config.TextColumn(width="large"),
                "Vektor": st.column_config.TextColumn(width="medium"),
                "Score": st.column_config.NumberColumn(width="small", format="%.6f")
            },
            hide_index=True
        )
        
        # Add a separator
        st.markdown("---")
    
    # Ensure this expander is not nested
    with st.expander("Detail Embedding", expanded=True):
        # Add a filter for top chunks
        col1, col2 = st.columns([1, 3])
        with col1:
            # Default to showing top 5 chunks if there are more than 5
            default_num_chunks = min(5, len(embedded_data))
            num_chunks = st.slider("Show top N chunks:", 1, len(embedded_data), default_num_chunks)
        
        with col2:
            # Display information about the number of chunks
            if num_chunks < len(embedded_data):
                # st.write(f"Showing {num_chunks} most relevant chunks out of {len(embedded_data)} total chunks")
                st.write(f"Showing {num_chunks} most relevant chunks out of 10 total chunks")
            
        # Filter data based on user selection
        filtered_data = embedded_data[:num_chunks] if num_chunks < len(embedded_data) else embedded_data
        
        # Prepare data for display
        data = {
            "No": range(1, len(filtered_data) + 1),
            "Chunk": [' '.join(chunk.split()) for chunk, _ in filtered_data],
            "Score": [score for _, score in filtered_data]
        }
        
        df = pd.DataFrame(data)
        # Display the dataframe with improved formatting
        st.dataframe(
            df,
            column_config={
                "No": st.column_config.NumberColumn(width="small"),
                "Chunk": st.column_config.TextColumn(width="large"),
                "Score": st.column_config.NumberColumn(width="small", format="%.6f")
            },
            hide_index=True
        )
        
        # Add a section to display the full raw text of filtered chunks
        st.subheader("Top Retrieved Content")
        
        # Format the chunks properly as paragraphs
        formatted_chunks = []
        for chunk, _ in filtered_data:
            # Clean up excessive whitespace and newlines
            cleaned_chunk = ' '.join(chunk.split())
            formatted_chunks.append(cleaned_chunk)
        
        # Join the formatted chunks with proper paragraph breaks
        full_content = "\n\n".join(formatted_chunks)
        
        # Display the formatted content
        st.text_area("Top filtered chunks:", full_content, height=200)
    
    st.success("Analisis informasi selesai!")

def format_dont_know_response(query):
    """
    Create a helpful and informative response when the system doesn't know the answer.
    
    Args:
        query (str): The user's original query
        
    Returns:
        str: A formatted response acknowledging lack of information
    """
    # Detect query type to give more specific recommendations
    query_lower = query.lower()
    
    # Base response parts (in Indonesian by default)
    greeting = "Mohon maaf, "
    acknowledgment = "saya tidak memiliki informasi spesifik tentang "
    
    # Analyze query to determine topic
    if "skripsi" in query_lower or "tugas akhir" in query_lower:
        topic = "proses skripsi atau tugas akhir di program studi tersebut"
        recommendations = [
            "Anda dapat mengunjungi website resmi Program Studi untuk informasi lengkap tentang prosedur skripsi",
            "Bagian akademik fakultas biasanya memiliki panduan lengkap tentang proses skripsi",
            "Dosen pembimbing akademik Anda juga dapat memberikan pengarahan tentang prosedur skripsi"
        ]
        clarification = "Saya hanya dapat memberikan informasi tentang proses skripsi di Program Studi Sistem Informasi Undiksha."
    else:
        topic = "topik tersebut"
        recommendations = [
            "Silakan cek website resmi institusi terkait untuk informasi yang akurat",
            "Bagian akademik atau administrasi kampus biasanya dapat membantu pertanyaan seperti ini",
            "Anda mungkin dapat menemukan informasi di sistem informasi akademik universitas"
        ]
        clarification = "Saya hanya memiliki informasi tentang Program Studi Sistem Informasi Undiksha."
    
    # Build the response
    response = f"{greeting}{acknowledgment}{topic}.\n\n"
    response += f"{clarification}\n\n"
    response += "Saran untuk mendapatkan informasi yang Anda cari:\n"
    
    for i, recommendation in enumerate(recommendations, 1):
        response += f"{i}. {recommendation}\n"
    
    response += "\nApakah ada hal lain yang dapat saya bantu terkait Program Studi Sistem Informasi Undiksha?"
    
    return response

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
            
            # Check if the response is a simple "I don't know" response
            dont_know_phrases = [
                "saya tidak tahu", 
                "tidak mengetahui", 
                "tidak memiliki informasi", 
                "tidak dapat memberikan informasi",
                "maaf, saya tidak",
                "i don't know",
                "i do not know",
                "i don't have",
                "i don't have specific information",
                "i don't have information",
                "i do not have information"
            ]
            
            # If it's a simple "don't know" response, replace it with our formatted response
            if any(phrase in answer.lower() for phrase in dont_know_phrases) and len(answer.split()) < 30:
                answer = format_dont_know_response(user_input)
                
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
            answer = "Maaf, terjadi kesalahan dalam memproses pertanyaan Anda. Silakan coba kembali beberapa saat lagi atau coba pertanyaan lain. Jika masalah berlanjut, silakan hubungi administrator sistem."
    
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
    
    # Initialize other session state variables for developer mode
    if "dev_mode_embedded_data" not in st.session_state:
        st.session_state.dev_mode_embedded_data = None
    
    if "dev_mode_query_embedding" not in st.session_state:
        st.session_state.dev_mode_query_embedding = None
    
    if "dev_mode_csv_path" not in st.session_state:
        st.session_state.dev_mode_csv_path = None
        
    # Store the original query
    if "dev_mode_query" not in st.session_state:
        st.session_state.dev_mode_query = None
        
    # Store the latest answer
    if "dev_mode_latest_answer" not in st.session_state:
        st.session_state.dev_mode_latest_answer = None
        
    # Flag to track if we're processing a new question
    if "processing_new_question" not in st.session_state:
        st.session_state.processing_new_question = False
    
    # Sidebar
    with st.sidebar:
        st.image("https://encrypted-tbn0.gstatic.com/images?q=tbn:ANd9GcSxGdjI58B_NuUd8eAWfRBlVms7f-2e2oI_SA&s", width=150)
        st.title("Virtual Assistant SI Undiksha")
        mode = st.selectbox("Mode", ["User Mode", "Developer Mode"])
        
        # Add CSV export option in developer mode
        export_to_csv = False
        if mode == "Developer Mode":
            export_to_csv = st.checkbox("Export retrieval data to CSV", value=False)
        
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

    show_process = mode == "Developer Mode"
    
    # In developer mode, we'll display differently to ensure answers appear at the bottom
    if show_process:
        # Create a container for displaying the conversation
        chat_container = st.container()
        
        # In developer mode, only display the questions in the main chat area
        with chat_container:
            # Show questions-only view
            for i, message in enumerate(st.session_state.chat_history):
                if message["role"] == "user":
                    with st.chat_message(message["role"]):
                        st.markdown(message["content"])
                        
        # Display the embedding process info (if available)
        if st.session_state.dev_mode_embedded_data is not None and not st.session_state.processing_new_question:
            display_embedding_process(
                st.session_state.dev_mode_embedded_data,
                st.session_state.dev_mode_query,
                st.session_state.dev_mode_query_embedding
            )
            
            # If we exported to CSV, show download button
            if st.session_state.dev_mode_csv_path is not None:
                st.success(f"Retrieval data exported to CSV: {st.session_state.dev_mode_csv_path}")
                
                # Provide download button for the CSV
                try:
                    with open(st.session_state.dev_mode_csv_path, 'r', encoding='utf-8') as file:
                        csv_content = file.read()
                    st.download_button(
                        label="Download CSV",
                        data=csv_content,
                        file_name=os.path.basename(st.session_state.dev_mode_csv_path),
                        mime="text/csv"
                    )
                except Exception as e:
                    st.warning(f"Could not load CSV file: {e}")
                    
        # Display the latest answer at the very bottom (after all processing)
        if st.session_state.dev_mode_latest_answer is not None and not st.session_state.processing_new_question:
            with st.chat_message("assistant"):
                st.markdown(st.session_state.dev_mode_latest_answer)
    else:
        # In user mode, display the full chat history normally
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
        
        # Set flag that we're processing a new question
        st.session_state.processing_new_question = True
        
        # Process the query
        if show_process:
            result = chunking_and_retrieval(user_input, show_process, export_to_csv)
            
            # Handle different return types
            if export_to_csv and len(result) == 3:
                embedded_data, query_embedding, csv_path = result
                st.session_state.dev_mode_csv_path = csv_path
            else:
                embedded_data, query_embedding = result
            
            # Store in session state for persistence
            st.session_state.dev_mode_embedded_data = embedded_data
            st.session_state.dev_mode_query_embedding = query_embedding
            st.session_state.dev_mode_query = user_input
        else:
            # In user mode, we don't need to store the results
            chunking_and_retrieval(user_input, show_process)
        
        # Generate response
        answer = generation(user_input, show_process)
        
        # Store the latest answer
        st.session_state.dev_mode_latest_answer = answer
        
        # Add assistant response to chat history
        st.session_state.chat_history.append({"role": "assistant", "content": answer})
        
        # Clear the processing flag since we're done
        st.session_state.processing_new_question = False
        
        # Rerun to update the UI
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
