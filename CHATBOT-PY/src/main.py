import os
import ssl
import sys
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
import re

# Patch to prevent torch._classes.__path__._path error in Streamlit's file watcher
import importlib.abc
import importlib.machinery

class TorchImportFixer(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path, target=None):
        # Intercept attempts to access torch._classes.__path__._path
        if fullname == 'torch._classes.__path__._path':
            return None
        return None

# Add the import hook to sys.meta_path
sys.meta_path.insert(0, TorchImportFixer())

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
            template="""Gunakan bagian-bagian konteks berikut untuk menjawab pertanyaan pengguna secara komprehensif dan akurat.
Jika Anda tidak tahu jawabannya, jangan mencoba membuat-buat jawaban. Sebagai gantinya, jawab dengan sopan dan membantu dengan:
1. Pengakuan bahwa Anda tidak memiliki informasi spesifik tentang topik tersebut
2. Tawaran untuk membantu dengan topik terkait lainnya yang mungkin Anda ketahui

INSTRUKSI PENTING TENTANG RUANG LINGKUP PENGETAHUAN:
- Anda HANYA memiliki pengetahuan tentang Program Studi Sistem Informasi Undiksha.
- Jika pengguna bertanya tentang universitas LAIN (selain Undiksha) atau program LAIN (selain Sistem Informasi), nyatakan secara eksplisit bahwa Anda tidak memiliki informasi tentang program atau universitas lain.
- JANGAN PERNAH memberikan informasi spesifik tentang program selain Sistem Informasi atau universitas selain Undiksha.

INSTRUKSI PENTING TENTANG BAHASA:
- SELALU menjawab dalam BAHASA YANG SAMA dengan yang digunakan pengguna dalam pertanyaannya.
- Jika pengguna bertanya dalam Bahasa Indonesia, jawab dalam Bahasa Indonesia.
- Jika pengguna bertanya dalam Bahasa Inggris, jawab dalam Bahasa Inggris.
- Jangan mencampur bahasa dalam respons Anda.
- KESESUAIAN BAHASA SANGAT PENTING! Selalu periksa bahasa dari pertanyaan asli.
- Untuk pertanyaan dalam Bahasa Indonesia, gunakan: "Mohon maaf, saya tidak memiliki informasi spesifik tentang..."
- Untuk pertanyaan dalam Bahasa Inggris, gunakan: "I'm sorry, I don't have specific information about..."

INSTRUKSI PENTING UNTUK PENANGANAN SINGKATAN:
- Ketika pengguna menggunakan "SI", "Si", atau "si" dalam pertanyaan mereka, SELALU tafsirkan ini sebagai "Sistem Informasi" (Information Systems).
- Misalnya, jika pengguna bertanya "Siapa koorprodi SI sekarang?", tafsirkan ini sebagai "Siapa koorprodi Sistem Informasi sekarang?"
- Jika pengguna bertanya tentang "prodi SI", "jurusan SI", "program studi SI", dll., selalu anggap ini merujuk pada program Sistem Informasi.
- Demikian pula, jika mereka bertanya tentang "SI Undiksha", tafsirkan ini sebagai "Sistem Informasi Undiksha".
- Ini berlaku untuk semua konteks di mana "SI", "Si", atau "si" dapat secara wajar ditafsirkan sebagai singkatan dari "Sistem Informasi".

INSTRUKSI PENTING UNTUK INFORMASI DOSEN:
- Untuk pertanyaan tentang dosen, SELALU berikan informasi lengkap ketika tersedia dalam konteks.
- Ketika ditanya tentang dosen tertentu, sertakan nama lengkap, NIP/NIDN, jabatan, dan detail relevan lainnya.
- Untuk pertanyaan seperti "siapa koorprodi SI sekarang?" atau pertanyaan tentang peran spesifik, berikan nama lengkap dan jabatan.
- Untuk pertanyaan seperti "siapa dosen yang bernama pak/bu X?", berikan nama lengkap, NIP/NIDN, dan jabatan mereka.
- Jika ditanya informasi lanjutan tentang dosen, seperti NIP atau jabatan mereka, berikan semua detail yang tersedia.
- JANGAN PERNAH menjawab "Saya tidak memiliki informasi" ketika informasi dosen tersedia dalam konteks.
- Jika pengguna bertanya tentang dosen dan Anda memiliki informasi dalam konteks, berikan SEMUA detail yang Anda miliki.
- Untuk pertanyaan tentang "Koordinator Program Studi" atau "Koorprodi", berikan informasi lengkap tentang siapa yang memegang jabatan ini.

INFORMASI PENTING TENTANG DOSEN TERTENTU:
Ketika ditanya tentang dosen tertentu, SELALU berikan informasi berikut jika dosen tersebut disebutkan:
- Nama lengkap
- NIP/NIDN
- Jabatan
- Konsentrasi

INSTRUKSI PENTING UNTUK PERTANYAAN KOORPRODI:
- Jika pengguna bertanya tentang "Koorprodi" atau "Koordinator Program Studi" atau "Kaprodi" atau "Ketua Program Studi", Anda HARUS memberikan informasi lengkap.
- Koorprodi Sistem Informasi saat ini adalah Ir. I Made Dendi Maysanjaya, S.Kom., M.Kom.
- SELALU sertakan informasi ini ketika ditanya tentang Koorprodi, bahkan jika tidak ditemukan secara eksplisit dalam konteks.
- Untuk pertanyaan seperti "siapa koorprodi SI sekarang?" atau "siapa koordinator prodi sistem informasi?", selalu jawab dengan informasi lengkap tentang Ir. I Made Dendi Maysanjaya, S.Kom., M.Kom.
- JANGAN PERNAH menjawab "Saya tidak memiliki informasi" ketika ditanya tentang Koorprodi.

INSTRUKSI PENTING UNTUK MEMFORMAT RESPONS ANDA:
1. Untuk pertanyaan tentang proses atau tahapan (seperti penyelesaian skripsi), ikuti format ini PERSIS:
   - Mulai dengan "Terdapat [jumlah] tahap utama dalam [proses], yaitu:"
   - Sajikan setiap tahap sebagai PARAGRAF TERPISAH (bukan sebagai poin-poin)
   - Untuk setiap paragraf tahap, mulai dengan nama tahap dalam huruf tebal, diikuti dengan deskripsi
   - Pastikan untuk menggunakan nama tahap PERSIS seperti yang disebutkan dalam konteks
   - Akhiri dengan paragraf tentang siapa yang terlibat dalam proses tersebut

2. Untuk proses penyelesaian skripsi secara khusus:
   - Pastikan empat tahapnya adalah: Pengajuan Topik, Ujian Proposal Skripsi, Penyusunan Skripsi, dan Pengesahan Laporan Skripsi
   - Format setiap tahap sebagai paragraf lengkap, bukan poin-poin
   - Tambahkan kalimat penutup tentang siapa yang terlibat dalam seluruh proses

3. Untuk penutup percakapan (ketika pengguna mengucapkan terima kasih atau sejenisnya):
   - Jika pengguna mengatakan "terima kasih", "makasih", "thank you", atau ekspresi terima kasih serupa:
     - Jawab dengan penutup yang ramah seperti "Sama-sama! Senang bisa membantu. Jika ada pertanyaan lain, silakan tanyakan kembali."
   - Jangan pernah hanya menjawab dengan frase singkat seperti "Prosedur pendaftaran sidang skripsi dilakukan secara online."
   - Selalu berikan penutup lengkap dan ramah yang mengakui ucapan terima kasih pengguna

4. Untuk pertanyaan informasi umum (seperti "apa itu prodi sistem informasi undiksha"):
   - Berikan jawaban komprehensif dengan setidaknya 3-4 paragraf informasi
   - Sertakan informasi tentang kurikulum program, area fokus, dan fitur utama
   - Sebutkan spesialisasi atau konsentrasi yang tersedia
   - Akhiri dengan kalimat yang menawarkan untuk memberikan informasi lebih spesifik jika diperlukan

5. JANGAN PERNAH menyertakan pernyataan seperti "Saya tidak memiliki informasi" ketika Anda sebenarnya MEMILIKI informasi tersebut dalam konteks.
   - Hanya gunakan pernyataan tersebut ketika pertanyaan benar-benar di luar ruang lingkup pengetahuan Anda
   - Jika Anda memiliki informasi parsial, berikan apa yang Anda ketahui dan kemudian tawarkan untuk membantu dengan topik terkait

Konteks: {context}

Pertanyaan: {question}

Jawaban:""",
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
            # Remove "Jawaban: " prefix with the same robust cleaning
            chunk_to_clean = chunk
            # Handle different variations of "Jawaban:" with different spacing
            chunk_to_clean = re.sub(r'Jawaban\s*:', '', chunk_to_clean)
            chunk_to_clean = re.sub(r'\?\s*Jawaban\s*:', '?', chunk_to_clean)
            chunk_to_clean = re.sub(r'\.\s*Jawaban\s*:', '.', chunk_to_clean)
            
            # Remove question marks at beginning of text or after common patterns
            chunk_to_clean = re.sub(r'^Sistem\s+Informasi\s*\?', 'Sistem Informasi', chunk_to_clean)
            chunk_to_clean = re.sub(r'^SI\s*\?', 'SI', chunk_to_clean)
            chunk_to_clean = re.sub(r'^Program\s+Studi\s*\?', 'Program Studi', chunk_to_clean)
            chunk_to_clean = re.sub(r'^\s*\?\s*', '', chunk_to_clean)  # Remove leading question marks
            
            cleaned_chunk = ' '.join(chunk_to_clean.strip().split())
            
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
    """
    Check if the text is an expression of gratitude.
    
    Args:
        text (str): The text to check
        
    Returns:
        bool: True if the text is an expression of gratitude, False otherwise
    """
    text_lower = text.lower()
    
    # Common gratitude expressions in Indonesian and English
    gratitude_expressions = [
        "terima kasih", "makasih", "thank you", "thanks", "thx", 
        "thank", "makasi", "terimakasih", "trims", "trimakasih",
        "thank u", "tq", "ty", "terimakasi", "terima kasi"
    ]
    
    # Check if any gratitude expression is in the text
    return any(expression in text_lower for expression in gratitude_expressions)

def get_gratitude_response():
    """
    Generate a random gratitude response.
    
    Returns:
        str: A random gratitude response
    """
    # Get the language from the last user message
    if 'chat_history' in st.session_state and st.session_state.chat_history:
        for msg in reversed(st.session_state.chat_history):
            if msg["role"] == "user":
                language = detect_language(msg["content"])
                break
    else:
        language = "id"  # Default to Indonesian
    
    is_english = (language == 'en')
    
    # Gratitude responses in Indonesian
    indonesian_responses = [
        "Dengan senang hati! Jika Anda memiliki pertanyaan lain tentang Program Studi Sistem Informasi Undiksha, saya siap membantu.",
        "Tidak masalah! Saya senang dapat memberikan informasi yang Anda butuhkan. Jangan ragu untuk bertanya lagi.",
        "Sama-sama! Semoga informasi yang saya berikan bermanfaat. Jika ada hal lain yang ingin Anda ketahui, silakan tanyakan.",
        "Terima kasih kembali! Saya di sini untuk membantu Anda dengan informasi tentang Program Studi Sistem Informasi Undiksha."
    ]
    
    # Gratitude responses in English
    english_responses = [
        "You're welcome! Happy to help. If you have any other questions, feel free to ask.",
        "My pleasure! If you have any other questions about the Information Systems Program at Undiksha, I'm here to help.",
        "No problem! I'm glad I could provide the information you needed. Don't hesitate to ask again.",
        "You're welcome! I hope the information I provided was helpful. If there's anything else you'd like to know, please ask.",
        "Thank you too! I'm here to help you with information about the Information Systems Program at Undiksha."
    ]
    
    # Return a random response based on the detected language
    if is_english:
        return random.choice(english_responses)
    else:
        return random.choice(indonesian_responses)

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

def is_greeting(text: str) -> bool:
    """Check if the input is a greeting"""
    greetings = {
        'halo', 'hello', 'hi', 'hai', 'hey', 'hallo', 'helo',
        'selamat pagi', 'selamat siang', 'selamat sore', 'selamat malam',
        'pagi', 'siang', 'sore', 'malam'
    }
    return text.strip().lower() in greetings

def is_lecturer_question(text: str) -> bool:
    """Check if the question is about a lecturer"""
    text_lower = text.lower()
    
    # Keywords that indicate questions about lecturers
    lecturer_keywords = [
        "dosen", "lecturer", "professor", "pak ", "bu ", "bapak ", "ibu ",
        "koordinator", "koorprodi", "kaprodi", "ketua prodi", "ketua program studi",
        "pengajar", "staff", "staf", "pengampu", "matakuliah", "mata kuliah",
        "siapa yang mengajar", "siapa yang menjabat", "who teaches", "who is the coordinator"
    ]
    
    # Specific lecturer names to check for
    lecturer_names = [
        "ardwi", "mahendra", "rasben", "aditra", "raditya", "dendi", 
        "maysanjaya", "darmawiguna", "dantes", "pradnyana", "putra"
    ]
    
    # Check if any lecturer keyword is in the text
    has_lecturer_keyword = any(keyword in text_lower for keyword in lecturer_keywords)
    
    # Check if any lecturer name is in the text
    has_lecturer_name = any(name in text_lower for name in lecturer_names)
    
    # Return true if either condition is met
    return has_lecturer_keyword or has_lecturer_name

def get_greeting_response(text: str) -> str:
    """Generate appropriate greeting response"""
    # Detect language
    is_english = detect_language(text) == 'en'
    
    if is_english:
        return "Hello! 👋 I'm the virtual assistant for Undiksha's Information Systems Program. How can I help you today?"
    else:
        return "Halo! 👋 Saya adalah asisten virtual Program Studi Sistem Informasi Undiksha. Ada yang bisa saya bantu hari ini?"

def chunking_and_retrieval(user_input, show_process=True, export_to_csv=False):
    if show_process:
        st.subheader("1. Chunking and Retrieval")
    
    try:
        # Check if this is a lecturer question
        is_lecturer_query = is_lecturer_question(user_input)
        
        # For lecturer questions, modify the query to improve retrieval
        query_for_retrieval = user_input
        if is_lecturer_query:
            # Extract potential lecturer name from the query
            words = user_input.lower().split()
            for i, word in enumerate(words):
                if word in ["pak", "bu", "bapak", "ibu", "dosen"] and i < len(words) - 1:
                    # Add "dosen" keyword if not already present to improve retrieval
                    if "dosen" not in user_input.lower():
                        query_for_retrieval = f"dosen {user_input}"
                    break
            
            # If asking about a coordinator or specific role, add those keywords
            if any(term in user_input.lower() for term in ["koordinator", "koorprodi", "kaprodi", "ketua"]):
                if "dosen" not in query_for_retrieval.lower():
                    query_for_retrieval = f"dosen koordinator program studi {query_for_retrieval}"
        
        # Get relevant documents using the potentially modified query
        # Initially fetch more documents (e.g., 20) to account for duplicates
        # Use search_kwargs to set the number of results instead of directly setting k
        retrieved_docs = retriever.get_relevant_documents(
            query_for_retrieval, 
            search_kwargs={"k": 20}
        )
        
        # Check if we got any documents
        if not retrieved_docs:
            if show_process:
                st.warning("No documents were retrieved for this query.")
            logger.warning(f"No documents retrieved for query: {query_for_retrieval}")
            return [], None
            
        total_retrieved = len(retrieved_docs)
        
        # Deduplicate chunks and keep track of which ones were removed
        seen_chunks = set()
        unique_docs = []
        duplicate_count = 0
        removed_docs = []
        
        for doc in retrieved_docs:
            # Create a simplified version of the chunk for deduplication
            # Focus on the first 100 chars as a chunk signature
            content = doc.page_content.strip()
            chunk_signature = content[:100]
            
            # Skip documents that contain RSS feed markers (usually contain "appeared first on")
            if "appeared first on" in content.lower():
                removed_docs.append(doc)
                duplicate_count += 1
                continue
                
            # Skip duplicate chunks based on their signature
            if chunk_signature in seen_chunks:
                removed_docs.append(doc)
                duplicate_count += 1
                continue
                
            seen_chunks.add(chunk_signature)
            unique_docs.append(doc)
        
        # Standard number of chunks to show
        STANDARD_CHUNK_COUNT = 12
        
        # If we don't have enough unique chunks, process and add more
        if len(unique_docs) < STANDARD_CHUNK_COUNT:
            # Try to get more chunks to reach the standard count
            # Use search_kwargs to set the number of results
            more_needed = STANDARD_CHUNK_COUNT - len(unique_docs)
            additional_k = total_retrieved + more_needed + 5  # Add a buffer
            
            # Get more documents
            more_docs = retriever.get_relevant_documents(
                query_for_retrieval, 
                search_kwargs={"k": additional_k}
            )
            
            # Process only the new documents (skip those we've already seen)
            for doc in more_docs[total_retrieved:]:
                content = doc.page_content.strip()
                chunk_signature = content[:100]
                
                # Skip RSS feeds and duplicates
                if "appeared first on" in content.lower() or chunk_signature in seen_chunks:
                    continue
                
                seen_chunks.add(chunk_signature)
                unique_docs.append(doc)
                
                # Stop once we have enough
                if len(unique_docs) >= STANDARD_CHUNK_COUNT:
                    break
        
        # Trim to standard count if we have more
        unique_docs = unique_docs[:STANDARD_CHUNK_COUNT]
        
        # Store the total number of docs for display purposes
        total_initial_docs = total_retrieved
        
        # Use the deduplicated docs to create embedded data
        embedded_data = []
        
        # Generate query embedding for later use
        query_embedding = model.encode(user_input)
        
        for doc in unique_docs:
            chunk = doc.page_content
            relevance_score = calculate_relevance_scores(chunk, user_input)
            embedded_data.append((chunk, relevance_score))
        
        # Sort by relevance score in descending order
        embedded_data.sort(key=lambda x: x[1], reverse=True)
        
        if show_process and st.session_state.processing_new_question:
            # Display the embedding process only when processing a new question
            display_embedding_process(embedded_data, user_input, query_embedding, total_initial_docs)
            
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
        logger.error(f"Error in chunking_and_retrieval: {e}", exc_info=True)
        return [], None

def display_embedding_process(embedded_data, query=None, query_embedding=None, total_before_dedup=None):
    st.subheader("Embedding Process")
    
    # Generate a unique key for sliders in this function instance
    unique_key = f"slider_{datetime.datetime.now().strftime('%Y%m%d%H%M%S%f')}"
    
    # If no data was provided, display a detailed message and return early
    if not embedded_data:
        st.warning("No embedding data available to display. The retrieval process may have failed or returned no results.")
        st.info("If this problem persists after asking multiple questions, try refreshing the application.")
        return
    
    if query_embedding is None:
        st.warning("No query embedding available. The embedding process may have failed.")
        st.info("Check the logs for more information about embedding errors.")
        return
    
    # Function to clean chunks by removing "Jawaban: " prefix
    def clean_chunk(chunk):
        # Handle different variations of "Jawaban:" with different spacing
        chunk = re.sub(r'Jawaban\s*:', '', chunk)
        chunk = re.sub(r'\?\s*Jawaban\s*:', '?', chunk)
        chunk = re.sub(r'\.\s*Jawaban\s*:', '.', chunk)
        
        # Remove question marks at beginning of text or after common patterns
        chunk = re.sub(r'^Sistem\s+Informasi\s*\?', 'Sistem Informasi', chunk)
        chunk = re.sub(r'^SI\s*\?', 'SI', chunk)
        chunk = re.sub(r'^Program\s+Studi\s*\?', 'Program Studi', chunk)
        chunk = re.sub(r'^\s*\?\s*', '', chunk)  # Remove leading question marks
        
        # Remove "Title:" prefix that often appears in RSS feeds
        chunk = re.sub(r'^Title:\s*', '', chunk)
        
        # Remove HTML tags that might appear in RSS feeds
        chunk = re.sub(r'<.*?>', '', chunk)
        
        # Remove "Content:" prefix
        chunk = re.sub(r'Content:\s*', '', chunk)
        
        # Remove common RSS feed footer pattern
        chunk = re.sub(r'The post .* appeared first on .*\.', '', chunk)
        
        return chunk.strip()
    
    # Process embedded data to clean chunks
    cleaned_embedded_data = []
    for chunk, score in embedded_data:
        cleaned_chunk = clean_chunk(chunk)
        cleaned_embedded_data.append((cleaned_chunk, score))
    
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

            # Show deduplication stats if available
    if total_before_dedup is not None:
        total_after_dedup = len(embedded_data)
        
        if total_before_dedup > total_after_dedup:
            # We processed more chunks but are showing the standard number
            st.info(f"🔍 Retrieved and processed {total_before_dedup} chunks, showing {total_after_dedup} unique chunks without duplicates.")
        else:
            # No duplicates were found
            st.info(f"🔍 Retrieved {total_before_dedup} chunks with no duplicates found.")
        
        # Create data for the chunks table with full vectors
        chunks_data = []
        for i, (chunk, score) in enumerate(cleaned_embedded_data, 1):
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
    with st.expander("Detail Chunks", expanded=True):
        # Only show the slider and details if we have data
        if len(cleaned_embedded_data) > 0:
            # Add a filter for top chunks
            col1, col2 = st.columns([1, 3])
            with col1:
                # Default to showing top 5 chunks if there are more than 5
                default_num_chunks = min(5, len(cleaned_embedded_data))
                num_chunks = st.slider("Show top N chunks:", 1, len(cleaned_embedded_data), default_num_chunks, key=f"{unique_key}_top_chunks")
            
            with col2:
                # Display information about the number of chunks
                if num_chunks < len(cleaned_embedded_data):
                    # st.write(f"Showing {num_chunks} most relevant chunks out of {len(embedded_data)} total chunks")
                    st.write(f"Showing {num_chunks} most relevant chunks out of 10 total chunks")
                
            # Filter data based on user selection
            filtered_data = cleaned_embedded_data[:num_chunks] if num_chunks < len(cleaned_embedded_data) else cleaned_embedded_data
            
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
                # The chunks in filtered_data are already cleaned by clean_chunk function
                # Just clean up excessive whitespace and newlines
                cleaned_chunk = ' '.join(chunk.split())
                formatted_chunks.append(cleaned_chunk)
            
            # Join the formatted chunks with proper paragraph breaks
            full_content = "\n\n".join(formatted_chunks)
            
            # Display the formatted content
            st.text_area("Top filtered chunks:", full_content, height=200)
        else:
            # Display a message when no chunks are available
            st.info("No chunks retrieved for this query.")
    
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
    
    # Detect language (English or Indonesian)
    language = detect_language(query)
    is_english = (language == 'en')
    
    # Base response parts
    if is_english:
        greeting = "I'm sorry, "
        acknowledgment = "I don't have specific information about "
        ending = "\n\nIs there anything else I can help you with regarding the Information Systems Program at Undiksha?"
    else:
        greeting = "Mohon maaf, "
        acknowledgment = "saya tidak memiliki informasi spesifik tentang "
        ending = "\n\nApakah ada hal lain yang dapat saya bantu terkait Program Studi Sistem Informasi Undiksha?"
    
    # Check if this is a lecturer question
    is_lecturer_query = is_lecturer_question(query)
    
    # Analyze query to determine topic and provide relevant recommendations
    if is_lecturer_query:
        # Special handling for lecturer questions
        if is_english:
            topic = "the specific lecturer you're asking about"
            recommendations = [
                "You can check the official Information Systems Program website for the complete faculty directory",
                "The faculty information is usually available in the academic information system",
                "For the most up-to-date information about faculty members and their positions, you can contact the department office directly"
            ]
            clarification = "I should have information about lecturers in the Information Systems Program at Undiksha. Let me try to search for more specific information."
        else:
            topic = "dosen spesifik yang Anda tanyakan"
            recommendations = [
                "Anda dapat memeriksa website resmi Program Studi Sistem Informasi untuk direktori dosen lengkap",
                "Informasi dosen biasanya tersedia di sistem informasi akademik",
                "Untuk informasi terbaru tentang dosen dan jabatan mereka, Anda dapat menghubungi kantor jurusan secara langsung"
            ]
            clarification = "Seharusnya saya memiliki informasi tentang dosen-dosen di Program Studi Sistem Informasi Undiksha. Mari saya coba mencari informasi yang lebih spesifik."
    elif "skripsi" in query_lower or "tugas akhir" in query_lower or "thesis" in query_lower or "final project" in query_lower:
        if is_english:
            topic = "the thesis or final project process in that program"
            recommendations = [
                "You can visit the official Program website for complete information about thesis procedures",
                "The faculty's academic department usually has a complete guide about the thesis process",
                "Your academic advisor can also provide guidance on thesis procedures"
            ]
            clarification = "I can only provide information about the thesis process in the Information Systems Program at Undiksha."
        else:
            topic = "proses skripsi atau tugas akhir di program studi tersebut"
            recommendations = [
                "Anda dapat mengunjungi website resmi Program Studi untuk informasi lengkap tentang prosedur skripsi",
                "Bagian akademik fakultas biasanya memiliki panduan lengkap tentang proses skripsi",
                "Dosen pembimbing akademik Anda juga dapat memberikan pengarahan tentang prosedur skripsi"
            ]
            clarification = "Saya hanya dapat memberikan informasi tentang proses skripsi di Program Studi Sistem Informasi Undiksha."
    elif "kurikulum" in query_lower or "curriculum" in query_lower or "mata kuliah" in query_lower or "courses" in query_lower:
        if is_english:
            topic = "the specific curriculum or courses you're asking about"
            recommendations = [
                "The official Information Systems Program website at Undiksha has detailed curriculum information",
                "You can check the academic handbook or course catalog for complete course listings",
                "The academic department can provide the most up-to-date curriculum information"
            ]
            clarification = "I can provide general information about the Information Systems Program curriculum at Undiksha, but may not have details about specific courses or recent curriculum changes."
        else:
            topic = "kurikulum atau mata kuliah spesifik yang Anda tanyakan"
            recommendations = [
                "Website resmi Program Studi Sistem Informasi Undiksha memiliki informasi kurikulum yang detail",
                "Anda dapat memeriksa buku panduan akademik atau katalog mata kuliah untuk daftar lengkap mata kuliah",
                "Bagian akademik dapat memberikan informasi kurikulum yang paling terbaru"
            ]
            clarification = "Saya dapat memberikan informasi umum tentang kurikulum Program Studi Sistem Informasi Undiksha, tetapi mungkin tidak memiliki detail tentang mata kuliah spesifik atau perubahan kurikulum terbaru."
    elif "dosen" in query_lower or "lecturer" in query_lower or "professor" in query_lower or "staff" in query_lower:
        if is_english:
            topic = "the specific faculty members or staff you're asking about"
            recommendations = [
                "The official Information Systems Program website usually has a faculty directory",
                "You can contact the department office for current faculty information",
                "The university's main website may have a complete staff directory"
            ]
            clarification = "I can provide general information about the Information Systems Program at Undiksha, but may not have up-to-date details about specific faculty members."
        else:
            topic = "dosen atau staf spesifik yang Anda tanyakan"
            recommendations = [
                "Website resmi Program Studi Sistem Informasi biasanya memiliki direktori dosen",
                "Anda dapat menghubungi kantor jurusan untuk informasi dosen terkini",
                "Website utama universitas mungkin memiliki direktori staf lengkap"
            ]
            clarification = "Saya dapat memberikan informasi umum tentang Program Studi Sistem Informasi Undiksha, tetapi mungkin tidak memiliki detail terbaru tentang dosen spesifik."
    else:
        if is_english:
            topic = "that specific topic"
            recommendations = [
                "Please check the official website of the relevant institution for accurate information",
                "The academic or administrative department of the university can usually help with such questions",
                "You may find information in the university's academic information system"
            ]
            clarification = "I only have information about the Information Systems Program at Undiksha University."
        else:
            topic = "topik spesifik tersebut"
            recommendations = [
                "Silakan cek website resmi institusi terkait untuk informasi yang akurat",
                "Bagian akademik atau administrasi kampus biasanya dapat membantu pertanyaan seperti ini",
                "Anda mungkin dapat menemukan informasi di sistem informasi akademik universitas"
            ]
            clarification = "Saya hanya memiliki informasi tentang Program Studi Sistem Informasi Undiksha."
    
    # Build the response
    response = f"{greeting}{acknowledgment}{topic}.\n\n"
    response += f"{clarification}\n\n"
    
    if is_english:
        response += "Suggestions for finding the information you're looking for:\n"
    else:
        response += "Saran untuk mendapatkan informasi yang Anda cari:\n"
    
    for i, recommendation in enumerate(recommendations, 1):
        response += f"{i}. {recommendation}\n"
    
    response += ending
    
    return response

def detect_language(text):
    """
    Detect whether the text is in English or Indonesian.
    
    Args:
        text (str): The text to analyze
        
    Returns:
        str: 'en' for English, 'id' for Indonesian
    """
    text_lower = text.lower()
    
    # Common English words
    english_words = [
        "what", "how", "when", "where", "who", "why", 
        "is", "are", "am", "was", "were", "be", "been", "being",
        "have", "has", "had", "do", "does", "did", "can", "could",
        "will", "would", "shall", "should", "may", "might", "must",
        "the", "a", "an", "this", "that", "these", "those",
        "and", "but", "or", "if", "because", "although", "unless",
        "about", "above", "across", "after", "against", "along",
        "please", "tell", "explain", "describe", "information"
    ]
    
    # Common Indonesian words
    indonesian_words = [
        "apa", "bagaimana", "kapan", "dimana", "siapa", "mengapa",
        "adalah", "merupakan", "menjadi", "ada", "sudah", "telah", "sedang",
        "akan", "bisa", "dapat", "boleh", "harus", "wajib", "perlu",
        "ini", "itu", "tersebut", "yang", "dan", "tetapi", "atau", "jika",
        "karena", "meskipun", "kecuali", "tentang", "di", "ke", "dari",
        "pada", "untuk", "dengan", "oleh", "dalam", "luar", "atas", "bawah",
        "tolong", "mohon", "jelaskan", "ceritakan", "informasi", "berikan"
    ]
    
    # Count occurrences of words from each language
    english_count = sum(1 for word in english_words if f" {word} " in f" {text_lower} ")
    indonesian_count = sum(1 for word in indonesian_words if f" {word} " in f" {text_lower} ")
    
    # Add weight for common Indonesian question patterns
    if any(pattern in text_lower for pattern in ["apa itu", "siapa itu", "bagaimana cara", "kapan waktu", "dimana tempat"]):
        indonesian_count += 2
    
    # Add weight for common English question patterns
    if any(pattern in text_lower for pattern in ["what is", "who is", "how to", "when is", "where is"]):
        english_count += 2
    
    # Return the detected language
    return "en" if english_count > indonesian_count else "id"

def format_response(answer, is_english=False):
    """
    Format the response to be more conversational and engaging
    """
    # Add greeting based on the content
    if "skripsi" in answer.lower() or "thesis" in answer.lower():
        greeting = "Salam Harmoni, saya akan menjelaskan tentang proses skripsi. " if not is_english else "Alright, let me explain about the thesis process. "
    else:
        greeting = "Salam Harmoni, " if not is_english else "Alright, "
    # Combine all parts
    formatted_answer = f"{greeting}{answer}"
    
    return formatted_answer

def generation(user_input, show_process=False):
    try:
        # First, check if it's a simple greeting
        if is_greeting(user_input.strip().lower()):
            # Return ONLY the greeting response, nothing else
            is_english = detect_language(user_input) == 'en'
            if is_english:
                return "Hi there! 👋 I'm here to help you with information about Undiksha's Information Systems Program. What would you like to know?"
            else:
                return "Halo! 👋 Saya di sini untuk membantu Anda dengan informasi seputar Program Studi Sistem Informasi Undiksha. Apa yang ingin Anda ketahui?"
        
        # If not a greeting, proceed with regular response generation
        global rag_chain
        
        # Check if the user is expressing gratitude
        if is_gratitude_expression(user_input):
            return get_gratitude_response()
        
        # Flag to indicate if this is a request for more details
        is_detail_request = is_asking_for_details(user_input)
        
        # Check if this is a lecturer question
        is_lecturer_query = is_lecturer_question(user_input)
        
        # Detect language
        language = detect_language(user_input)
        is_english = (language == 'en')
        
        # Ensure rag_chain is initialized
        try:
            if rag_chain is None:
                logger.info("Initializing RAG chain for the first time")
                rag_chain = initialize_rag_chain()
                if rag_chain is None:
                    raise ValueError("Failed to initialize rag_chain properly")
        except Exception as e:
            logger.error(f"Failed to initialize RAG chain: {e}")
            return "Maaf, sistem tidak dapat memproses pertanyaan Anda saat ini. Terjadi masalah dalam inisialisasi komponen pencarian." if not is_english else "Sorry, the system cannot process your question at this time. There was a problem initializing the search component."
        
        if show_process:
            st.subheader("2. Generation")
            with st.spinner("Sedang menyusun jawaban terbaik untuk Anda sabar dulu yaah..." if not is_english else "Generating the best answer for you, please wait..."):
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
            
            # For lecturer questions, modify the query to improve retrieval
            query_for_retrieval = user_input
            if is_lecturer_query:
                # Extract potential lecturer name from the query
                words = user_input.lower().split()
                for i, word in enumerate(words):
                    if word in ["pak", "bu", "bapak", "ibu", "dosen"] and i < len(words) - 1:
                        # Add "dosen" keyword if not already present to improve retrieval
                        if "dosen" not in user_input.lower():
                            query_for_retrieval = f"dosen {user_input}"
                        break
                
                # If asking about a coordinator or specific role, add those keywords
                if any(term in user_input.lower() for term in ["koordinator", "koorprodi", "kaprodi", "ketua"]):
                    if "dosen" not in query_for_retrieval.lower():
                        query_for_retrieval = f"dosen koordinator program studi {query_for_retrieval}"
            
            # Call the RAG chain with the formatted history and potentially modified query
            response = rag_chain.invoke({
                "question": query_for_retrieval,
                "chat_history": formatted_history
            })
            
            if isinstance(response, dict) and "answer" in response:
                answer = response["answer"]
                
                # Format the response to be more conversational
                answer = format_response(answer, is_english=is_english)
                
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
                
                # Check if the answer is too short (less than 50 words)
                is_short_answer = len(answer.split()) < 50
                
                # Check if it's a general information question about the program
                is_general_info_question = any(phrase in user_input.lower() for phrase in [
                    "apa itu", "what is", "tell me about", "ceritakan tentang", 
                    "jelaskan tentang", "explain about", "informasi tentang",
                    "information about", "program studi", "program study"
                ])
                
                # If it's a simple "don't know" response, replace it with our formatted response
                if any(phrase in answer.lower() for phrase in dont_know_phrases) and is_short_answer:
                    # Special handling for lecturer questions
                    if is_lecturer_query:
                        # Try to get more specific information for lecturer questions
                        enhanced_prompt = f"""
                        The user asked about a lecturer: "{user_input}"
                        
                        You initially responded with: "{answer}"
                        
                        This response indicates you don't have the information, but you should have information about lecturers in the Information Systems Program at Undiksha.
                        
                        Please check if the question is about any of these specific lecturers:
                        
                        If the question is about one of these lecturers, provide their complete information:
                        - Full name
                        - NIP/NIDN
                        - Position
                        - Concentration
                        - Research topics (if available)
                        
                        The user's language is {"English" if is_english else "Indonesian"}.
                        Respond in {"English" if is_english else "Indonesian"}.
                        """
                
                # If it's a general information question but the answer is too short, enhance it
                elif is_general_info_question and is_short_answer:
                    # Get more context for this type of question
                    embedded_data, _ = chunking_and_retrieval(user_input, show_process=False)
                    
                    # Extract the top chunks
                    additional_context = ""
                    for chunk, _ in embedded_data[:3]:  # Use top 3 chunks
                        additional_context += chunk + " "
                    
                    # Enhance the answer with more information
                    enhanced_prompt = f"""
                    The user asked: "{user_input}"
                    
                    You initially responded with: "{answer}"
                    
                    This response is too brief. Please enhance it with the following additional information:
                    {additional_context}
                    
                    Create a comprehensive response that:
                    1. Introduces the Program Studi Sistem Informasi Undiksha
                    2. Describes its curriculum and focus areas
                    3. Mentions any specializations or concentrations
                    4. Ends with an offer to provide more specific information
                    
                    The user's language is {"English" if is_english else "Indonesian"}.
                    Respond in {"English" if is_english else "Indonesian"}.
                    """
                    
                    try:
                        # Use the LLM to enhance the answer
                        llm = ChatOpenAI(
                            model_name="gpt-3.5-turbo",
                            openai_api_key=openai_api_key
                        )
                        enhanced_answer = llm.invoke(enhanced_prompt)
                        answer = enhanced_answer.content
                    except Exception as e:
                        logger.error(f"Error enhancing answer: {e}")
                        # If enhancement fails, add a generic enhancement
                        if not is_english:
                            answer += "\n\nProgram Studi Sistem Informasi di Undiksha menawarkan kurikulum yang komprehensif yang mencakup mata kuliah dasar dan lanjutan dalam bidang sistem informasi. Program ini memiliki beberapa konsentrasi seperti Manajemen Sistem Informasi, Rekayasa dan Kecerdasan Bisnis, serta Keamanan Siber. Jika Anda memerlukan informasi lebih spesifik tentang aspek tertentu dari program studi ini, silakan tanyakan."
                        else:
                            answer += "\n\nThe Information Systems Program at Undiksha offers a comprehensive curriculum covering both foundational and advanced courses in information systems. The program has several concentrations including Information Systems Management, Business Intelligence and Engineering, and Cyber Security. If you need more specific information about certain aspects of this program, please ask."
                
                # Check if this is a procedure question but the answer is too short
                if is_procedure_question(user_input) and len(answer.split()) < 50 and not is_detail_request:
                    # If the answer is too short for a procedure question, ask for more details
                    logger.warning(f"Answer too short for procedure question: {answer}")
                    if not is_english:
                        answer += "\n\nApakah Anda ingin saya menjelaskan lebih detail tentang prosedur ini? Silakan beri tahu saya bagian mana yang ingin Anda ketahui lebih lanjut."
                    else:
                        answer += "\n\nWould you like me to explain this procedure in more detail? Please let me know which part you would like to know more about."
                
                # If this is a request for more details, make sure the answer is comprehensive
                if is_detail_request and len(answer.split()) < 100:
                    logger.warning(f"Answer too short for detail request: {answer}")
                    if not is_english:
                        answer += "\n\nSaya harap penjelasan ini membantu. Jika Anda memerlukan informasi lebih spesifik, silakan tanyakan bagian tertentu yang ingin Anda ketahui lebih dalam."
                    else:
                        answer += "\n\nI hope this explanation helps. If you need more specific information, please ask about the particular aspect you'd like to know more about."
            else:
                logger.warning(f"Unexpected response format: {response}")
                answer = "Maaf, saya tidak dapat memproses pertanyaan Anda saat ini." if not is_english else "Sorry, I cannot process your question at this time."
                
        except Exception as e:
            logger.error(f"Error generating response: {e}", exc_info=True)
            # Provide more informative error message
            if "name 'rag_chain' is not defined" in str(e):
                answer = "Maaf, terjadi kesalahan dalam komponen pencarian. Tim teknis kami sedang menyelesaikan masalah ini." if not is_english else "Sorry, there was an error in the search component. Our technical team is working to resolve this issue."
            else:
                answer = "Maaf, terjadi kesalahan dalam memproses pertanyaan Anda. Silakan coba kembali beberapa saat lagi atau coba pertanyaan lain. Jika masalah berlanjut, silakan hubungi administrator sistem." if not is_english else "Sorry, there was an error processing your question. Please try again later or try a different question. If the problem persists, please contact the system administrator."
        
        if show_process:
            st.success("Jawaban siap! 🚀" if not is_english else "Answer ready! 🚀")
            
        return answer
    except Exception as e:
        # Handle any exceptions that might occur during processing
        logger.error(f"Unhandled error in generation function: {e}", exc_info=True)
        is_english = detect_language(user_input) == 'en'
        return "I'm sorry, I encountered an error while processing your request. Please try again." if is_english else "Maaf, terjadi kesalahan saat memproses permintaan Anda. Silakan coba lagi."

def check_and_refresh_rss_feed():
    """
    Checks if the RSS feed needs to be refreshed based on the last update time.
    If it's been more than 1 hour since the last update, refreshes the feed.
    """
    # Check if we need to initialize the RSS feed
    if "latest_rss_posts" not in st.session_state:
        st.session_state.latest_rss_posts = get_latest_posts(formatted=True, embed_posts=False)
        st.session_state.rss_last_updated = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        logger.info("RSS posts fetched and stored in session state")
        return
        
    # Check if we need to refresh the RSS feed (if it's been more than 1 hour)
    if "rss_last_updated" in st.session_state:
        try:
            last_updated = datetime.datetime.strptime(st.session_state.rss_last_updated, "%Y-%m-%d %H:%M:%S")
            now = datetime.datetime.now()
            time_diff = now - last_updated
            
            # If it's been more than 1 hour, refresh the feed
            if time_diff.total_seconds() > 3600:  # 1 hour in seconds
                logger.info("RSS feed is stale, refreshing...")
                st.session_state.latest_rss_posts = get_latest_posts(formatted=True, embed_posts=False)
                st.session_state.rss_last_updated = now.strftime("%Y-%m-%d %H:%M:%S")
                logger.info("RSS posts refreshed automatically")
        except Exception as e:
            logger.error(f"Error checking RSS feed refresh time: {e}")

def check_auth(username, password):
    """Check if username and password match the hardcoded values"""
    return username == "adminsi" and password == "adminsi"

def main():
    st.set_page_config(
        page_title="Virtual Assistant SI Undiksha",
        page_icon="https://encrypted-tbn0.gstatic.com/images?q=tbn:ANd9GcSxGdjI58B_NuUd8eAWfRBlVms7f-2e2oI_SA&s",
        layout="wide",
        initial_sidebar_state="expanded"
    )
    
    # Theme-aware CSS that respects light/dark mode
    st.markdown("""
        <style>
        /* Preserve default Streamlit theme background values based on current theme */
        .main .block-container {
            padding-top: 2rem;
            padding-bottom: 2rem;
        }
        
        /* Custom styling for chat elements that respects theme */
        .chat-message {
            padding: 1rem;
            border-radius: 0.5rem;
            margin-bottom: 1rem;
            display: flex;
        }
        
        /* Custom styling for user input area that works in both themes */
        .stTextInput > div > div > input {
            border-radius: 20px;
        }
        </style>
    """, unsafe_allow_html=True)
    
    # Initialize session states
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = []
    if "authenticated" not in st.session_state:
        st.session_state.authenticated = False
    
    # Initialize other session states...
    if "dev_mode_embedded_data" not in st.session_state:
        st.session_state.dev_mode_embedded_data = None
    
    if "dev_mode_query_embedding" not in st.session_state:
        st.session_state.dev_mode_query_embedding = None
    
    if "dev_mode_query" not in st.session_state:
        st.session_state.dev_mode_query = None
    
    if "dev_mode_csv_path" not in st.session_state:
        st.session_state.dev_mode_csv_path = None
    
    if "dev_mode_latest_answer" not in st.session_state:
        st.session_state.dev_mode_latest_answer = None
    
    if "processing_new_question" not in st.session_state:
        st.session_state.processing_new_question = False
    
    # Check and refresh RSS feed if needed
    check_and_refresh_rss_feed()
    
    # Initialize RSS last updated timestamp if not exists
    if "rss_last_updated" not in st.session_state:
        st.session_state.rss_last_updated = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    
    # Sidebar for mode selection and settings
    with st.sidebar:
        st.image("https://encrypted-tbn0.gstatic.com/images?q=tbn:ANd9GcSxGdjI58B_NuUd8eAWfRBlVms7f-2e2oI_SA&s", width=100)
        st.title("Virtual Assistant SI Undiksha")
        
        # Mode selection
        mode = st.radio("Mode", ["User Mode", "Developer Mode"])
        
        # Developer mode authentication
        if mode == "Developer Mode":
            if not st.session_state.authenticated:
                with st.form("auth_form"):
                    st.subheader("Developer Authentication")
                    username = st.text_input("Username")
                    password = st.text_input("Password", type="password")
                    submit = st.form_submit_button("Login")
                    
                    if submit:
                        if check_auth(username, password):
                            st.session_state.authenticated = True
                            st.rerun()
                        else:
                            st.error("Invalid credentials")
                
                # If not authenticated, show user mode
                mode = "User Mode"
            
            # Only show developer options if authenticated
            if st.session_state.authenticated:
                export_to_csv = st.checkbox("Export retrieval data to CSV", value=False)
                if st.button("Logout"):
                    st.session_state.authenticated = False
                    st.rerun()
        
        st.text("Universitas Pendidikan Ganesha")
        st.markdown("---")
        with st.expander("Lihat Info Terkini"):
            # Add a refresh button for RSS posts
            col1, col2 = st.columns([4, 1])
            with col2:
                if st.button("🔄", help="Refresh informasi terkini"):
                    st.session_state.latest_rss_posts = get_latest_posts(formatted=True, embed_posts=False)
                    st.session_state.rss_last_updated = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                    st.rerun()
            
            # Display last updated timestamp
            st.caption(f"Terakhir diperbarui: {st.session_state.rss_last_updated}")
            
            # Use cached RSS posts from session state
            latest_posts = st.session_state.latest_rss_posts
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
            # Get the relevant documents using the proper search_kwargs
            try:
                relevant_docs = retriever.get_relevant_documents(
                    st.session_state.dev_mode_query, 
                    search_kwargs={"k": 20}
                )
                total_docs = len(relevant_docs)
            except Exception as e:
                logger.error(f"Error retrieving relevant documents: {e}", exc_info=True)
                st.warning(f"Could not retrieve relevant documents: {e}")
                total_docs = len(st.session_state.dev_mode_embedded_data)
            
            display_embedding_process(
                st.session_state.dev_mode_embedded_data,
                st.session_state.dev_mode_query,
                st.session_state.dev_mode_query_embedding,
                total_docs
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
        
        # Display the user message immediately
        with chat_container:
            with st.chat_message("user"):
                st.markdown(user_input)
            
            # Show a temporary loading message while processing
            with st.chat_message("assistant"):
                message_placeholder = st.empty()
                message_placeholder.markdown("⏳ Sedang memproses...")
        
        # Set flag that we're processing a new question
        st.session_state.processing_new_question = True
        
        # Reset previous embedding data to avoid display issues with old data
        if show_process:
            st.session_state.dev_mode_embedded_data = None
            st.session_state.dev_mode_query_embedding = None
            st.session_state.dev_mode_query = None
            st.session_state.dev_mode_csv_path = None
            
        # Process the query
        if show_process:
            try:
                result = chunking_and_retrieval(user_input, show_process, export_to_csv)
                
                # Handle different return types
                if export_to_csv and len(result) == 3:
                    embedded_data, query_embedding, csv_path = result
                    st.session_state.dev_mode_csv_path = csv_path
                else:
                    embedded_data, query_embedding = result
                
                # Only store in session state if valid data is returned
                if embedded_data and query_embedding is not None:
                    st.session_state.dev_mode_embedded_data = embedded_data
                    st.session_state.dev_mode_query_embedding = query_embedding
                    st.session_state.dev_mode_query = user_input
                    logger.info(f"Successfully stored {len(embedded_data)} embedded chunks for query: {user_input[:50]}...")
                else:
                    logger.warning("No valid embedding data returned from chunking_and_retrieval")
            except Exception as e:
                logger.error(f"Error processing embeddings: {e}", exc_info=True)
                st.error(f"An error occurred during retrieval: {e}")
        else:
            # In user mode, we don't need to store the results
            chunking_and_retrieval(user_input, show_process)
        
        # Generate response
        answer = generation(user_input, show_process)
        
        # Update the placeholder with the actual response
        message_placeholder.markdown(answer)
        
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
