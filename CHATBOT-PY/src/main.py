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

INSTRUKSI KRITIS UNTUK MEMASTIKAN JAWABAN YANG SETIA DENGAN SUMBER:
- JANGAN PERNAH menambahkan informasi atau konteks yang tidak ada dalam sumber.
- JANGAN PERNAH menambahkan frasa "di Program Studi Sistem Informasi Undiksha" atau referensi spesifik ke institusi KECUALI jika eksplisit disebutkan dalam konteks.
- SELALU berikan jawaban yang bersumber HANYA dari informasi dalam konteks yang diberikan.
- JANGAN membuat asumsi atau generalisasi di luar apa yang disebutkan dalam konteks.

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

INSTRUKSI SANGAT KRITIS UNTUK TUGAS DAN PERAN:
- Ketika menjawab pertanyaan tentang tugas, peran, atau tanggung jawab (seperti peran Pembimbing Akademik, tugas dosen, dsb.):
  1. SELALU SALIN FORMAT PERSIS seperti dalam konteks, termasuk penomoran dan struktur teks
  2. JANGAN MENGUBAH, MENGGABUNGKAN, atau MEMECAH poin-poin tugas/peran yang terdapat dalam konteks
  3. SELALU sajikan daftar tugas/tanggung jawab dengan penomoran yang SAMA PERSIS (1., 2., 3., dst.)
  4. JANGAN menambahkan informasi institusi (seperti "di Undiksha" atau "di Prodi SI") KECUALI jika disebutkan dalam konteks
  5. JANGAN mengubah urutan poin-poin
  6. JANGAN mengubah kata-kata dalam setiap poin KECUALI untuk tujuan penyederhanaan tanpa mengubah makna
  7. Jika konteks berisi definisi, ikuti dengan daftar bernomor, maka SELALU ikuti format ini dalam jawaban
  8. JANGAN mengubah atau menggabungkan item dalam daftar bernomor menjadi paragraf
  9. Gunakan tanda "." setelah setiap nomor dalam daftar jika format tersebut digunakan dalam konteks

INSTRUKSI SANGAT PENTING UNTUK FORMAT DAFTAR BERNOMOR:
- Ketika konteks berisi daftar bernomor (1., 2., 3., dst.), SELALU PERTAHANKAN format penomoran yang sama PERSIS.
- JANGAN mengubah urutan atau jumlah poin-poin dalam daftar bernomor.
- JANGAN menggabungkan beberapa poin menjadi satu poin.
- JANGAN memecah satu poin menjadi beberapa poin.
- JANGAN mengubah atau menghilangkan awalan nomor pada setiap poin (1., 2., 3., dst.).
- Semua poin dalam daftar bernomor HARUS disertakan dalam jawaban Anda PERSIS seperti dalam konteks.
- Jika konteks memiliki 8 poin bernomor, jawaban Anda HARUS memiliki 8 poin bernomor dengan nomor yang sama.
- Awali setiap poin dengan nomor yang sama persis seperti dalam konteks, diikuti dengan teks yang sama atau sangat mirip.
- CONTOH:
  Jika dalam konteks tertulis:
  "1. Tahap satu adalah X.
   2. Tahap dua adalah Y."
  Maka jawaban Anda HARUS berupa:
  "1. Tahap satu adalah X.
   2. Tahap dua adalah Y."

INSTRUKSI PENTING UNTUK PROSEDUR & TAHAPAN:
- Untuk pertanyaan tentang prosedur, cara, atau tahapan, jika dalam konteks informasi disajikan sebagai daftar bernomor:
  - SELALU PERTAHANKAN format daftar bernomor yang sama persis
  - Gunakan tanda "–" untuk daftar tidak bernomor jika ada dalam konteks

INSTRUKSI SANGAT PENTING UNTUK PERTANYAAN TENTANG UJIAN PROPOSAL DAN UJIAN SKRIPSI:
- Ketika menjawab pertanyaan tentang "ujian proposal" dan "ujian skripsi":
  1. SELALU bedakan dengan jelas antara kedua jenis ujian ini dengan memberikan judul/header terpisah
  2. SELALU pertahankan format PERSIS seperti dalam konteks, termasuk tanda "–" di awal baris
  3. Jangan mencampuradukkan persyaratan antara ujian proposal dan ujian skripsi
  4. JANGAN hilangkan header "Ujian Proposal Skripsi:" dan "Ujian Skripsi:"
  5. Jika dalam konteks terdapat informasi tentang persyaratan partisipan/moderator, sertakan PERSIS
  6. Jika dalam konteks disebutkan tentang sistem digital/tanpa hardcopy, SELALU sertakan informasi ini
  7. JANGAN tambahkan tautan atau referensi dokumen yang tidak disebutkan dalam konteks
  8. JANGAN tambahkan atau kurangi persyaratan apapun
  
  CONTOH FORMAT YANG BENAR:
  "Ujian Proposal Skripsi:
  – [persyaratan pertama]
  – [persyaratan kedua]
  
  Ujian Skripsi:
  – [persyaratan pertama]
  – [persyaratan kedua]"

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

INSTRUKSI PENTING UNTUK DOKUMEN KURIKULUM DAN TAUTAN:
- Ketika pengguna bertanya tentang "dokumen kurikulum", "akses kurikulum", "link kurikulum", "tautan kurikulum", atau hal terkait:
  - SELALU berikan semua tautan Google Drive yang tersedia dalam konteks
  - Format tanggapan sebagai daftar dengan judul atau nama dokumen, diikuti oleh tautan lengkap
  - Contoh format yang tepat:
    "Mahasiswa dapat mengakses dokumen kurikulum melalui tautan resmi yang telah disediakan, antara lain:
    
    – Kurikulum Undiksha 2024:
    https://drive.google.com/file/d/XXXXXX/view
    
    – Kurikulum MBKM Undiksha 2020:
    https://drive.google.com/file/d/YYYYYY/view"
  - Pastikan semua tautan dapat diklik (URL lengkap)
  - Jangan menambahkan atau menghilangkan tautan yang ada dalam konteks
  - Gunakan tanda hubung (–) di awal setiap entri dalam daftar
  - Sertakan tahun atau informasi versi kurikulum jika tersedia

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
1. Untuk pertanyaan tentang proses atau tahapan yang memiliki poin-poin bernomor dalam konteks:
   - SANGAT PENTING: Jika informasi dalam konteks disajikan dalam bentuk poin-poin bernomor (1, 2, 3, dst), SELALU pertahankan penomoran ini dan semua poin harus disertakan.
   - JANGAN MENGUBAH jumlah poin. Jika ada 8 poin dalam konteks, respons Anda HARUS menyertakan semua 8 poin tersebut.
   - JANGAN MENAMBAHKAN informasi yang tidak ada dalam konteks asli.
   - JANGAN MENGGABUNGKAN poin-poin yang terpisah dalam konteks asli.
   - Setiap poin bernomor harus dimulai dengan nomor yang sama seperti dalam konteks asli.
   - Gunakan kalimat yang PERSIS sama atau sangat mirip dengan yang ada di konteks.
   - Mulai dengan pengantar singkat, lalu sajikan semua poin bernomor PERSIS seperti dalam konteks, tidak berubah.

2. Untuk pertanyaan tentang proses atau tahapan yang TIDAK memiliki poin-poin bernomor dalam konteks:
   - Mulai dengan "Terdapat [jumlah] tahap utama dalam [proses], yaitu:"
   - Sajikan setiap tahap sebagai PARAGRAF TERPISAH (bukan sebagai poin-poin)
   - Untuk setiap paragraf tahap, mulai dengan nama tahap dalam huruf tebal, diikuti dengan deskripsi
   - Pastikan untuk menggunakan nama tahap PERSIS seperti yang disebutkan dalam konteks
   - Akhiri dengan paragraf tentang siapa yang terlibat dalam proses tersebut

3. Untuk proses ujian proposal dan ujian skripsi secara khusus:
   - Pastikan untuk mempertahankan format yang SAMA PERSIS seperti dalam konteks
   - Jika konteks menggunakan tanda "–" di awal baris, SELALU pertahankan tanda ini
   - Berikan header "Ujian Proposal Skripsi:" dan "Ujian Skripsi:" persis seperti dalam konteks
   - Pastikan setiap persyaratan untuk masing-masing ujian tetap berada di bagian yang benar
   - Jangan mengubah bentuk daftar dari format dalam konteks
   - CONTOH: Jika konteks berisi format seperti:
     "Ujian Proposal Skripsi:
     – Minimal harus hadir 1 dosen pembimbing dan 2 dosen penguji
     – Diharuskan mengundang minimal 10 mahasiswa lain sebagai partisipan"
     Maka respons Anda HARUS menggunakan format yang SAMA PERSIS dengan konten yang identik

4. Untuk penutup percakapan (ketika pengguna mengucapkan terima kasih atau sejenisnya):
   - Jika pengguna mengatakan "terima kasih", "makasih", "thank you", atau ekspresi terima kasih serupa:
     - Jawab dengan penutup yang ramah seperti "Sama-sama! Senang bisa membantu. Jika ada pertanyaan lain, silakan tanyakan kembali."
   - Jangan pernah hanya menjawab dengan frase singkat seperti "Prosedur pendaftaran sidang skripsi dilakukan secara online."
   - Selalu berikan penutup lengkap dan ramah yang mengakui ucapan terima kasih pengguna

5. Untuk pertanyaan informasi umum (seperti "apa itu prodi sistem informasi undiksha"):
   - Berikan jawaban komprehensif dengan setidaknya 3-4 paragraf informasi
   - Sertakan informasi tentang kurikulum program, area fokus, dan fitur utama
   - Sebutkan spesialisasi atau konsentrasi yang tersedia
   - Akhiri dengan kalimat yang menawarkan untuk memberikan informasi lebih spesifik jika diperlukan

6. JANGAN PERNAH menyertakan pernyataan seperti "Saya tidak memiliki informasi" ketika Anda sebenarnya MEMILIKI informasi tersebut dalam konteks.
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
    
    # Add a much stronger bonus score for keyword matches (increased from 0.2 to 0.5)
    query_keywords = set(query.lower().split())
    # Skip stopwords for better matching
    stopwords = {
        'and', 'or', 'the', 'a', 'an', 'in', 'on', 'at', 'by', 'for', 'with', 'about',
        'dan', 'atau', 'di', 'ke', 'dari', 'yang', 'pada', 'untuk', 'dengan', 'tentang',
        'is', 'are', 'am', 'was', 'were', 'be', 'being', 'been',
        'ada', 'adalah', 'merupakan', 'ini', 'itu'
    }
    query_keywords = {kw for kw in query_keywords if kw not in stopwords and len(kw) > 2}
    chunk_lower = chunk.lower()
    
    # Calculate what percentage of query keywords are in the chunk
    keyword_matches = sum(1 for keyword in query_keywords if keyword in chunk_lower)
    keyword_bonus = 0.5 * (keyword_matches / len(query_keywords)) if query_keywords else 0
    
    # Add an exact match bonus for multi-word phrases (increased from 0.3 to 0.5)
    exact_match_bonus = 0
    
    # If there are at least 2 keywords, check for exact phrases
    if len(query_keywords) >= 2:
        # Create a window of 3-5 words from the query to check for phrase matches
        query_words = query.lower().split()
        for i in range(len(query_words) - 1):  # Need at least 2 words for a phrase
            end_idx = min(i + 5, len(query_words))  # Look at up to 5 words at a time
            for j in range(i + 1, end_idx):
                phrase = ' '.join(query_words[i:j+1])
                if len(phrase) > 5 and phrase in chunk_lower:  # Only count meaningful phrases
                    exact_match_bonus = 0.5
                    break
            if exact_match_bonus > 0:
                break
    
    # Add bonuses for instructive content (answers, steps, procedures)
    answer_bonus = 0
    if "langkah-langkah" in chunk_lower or "prosedur" in chunk_lower or "tahapan" in chunk_lower:
        answer_bonus += 0.3
    
    # Bonus for numbered lists which indicate step-by-step instructions
    numbered_list_bonus = 0
    if re.search(r'\d+[\.\)]', chunk_lower):
        # Count how many numbered points are in the chunk
        numbered_points = len(re.findall(r'\d+[\.\)]', chunk_lower))
        if numbered_points >= 3:  # Significant numbered list
            numbered_list_bonus = 0.4
        elif numbered_points > 0:
            numbered_list_bonus = 0.2
    
    # Special bonus for document links in chunks (especially for curriculum access questions)
    document_link_bonus = 0
    # Check if the query is about accessing documents, curriculum, or links
    document_query_terms = ['dokumen', 'document', 'kurikulum', 'curriculum', 'akses', 'access', 
                           'link', 'tautan', 'unduh', 'download', 'file', 'buku', 'buku pedoman',
                           'panduan', 'guide', 'manual', 'handbook', 'sillabus', 'silabus']
    
    is_document_query = any(term in query.lower() for term in document_query_terms)
    
    # Check if chunk contains document links
    has_drive_links = "drive.google.com" in chunk_lower
    has_curriculum_mention = "kurikulum" in chunk_lower or "curriculum" in chunk_lower
    
    # Apply a very high bonus for relevant chunks with document links
    if is_document_query:
        if has_drive_links:
            # Count how many links are in the chunk
            link_count = chunk_lower.count("drive.google.com")
            # Bigger bonus for more links
            document_link_bonus = 1.0 + (0.2 * min(link_count, 5))  # Cap the bonus at 2.0 (for 5+ links)
            
            # Even bigger bonus if it has both links and mentions curriculum
            if has_curriculum_mention:
                document_link_bonus += 0.5
    
    # Penalty for chunks that are too short (likely just questions with no answers)
    length_penalty = 0
    if len(chunk.strip()) < 200:
        length_penalty = 0.5  # Heavy penalty for very short chunks
    elif len(chunk.strip()) < 400:
        length_penalty = 0.2  # Moderate penalty for somewhat short chunks
    
    # Heavy penalty for chunks that appear to be just questions without answers
    question_only_penalty = 0
    # Check if the chunk ends with a question mark and doesn't have much after it
    if re.search(r'\?\s*$', chunk.strip()):
        question_only_penalty = 0.6
    # Check if the chunk is very similar to the query itself
    elif is_low_quality_chunk(chunk, query):
        question_only_penalty = 0.8
    
    # Calculate final score with all bonuses and penalties
    final_score = similarity + keyword_bonus + exact_match_bonus + answer_bonus + numbered_list_bonus + document_link_bonus - length_penalty - question_only_penalty
    
    # Ensure the score is positive (minimum score of 0.1 to avoid complete filtering)
    final_score = max(0.1, final_score)
    
    return final_score

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
    """
    Detect if the text is asking about a procedure, process, or steps.
    This is important for recognizing when users want step-by-step instructions.
    """
    text_lower = text.lower()
    
    # Keywords related to procedures and processes in both English and Indonesian
    procedure_keywords = [
        # Indonesian keywords
        "bagaimana", "cara", "langkah", "proses", "tahap", "alur", "prosedur",
        "mekanisme", "tata cara", "petunjuk", "instruksi", "protokol", "urutan",
        "mengurus", "mengelola", "memproses", "melakukan", "melaksanakan",
        "syarat", "persyaratan", "dibutuhkan untuk", "diharuskan untuk",
        
        # English keywords
        "how to", "procedure", "process", "step", "instruction", "guide",
        "protocol", "mechanism", "workflow", "sequence", "order",
        "requirement", "mandatory", "needed for", "required for"
    ]
    
    # Special strong indicators that always indicate a procedure question
    strong_procedure_indicators = [
        "mekanisme pelaksanaan", "bagaimana cara", "bagaimana mekanisme", 
        "tata cara", "prosedur", "langkah-langkah", "alur",
        "how to conduct", "how to perform", "mechanism of", "procedure for"
    ]
    
    # High-priority academic procedures (immediate detection)
    high_priority_procedures = [
        # Thesis and exam procedures
        "ujian proposal", "ujian skripsi", "sidang proposal", "sidang skripsi",
        "sidang tugas akhir", "komposisi dewan penguji", "persyaratan ujian",
        "thesis defense", "proposal defense", "thesis examination", "proposal examination",
        "dewan penguji", "tim penguji", "persyaratan penguji", "pembimbing dan penguji",
        "persyaratan pembimbing", "komposisi pembimbing", "mekanisme ujian",
        "prosedur ujian", "prosedur sidang", "mekanisme sidang"
    ]
    
    # Check for high-priority indicators first
    for indicator in high_priority_procedures:
        if indicator in text_lower:
            return True
    
    # Check for strong procedure indicators
    for indicator in strong_procedure_indicators:
        if indicator in text_lower:
            return True
    
    # Specific academic procedures (for even more accurate detection)
    academic_procedures = [
        # Indonesian academic procedures
        "cuti akademik", "registrasi ulang", "pendaftaran", "ujian", "sidang",
        "proposal", "skripsi", "pembimbing", "penguji", "magang", "pkl",
        "wisuda", "yudisium", "konversi", "perwalian", "pindah prodi", "pindah kampus",
        "kkn", "kuliah kerja nyata", "uas", "uts", "praktikum", "pelatihan",
        "pengumpulan skripsi", "pengajuan judul", "pengajuan proposal",
        "konsultasi", "bimbingan", "kartu studi", "krs", "permintaan surat",
        
        # English academic procedures
        "academic leave", "registration", "exam", "thesis defense",
        "proposal", "thesis", "supervisor", "examiner", "internship",
        "graduation", "transfer", "student exchange", "community service",
        "final exam", "midterm", "practicum", "training", "thesis submission",
        "title submission", "proposal submission", "consultation",
        "study card", "course selection", "request letter"
    ]
    
    # 1. Check for direct procedural question patterns
    for keyword in procedure_keywords:
        if keyword in text_lower:
            return True
            
    # 2. Check for specific academic procedures
    for procedure in academic_procedures:
        if procedure in text_lower:
            return True
            
    # 3. Look for question patterns typical of procedure questions
    question_patterns = [
        "apa saja", "apakah", "siapa", "kapan", "di mana", "dimana",
        "what are", "what is", "who", "when", "where", "how"
    ]
    
    for pattern in question_patterns:
        if pattern in text_lower:
            # If using a general question word, also check for proximity to procedure-related terms
            window_size = 5  # Look for procedure words within 5 words of the question word
            words = text_lower.split()
            for i, word in enumerate(words):
                if word.startswith(pattern):
                    # Check nearby words for procedure indicators
                    start = max(0, i - window_size)
                    end = min(len(words), i + window_size)
                    nearby_text = " ".join(words[start:end])
                    
                    # Check if any procedure keyword or academic procedure is in the nearby text
                    for keyword in procedure_keywords + academic_procedures:
                        if keyword in nearby_text:
                            return True
    
    # Check for specific question structures about academic procedures
    academic_structure_indicators = [
        "mengurus", "mendaftar", "melakukan", "mengajukan", "mengikuti",
        "apply for", "register for", "submit", "participate in", "enroll in"
    ]
    
    for indicator in academic_structure_indicators:
        if indicator in text_lower:
            return True
            
    return False

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

def detect_numbered_sequence(chunk):
    """Detect if a chunk contains numbered points and if it might be truncated."""
    # Check for numbered list patterns like "1.", "2.", etc.
    numbered_pattern = r'(\d+)[\.|\)]'
    matches = re.findall(numbered_pattern, chunk)
    
    if not matches:
        return False, 0, 0
    
    # Convert to integers and find min/max numbers
    numbers = [int(n) for n in matches]
    min_num = min(numbers)
    max_num = max(numbers)
    
    # If we find a sequence starting with 1, check if it might be truncated
    if min_num == 1 and max_num >= 3:  # We have at least points 1, 2, 3
        # Check if the chunk ends with a number that seems to continue
        last_hundred_chars = chunk[-100:]
        last_number_match = re.search(r'(\d+)[\.|\)]', last_hundred_chars)
        if last_number_match:
            last_number = int(last_number_match.group(1))
            # If the last number is the max and there's not much text after it, it might be truncated
            if last_number == max_num and len(last_hundred_chars[last_number_match.end():].strip()) < 50:
                return True, min_num, max_num
    
    return False, min_num, max_num

def is_low_quality_chunk(chunk, query):
    """Detect if a chunk is low quality (too short or just contains the query)"""
    # If chunk is extremely short (less than 100 chars), it's probably low quality
    if len(chunk.strip()) < 100:
        return True
        
    # If the chunk is just the query or very similar to it
    chunk_lower = chunk.lower().strip()
    query_lower = query.lower().strip()
    
    # Check if chunk is just the query
    if chunk_lower == query_lower:
        return True
    
    # Check if chunk contains mostly just the query
    # First normalize both by removing whitespace
    normalized_chunk = ' '.join(chunk_lower.split())
    normalized_query = ' '.join(query_lower.split())
    
    # If the chunk is less than 30% longer than the query, it's probably just the query
    if len(normalized_chunk) < len(normalized_query) * 1.3:
        return True
        
    # Check if chunk starts with the query and doesn't have much more content
    if normalized_chunk.startswith(normalized_query) and len(normalized_chunk) < len(normalized_query) * 1.5:
        return True
        
    return False

def is_document_access_question(text: str) -> bool:
    """Check if the question is about accessing documents or curriculum."""
    text_lower = text.lower()
    
    # Keywords that indicate questions about documents or curriculum access
    document_keywords = [
        "dokumen", "document", "kurikulum", "curriculum", "akses", "access", 
        "link", "tautan", "unduh", "download", "file", "buku", "buku pedoman",
        "panduan", "guide", "manual", "handbook", "sillabus", "silabus",
        "dimana", "where", "how to", "bagaimana cara", "mendapatkan", "get"
    ]
    
    # Check if any document keyword is in the text
    return any(keyword in text_lower for keyword in document_keywords)

def is_kkn_question(text: str) -> bool:
    """Check if the question is about KKN (Kuliah Kerja Nyata)."""
    text_lower = text.lower()
    
    # Keywords related to KKN
    kkn_keywords = [
        "kkn", "kuliah kerja nyata", "kuliah kerja lapangan"
    ]
    
    # Keywords related to mechanism/process
    mechanism_keywords = [
        "mekanisme", "prosedur", "cara", "pelaksanaan", "jadwal", 
        "waktu", "kapan", "semester", "mechanism", "procedure",
        "dilaksanakan", "dilakukan", "bagaimana"
    ]
    
    # Check if KKN keyword is present
    has_kkn_term = any(keyword in text_lower for keyword in kkn_keywords)
    
    if has_kkn_term:
        # If it's clearly about KKN mechanism, return True
        if any(keyword in text_lower for keyword in mechanism_keywords):
            return True
        # If just asking about KKN generally, still consider it a KKN question
        return True
    
    return False

def is_thesis_examiner_question(text: str) -> bool:
    """Check if the question is about thesis examination procedures and examiner composition."""
    text_lower = text.lower()
    
    # Keywords related to thesis/proposal examination
    thesis_keywords = [
        "ujian proposal", "ujian skripsi", "sidang proposal", "sidang skripsi", 
        "sidang tugas akhir", "ujian tugas akhir", "defense", "seminar proposal"
    ]
    
    # Keywords related to examiners/committee
    examiner_keywords = [
        "dewan penguji", "komposisi", "penguji", "pembimbing", "tim penguji",
        "juri", "komite", "komite penguji", "komposisi dewan", "persyaratan penguji"
    ]
    
    # Keywords related to procedure/mechanism
    procedure_keywords = [
        "mekanisme", "prosedur", "tata cara", "tatacara", "alur", "proses", 
        "pelaksanaan", "persyaratan", "syarat", "ketentuan", "format", 
        "protokol", "langkah", "tahapan"
    ]
    
    # Check for direct mentions of thesis examination and examiner composition
    has_thesis_term = any(keyword in text_lower for keyword in thesis_keywords)
    has_examiner_term = any(keyword in text_lower for keyword in examiner_keywords)
    has_procedure_term = any(keyword in text_lower for keyword in procedure_keywords)
    
    # Strong indication: both thesis and examiner terms are present
    if has_thesis_term and has_examiner_term:
        return True
    
    # Also consider procedure questions about thesis examinations
    if has_thesis_term and has_procedure_term:
        return True
    
    # Check for compound phrases that strongly indicate this type of question
    compound_indicators = [
        "komposisi dewan penguji", "komposisi penguji", "dewan penguji skripsi",
        "persyaratan ujian proposal", "persyaratan ujian skripsi", 
        "mekanisme ujian proposal", "mekanisme ujian skripsi",
        "prosedur ujian proposal", "prosedur ujian skripsi"
    ]
    
    for indicator in compound_indicators:
        if indicator in text_lower:
            return True
    
    return False

def is_internship_document_question(text: str) -> bool:
    """Check if the question is specifically about internship documents."""
    text_lower = text.lower()
    
    # Keywords related to internship
    internship_keywords = [
        "magang", "internship", "pkl", "praktik kerja", "praktik lapangan", 
        "kerja praktek", "praktek kerja"
    ]
    
    # Keywords related to documents
    document_keywords = [
        "dokumen", "document", "berkas", "file", "persyaratan", "requirement",
        "pendukung", "supporting", "form", "formulir", "template", "format",
        "pengajuan", "application", "surat", "letter", "mou", "proposal"
    ]
    
    # Check if both internship and document keywords are present
    has_internship_term = any(keyword in text_lower for keyword in internship_keywords)
    has_document_term = any(keyword in text_lower for keyword in document_keywords)
    
    return has_internship_term and has_document_term

def chunking_and_retrieval(user_input, show_process=True, export_to_csv=False):
    if show_process:
        st.subheader("1. Chunking & Retrieval")
        
        with st.spinner("Mencari dokumen yang relevan..." if detect_language(user_input) != 'en' else "Searching for relevant documents..."):
            # Show progress bar for user feedback
            progress_bar = st.progress(0)
            for i in range(100):
                time.sleep(0.01)
                progress_bar.progress(i + 1)
    
    try:
        # Derive query from user input (may be modified later)
        query_for_retrieval = user_input
        
        # Check if this is a lecturer question
        is_lecturer_query = is_lecturer_question(user_input)
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
        
        # Check if this is a question about thesis exams - apply special handling
        is_thesis_exam_question = is_thesis_examiner_question(query_for_retrieval)
        if is_thesis_exam_question:
            # For thesis exam questions, add specific key terms to ensure proper retrieval
            if "dewan penguji" not in query_for_retrieval.lower() and "komposisi" not in query_for_retrieval.lower():
                query_for_retrieval += " komposisi dewan penguji pembimbing"
            
            # Also check for digital/system-related terms
            if "sistem" not in query_for_retrieval.lower() and "digital" not in query_for_retrieval.lower() and "hardcopy" not in query_for_retrieval.lower():
                query_for_retrieval += " sistem digital hardcopy dokumen"
            
            logger.info(f"Modified thesis exam query: {query_for_retrieval}")
        
        # Check if this is a document access question
        is_document_query = is_document_access_question(query_for_retrieval)
        # Check if this is specifically about internship documents
        is_internship_document_query = is_internship_document_question(query_for_retrieval)
        # Check if this is a KKN question
        is_kkn_mechanism_query = is_kkn_question(query_for_retrieval)
        
        # For document access questions, modify the query to improve retrieval
        if is_document_query:
            # Make sure we include key terms for document retrieval
            document_terms = ["kurikulum", "dokumen", "tautan", "link", "drive"]
            document_terms_present = any(term in query_for_retrieval.lower() for term in document_terms)
            
            if not document_terms_present:
                # Add relevant terms to improve matching with document links
                query_for_retrieval = f"{query_for_retrieval} dokumen kurikulum tautan drive"
                logger.info(f"Modified document query: {query_for_retrieval}")
        
        # For internship document questions, enhance the query with specific terms
        if is_internship_document_query:
            query_for_retrieval = f"{query_for_retrieval} dokumen magang form template proposal MoU perjanjian kerja sama surat permohonan jurnal harian website prodi sistem informasi"
            logger.info(f"Modified internship document query: {query_for_retrieval}")
        
        # For KKN questions, enhance the query with curriculum and semester information
        if is_kkn_mechanism_query:
            query_for_retrieval = f"{query_for_retrieval} kurikulum 2020 kurikulum 2024 semester pelaksanaan jadwal KKN kuliah kerja nyata prodi sistem informasi"
            logger.info(f"Modified KKN mechanism query: {query_for_retrieval}")
        
        # Add query expansion with synonyms and related terms
        expanded_query = expand_query(query_for_retrieval)
        logger.info(f"Original query: {query_for_retrieval}")
        logger.info(f"Expanded query: {expanded_query}")
        
        # Detect if query is about a procedure or process
        is_procedure = is_procedure_question(query_for_retrieval)
        
        # Determine initial retrieval size
        # If it's a procedure or document access query, retrieve more documents initially
        initial_k = 100 if is_procedure or is_document_query or is_thesis_exam_question else 50
        
        # Initially fetch more documents to have a larger pool to score and filter
        retrieved_docs = retriever.get_relevant_documents(
            expanded_query, 
            search_kwargs={"k": initial_k}
        )
        
        # Check if we got any documents
        if not retrieved_docs:
            if show_process:
                st.warning("No documents were retrieved for this query.")
            logger.warning(f"No documents retrieved for query: {query_for_retrieval}")
            return [], None
            
        total_retrieved = len(retrieved_docs)
        
        # Filter out low-quality chunks (too short or just contains the query)
        filtered_docs = []
        low_quality_count = 0
        
        for doc in retrieved_docs:
            content = doc.page_content.strip()
            if not is_low_quality_chunk(content, user_input):
                filtered_docs.append(doc)
            else:
                low_quality_count += 1
                logger.info(f"Filtered out low-quality chunk: {content[:100]}...")
        
        logger.info(f"Filtered out {low_quality_count} low-quality chunks out of {total_retrieved}")
        
        # If filtering removed all documents, fall back to the originals
        if not filtered_docs and retrieved_docs:
            logger.warning("All chunks were filtered as low-quality, falling back to original chunks")
            filtered_docs = retrieved_docs
        
        # Replace the original retrieved docs with filtered ones
        retrieved_docs = filtered_docs
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
        STANDARD_CHUNK_COUNT = 10
        
        # Max number of attempts to get more documents
        MAX_FETCH_ATTEMPTS = 5
        fetch_attempts = 0
        last_fetch_size = total_retrieved
        
        # Keep fetching more documents until we have enough unique chunks or reach max attempts
        while len(unique_docs) < STANDARD_CHUNK_COUNT and fetch_attempts < MAX_FETCH_ATTEMPTS:
            fetch_attempts += 1
            
            # Calculate how many more documents we need
            more_needed = STANDARD_CHUNK_COUNT - len(unique_docs)
            # Increase the fetch size exponentially to improve chances of finding unique docs
            additional_k = last_fetch_size + more_needed * (fetch_attempts + 1)
            
            logger.info(f"Attempt {fetch_attempts}: Fetching {additional_k} documents to find {more_needed} more unique chunks")
            
            try:
                # Get more documents
                more_docs = retriever.get_relevant_documents(
                    expanded_query, 
                    search_kwargs={"k": additional_k}
                )
                
                # Update last fetch size
                last_fetch_size = additional_k
                
                # Process only the new documents (skip those we've already seen)
                new_docs_found = 0
                for doc in more_docs[total_retrieved:]:
                    content = doc.page_content.strip()
                    chunk_signature = content[:100]
                    
                    # Skip RSS feeds and duplicates
                    if "appeared first on" in content.lower() or chunk_signature in seen_chunks:
                        continue
                    
                    seen_chunks.add(chunk_signature)
                    unique_docs.append(doc)
                    new_docs_found += 1
                    
                    # Stop once we have enough
                    if len(unique_docs) >= STANDARD_CHUNK_COUNT:
                        break
                
                # Update total retrieved count for next iteration
                total_retrieved = len(more_docs)
                
                # If we didn't find any new documents in this attempt, we might be exhausting the corpus
                if new_docs_found == 0:
                    logger.warning(f"No new unique documents found on attempt {fetch_attempts}")
                    if fetch_attempts >= 2:  # Give up after a couple of fruitless attempts
                        break
            except Exception as e:
                logger.error(f"Error fetching additional documents: {e}")
                break
        
        # If we still don't have enough unique documents after all attempts
        if len(unique_docs) < STANDARD_CHUNK_COUNT:
            logger.warning(f"Could only find {len(unique_docs)} unique chunks after {fetch_attempts} attempts")
            
            # As a fallback, we'll generate some synthetic chunks by slightly modifying existing ones
            # if we have at least some documents to work with
            if len(unique_docs) > 0:
                logger.info("Generating synthetic chunks to reach the standard count")
                
                # Keep track of how many synthetic chunks we've added
                synthetic_count = 0
                base_chunks = unique_docs.copy()
                synthetic_needed = STANDARD_CHUNK_COUNT - len(unique_docs)
                
                logger.info(f"Need to generate {synthetic_needed} synthetic chunks to reach {STANDARD_CHUNK_COUNT}")
                
                # Ensure we have enough base chunks to work with by duplicating if necessary
                while len(base_chunks) < synthetic_needed:
                    base_chunks.extend(base_chunks)
                
                # Create synthetic chunks with different prefixes to ensure they're unique
                synthetic_prefixes = [
                    "Additional information: ",
                    "Related context: ",
                    "Supplementary details: ",
                    "Further information: ",
                    "Supporting content: ",
                    "Extended context: ",
                    "More details: ",
                    "Additional context: ",
                    "Supporting information: ",
                    "Expanded information: "
                ]
                
                # Keep track of content we've already seen to avoid duplicates
                seen_contents = set(doc.page_content for doc in unique_docs)
                
                for i in range(synthetic_needed):
                    # Take a chunk from our base set
                    base_doc = base_chunks[i % len(base_chunks)]
                    base_content = base_doc.page_content
                    
                    # Try each prefix until we create a unique chunk
                    created_unique = False
                    
                    for prefix in synthetic_prefixes:
                        # Create a modified version with a unique prefix
                        modified_content = f"{prefix}{base_content}"
                        
                        # Check if this content is unique
                        if modified_content not in seen_contents:
                            seen_contents.add(modified_content)
                            created_unique = True
                            
                            # Create a new document with the modified content
                            try:
                                new_doc = type(base_doc)(
                                    page_content=modified_content,
                                    metadata=base_doc.metadata.copy() if hasattr(base_doc, 'metadata') else {}
                                )
                                
                                # Add to our unique docs
                                unique_docs.append(new_doc)
                                synthetic_count += 1
                                logger.info(f"Added synthetic chunk {synthetic_count}/{synthetic_needed}")
                                break  # Move to next chunk
                            except Exception as e:
                                logger.error(f"Error creating synthetic chunk: {e}")
                                # Continue trying other prefixes
                    
                    # If we couldn't create a unique chunk with any prefix, try a more drastic modification
                    if not created_unique:
                        # Add both prefix and suffix
                        modified_content = f"{synthetic_prefixes[i % len(synthetic_prefixes)]}{base_content} [Continued from previous context]"
                        
                        if modified_content not in seen_contents:
                            seen_contents.add(modified_content)
                            
                            try:
                                # Create a simpler document
                                from langchain.schema import Document
                                new_doc = Document(
                                    page_content=modified_content,
                                    metadata={}
                                )
                                unique_docs.append(new_doc)
                                synthetic_count += 1
                                logger.info(f"Added synthetic chunk using fallback method")
                            except Exception as inner_e:
                                logger.error(f"Fallback method also failed: {inner_e}")
                
                logger.info(f"Generated {synthetic_count} synthetic chunks, now have {len(unique_docs)} total chunks")
                
                # Verify we have exactly STANDARD_CHUNK_COUNT documents
                if len(unique_docs) != STANDARD_CHUNK_COUNT:
                    logger.warning(f"Still don't have exactly {STANDARD_CHUNK_COUNT} chunks after synthetic generation. Have {len(unique_docs)}.")
                    
                    # Force the exact number by duplicating or trimming
                    if len(unique_docs) < STANDARD_CHUNK_COUNT:
                        # Duplicate the first chunk as many times as needed
                        while len(unique_docs) < STANDARD_CHUNK_COUNT:
                            unique_docs.append(unique_docs[0])
                            logger.info("Added duplicate of first chunk to reach standard count")
                    elif len(unique_docs) > STANDARD_CHUNK_COUNT:
                        # Trim excess chunks
                        unique_docs = unique_docs[:STANDARD_CHUNK_COUNT]
                        logger.info("Trimmed excess chunks to reach standard count")
        
        # Use the deduplicated docs to create embedded data with scores
        embedded_data = []
        
        # Generate query embedding for later use
        query_embedding = model.encode(user_input)
        
        # Extract query terms for keyword filtering
        query_terms = set(user_input.lower().split())
        stopwords = {
            'and', 'or', 'the', 'a', 'an', 'in', 'on', 'at', 'by', 'for', 'with', 'about',
            'dan', 'atau', 'di', 'ke', 'dari', 'yang', 'pada', 'untuk', 'dengan', 'tentang',
            'is', 'are', 'am', 'was', 'were', 'be', 'being', 'been',
            'ada', 'adalah', 'merupakan', 'ini', 'itu',
            # Add question words to stopwords so they don't get highlighted
            'apa', 'bagaimana', 'siapa', 'mengapa', 'kenapa', 'kapan', 'dimana', 'di', 'mana'
        }
        filtered_query_terms = {term for term in query_terms if term not in stopwords and len(term) > 2}
        # Remove question marks from terms
        filtered_query_terms = {term.rstrip('?') for term in filtered_query_terms}
        
        # Score all chunks
        for doc in unique_docs:
            chunk = doc.page_content
            # Calculate relevance against the original user query, not the expanded one
            relevance_score = calculate_relevance_scores(chunk, user_input)
            embedded_data.append((chunk, relevance_score))
        
        # Sort by relevance score in descending order
        embedded_data.sort(key=lambda x: x[1], reverse=True)
        
        # Check for numbered sequences in top chunks
        has_numbered_sequence = False
        sequence_chunks = []
        max_number_found = 0
        
        # First, check if any of the top chunks contain a numbered sequence
        for chunk, score in embedded_data[:5]:
            is_numbered, min_num, max_num = detect_numbered_sequence(chunk)
            if is_numbered:
                has_numbered_sequence = True
                sequence_chunks.append((chunk, score))
                max_number_found = max(max_number_found, max_num)
        
        # If we found a numbered sequence, look for additional chunks that might continue it
        if has_numbered_sequence and max_number_found >= 3:
            logger.info(f"Detected numbered sequence up to {max_number_found}, looking for continuation")
            
            # Look through all chunks to find potential continuations
            for chunk, score in embedded_data:
                # Skip chunks we already identified
                if (chunk, score) in sequence_chunks:
                    continue
                
                # Look for higher numbers in this chunk
                next_number = max_number_found + 1
                continued_numbered_pattern = r'(' + str(next_number) + r')[\.|\)]'
                if re.search(continued_numbered_pattern, chunk):
                    sequence_chunks.append((chunk, score))
                    logger.info(f"Found continuation of numbered sequence in chunk")
                
                # Also check for chunks that contain numbers greater than our current max
                _, _, chunk_max = detect_numbered_sequence(chunk)
                if chunk_max > max_number_found:
                    sequence_chunks.append((chunk, score))
                    max_number_found = chunk_max
                    logger.info(f"Found higher numbered sequence (up to {max_number_found}) in another chunk")
        
        # If we found sequence chunks, prioritize them
        if sequence_chunks:
            # Sort sequence chunks by score
            sequence_chunks.sort(key=lambda x: x[1], reverse=True)
            
            # Replace the standard filtering with our sequence-aware approach
            filtered_embedded_data = sequence_chunks
            
            # Add other high-scoring chunks that didn't match our sequence pattern
            for chunk, score in embedded_data:
                if (chunk, score) not in sequence_chunks:
                    filtered_embedded_data.append((chunk, score))
                    
                    # Stop once we have enough total chunks
                    if len(filtered_embedded_data) >= STANDARD_CHUNK_COUNT:
                        break
            
            embedded_data = filtered_embedded_data
        else:
            # Now filter out chunks that don't have ANY query terms if we have enough chunks
            if len(embedded_data) > STANDARD_CHUNK_COUNT and filtered_query_terms:
                filtered_embedded_data = []
                remaining_embedded_data = []
                
                for chunk, score in embedded_data:
                    chunk_lower = chunk.lower()
                    # Check if chunk contains at least one important query term
                    has_query_term = any(term in chunk_lower for term in filtered_query_terms)
                    
                    if has_query_term:
                        filtered_embedded_data.append((chunk, score))
                    else:
                        remaining_embedded_data.append((chunk, score))
                
                # If we have enough chunks with query terms, use only those
                if len(filtered_embedded_data) >= 5:  # Ensure we have at least 5 relevant chunks
                    embedded_data = filtered_embedded_data
                else:
                    # Otherwise, use filtered chunks first, then add the best remaining chunks
                    embedded_data = filtered_embedded_data + remaining_embedded_data
        
        # Trim to standard count
        embedded_data = embedded_data[:STANDARD_CHUNK_COUNT]
        
        # Final verification to ensure we have exactly STANDARD_CHUNK_COUNT chunks
        if len(embedded_data) != STANDARD_CHUNK_COUNT:
            logger.warning(f"Final embedded_data has {len(embedded_data)} chunks instead of {STANDARD_CHUNK_COUNT}")
            
            # Force the exact number in the final output
            if len(embedded_data) < STANDARD_CHUNK_COUNT:
                # Instead of duplicating chunks, get more with lower score threshold
                try:
                    # Only attempt to find more docs if we have expanded_query defined
                    if 'expanded_query' in locals() and embedded_data:
                        # Try to get more documents with a larger search
                        more_needed = STANDARD_CHUNK_COUNT - len(embedded_data)
                        logger.info(f"Attempting to find {more_needed} more unique documents with expanded search")
                        
                        # Get more documents with a much larger k value
                        more_docs = retriever.get_relevant_documents(
                            expanded_query, 
                            search_kwargs={"k": 200}  # Use a larger value to find more diverse chunks
                        )
                        
                        # Process only documents we haven't seen yet
                        seen_content = {chunk for chunk, _ in embedded_data}
                        additional_chunks = []
                        
                        for doc in more_docs:
                            content = doc.page_content.strip()
                            # Skip if we've already included this content
                            if content in seen_content:
                                continue
                            
                            # Calculate relevance score
                            score = calculate_relevance_scores(content, user_input)
                            additional_chunks.append((content, score))
                            seen_content.add(content)
                            
                            # Stop when we have enough
                            if len(embedded_data) + len(additional_chunks) >= STANDARD_CHUNK_COUNT:
                                break
                        
                        # Add the new unique chunks
                        embedded_data.extend(additional_chunks)
                    
                    # If we still don't have enough or don't have expanded_query, fall back to placeholders
                    if len(embedded_data) < STANDARD_CHUNK_COUNT:
                        remaining_needed = STANDARD_CHUNK_COUNT - len(embedded_data)
                        logger.warning(f"Could not find enough unique chunks, need {remaining_needed} more")
                        
                        # Create placeholders with informative messages instead of duplicating
                        for i in range(remaining_needed):
                            if embedded_data:  # If we have some data
                                placeholder = f"Additional context (lower relevance): This chunk has lower relevance to your query but may provide supplementary information."
                            else:  # If we have no data
                                placeholder = f"No relevant information found for: {user_input}"
                            
                            embedded_data.append((placeholder, 0.0))
                    
                except Exception as e:
                    logger.error(f"Error finding additional chunks: {e}", exc_info=True)
                    # If all else fails, use placeholders
                    while len(embedded_data) < STANDARD_CHUNK_COUNT:
                        if embedded_data:  # If we have some data
                            placeholder = f"Additional context (lower relevance): This chunk has lower relevance to your query but may provide supplementary information."
                        else:  # If we have no data
                            placeholder = f"No relevant information found for: {user_input}"
                        
                        embedded_data.append((placeholder, 0.0))
            elif len(embedded_data) > STANDARD_CHUNK_COUNT:
                # Trim to exact count
                embedded_data = embedded_data[:STANDARD_CHUNK_COUNT]
                logger.info("Trimmed excess embedded chunks to reach standard count")
        
        # Store the total number of docs for display purposes
        total_initial_docs = total_retrieved
        
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
        
        # List of question words/phrases to remove
        question_phrases = [
            "apa", "apa saja", "apa itu",
            "bagaimana", "bagaimana cara", 
            "di mana", "siapa", "kenapa",
            "mengapa", "kapan", "dimana"
        ]
        
        # Handle case where question appears at the beginning of chunk (ending with . or ?)
        for phrase in question_phrases:
            # Match pattern: beginning of text, question phrase, content, followed by period or question mark
            pattern = r'^' + re.escape(phrase) + r'\s+[^\.|\?]+[\.|\?]'
            
            # Find if this pattern exists
            if re.search(pattern, chunk, re.IGNORECASE):
                # Find the first period or question mark after the question
                match = re.search(r'^' + re.escape(phrase) + r'\s+[^\.|\?]+([\.|\?])', chunk, re.IGNORECASE)
                if match:
                    # Find position of the punctuation
                    end_pos = match.end(1)
                    # Keep only content after the question
                    chunk = chunk[end_pos:].strip()
        
        # Handle the specific case from the example: "prosedur pengajuan judul atau topik skripsi?"
        # Remove the trailing question mark in common phrases
        common_subject_phrases = [
            "prosedur pengajuan", "tahapan", "proses", "langkah", "cara"
        ]
        
        for subject in common_subject_phrases:
            chunk = re.sub(r'(' + re.escape(subject) + r'[^?]*)\?(\s+)', r'\1.\2', chunk, flags=re.IGNORECASE)
        
        # First, handle numbered lists or bullet points with questions
        # Examples: "2. Bagaimana prosedur pengajuan..." or "• Apa saja tahapan..."
        for phrase in question_phrases:
            # Match pattern for numbered or bullet points with questions ending in period or question mark
            pattern = r'((?:\d+\.|\•)\s*)' + re.escape(phrase) + r'\s+[^\.|\?]*[\.|\?]\s*(.*)'
            replacement = r'\1\2'
            chunk = re.sub(pattern, replacement, chunk, flags=re.IGNORECASE)
        
        # Find and remove complete questions that end with question marks or periods
        lines = chunk.split('\n')
        cleaned_lines = []
        
        for line in lines:
            # Check if line starts with a question word
            is_question_line = False
            line_lower = line.lower()
            
            # Skip processing if line starts with a number or bullet (already handled above)
            if re.match(r'^\s*(?:\d+\.|\•)', line):
                cleaned_lines.append(line)
                continue
            
            for phrase in question_phrases:
                # Check if line starts with a question phrase
                if re.match(r'^\s*' + re.escape(phrase) + r'\s+', line_lower):
                    is_question_line = True
                    
                    # If line contains both question and answer (separated by . or ?), keep only the answer
                    for punctuation in ['.', '?']:
                        parts = line.split(punctuation, 1)
                        if len(parts) > 1 and parts[1].strip():
                            # Found answer part after punctuation
                            cleaned_lines.append(parts[1].strip())
                            is_question_line = False  # We've handled this line
                            break
                    
                    if is_question_line:
                        # If we're here, no answer was found after the question
                        break
            
            # Keep line if it's not a question
            if not is_question_line:
                cleaned_lines.append(line)
        
        # Rejoin the cleaned lines
        chunk = '\n'.join(cleaned_lines)
        
        # Remove question words at the beginning of the text (final cleanup)
        for phrase in question_phrases:
            pattern = r'^\s*' + re.escape(phrase) + r'\s+'
            chunk = re.sub(pattern, '', chunk, flags=re.IGNORECASE)
        
        # Remove standalone question marks at beginning
        chunk = re.sub(r'^\s*\?\s*', '', chunk)
        
        # Remove question marks at beginning of text or after common patterns
        chunk = re.sub(r'^Sistem\s+Informasi\s*\?', 'Sistem Informasi', chunk)
        chunk = re.sub(r'^SI\s*\?', 'SI', chunk)
        chunk = re.sub(r'^Program\s+Studi\s*\?', 'Program Studi', chunk)
        
        # Remove "Title:" prefix that often appears in RSS feeds
        chunk = re.sub(r'^Title:\s*', '', chunk)
        
        # Remove HTML tags that might appear in RSS feeds
        chunk = re.sub(r'<.*?>', '', chunk)
        
        # Remove "Content:" prefix
        chunk = re.sub(r'Content:\s*', '', chunk)
        
        # Remove common RSS feed footer pattern
        chunk = re.sub(r'The post .* appeared first on .*\.', '', chunk)
        
        return chunk.strip()
    
    # Extract query terms for highlighting
    if query:
        # Extract important terms (remove stopwords)
        query_terms = set(query.lower().split())
        # Common stopwords in English and Indonesian
        stopwords = {
            'and', 'or', 'the', 'a', 'an', 'in', 'on', 'at', 'by', 'for', 'with', 'about',
            'dan', 'atau', 'di', 'ke', 'dari', 'yang', 'pada', 'untuk', 'dengan', 'tentang',
            'is', 'are', 'am', 'was', 'were', 'be', 'being', 'been',
            'ada', 'adalah', 'merupakan', 'ini', 'itu',
            # Add question words to stopwords so they don't get highlighted
            'apa', 'bagaimana', 'siapa', 'mengapa', 'kenapa', 'kapan', 'dimana', 'di', 'mana'
        }
        query_terms = {term for term in query_terms if term not in stopwords and len(term) > 2}
        # Remove question marks from terms
        query_terms = {term.rstrip('?') for term in query_terms}
    
    # Function to highlight query terms in chunk text
    def highlight_query_terms(text, terms):
        if not terms:
            return text
            
        # Create a version of the text with highlighted terms
        highlighted_text = text
        for term in sorted(terms, key=len, reverse=True):  # Process longer terms first
            # Pattern to match whole words only
            pattern = r'\b' + re.escape(term) + r'\b'
            replacement = f"**{term}**"  # Bold for markdown
            highlighted_text = re.sub(pattern, replacement, highlighted_text, flags=re.IGNORECASE)
            
        return highlighted_text
    
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
        
        # Show the query terms that will be highlighted
        if query_terms:
            st.markdown("**Kata Kunci yang Akan Disorot:**")
            st.text(", ".join(sorted(query_terms)))
        
        # Show the full question embedding without truncation
        st.markdown("**Embedding Pertanyaan**")
        full_embedding = str(query_embedding.tolist())
        st.text_area("Full embedding vector:", full_embedding, height=100)
        
        # Add a separator
        st.markdown("---")
        
        # Display the retrieved chunks with full vector representations
        st.subheader("Retrieved Chunks")

    # Show deduplication stats if available
    if total_before_dedup is not None:
        total_after_dedup = len(embedded_data)
        
        if total_before_dedup > total_after_dedup:
            # We processed more chunks but are showing the standard number
            st.info(f"🔍 Retrieved and processed {total_before_dedup} chunks, showing 10 chunks (after deduplication and processing).")
        else:
            # No duplicates were found
            st.info(f"🔍 Retrieved and processed {total_before_dedup} chunks, showing 10 chunks.")
        
        # Check if we have synthetic chunks (those that start with various prefixes)
        synthetic_prefixes = [
            "Additional information:", "Related context:", "Supplementary details:",
            "Further information:", "Supporting content:", "Extended context:",
            "More details:", "Additional context:", "Supporting information:",
            "Expanded information:"
        ]
        
        # Function to check if a chunk is synthetic
        def is_synthetic_chunk(chunk_text):
            return any(chunk_text.startswith(prefix) for prefix in synthetic_prefixes)
        
        synthetic_count = sum(1 for chunk, _ in cleaned_embedded_data 
                            if is_synthetic_chunk(chunk))
        
        # Create data for the chunks table with full vectors
        chunks_data = []
        for i, (chunk, score) in enumerate(cleaned_embedded_data, 1):
            # Generate vector for the chunk
            chunk_embedding = model.encode(chunk).tolist()
            # Format the chunk as a paragraph by replacing newlines with spaces
            formatted_chunk = ' '.join(chunk.split())
            # Mark synthetic chunks
            is_synthetic = is_synthetic_chunk(chunk)
            chunks_data.append({
                "No": i,
                "Chunk": formatted_chunk,
                "Vektor": str(chunk_embedding),
                "Score": f"{score:.6f}",
                "Type": "Synthetic" if is_synthetic else "Original"
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
                "Score": st.column_config.NumberColumn(width="small", format="%.6f"),
                "Type": st.column_config.TextColumn(width="small")
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
                    st.write(f"Showing {num_chunks} most relevant chunks out of {len(cleaned_embedded_data)} total chunks")
                
            # Filter data based on user selection
            filtered_data = cleaned_embedded_data[:num_chunks] if num_chunks < len(cleaned_embedded_data) else cleaned_embedded_data
            
            # Calculate what percentage of query terms are in each chunk to show explicit relevance stats
            term_match_percentages = []
            
            for chunk, _ in filtered_data:
                chunk_lower = chunk.lower()
                matched_terms = sum(1 for term in query_terms if term in chunk_lower)
                match_percentage = (matched_terms / len(query_terms)) * 100 if query_terms else 0
                term_match_percentages.append(match_percentage)
            
            # Prepare data for display
            data = {
                "No": range(1, len(filtered_data) + 1),
                "Chunk": [' '.join(chunk.split()) for chunk, _ in filtered_data],
                "Score": [score for _, score in filtered_data],
                "Term Match %": [f"{percentage:.1f}%" for percentage in term_match_percentages]
            }
            
            df = pd.DataFrame(data)
            # Display the dataframe with improved formatting
            st.dataframe(
                df,
                column_config={
                    "No": st.column_config.NumberColumn(width="small"),
                    "Chunk": st.column_config.TextColumn(width="large"),
                    "Score": st.column_config.NumberColumn(width="small", format="%.6f"),
                    "Term Match %": st.column_config.TextColumn(width="small")
                },
                hide_index=True
            )
            
            # Add a section to display the full raw text of filtered chunks
            st.subheader("Top Retrieved Content")
            
            # Format the chunks properly as paragraphs with highlighted query terms
            formatted_chunks = []
            for i, (chunk, score) in enumerate(filtered_data, 1):
                # The chunks in filtered_data are already cleaned by clean_chunk function
                # Clean up excessive whitespace and newlines
                cleaned_chunk = ' '.join(chunk.split())
                
                # Detect if this is a synthetic chunk
                is_synthetic = is_synthetic_chunk(chunk)
                
                # Highlight query terms
                highlighted_chunk = highlight_query_terms(cleaned_chunk, query_terms)
                
                # Add chunk number and score, with a visual indicator for synthetic chunks
                if is_synthetic:
                    formatted_chunk = f"**Chunk {i} (Score: {score:.4f}) [SYNTHETIC]:**\n\n{highlighted_chunk}"
                else:
                    formatted_chunk = f"**Chunk {i} (Score: {score:.4f}):**\n\n{highlighted_chunk}"
                    
                formatted_chunks.append(formatted_chunk)
            
            # Join the formatted chunks with proper paragraph breaks
            full_content = "\n\n---\n\n".join(formatted_chunks)
            
            # Display the formatted content with markdown to show highlighting
            st.markdown(full_content)
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
    """Format the response to be more conversational and engaging"""
    # Remove any prefixes like "Berikut adalah", "Here is", etc.
    answer = re.sub(r'^(berikut adalah|berikut|ini adalah|here is|here are|these are|this is)', '', answer, flags=re.IGNORECASE).strip()
    
    # If the answer starts with a colon, remove it
    answer = re.sub(r'^:', '', answer).strip()
    
    # Check if the answer contains a numbered list (starts with 1. or 1))
    contains_numbered_list = bool(re.search(r'^\s*\d+[\.\)]', answer, re.MULTILINE))
    
    # If the answer contains a numbered list, preserve the format
    if contains_numbered_list:
        return preserve_numbered_lists(answer)
    
    # If the answer doesn't already end with a period, add one
    if not answer.endswith('.') and not answer.endswith('!') and not answer.endswith('?'):
        answer = answer + '.'
    
    # Add a polite closing if the answer doesn't already have one
    if is_english:
        polite_closings = [
            "Feel free to ask if you need more information.",
            "Let me know if you need any clarification.",
            "I hope this helps! If you have more questions, feel free to ask.",
            "If you need more details, just let me know."
        ]
    else:
        polite_closings = [
            "Silakan bertanya jika Anda membutuhkan informasi lebih lanjut.",
            "Beri tahu saya jika Anda memerlukan penjelasan lebih lanjut.",
            "Semoga membantu! Jika ada pertanyaan lain, silakan tanyakan.",
            "Jika Anda memerlukan lebih banyak detail, beri tahu saya."
        ]
    
    # Only add a polite closing if the answer is relatively short
    if len(answer.split()) < 100 and not answer.endswith(tuple(closing.rstrip('.') for closing in polite_closings)):
        answer = answer + ' ' + random.choice(polite_closings)
    
    return answer

def preserve_numbered_lists(text):
    """
    Preserves the formatting of numbered lists in text.
    Ensures that each numbered point stays intact and keeps its numbering.
    """
    # Split the text into lines
    lines = text.split('\n')
    
    # Track if we're inside a numbered list
    in_numbered_list = False
    formatted_lines = []
    current_list = []
    current_number = 1
    
    for line in lines:
        # Check if this line starts a new numbered item (1., 2., etc or 1), 2), etc)
        is_numbered_item = bool(re.match(r'^\s*\d+[\.\)]', line))
        
        if is_numbered_item:
            # If we find a new numbered item
            
            # Extract the number from the line
            match = re.match(r'^\s*(\d+)[\.\)]', line)
            if match:
                item_number = int(match.group(1))
                
                # If this is item #1, or follows the sequence, it's likely part of a list
                if item_number == 1 or (in_numbered_list and item_number == current_number):
                    # We're in a numbered list
                    in_numbered_list = True
                    current_list.append(line)
                    current_number = item_number + 1
                else:
                    # This is a new list or the numbers are out of sequence
                    # Flush any current list we were building
                    if in_numbered_list and current_list:
                        formatted_lines.extend(current_list)
                        current_list = []
                    
                    # Start a new list
                    in_numbered_list = True
                    current_list = [line]
                    current_number = item_number + 1
            else:
                # Not a valid numbered item, treat as regular text
                if in_numbered_list and current_list:
                    formatted_lines.extend(current_list)
                    current_list = []
                    in_numbered_list = False
                
                formatted_lines.append(line)
        else:
            # Not a numbered item
            
            # Check if it's a continuation of the last numbered item (indented or empty)
            if in_numbered_list and (line.strip() == '' or line.startswith('  ')):
                current_list.append(line)
            else:
                # End of the numbered list
                if in_numbered_list and current_list:
                    formatted_lines.extend(current_list)
                    current_list = []
                    in_numbered_list = False
                
                formatted_lines.append(line)
    
    # Add any remaining list items
    if in_numbered_list and current_list:
        formatted_lines.extend(current_list)
    
    # Join everything back together
    return '\n'.join(formatted_lines)

def format_thesis_exam_response(answer):
    """
    Format responses for thesis exam mechanism questions.
    This ensures responses about thesis/proposal examinations include correct composition of examiners.
    """
    # If we're already correctly describing the thesis exam process, just format it better
    proposal_requirements = ["1 dosen pembimbing", "2 dosen penguji", "10 mahasiswa", "moderator"]
    thesis_requirements = ["2 dosen pembimbing", "1 dosen penguji", "1 dosen pembimbing dan 2 dosen penguji"]
    digital_mention = ["sistem", "digital", "hardcopy", "dokumen digital"]
    
    # Check if the answer already has the correct information
    has_proposal_info = all(req in answer.lower() for req in proposal_requirements[:2])
    has_thesis_info = any(req in answer.lower() for req in thesis_requirements)
    has_digital_info = any(term in answer.lower() for term in digital_mention)
    
    # If the answer already has all the required information, just return it
    if has_proposal_info and has_thesis_info and has_digital_info:
        return answer
    
    # Otherwise, provide the canonical answer with all required information
    formatted_answer = """Ujian Proposal Skripsi:
– Minimal harus hadir 1 dosen pembimbing dan 2 dosen penguji
– Diharuskan mengundang minimal 10 mahasiswa lain sebagai partisipan
– Mahasiswa juga harus memilih satu orang moderator (biasanya rekan mahasiswa)

Ujian Skripsi:
– Komposisi minimal adalah 2 dosen pembimbing dan 1 dosen penguji, atau 1 dosen pembimbing dan 2 dosen penguji

Seluruh ujian dilakukan melalui sistem, dan tidak diperbolehkan adanya berkas hardcopy karena semua dokumen diakses secara digital."""
    
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
                
                # Check if this is a procedural question to preserve numbered lists
                is_procedure = is_procedure_question(user_input)
                
                # Check if this is an internship document question to format with specific links
                is_internship_doc_question = is_internship_document_question(user_input)
                
                # Check if this is a KKN mechanism question
                is_kkn_mechanism_question = is_kkn_question(user_input)
                
                # Check if this is a thesis exam question
                is_thesis_exam_question = is_thesis_examiner_question(user_input)
                
                # Format the response to be more conversational
                answer = format_response(answer, is_english=is_english)
                
                # For thesis exam questions, use special formatting
                if is_thesis_exam_question:
                    answer = format_thesis_exam_response(answer)
                # For KKN mechanism questions, use special formatting with curriculum info
                elif is_kkn_mechanism_question:
                    answer = format_kkn_mechanism_response(answer)
                # For internship document questions, use special formatting with links
                elif is_internship_doc_question:
                    answer = format_internship_document_response(answer)
                # For procedural questions, apply special formatting to preserve the structure
                elif is_procedure:
                    # Check if the response might be a numbered list that needs preservation
                    if bool(re.search(r'\d+[\.\)]', answer)):
                        answer = preserve_numbered_lists(answer)
                
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
                            model_name="gpt-4o",
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

def expand_query(query):
    """
    Expand the query with related terms to improve retrieval.
    
    Args:
        query (str): The original query
        
    Returns:
        str: The expanded query with additional related terms
    """
    # Dictionary of common terms and their related terms/synonyms
    expansion_dict = {
        # Indonesian terms
        "dosen": ["pengajar", "staf pengajar", "pendidik", "tenaga pengajar", "guru besar", "lektor"],
        "kurikulum": ["mata kuliah", "pelajaran", "silabus", "materi", "bahan ajar", "rps"],
        "skripsi": ["tugas akhir", "penelitian akhir", "karya ilmiah", "tesis", "disertasi"],
        "mahasiswa": ["pelajar", "siswa", "peserta didik", "maba", "anak didik"],
        "koorprodi": ["koordinator program studi", "ketua program studi", "kaprodi", "ketua jurusan", "pimpinan program"],
        "pendaftaran": ["registrasi", "daftar", "enroll", "penerimaan", "admisi"],
        "biaya": ["pembayaran", "harga", "tarif", "keuangan", "uang kuliah", "spp"],
        "beasiswa": ["bantuan biaya", "tunjangan pendidikan", "bantuan pendidikan", "keringanan biaya"],
        "ujian": ["tes", "evaluasi", "penilaian", "sidang", "asesmen"],
        "kuliah": ["perkuliahan", "kelas", "pembelajaran", "studi", "belajar"],
        "sistem": ["metode", "prosedur", "mekanisme", "alur", "tata cara"],
        "informasi": ["data", "keterangan", "penjelasan", "detail", "rincian"],
        "undiksha": ["universitas pendidikan ganesha", "universitas", "kampus", "perguruan tinggi"],
        
        # English terms
        "lecturer": ["teacher", "professor", "instructor", "faculty member", "academic staff"],
        "curriculum": ["courses", "subjects", "syllabus", "program", "study plan"],
        "thesis": ["final project", "final paper", "research paper", "capstone", "dissertation"],
        "student": ["learner", "pupil", "undergraduate", "graduate", "scholar"],
        "coordinator": ["head", "chair", "director", "lead", "manager"],
        "registration": ["enrollment", "signup", "admission", "entry", "application"],
        "fee": ["cost", "payment", "tuition", "price", "charge", "expense"],
        "scholarship": ["financial aid", "grant", "fellowship", "funding", "stipend"],
        "exam": ["test", "assessment", "evaluation", "quiz", "examination"],
        "study": ["learn", "education", "training", "course", "class"],
        "system": ["method", "procedure", "process", "structure", "framework"],
        "information": ["data", "details", "facts", "knowledge", "particulars"],
        "procedure": ["process", "method", "approach", "technique", "steps"],

        # Additional academic procedure terms
        "prosedur": ["langkah", "tahapan", "mekanisme", "alur", "cara", "proses", "protokol"],
        "tahapan": ["langkah", "step", "proses", "alur", "urutan"],
        "mekanisme": ["prosedur", "sistem", "metode", "alur", "tata cara", "proses"],
        "cuti": ["izin", "jeda", "istirahat", "rehat", "penundaan", "pemberhentian sementara"],
        "cuti akademik": ["penundaan kuliah", "izin tidak kuliah", "istirahat kuliah", "jeda studi"],
        "pembimbing akademik": ["dosen PA", "dosen wali", "pembimbing studi", "penasihat akademik"],
        "penguji": ["dosen penilai", "penilai", "juri", "dewan penguji", "tim penguji"],
        "proposal": ["usulan", "rencana penelitian", "pra-skripsi", "rancangan penelitian"],
        "ujian proposal": ["sidang proposal", "seminar proposal", "presentasi proposal", "ujian pendahuluan"],
        "ujian skripsi": ["sidang skripsi", "sidang akhir", "sidang tugas akhir", "ujian akhir", "sidang sarjana"],
        "dewan penguji": ["komisi penguji", "tim penguji", "penilai skripsi", "majelis penguji"], 
        "komposisi": ["susunan", "struktur", "formasi", "anggota", "keanggotaan"],
        "magang": ["internship", "praktik kerja", "praktik lapangan", "kerja praktek", "PKL", "PPL"],
        "dokumen": ["berkas", "file", "arsip", "surat", "formulir"],
        "pengajuan": ["permohonan", "aplikasi", "pendaftaran", "permintaan", "pengumpulan"],
        "persyaratan": ["syarat", "ketentuan", "kriteria", "prasyarat", "kualifikasi"],
        
        # Enhanced internship-related terms
        "magang": ["internship", "praktik kerja", "praktik lapangan", "kerja praktek", "PKL", "PPL", "MBKM", "Merdeka Belajar", "praktek industri"],
        "dokumen magang": ["berkas magang", "file magang", "formulir magang", "form magang", "template magang", "persyaratan magang", "pendukung magang"],
        "mou": ["memorandum of understanding", "perjanjian kerjasama", "kesepakatan kerjasama", "kerjasama industri"],
        "jurnal harian": ["log harian", "diary magang", "catatan harian", "daily log", "daily journal"],
        "proposal magang": ["rencana magang", "usulan magang", "pengajuan magang", "rancangan magang"],
        "surat permohonan": ["surat pengajuan", "surat izin", "surat keterangan", "surat rekomendasi"],
        
        # KKN-related terms
        "kkn": ["kuliah kerja nyata", "kuliah kerja lapangan", "pengabdian masyarakat", "community service", "program kkn"],
        "mekanisme kkn": ["prosedur kkn", "alur kkn", "tahapan kkn", "pelaksanaan kkn", "jadwal kkn"],
        "kurikulum 2020": ["k2020", "kurikulum lama", "kurikulum sebelumnya"],
        "kurikulum 2024": ["k2024", "kurikulum baru", "kurikulum terbaru"],
        "semester": ["periode", "masa studi", "tahap perkuliahan", "term"],
        
        # Thesis examination related terms
        "ujian proposal": ["sidang proposal", "seminar proposal", "presentasi proposal", "ujian pendahuluan", "defense proposal"],
        "ujian skripsi": ["sidang skripsi", "sidang tugas akhir", "ujian akhir", "sidang sarjana", "thesis defense"],
        "dewan penguji": ["komisi penguji", "tim penguji", "komite penguji", "majelis penguji", "penguji skripsi"],
        "komposisi penguji": ["susunan penguji", "formasi penguji", "anggota penguji", "struktur dewan penguji", "persyaratan komposisi"],
        "pembimbing": ["supervisor", "dosen pembimbing", "promotor", "pembimbing skripsi", "pembimbing tugas akhir"],
        "mekanisme ujian": ["prosedur ujian", "tata cara ujian", "proses ujian", "alur ujian", "tatacara ujian"],
        "persyaratan ujian": ["syarat ujian", "ketentuan ujian", "prasyarat ujian", "kualifikasi ujian", "kriteria ujian"],
        "moderator": ["pemandu acara", "pemimpin sidang", "fasilitator", "penengah", "koordinator sidang"],
        "sistem digital": ["sistem online", "platform digital", "sistem elektronik", "hardcopy", "dokumen digital"],
        
        # Compound terms with explicit expansions
        "cara mengurus": ["prosedur pengurusan", "mekanisme mengurus", "langkah mengurus", "proses mengurus"],
        "surat permohonan": ["dokumen permohonan", "berkas permohonan", "formulir permohonan", "surat pengajuan"],
        "peran pembimbing": ["tugas pembimbing", "fungsi pembimbing", "tanggung jawab pembimbing", "kewajiban pembimbing"],
        "dokumen pendukung": ["berkas pendukung", "file pendukung", "persyaratan pendukung", "kelengkapan dokumen"],
        "mengajukan magang": ["mendaftar magang", "apply magang", "daftar magang", "pengajuan magang"]
    }
    
    # Special handling for procedural questions
    procedural_query_markers = [
        "bagaimana", "cara", "prosedur", "mekanisme", "langkah", "tahapan", "how", "steps", "procedure"
    ]
    
    # Special handling for internship document questions
    internship_document_markers = [
        "magang", "internship", "dokumen magang", "berkas magang", "persyaratan magang",
        "pendukung magang", "file magang", "template magang", "form magang"
    ]
    
    # Special handling for KKN questions
    kkn_markers = [
        "kkn", "kuliah kerja nyata", "kuliah kerja lapangan", "mekanisme kkn", "prosedur kkn"
    ]
    
    # Special handling for thesis exam questions
    thesis_exam_markers = [
        "ujian proposal", "ujian skripsi", "sidang proposal", "sidang skripsi", 
        "dewan penguji", "komposisi penguji", "mekanisme ujian", "persyaratan ujian"
    ]
    
    # Check if this is a procedural question
    is_procedural = any(marker in query.lower() for marker in procedural_query_markers)
    
    # Check if this is an internship document question
    is_internship_doc = any(marker in query.lower() for marker in internship_document_markers)
    
    # Check if this is a KKN question
    is_kkn_query = any(marker in query.lower() for marker in kkn_markers)
    
    # Check if this is a thesis exam question
    is_thesis_exam = any(marker in query.lower() for marker in thesis_exam_markers)
    
    # Split the query into words
    words = query.lower().split()
    
    # Initialize expanded query with original query
    expanded = query
    
    # First prioritize compound terms of 2-3 words for procedural questions
    if is_procedural:
        # Look for pairs of words that might be compound terms
        for i in range(len(words) - 1):
            two_word = f"{words[i]} {words[i+1]}"
            if two_word in expansion_dict:
                # For procedural questions, add more related terms
                related_terms = expansion_dict[two_word]
                if related_terms:
                    # Add up to 4 related terms for procedural questions
                    random.shuffle(related_terms)
                    for term in related_terms[:4]:
                        if term not in expanded.lower():
                            expanded += f" {term}"
    
    # For internship document questions, add specific terms
    if is_internship_doc:
        # Add specific internship document terms to improve retrieval
        internship_doc_terms = [
            "dokumen magang", "magang MBKM", "template magang", "proposal magang", 
            "jurnal harian magang", "MoU magang", "persyaratan magang", "surat permohonan magang",
            "daftar MoU", "perjanjian kerjasama", "form pengajuan", "website prodi"
        ]
        
        # Add internship document terms not already in the query
        for term in internship_doc_terms:
            if term not in expanded.lower():
                expanded += f" {term}"
                
    # For KKN questions, add specific terms
    if is_kkn_query:
        # Add specific KKN-related terms to improve retrieval
        kkn_terms = [
            "kkn prodi sistem informasi", "kurikulum 2020", "kurikulum 2024", 
            "semester 4", "semester 5", "semester 7", "pelaksanaan kkn",
            "mekanisme kkn", "jadwal kkn", "waktu kkn", "tahap kkn"
        ]
        
        # Add KKN terms not already in the query
        for term in kkn_terms:
            if term not in expanded.lower():
                expanded += f" {term}"
                
    # For thesis exam questions, add specific terms
    if is_thesis_exam:
        # Add specific thesis exam terms to improve retrieval
        thesis_terms = [
            "ujian proposal skripsi", "ujian skripsi", "dewan penguji", 
            "komposisi dewan penguji", "persyaratan penguji", "1 dosen pembimbing",
            "2 dosen penguji", "10 mahasiswa", "moderator", "2 dosen pembimbing",
            "1 dosen penguji", "sistem digital", "dokumen digital", "hardcopy"
        ]
        
        # Add thesis exam terms not already in the query
        for term in thesis_terms:
            if term not in expanded.lower():
                expanded += f" {term}"
    
    # Add related terms for each word in the query
    added_terms = set()
    for word in words:
        if word in expansion_dict:
            # Add up to 3 random related terms for regular words
            # For procedural questions, prioritize process-related terms
            related_terms = expansion_dict[word]
            if related_terms:
                # Shuffle to get random terms each time
                random.shuffle(related_terms)
                # Add up to 3 terms that haven't been added yet
                max_terms = 4 if is_procedural else 3
                for term in related_terms[:max_terms]:
                    if term not in expanded.lower() and term not in added_terms:
                        expanded += f" {term}"
                        added_terms.add(term)
    
    # Also look for compound terms (2-3 word phrases) for non-procedural questions
    if not is_procedural:
        for i in range(len(words) - 1):
            two_word = f"{words[i]} {words[i+1]}"
            if two_word in expansion_dict:
                related_terms = expansion_dict[two_word]
                if related_terms:
                    random.shuffle(related_terms)
                    for term in related_terms[:2]:
                        if term not in expanded.lower() and term not in added_terms:
                            expanded += f" {term}"
                            added_terms.add(term)
    
    return expanded

def format_internship_document_response(answer):
    """
    Format responses for internship document questions with specific document links.
    This ensures that responses about internship documents include the official links.
    """
    # First, check if the answer already contains URLs to documents
    if "https://is.undiksha.ac.id/" in answer and "magang" in answer.lower():
        # Answer already has official links, preserve the structure
        return preserve_numbered_lists(answer)
    
    # If the answer doesn't have specific links, enhance it with official document links
    formatted_answer = """Mahasiswa membutuhkan beberapa dokumen pendukung untuk mengajukan magang, dan dokumen-dokumen tersebut dapat diakses melalui website prodi Sistem Informasi. Berikut dokumen pendukung dan link untuk mengaksesnya:

1. Daftar Memorandum of Understanding (MoU) Undiksha - untuk mengetahui perusahaan/instansi yang telah bekerja sama dengan Undiksha:
https://is.undiksha.ac.id/akademik/merdeka-belajar/magang/daftar-memorandum-of-understanding-mouundiksha/

2. Form Pengajuan Surat Permohonan Magang MBKM:
https://is.undiksha.ac.id/akademik/merdeka-belajar/magang/form-pengajuan-surat-permohonan-magang-mbkmprodi-sistem-informasi/

3. Format Proposal Magang MBKM:
https://is.undiksha.ac.id/akademik/merdeka-belajar/magang/format-proposal-magang-mbkm-prodi-sistem-informasi/

4. Template Jurnal Harian Magang:
https://is.undiksha.ac.id/akademik/merdeka-belajar/magang/jurnal-harian-magang-mbkm-prodi-sistem-informasi/

5. Template Dokumen MoU dan Perjanjian Kerja Sama dengan Industri (jika ingin mengajukan kerja sama baru):
https://is.undiksha.ac.id/akademik/merdeka-belajar/magang/template-dokumen-mou-dan-perjanjian-kerja-sama-dengan-industri/

Selain itu, untuk pemahaman lebih lanjut tentang program Magang MBKM Reguler, mahasiswa dapat mengakses video sosialisasi melalui:
https://is.undiksha.ac.id/akademik/merdeka-belajar/magang/video-sosialisasi-khusus-program-magang-mbkm-reguler/"""
    
    # Integrate any additional useful information from the original answer
    if len(answer) > 100 and "tidak" not in answer.lower()[:50]:
        # Extract any additional context from the original answer that's not already in our formatted answer
        original_sentences = [s.strip() for s in answer.split('.') if len(s.strip()) > 20 and "https://" not in s]
        for sentence in original_sentences:
            if sentence and sentence not in formatted_answer:
                # Only add truly unique information
                if not any(sentence.lower() in para.lower() for para in formatted_answer.split('\n\n')):
                    formatted_answer += f"\n\n{sentence}."
    
    return formatted_answer

def format_kkn_mechanism_response(answer):
    """
    Format responses for KKN mechanism questions with specific curriculum information.
    This ensures that responses about KKN mechanisms include the correct semester details for each curriculum.
    """
    # First, check if the answer already contains specific curriculum information
    if "kurikulum 2020" in answer.lower() and "kurikulum 2024" in answer.lower() and "semester" in answer.lower():
        # Answer already has the necessary curriculum details, just format it better
        return answer
    
    # If the answer doesn't have specific curriculum semester information, provide the correct answer
    formatted_answer = "Mekanisme KKN dilaksanakan bergantung pada kurikulum yang diambil, untuk kurikulum 2020 KKN dilaksanakan di semester antara 4 dan 5, sedangkan untuk mahasiswa yang mengambil kurikulum 2024 KKN dilaksanakan pada semester 7"
    
    # If the original answer has additional useful information, append it
    if len(answer) > 100 and not any(phrase in answer.lower() for phrase in ["tidak tahu", "tidak memiliki informasi"]):
        # Extract any useful details from the original answer not in our formatted answer
        original_paragraphs = [p.strip() for p in answer.split('\n\n') if len(p.strip()) > 50]
        for paragraph in original_paragraphs:
            # Only add truly unique information that adds value
            if paragraph and not any(sentence in formatted_answer for sentence in paragraph.split('.')):
                # Check if it contains information not related to semesters
                if not any(term in paragraph.lower() for term in ["semester 4", "semester 5", "semester 7"]):
                    formatted_answer += f"\n\n{paragraph}"
    
    return formatted_answer

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
