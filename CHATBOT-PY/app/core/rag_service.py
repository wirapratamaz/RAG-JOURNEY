import os
import logging
import re
from typing import Dict, List
from dotenv import load_dotenv
import chromadb
from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_community.vectorstores import Chroma

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Load environment variables
load_dotenv()

# ChromaDB Configuration
CHROMA_URL = os.getenv("CHROMA_URL", "https://chroma-production-0925.up.railway.app")
CHROMA_TOKEN = os.getenv("CHROMA_TOKEN", "qkj1zn522ppyn02i8wk5athfz98a1f4i")

# Global client to ensure consistent settings
_CHROMA_CLIENT = None

# Lecturer information - kept out of prompt for better token management
LECTURER_INFO = {
    "edy listartha": """
Ir. I Made Edy Listartha, S.Kom., M.Kom.
NIP. 198608122019031005 NIDN. 0012088606 Jabatan Fungsional : Lektor Google Scholar :
Tautan 🔗 SCOPUS : Tautan 🔗 Konsentrasi : KS Topik Riset : Keamanan Sistem Informasi;
Jaringan Komputer; Internet of Things, Robotik, dan Social Engineering
"""
}

def get_chroma_client():
    """Get or initialize the ChromaDB client"""
    global _CHROMA_CLIENT
    if _CHROMA_CLIENT is None:
        try:
            # Connect to Railway ChromaDB
            chroma_url = CHROMA_URL
            host = chroma_url.replace("https://", "").replace("http://", "")
            _CHROMA_CLIENT = chromadb.HttpClient(
                host=host,
                port=443,
                ssl=True,
                headers={"Authorization": f"Bearer {CHROMA_TOKEN}"}
            )
            logger.info(f"Connected to Remote ChromaDB at {CHROMA_URL}")
        except Exception as e:
            logger.error(f"Error connecting to Remote ChromaDB: {e}")
            # Fallback to local ChromaDB
            try:
                _CHROMA_CLIENT = chromadb.PersistentClient(path="./chroma_db")
                logger.info("Fallback to local ChromaDB client")
            except Exception as inner_e:
                logger.error(f"Failed to initialize any ChromaDB client: {inner_e}")
                raise
    return _CHROMA_CLIENT

class RAGService:
    """RAG Service for processing queries"""
    
    def __init__(self):
        """Initialize the RAG service"""
        self.openai_api_key = os.getenv("OPENAI_API_KEY")
        if not self.openai_api_key:
            raise ValueError("OpenAI API key not found in environment variables")
        
        self.embeddings = OpenAIEmbeddings()
        self.main_vector_store = None
        self.llm = ChatOpenAI(model="o3")
        self.initialize_vector_store()
        
    def initialize_vector_store(self):
        """Initialize the vector store"""
        try:
            # Use the client and a specific collection name
            chroma_client = get_chroma_client()
            
            # Main collection for general knowledge
            collection_name = "standalone_api"
            
            # Check if collection exists
            try:
                collections = chroma_client.list_collections()
                logger.info(f"Available collections: {collections}")
            except Exception as e:
                logger.error(f"Error listing collections: {e}")
                collections = []
            
            # Try to use an existing collection first
            try:
                # Instead of checking if the collection exists, try to get it directly
                logger.info(f"Attempting to use collection: {collection_name}")
                self.main_vector_store = Chroma(
                    client=chroma_client,
                    collection_name=collection_name,
                    embedding_function=self.embeddings
                )
                logger.info(f"Successfully connected to collection: {collection_name}")
            except Exception as e:
                logger.warning(f"Error connecting to collection {collection_name}: {e}")
                
                # Check if we should try to use another existing collection
                if collections:
                    # Try to use the first available collection
                    alt_collection = collections[0]
                    logger.info(f"Trying alternative collection: {alt_collection}")
                    try:
                        self.main_vector_store = Chroma(
                            client=chroma_client,
                            collection_name=alt_collection,
                            embedding_function=self.embeddings
                        )
                        logger.info(f"Using alternative collection: {alt_collection}")
                    except Exception as alt_e:
                        logger.error(f"Error using alternative collection: {alt_e}")
                        raise ValueError(f"Could not initialize vector store with any collection: {e}")
                else:
                    # Fallback to local ChromaDB
                    logger.warning(f"No collections available. Using local ChromaDB.")
                    try:
                        local_client = chromadb.PersistentClient(path="./chroma_db")
                        # Don't try to create a collection, just initialize an empty one
                        self.main_vector_store = Chroma(
                            client=local_client,
                            collection_name="local_fallback",
                            embedding_function=self.embeddings
                        )
                        logger.info("Initialized local ChromaDB fallback")
                    except Exception as local_e:
                        logger.error(f"Error initializing local fallback: {local_e}")
                        raise
            
            logger.info("Vector store initialized successfully")
        except Exception as e:
            logger.error(f"Error initializing vector store: {e}")
            raise
    
    def setup_retriever(self):
        """Set up a retriever for the vector store"""
        if not self.main_vector_store:
            logger.warning("Vector store not initialized")
            return None
            
        return self.main_vector_store.as_retriever(
            search_type="mmr",  # Maximum Marginal Relevance
            search_kwargs={
                "k": 3,  # Reduced from 5 to focus on more relevant results
                "fetch_k": 8,  # Reduced from 10
                "lambda_mult": 0.8,  # Increased diversity weight to avoid repetitive documents
                "score_threshold": 0.5  # Increased relevance threshold to filter out less relevant chunks
            }
        )

    def extract_links(self, text):
        """Extract web links from text"""
        web_pattern = r'https://[^\s<>"\']+|http://[^\s<>"\']+'
        return re.findall(web_pattern, text)
        
    def check_for_lecturer_query(self, question: str) -> str:
        """
        Check if the query is asking about a specific lecturer and return their information.
        Returns None if not a lecturer query.
        """
        # Convert to lowercase for case-insensitive matching
        question_lower = question.lower()
        
        # Check for lecturer names in the query
        for lecturer_name, info in LECTURER_INFO.items():
            if lecturer_name in question_lower:
                return info
                
        # No lecturer match found
        return None
        
    def query(self, question: str) -> Dict:
        """Process a query and return the answer"""
        logger.info(f"Processing query: {question}")
        
        # First check if this is a lecturer-specific query
        lecturer_info = self.check_for_lecturer_query(question)
        if lecturer_info:
            logger.info("Found lecturer information for direct response")
            return {
                "answer": lecturer_info.strip(),
                "sources": ["dosen.md"]
            }
        
        if not self.main_vector_store:
            logger.error("Vector store not initialized")
            return {
                "answer": "Sorry, the knowledge base is not available right now. Please try again later.",
                "sources": []
            }
        
        # Set up prompt template with the super prompt from CTO
        prompt_template = """SUPER CRITICAL RULE: If the {context} contains a direct and complete answer to the user's specific {question} (e.g., it matches an FAQ entry), you MUST COPY that answer VERBATIM from the {context}. DO NOT SUMMARIZE, REPHRASE, or CHANGE IT IN ANY WAY. For example, if the question is 'Apa yang harus dilakukan mahasiswa setelah laporan skripsi disetujui (ACC) oleh dosen pembimbing?' and the context contains the full answer starting with 'Saya akan menjelaskan...' and including the link 'https://go.undiksha.ac.id/RegSidang-TI', you MUST output that exact text.

ADDITIONAL CRITICAL RULE FOR INTERNSHIP DELIVERABLES: If the user asks about 'tagihan magang', 'kewajiban magang', 'apa saja yang harus diselesaikan saat magang', or similar, you MUST find the context listing the required items (starting with '1. Proposal Magang', '2. Input Jurnal harian...', etc.) and provide that EXACT numbered list and any concluding sentences from that specific context. DO NOT provide information about conduct ('tata tertib') instead.

---

INFORMASI DOSEN PROGRAM STUDI SISTEM INFORMASI:

Prof. Dr. Gede Rasben Dantes, S.T, M.T.I.
NIP. 197502212003121001 NIDN. 0021027503 Jabatan Fungsional : Profesor Google Scholar :
Tautan 🔗 SCOPUS : Tautan 🔗 Konsentrasi : MSI Topik Riset : Enterprise Resource Planning
(ERP); E-learning dan Teknologi Pembelajaran; Pengembangan Sistem Informasi dan
Teknologi; Data Mining dan Artificial Intelligence

Ir. I Gede Mahendra Darmawiguna, S.Kom., M.Sc.
NIP. 198501042010121004 NIDN. 0004018502 Jabatan Fungsional : Lektor Google Scholar :
Tautan 🔗 SCOPUS : Tautan 🔗 Konsentrasi : RIB, MSI Topik Riset : Augmented Reality (AR)
& Virtual Reality (VR); Immersive Learning; Data Science; Sistem Pendukung Keputusan;
Information System; E- Learning

Ir. I Made Ardwi Pradnyana, S.T., M.T.
NIP. 198611182015041001 NIDN. 0818118602 Jabatan Fungsional : Lektor Google Scholar :
Tautan 🔗 SCOPUS : Tautan 🔗 Konsentrasi : MSI Topik Riset : Enterprise Information System;
Human Computer Interaction; Green IT; IT Service Management; Business Process
Management

Gede Aditra Pradnyana, S.Kom., M.Kom.
NIP. 198901192015041004 NIDN. 0819018901 Jabatan Fungsional : Lektor Google Scholar :
Tautan 🔗 SCOPUS : Tautan 🔗 Konsentrasi : RIB Topik Riset : Data Science; Text Mining;
Natural Language Processing (NLP); Machine Learning; Data Mining; Educational Technology

I Gusti Lanang Agung Raditya Putra, S.Pd., M.T.
NIP. 198908272019031008 NIDN. 0827088901 Jabatan Fungsional : Lektor Google Scholar :
Tautan 🔗 SCOPUS : - Konsentrasi : MSI Topik Riset : Enterprise Information System; IT
Governance; E-Government

Ir. I Made Edy Listartha, S.Kom., M.Kom.
NIP. 198608122019031005 NIDN. 0012088606 Jabatan Fungsional : Lektor Google Scholar :
Tautan 🔗 SCOPUS : Tautan 🔗 Konsentrasi : KS Topik Riset : Keamanan Sistem Informasi;
Jaringan Komputer; Internet of Things, Robotik, dan Social Engineering

Ir. I Made Dendi Maysanjaya, S.Pd., M.Eng.
NIP. 199005152019031008 NIDN. 0015059007 Jabatan Fungsional : Lektor Google Scholar :
Tautan 🔗 SCOPUS : Tautan 🔗 Konsentrasi : RIB Topik Riset : Kecerdasan Buatan, Sistem
Pakar, Sistem Pendukung Keputusan, Teknik Biomedis

Ir. I Gusti Ayu Agung Diatri Indradewi, S.Kom., M.T.
NIP. 198907112020122004 NIDN. 0811078901 Jabatan Fungsional : Lektor Google Scholar :
Tautan 🔗 SCOPUS : Tautan 🔗 Konsentrasi : RIB Topik Riset : Digital Image Processing;
Pattern Recognition; Information System; Machine Learning; Expert System

Ir. Gede Arna Jude Saskara, S.T., M.T.
NIP. 199105152020121003 NIDN. 0815059102 Jabatan Fungsional : Lektor Google Scholar :
Tautan 🔗 SCOPUS : Tautan 🔗 Konsentrasi : KS Topik Riset : Network; Network Security

Ir. Putu Yudia Pratiwi, S.Pd., M.Eng.
NIP. 199308042020122008 NIDN. 0004089302 Jabatan Fungsional : Lektor Google Scholar :
Tautan 🔗 SCOPUS : Tautan 🔗 Konsentrasi : MSI Topik Riset : Human Computer Interaction;
User Experience

Ir. Gede Surya Mahendra, S.Pd., M.Kom.
NIP. 199003132022031009 NIDN. 0813039004 Jabatan Fungsional : Asisten Ahli Google
Scholar : Tautan 🔗 SCOPUS : - Konsentrasi : RIB, MSI Topik Riset : Sistem Pendukung
Keputusan, Data Mining, Data Science, Teknologi Budaya, Mobile Apps

Ir. I Nyoman Tri Anindia Putra, S.Kom., M.Cs.
NIP. 199111302024061001 NIDN. 0830119102 Jabatan Fungsional : Dosen Google Scholar :
Tautan 🔗 SCOPUS : Tautan �� Konsentrasi : MSI, RIB Topik Riset : Computer Science;
Digital Heritage; Information System; Computer Vision

Putu Buddhi Prameswara, M.Kom.
NIP. 198804232024211001 NIDN. 0023048804 Jabatan Fungsional : Asisten Ahli Google
Scholar : Tautan 🔗 SCOPUS : - Konsentrasi : MSI Topik Riset : IS/IT Evaluation; Education
Technology

INFORMASI PENTING:
Koorprodi Sistem Informasi saat ini adalah Ir. I Made Dendi Maysanjaya, S.Pd., M.Eng. mulai dilantik tahun 2023.

---

Gunakan bagian-bagian konteks berikut untuk menjawab pertanyaan pengguna secara komprehensif dan akurat.
Jika Anda tidak tahu jawabannya, jangan mencoba membuat-buat jawaban. Sebagai gantinya, jawab dengan sopan dan membantu dengan:
1. Pengakuan bahwa Anda tidak memiliki informasi spesifik tentang topik tersebut
2. Tawaran untuk membantu dengan topik terkait lainnya yang mungkin Anda ketahui

INSTRUKSI KRITIS UNTUK MEMASTIKAN JAWABAN YANG SETIA DENGAN SUMBER:
- JANGAN PERNAH menambahkan informasi atau konteks yang tidak ada dalam sumber.
- JANGAN PERNAH menambahkan frasa "di Program Studi Sistem Informasi Undiksha" atau referensi spesifik ke institusi KECUALI jika eksplisit disebutkan dalam konteks.
- SELALU berikan jawaban yang bersumber HANYA dari informasi dalam konteks yang diberikan.
- PENTING: Jika konteks berisi jawaban yang LENGKAP dan LANGSUNG untuk pertanyaan spesifik pengguna (misalnya, dari daftar FAQ atau prosedur detail), prioritaskan untuk menggunakan TEKS PERSIS dari konteks tersebut, termasuk semua detail, tautan (link), dan struktur aslinya. JANGAN meringkas atau mengubah formulasi jawaban langsung ini.
- JANGAN membuat asumsi atau generalisasi di luar apa yang disebutkan dalam konteks.
- JANGAN PERNAH menyertakan sitasi sumber atau referensi dokumen (seperti "Sumber: [nama file].pdf" atau URL) dalam jawaban akhir Anda KECUALI jika secara eksplisit diminta untuk memberikan tautan dokumen.

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
- PENTING: Jika konteks memberikan jawaban langsung dalam format daftar bernomor untuk pertanyaan prosedural pengguna (seperti 'bagaimana cara...', 'apa langkah-langkah...', 'prosedur pemilihan konsentrasi'), SALIN LANGSUNG teks dan struktur daftar bernomor tersebut dari konteks sebagai jawaban Anda. JANGAN MERINGKAS atau MERUBAH FORMULASINYA.
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
- Gunakan tanda "–" untuk daftar tidak bernomor HANYA JIKA tanda "–" tersebut secara eksplisit digunakan dalam konteks. Jika konteks menggunakan format lain (seperti baris baru tanpa awalan), PERTAHANKAN format asli tersebut.
- PENTING: Jika konteks berisi langkah-langkah atau prosedur untuk pertanyaan pengguna (terutama untuk topik seperti 'cuti akademik', 'pendaftaran', 'pengajuan skripsi', 'ujian', dsb.), SELALU berikan jawaban berdasarkan langkah-langkah yang ditemukan dalam konteks tersebut. JANGAN menjawab 'tidak tahu' atau 'tidak memiliki informasi' jika prosedur yang relevan ada dalam konteks.

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
  - PERHATIAN KHUSUS: Jika pengguna bertanya tentang 'dokumen kurikulum' atau 'tautan kurikulum' dan konteks berisi daftar tautan Google Drive, Anda WAJIB menyalin daftar tersebut PERSIS seperti dalam konteks, termasuk semua tautan LENGKAP dan BENAR (seperti `https://drive.google.com/file/d/1jUQ5aIuC4H52ju9BCDZmYOT3sKU1mQG4/view` untuk Kurikulum 2024). JANGAN gunakan placeholder atau URL yang tidak lengkap. Pertahankan format daftar (misalnya, menggunakan `–` jika ada di konteks).
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
- Koorprodi Sistem Informasi saat ini adalah Ir. I Made Dendi Maysanjaya, S.Pd., M.Eng.
- SELALU sertakan informasi ini ketika ditanya tentang Koorprodi, bahkan jika tidak ditemukan secara eksplisit dalam konteks.
- Untuk pertanyaan seperti "siapa koorprodi SI sekarang?" atau "siapa koordinator prodi sistem informasi?", selalu jawab dengan informasi lengkap tentang Ir. I Made Dendi Maysanjaya, S.Pd., M.Eng.
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

Question: {question}
Context: {context}

Answer:"""

        # Create QA chain using retriever
        retriever = self.setup_retriever()
        if retriever:
            PROMPT = PromptTemplate(
                template=prompt_template,
                input_variables=["context", "question"]
            )

            qa_chain = RetrievalQA.from_chain_type(
                llm=self.llm,
                chain_type="stuff",
                retriever=retriever,
                return_source_documents=True,
                chain_type_kwargs={
                    "prompt": PROMPT
                }
            )
            
            # Process the query
            try:
                response = qa_chain({"query": question})
                result = response["result"]
                source_docs = response["source_documents"]
                
                return self.process_answer(result, source_docs, question)
            except Exception as e:
                logger.error(f"Error in query processing: {e}")
                return {
                    "answer": f"I encountered an error while processing your query. Please try again or rephrase your question.",
                    "sources": []
                }
        else:
            logger.error("No retriever available")
            return {
                "answer": "Sorry, I cannot retrieve information right now. Please try again later.",
                "sources": []
            }
    
    def process_answer(self, result, source_docs, question=""):
        """Process the answer, adding links if necessary"""
        # Extract links from source documents
        all_links = set()
        google_drive_links = set()
        
        # Track unique source files
        source_files = set()
        
        for doc in source_docs:
            # Extract links from the content
            extracted_links = self.extract_links(doc.page_content)
            all_links.update(extracted_links)
            
            # Specifically identify Google Drive links
            for link in extracted_links:
                if "drive.google.com" in link:
                    google_drive_links.add(link)
            
            # Get source file name from metadata if available
            if hasattr(doc, 'metadata') and 'source' in doc.metadata:
                source_name = doc.metadata['source']
                # Remove path prefix and file extension if present
                if isinstance(source_name, str):
                    source_name = os.path.basename(source_name)
                    source_name = os.path.splitext(source_name)[0]
                    source_files.add(source_name)
        
        # Check if the answer already includes links
        links_in_result = any(link in result for link in all_links)
        
        # Determine if we should add document links based on the content
        should_add_links = False
        
        # Keywords suggesting documentation might be needed
        doc_reference_indicators = [
            "kurikulum", "dokumen", "panduan", "pedoman", "manual", 
            "form", "formulir", "unduh", "download", "akses", "lengkap",
            "tautan", "link", "url", "informasi lebih lanjut"
        ]
        
        # Check if the question explicitly asks for documentation/links
        question_lower = question.lower() if question else ""
        result_lower = result.lower()
        
        # Only include links if the question explicitly asks for documentation
        if any(f"di{indicator}" in question_lower.replace(" ", "") for indicator in ["mana", "mna"]):
            if any(doc_term in question_lower for doc_term in ["dokumen", "file", "berkas", "link", "tautan", "akses"]):
                should_add_links = True
        
        # For specific queries about where to find information
        if any(term in question_lower for term in ["dimana", "di mana", "bagaimana cara", "akses", "dapatkan", "mendapatkan"]):
            if any(doc_term in question_lower for doc_term in ["dokumen", "file", "berkas", "kurikulum", "panduan"]):
                should_add_links = True
        
        # Only add links if the result also suggests documentation is relevant
        if should_add_links and any(indicator in result_lower for indicator in doc_reference_indicators):
            # Only now are we confident links should be included
            pass
        else:
            # Factual questions or non-document queries should not include links
            should_add_links = False
        
        # Add missing links to the response if not already included AND they are relevant
        if not links_in_result and google_drive_links and should_add_links:
            # If the result doesn't end with a period, add one
            if result and not result.endswith(('.', '!', '?')):
                result += '.'
            
            # Add a newline if there isn't one already
            if not result.endswith('\n'):
                result += '\n\n'
            else:
                result += '\n'
            
            result += "Informasi lengkap dapat diakses melalui link:\n"
            
            # Add links - prioritize Google Drive links only when they're relevant
            for link in google_drive_links:
                result += f"{link}\n"
        
        # Add source citation at the end
        if source_files:
            # Ensure proper spacing before adding source citation
            if not result.endswith('\n'):
                result += '\n\n'
            elif not result.endswith('\n\n'):
                result += '\n'
            
            # Format source names
            sources_text = ", ".join(source_files)
            result += f"Sumber data: {sources_text}"
        
        # Return the final answer and sources
        return {
            "answer": result,
            "sources": [doc.page_content for doc in source_docs]
        } 