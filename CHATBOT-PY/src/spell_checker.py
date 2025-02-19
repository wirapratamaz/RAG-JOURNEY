import logging
from typing import Dict, Tuple, List
from textdistance import levenshtein

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class SpellChecker:
    def __init__(self):
        self.common_words = self._load_common_words()
        self.max_distance = 2  # Maximum Levenshtein distance for suggestions
        self.min_word_length = 3  # Minimum word length to consider for correction
        self.context_aware = True  # Enable context-aware corrections

    def _load_common_words(self) -> Dict[str, str]:
        """
        Load common Indonesian words, abbreviations, and their corrections
        Including academic terms, chat language, and common typos
        """
        base_words = {
            # Basic pronouns and common words
            "sya": "saya",
            "aq": "aku",
            "gw": "saya",
            # Additional academic/campus terms
            "jrsan": "jurusan",
            "prodi": "program studi", 
            "dospem": "dosen pembimbing",
            "dosbing": "dosen pembimbing",
            "doswal": "dosen wali",
            "smt": "semester",
            "smstr": "semester",
            "nilai": "nilai",
            "nilainya": "nilainya",
            "kurikulum": "kurikulum",
            "krs": "kartu rencana studi",
            "khs": "kartu hasil studi",
            "wisuda": "wisuda",
            "sidang": "sidang",
            "semhas": "seminar hasil",
            "sempro": "seminar proposal",
            "kompre": "komprehensif",
            "pkl": "praktik kerja lapangan",
            "magang": "magang",
            
            # Additional informal/chat terms
            "bls": "balas",
            "jwb": "jawab",
            "jwbn": "jawaban",
            "pndpt": "pendapat",
            "mslh": "masalah",
            "byk": "banyak",
            "sdikit": "sedikit",
            "dkit": "sedikit",
            "bsr": "besar",
            "kcil": "kecil",
            "ktmu": "ketemu",
            "ktemu": "ketemu",
            
            # Technology terms
            "app": "aplikasi",
            "apps": "aplikasi",
            "web": "website",
            "website": "website",
            "login": "login",
            "logout": "logout",
            "pass": "password",
            "pwd": "password",
            "usr": "username",
            "pic": "picture",
            "foto": "foto",
            "vid": "video",
            "link": "link",
            
            # Status/condition words
            "ok": "oke",
            "oke": "oke",
            "gpp": "tidak apa-apa",
            "gpapa": "tidak apa-apa",
            "gak papa": "tidak apa-apa",
            "sibuk": "sibuk",
            "free": "kosong",
            "kosong": "kosong",
            "full": "penuh",
            "penuh": "penuh",
            
            # Additional location terms
            "kamp": "kampus",
            "kampus": "kampus",
            "fak": "fakultas",
            "perpus": "perpustakaan",
            "lab": "laboratorium",
            "kls": "kelas",
            "ruang": "ruangan",
            "gedung": "gedung",
            "lantai": "lantai",
            
            # Time-related additions
            "hr ini": "hari ini",
            "bsk": "besok",
            "lusa": "lusa",
            "kmrn": "kemarin",
            "kmrin": "kemarin",
            "stlh": "setelah",
            "sblm": "sebelum",
            "jam": "jam",
            "menit": "menit",
            "dtk": "detik",
            
            # Document types
            "doc": "dokumen",
            "pdf": "pdf",
            "file": "file",
            "berkas": "berkas",
            "surat": "surat",
            "formulir": "formulir",
            "form": "formulir",
            
            # Additional status words
            "acc": "disetujui",
            "approved": "disetujui",
            "reject": "ditolak",
            "pending": "menunggu",
            "done": "selesai",
            "finish": "selesai",
            "proses": "proses",
        }
        
        if hasattr(self, 'common_words'):
            base_words.update(self.common_words)
            
        return base_words

    def detect_language_style(self, text: str) -> str:
        """
        Detect the language style (formal Indonesian, informal Indonesian, or English)
        based on the text content and common patterns
        """
        text = text.lower()
        words = text.split()
        
        # Count indicators for each style
        informal_indicators = sum(1 for word in words if word in ["aku", "gue", "gw", "lu", "km", "kamu", "nih", "dong", "sih"])
        english_indicators = sum(1 for word in words if all(ord(c) < 128 for c in word))
        
        # Calculate proportions
        total_words = len(words)
        informal_ratio = informal_indicators / total_words if total_words > 0 else 0
        english_ratio = english_indicators / total_words if total_words > 0 else 0
        
        if english_ratio > 0.7:
            return "english"
        elif informal_ratio > 0.3:
            return "casual"
        return "formal"

    def get_context_aware_suggestions(self, word: str, context: List[str]) -> List[str]:
        """Get spelling suggestions considering surrounding context"""
        base_suggestions = self.get_suggestions(word)
        if not context or not base_suggestions:
            return base_suggestions
            
        # Weight suggestions based on context
        weighted_suggestions = []
        for suggestion in base_suggestions:
            weight = sum(1 for ctx_word in context 
                        if ctx_word in self.common_words.values() 
                        and any(w in suggestion for w in ctx_word.split()))
            weighted_suggestions.append((suggestion, weight))
            
        return [s for s, _ in sorted(weighted_suggestions, key=lambda x: (-x[1], len(x[0])))]

    def get_suggestions(self, word: str) -> List[str]:
        """Get spelling suggestions for a word using Levenshtein distance"""
        suggestions = []
        word = word.lower()
        
        # First check if it's a common misspelling
        if word in self.common_words:
            return [self.common_words[word]]
        
        # Then check for similar words using Levenshtein distance
        for correct_word in set(self.common_words.values()):
            distance = levenshtein.distance(word, correct_word)
            if distance <= self.max_distance:
                suggestions.append(correct_word)
        
        return sorted(suggestions, key=lambda x: levenshtein.distance(word, x))

    def correct_text(self, text: str) -> Tuple[str, List[Tuple[str, str]]]:
        """
        Correct spelling mistakes in the given text with context awareness
        Returns tuple of (corrected_text, list of corrections)
        """
        words = text.lower().split()
        corrected_words = []
        corrections = []
        
        for i, word in enumerate(words):
            # Skip correction for very short words unless they're common abbreviations
            if len(word) < self.min_word_length and word not in self.common_words:
                corrected_words.append(word)
                continue
                
            if word in self.common_words:
                correction = (word, self.common_words[word])
                corrections.append(correction)
                corrected_words.append(self.common_words[word])
            else:
                # Get context (surrounding words)
                context = words[max(0, i-2):i] + words[i+1:min(len(words), i+3)]
                
                if self.context_aware:
                    suggestions = self.get_context_aware_suggestions(word, context)
                else:
                    suggestions = self.get_suggestions(word)
                    
                if suggestions and levenshtein.distance(word, suggestions[0]) <= self.max_distance:
                    correction = (word, suggestions[0])
                    corrections.append(correction)
                    corrected_words.append(suggestions[0])
                else:
                    corrected_words.append(word)
        
        return " ".join(corrected_words), corrections

    def format_corrections(self, corrections: List[Tuple[str, str]], style: str = "formal") -> str:
        """Format spelling corrections in a user-friendly way with improved formatting"""
        if not corrections:
            return ""
        
        if style == "formal":
            header = "Beberapa kata yang mungkin Anda maksud:\n"
            footer = "\nSilakan gunakan kata-kata formal dalam konteks akademik untuk komunikasi yang lebih baik."
        elif style == "casual":
            header = "Sepertinya ada beberapa kata yang bisa diperbaiki nih:\n"
            footer = "\nTips: Kamu bisa pakai kata-kata yang lebih formal kalau lagi komunikasi sama dosen ya! 😊"
        else:  # English
            header = "I noticed some words that could be corrected:\n"
            footer = "\nTip: Using formal language is recommended for academic communication."
            
        correction_text = header
        
        # Group corrections by category
        academic_terms = []
        chat_terms = []
        other_terms = []
        
        for original, corrected in corrections:
            if any(term in corrected for term in ["mahasiswa", "dosen", "kuliah", "semester", "skripsi"]):
                academic_terms.append((original, corrected))
            elif any(term in original for term in ["gw", "lu", "km", "bs", "ga"]):
                chat_terms.append((original, corrected))
            else:
                other_terms.append((original, corrected))
        
        # Add corrections by category
        if academic_terms:
            correction_text += "\nIstilah akademik:\n"
            for original, corrected in academic_terms:
                correction_text += f"• '{original}' → '{corrected}'\n"
                
        if chat_terms:
            correction_text += "\nBahasa informal:\n"
            for original, corrected in chat_terms:
                correction_text += f"• '{original}' → '{corrected}'\n"
                
        if other_terms:
            correction_text += "\nKata-kata umum:\n"
            for original, corrected in other_terms:
                correction_text += f"• '{original}' → '{corrected}'\n"
        
        return correction_text + footer

# Example usage
if __name__ == "__main__":
    spell_checker = SpellChecker()
    
    # Test cases
    test_texts = [
        "sya mhs undiksa mau tnya ttg skripsi dgn dosen2",
        "gmn cara daftar matkul di sifo",
        "kpn jadwal uas smt ini dimulai"
    ]
    
    for text in test_texts:
        print(f"\nOriginal text: {text}")
        corrected_text, corrections = spell_checker.correct_text(text)
        print("Corrections:")
        print(spell_checker.format_corrections(corrections, "casual"))
        print(f"Corrected text: {corrected_text}") 