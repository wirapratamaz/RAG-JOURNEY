import os
import logging
from dotenv import load_dotenv
import pandas as pd
import numpy as np
from ragas import evaluate
from ragas.metrics import (
    faithfulness,
    answer_relevancy,
    context_precision,
    context_recall
)
from datasets import Dataset
from langchain_openai import ChatOpenAI

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Load environment variables
load_dotenv()
openai_api_key = os.getenv("OPENAI_API_KEY")

# Check if the API key is set
if not openai_api_key:
    logger.error("OpenAI API key not set. Please set it in the .env file")
    exit(1)

# Set OpenAI API key for Ragas
os.environ["OPENAI_API_KEY"] = openai_api_key

# Define the test data
test_data = {
    "question": [
        "Apa saja tahapan utama dalam proses penyelesaian skripsi di Program Studi Sistem Informasi?"
    ],
    "answer": [
        """Terdapat empat tahap utama dalam proses penyelesaian skripsi di Program Studi Sistem Informasi, yaitu:

Pengajuan Topik: Tahap pertama adalah pengajuan topik skripsi, di mana mahasiswa mengusulkan topik skripsi melalui sistem karya akhir dengan melengkapi Formulir Usulan Topik Skripsi.

Ujian Proposal Skripsi: Setelah pengajuan topik, mahasiswa akan menjalani Ujian Proposal Skripsi di mana mereka akan mempresentasikan proposal skripsi mereka kepada dosen pembimbing dan dosen penguji.

Penyusunan Skripsi: Tahap ini meliputi penyusunan skripsi yang mencakup penulisan skripsi serta proses bimbingan dan revisi dari dosen pembimbing.

Pengesahan Laporan Skripsi: Tahap terakhir adalah pengesahan Laporan Skripsi setelah proses penyusunan selesai dan laporan skripsi disetujui oleh dosen pembimbing.

Setiap tahap memiliki prosedur khusus yang melibatkan peran mahasiswa, dosen pembimbing, dosen penguji, koordinator program studi, dan ketua jurusan dalam mendukung kelancaran penyelesaian skripsi mahasiswa."""
    ],
    "contexts": [
        [
            """=== End of DOSEN.pdf ===

=== Start of Karya Akhir.pdf ===
1. Apa saja tahapan utama dalam proses penyelesaian skripsi di Program Studi Sistem Informasi?
Jawaban: Proses penyelesaian skripsi terdiri dari empat tahap utama:
• Pengajuan Topik
• Ujian Proposal Skripsi
• Penyusunan Skripsi (termasuk bimbingan dan revisi)
• Pengesahan Laporan Skripsi
Setiap tahap memiliki prosedur khusus yang melibatkan mahasiswa, dosen pembimbing (termasuk dosen penguji), koordinator program studi, dan ketua jurusan."""
        ]
    ],
    "reference": [
        """Proses penyelesaian skripsi terdiri dari empat tahap utama:
• Pengajuan Topik
• Ujian Proposal Skripsi
• Penyusunan Skripsi (termasuk bimbingan dan revisi)
• Pengesahan Laporan Skripsi
Setiap tahap memiliki prosedur khusus yang melibatkan mahasiswa, dosen pembimbing (termasuk dosen penguji), koordinator program studi, dan ketua jurusan."""
    ]
}

# Convert to Dataset
dataset = Dataset.from_dict(test_data)

# Run evaluation
try:
    # Define metrics
    metrics = [
        faithfulness,
        answer_relevancy,
        context_precision,
        context_recall
    ]
    
    # Evaluate
    result = evaluate(
        dataset=dataset,
        metrics=metrics
    )
    
    # Print results
    print("\n=== RAGAS Evaluation Results ===")
    
    # Extract scores
    faithfulness_score = float(result['faithfulness'][0]) if isinstance(result['faithfulness'], list) else float(result['faithfulness'])
    answer_relevancy_score = float(result['answer_relevancy'][0]) if isinstance(result['answer_relevancy'], list) else float(result['answer_relevancy'])
    context_precision_score = float(result['context_precision'][0]) if isinstance(result['context_precision'], list) else float(result['context_precision'])
    context_recall_score = float(result['context_recall'][0]) if isinstance(result['context_recall'], list) else float(result['context_recall'])
    
    print(f"Faithfulness: {faithfulness_score:.4f}")
    print(f"Answer Relevancy: {answer_relevancy_score:.4f}")
    print(f"Context Precision: {context_precision_score:.4f}")
    print(f"Context Recall: {context_recall_score:.4f}")
    
    # Calculate overall score
    overall_score = (
        faithfulness_score +
        answer_relevancy_score +
        context_precision_score +
        context_recall_score
    ) / 4
    
    print(f"Overall Score: {overall_score:.4f}")
    
    # Save results to CSV
    result_df = pd.DataFrame({
        'Metric': ['Faithfulness', 'Answer Relevancy', 'Context Precision', 'Context Recall', 'Overall'],
        'Score': [
            faithfulness_score,
            answer_relevancy_score,
            context_precision_score,
            context_recall_score,
            overall_score
        ]
    })
    
    result_df.to_csv('ragas_evaluation_results.csv', index=False)
    print("Results saved to ragas_evaluation_results.csv")
    
except Exception as e:
    logger.error(f"Error during evaluation: {e}")
    print(f"Error during evaluation: {e}") 