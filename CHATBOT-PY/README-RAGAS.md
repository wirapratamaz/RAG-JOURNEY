# RAGAS Evaluation Framework

<div align="center">
  <h3>Evaluation Framework for Retrieval Augmented Generation Systems</h3>
</div>

## 📋 Overview

RAGAS (Retrieval Augmented Generation Assessment) is a framework for evaluating RAG (Retrieval Augmented Generation) systems. It provides a set of metrics to assess different aspects of RAG performance, helping developers identify strengths and weaknesses in their implementations.

## 🔍 Key Metrics

RAGAS offers several metrics to evaluate different aspects of RAG systems:

### Retrieval Metrics
- **Context Precision**: Measures how much of the retrieved context is relevant to the query
- **Context Recall**: Measures how much of the necessary information is present in the retrieved context
- **Context Relevancy**: Evaluates the relevance of retrieved documents to the query

### Generation Metrics
- **Faithfulness**: Measures factual consistency between the generated response and retrieved context
- **Answer Relevancy**: Assesses how well the generated response answers the query
- **Contextual Precision**: Evaluates how precisely the response uses information from the context

### Other Metrics
- **Hallucination**: Detects fabricated information not present in the context
- **Coherence**: Evaluates the logical flow and readability of the response
- **Multimodal Metrics**: For evaluating systems with image, audio, or video content

## 🧪 Sample Evaluation

Below is a sample evaluation of our RAG system's response to a user query about thesis completion steps in the Information Systems Study Program.

### Query
> Apa saja tahapan utama dalam proses penyelesaian skripsi di Program Studi Sistem Informasi?

### Response
> Terdapat empat tahap utama dalam proses penyelesaian skripsi di Program Studi Sistem Informasi, yaitu:
> 
> **Pengajuan Topik**: Tahap pertama adalah pengajuan topik skripsi, di mana mahasiswa mengusulkan topik skripsi melalui sistem karya akhir dengan melengkapi Formulir Usulan Topik Skripsi.
> 
> **Ujian Proposal Skripsi**: Setelah pengajuan topik, mahasiswa akan menjalani Ujian Proposal Skripsi di mana mereka akan mempresentasikan proposal skripsi mereka kepada dosen pembimbing dan dosen penguji.
> 
> **Penyusunan Skripsi**: Tahap ini meliputi penyusunan skripsi yang mencakup penulisan skripsi serta proses bimbingan dan revisi dari dosen pembimbing.
> 
> **Pengesahan Laporan Skripsi**: Tahap terakhir adalah pengesahan Laporan Skripsi setelah proses penyusunan selesai dan laporan skripsi disetujui oleh dosen pembimbing.
> 
> Setiap tahap memiliki prosedur khusus yang melibatkan peran mahasiswa, dosen pembimbing, dosen penguji, koordinator program studi, dan ketua jurusan dalam mendukung kelancaran penyelesaian skripsi mahasiswa.

### Evaluation Results

<div style="display: flex; flex-wrap: wrap; gap: 20px; margin-bottom: 30px;">
  <div style="flex: 1; min-width: 300px; border: 1px solid #ccc; border-radius: 8px; padding: 15px; background-color: #f0f0f0; box-shadow: 0 2px 4px rgba(0,0,0,0.1);">
    <h3 style="margin-top: 0; color: #2c3e50; border-bottom: 2px solid #4CAF50; padding-bottom: 5px;">Context Precision</h3>
    <div style="display: flex; align-items: center; margin-bottom: 10px;">
      <div style="flex-grow: 1; height: 12px; background-color: #e0e0e0; border-radius: 6px; box-shadow: inset 0 1px 3px rgba(0,0,0,0.2);">
        <div style="width: 90%; height: 100%; background-color: #4CAF50; border-radius: 6px;"></div>
      </div>
      <span style="margin-left: 10px; font-weight: bold; color: #333;">9/10</span>
    </div>
    <p style="color: #333; margin-top: 15px;">The chatbot's response uses information directly from the retrieved context without including irrelevant information. All four main stages of thesis completion are accurately presented, and the explanation of each stage is concise and relevant to the question.</p>
  </div>

  <div style="flex: 1; min-width: 300px; border: 1px solid #ccc; border-radius: 8px; padding: 15px; background-color: #f0f0f0; box-shadow: 0 2px 4px rgba(0,0,0,0.1);">
    <h3 style="margin-top: 0; color: #2c3e50; border-bottom: 2px solid #4CAF50; padding-bottom: 5px;">Context Recall</h3>
    <div style="display: flex; align-items: center; margin-bottom: 10px;">
      <div style="flex-grow: 1; height: 12px; background-color: #e0e0e0; border-radius: 6px; box-shadow: inset 0 1px 3px rgba(0,0,0,0.2);">
        <div style="width: 90%; height: 100%; background-color: #4CAF50; border-radius: 6px;"></div>
      </div>
      <span style="margin-left: 10px; font-weight: bold; color: #333;">9/10</span>
    </div>
    <p style="color: #333; margin-top: 15px;">The response successfully recalls all the key information from the source document: all four main stages, the involvement of key stakeholders, and the procedural nature of each stage.</p>
  </div>

  <div style="flex: 1; min-width: 300px; border: 1px solid #ccc; border-radius: 8px; padding: 15px; background-color: #f0f0f0; box-shadow: 0 2px 4px rgba(0,0,0,0.1);">
    <h3 style="margin-top: 0; color: #2c3e50; border-bottom: 2px solid #4CAF50; padding-bottom: 5px;">Response Relevancy</h3>
    <div style="display: flex; align-items: center; margin-bottom: 10px;">
      <div style="flex-grow: 1; height: 12px; background-color: #e0e0e0; border-radius: 6px; box-shadow: inset 0 1px 3px rgba(0,0,0,0.2);">
        <div style="width: 100%; height: 100%; background-color: #4CAF50; border-radius: 6px;"></div>
      </div>
      <span style="margin-left: 10px; font-weight: bold; color: #333;">10/10</span>
    </div>
    <p style="color: #333; margin-top: 15px;">The answer is highly relevant to the user's question. The response directly addresses "tahapan utama dalam proses penyelesaian skripsi" with a structured explanation of each stage. The answer is complete and doesn't contain redundant information.</p>
  </div>

  <div style="flex: 1; min-width: 300px; border: 1px solid #ccc; border-radius: 8px; padding: 15px; background-color: #f0f0f0; box-shadow: 0 2px 4px rgba(0,0,0,0.1);">
    <h3 style="margin-top: 0; color: #2c3e50; border-bottom: 2px solid #4CAF50; padding-bottom: 5px;">Faithfulness</h3>
    <div style="display: flex; align-items: center; margin-bottom: 10px;">
      <div style="flex-grow: 1; height: 12px; background-color: #e0e0e0; border-radius: 6px; box-shadow: inset 0 1px 3px rgba(0,0,0,0.2);">
        <div style="width: 90%; height: 100%; background-color: #4CAF50; border-radius: 6px;"></div>
      </div>
      <span style="margin-left: 10px; font-weight: bold; color: #333;">9/10</span>
    </div>
    <p style="color: #333; margin-top: 15px;">The response is faithful to the source document and doesn't introduce information that isn't supported by the context. The chatbot has maintained factual accuracy while expanding slightly on each stage to provide a more comprehensive explanation.</p>
  </div>
</div>

### Overall Assessment

<div style="border: 1px solid #ccc; border-radius: 8px; padding: 20px; background-color: #e8f4fd; margin-bottom: 30px; box-shadow: 0 2px 4px rgba(0,0,0,0.1);">
  <h3 style="margin-top: 0; color: #0d47a1; border-bottom: 2px solid #0d47a1; padding-bottom: 5px;">Summary</h3>
  <p style="color: #333;">The chatbot has performed excellently in answering this question. The response is:</p>
  <ul style="color: #333;">
    <li><span style="color: #4CAF50; font-weight: bold;">✅</span> Comprehensive and well-structured</li>
    <li><span style="color: #4CAF50; font-weight: bold;">✅</span> Faithful to the source material</li>
    <li><span style="color: #4CAF50; font-weight: bold;">✅</span> Directly relevant to the user's question</li>
    <li><span style="color: #4CAF50; font-weight: bold;">✅</span> Includes all key information from the context</li>
  </ul>
  <p style="color: #333;">The answer demonstrates effective retrieval and generation capabilities, providing the user with accurate and complete information about thesis completion steps in the Information Systems Study Program.</p>
</div>

## 📊 How to Use RAGAS

### Installation

```bash
pip install ragas
```

### Basic Usage

```python
from ragas.metrics import (
    faithfulness,
    answer_relevancy,
    context_precision,
    context_recall
)

# Define your evaluation dataset
eval_dataset = [
    {
        "question": "Apa saja tahapan utama dalam proses penyelesaian skripsi di Program Studi Sistem Informasi?",
        "answer": "Terdapat empat tahap utama dalam proses penyelesaian skripsi...",
        "contexts": ["Proses penyelesaian skripsi terdiri dari empat tahap utama..."]
    }
]

# Create an evaluator with the metrics you want to use
from ragas import evaluate
result = evaluate(
    eval_dataset,
    metrics=[
        faithfulness,
        answer_relevancy,
        context_precision,
        context_recall
    ]
)

# View the results
print(result)
```

## 🔗 Resources

- [RAGAS Documentation](https://docs.ragas.io/)
- [GitHub Repository](https://github.com/explodinggradients/ragas)
- [Paper: RAGAS: Automated Evaluation of Retrieval Augmented Generation](https://arxiv.org/abs/2309.15217)

## 📝 License

RAGAS is released under the [MIT License](https://github.com/explodinggradients/ragas/blob/main/LICENSE). 