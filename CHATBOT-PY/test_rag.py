import pytest
from deepeval import assert_test
from deepeval.metrics.ragas import (
    RAGASContextualPrecisionMetric,
    RAGASFaithfulnessMetric,
    RAGASContextualRecallMetric,
    RAGASAnswerRelevancyMetric,
)
from deepeval.metrics import BiasMetric
from deepeval.test_case import LLMTestCase

#######################################
# Initialize metrics with thresholds ##
#######################################
context_precision = RAGASContextualPrecisionMetric(threshold=0.5)
context_recall = RAGASContextualRecallMetric(threshold=0.5)
response_relevancy = RAGASAnswerRelevancyMetric(threshold=0.5)
faithfulness = RAGASFaithfulnessMetric(threshold=0.5)

#######################################
# Specify evaluation metrics to use ###
#######################################
evaluation_metrics = [
  context_precision,
  context_recall,
  response_relevancy,
  faithfulness
]

#######################################
# Specify inputs to test RAG app on ###
#######################################
input_output_pairs = [
    {
        "input": "Apa status akreditasi dari program Sistem Informasi?",
        "expected_output": "Program Sistem Informasi di Universitas Pendidikan Ganesha (Undiksha) terakreditasi pada tahun 2021 dan menerima peringkat akreditasi 'Baik'. Undiksha sendiri memperoleh akreditasi 'Unggul' pada tahun 2018."
    },
    {
        "input": "Kapan program Sistem Informasi ini didirikan?",
        "expected_output": "Program Sistem Informasi adalah program sarjana (S-1) baru yang diusulkan pada 5 September 2017, dan secara resmi dibuka sesuai dengan Peraturan Menteri Riset, Teknologi, dan Pendidikan Tinggi No. 116/KPT/I/2018 pada 2 Februari 2018."
    },
    {
        "input": "Bagaimana profil dosen di Program Studi Sistem Informasi dan apa saja kualifikasi yang mereka miliki?",
        "expected_output": "Program Studi Sistem Informasi di Undiksha didukung oleh tim dosen yang berkompeten dan berpengalaman di berbagai bidang keilmuan terkait. Berikut adalah profil beberapa dosen utama beserta kualifikasi mereka:\n\n1. I Made Ardwi Pradnyana, S.T., M.T.: \n   - NIP: 198611182015041001 \n   - NIDN: 0818118602 \n   - Identitas Peneliti: Scopus Author ID: 57202607891, Google Scholar ID: [Link](https://scholar.google.co.id/citations?user=u_6QpjIAAAAJ&hl=en) \n   - ORCID iD: [Link](https://orcid.org/0000-0002-2076-5129) \n   - Research Interest: Enterprise Information Systems, Human Computer Interaction, Green IT\n\n2. I Gede Mahendra Darmawiguna, S.Kom., M.Sc.: \n   - NIP: 198501042010121004 \n   - NIDN: 0004018502 \n   - Identitas Peneliti: Scopus Author ID: 56912388700, Google Scholar ID: [Link](https://scholar.google.co.id/citations?user=TYNXprAAAAAJ&hl=en) \n   - ORCID iD: [Link](https://orcid.org/0000-0002-4368-580X) \n   - Research Interest: Data Science, Sistem Pendukung Keputusan, Information System, E-Learning, AR & VR, Immersive Learning\n\n3. I Gusti Lanang Agung Raditya Putra, S.Pd., M.T.: \n   - NIP: 198908272019031008 \n   - NIDN: 0827088901 \n   ...(line too long; chars omitted)"
    },
    {
        "input": "Apa itu Program International Virtual Summer School (IVSS) di Program Studi Sistem Informasi?",
        "expected_output": "International Virtual Summer School (IVSS) adalah inisiatif pendidikan yang dirancang oleh Universitas Pendidikan Ganesha (Undiksha) untuk memberikan kesempatan belajar tambahan kepada mahasiswa selama pandemi Covid-19. Berikut adalah detail mengenai proses dan syarat untuk mengikuti IVSS."
    },
    {
        "input": "Apa saja konsentrasi bidang keilmuan yang tersedia di Program Studi Sistem Informasi dan apa fokus masing-masing konsentrasi?",
        "expected_output": "Program Studi Sistem Informasi (SI) di Universitas Pendidikan Ganesha (Undiksha) menawarkan tiga konsentrasi bidang keilmuan yang dapat dipilih oleh mahasiswa sesuai minat dan bakat mereka. Pemilihan konsentrasi ini memungkinkan mahasiswa untuk mengarahkan ide topik skripsi dan mengembangkan kompetensi spesifik yang relevan dengan jenis konsentrasi yang dipilih. Berikut adalah ketiga konsentrasi tersebut:\n\n1. Konsentrasi Manajemen Sistem Informasi (MSI):\n   - Fokus: Pengelolaan dan pengembangan sistem informasi yang mendukung operasi dan strategi bisnis organisasi.\n   - Bidang Penelitian: Perencanaan strategis TI, tata kelola TI, manajemen proyek sistem informasi, dan akuisisi teknologi informasi yang tepat guna.\n\n2. Konsentrasi Rekayasa dan Kecerdasan Bisnis (RIB):\n   - Fokus: Pemanfaatan data untuk mendukung analisis bisnis dan organisasi, serta transformasi data menjadi pengetahuan bermakna untuk pengambilan keputusan strategis.\n   - Bidang Penelitian: Business Analytic, Data Management, Computerized Decision Support, Intelligent Systems.\n\n3. Konsentrasi Keamanan Siber (KS):\n   - Fokus: Praktik melindungi sistem, jaringan, dan program dari serangan digital untuk memastikan keamanan dan integritas informasi.\n   - Bidang Penelitian: Manajemen Identitas, Privasi, dan Kepercayaan, Perangkat Lunak Perusak, Biometrik, Keamanan Awan, Forensik Komputer."
    }
]

#######################################
# Loop through input output pairs #####
#######################################
@pytest.mark.parametrize(
    "input_output_pair",
    input_output_pairs,
)
def test_llamaindex(input_output_pair: Dict):
    input = input_output_pair.get("input", None)
    expected_output = input_output_pair.get("expected_output", None)

    # Hypothentical RAG application for demonstration only. 
    # Replace this with your own RAG implementation.
    # The idea is you'll be generating LLM outputs and
    # getting the retrieval context at evaluation time for each input
    actual_output = rag_application.query(input)
    retrieval_context = rag_application.get_retrieval_context()

    test_case = LLMTestCase(
        input=input,
        actual_output=actual_output,
        retrieval_context=retrieval_context,
        expected_output=expected_output
    )
    # assert test case
    assert_test(test_case, evaluation_metrics)