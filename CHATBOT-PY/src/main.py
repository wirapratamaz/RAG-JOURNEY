import os
import ssl
import streamlit as st
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain.chains import ConversationalRetrievalChain
from langchain.memory import ConversationBufferMemory
from retriever import retriever
import pandas as pd
import time

# Bypass SSL verification if needed
ssl._create_default_https_context = ssl._create_unverified_context

# Load environment variables
load_dotenv()
openai_api_key = os.getenv("OPENAI_API_KEY")

# Check if the API key is set
if not openai_api_key:
    raise ValueError("OpenAI API key not set. Please set it in the .env file")

# Initialize the OpenAI chat model
llm = ChatOpenAI(
    model_name="gpt-3.5-turbo",
    openai_api_key=openai_api_key
)

# Initialize the Conversational Retrieval Chain
memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)
qa_chain = ConversationalRetrievalChain.from_llm(
    llm=llm,
    retriever=retriever,
    memory=memory,
    return_source_documents=False
)

def chunking_and_retrieval(user_input):
    st.subheader("1. Chunking and Retrieval")
    with st.expander("Pemecahan Informasi", expanded=True):
        # Simulate chunking process
        chunks = [
            "Visi dan Misi Prodi Sistem Informasi Undiksha",
            "Kurikulum Terbaru Sistem Informasi 2023",
            "Fasilitas Lab Unggulan: AI dan Big Data",
            "Peluang Karir Lulusan Sistem Informasi",
            "Program Magang di Perusahaan Teknologi Terkemuka"
        ]
        for i, chunk in enumerate(chunks):
            st.text(f"Bagian {i+1}: {chunk}")
            time.sleep(0.5)
    
    with st.expander("Analisis Konteks", expanded=True):
        # Simulate embedding process
        data = {
            "No": range(1, 6),
            "Konten": chunks,
            "Relevansi (3 nilai teratas)": [
                [0.95, 0.88, 0.82],
                [0.91, 0.87, 0.85],
                [0.89, 0.86, 0.83],
                [0.93, 0.90, 0.87],
                [0.92, 0.89, 0.86]
            ]
        }
        df = pd.DataFrame(data)
        st.dataframe(df)
    
    st.success("Analisis informasi selesai! 🎉")

def generation(user_input):
    st.subheader("2. Generation")
    with st.spinner("Sedang menyusun jawaban terbaik untuk Anda sabar dulu yaah..."):
        # Simulate processing with a progress bar
        progress_bar = st.progress(0)
        for i in range(100):
            time.sleep(0.01)
            progress_bar.progress(i + 1)
        
        # Get response from the chain
        response = qa_chain.invoke({"question": user_input})
        answer = response["answer"]
    
    st.success("Jawaban siap! 🚀")
    return answer

def main():
    st.set_page_config(page_title="SisInfoBot Undiksha", page_icon="https://encrypted-tbn0.gstatic.com/images?q=tbn:ANd9GcSxGdjI58B_NuUd8eAWfRBlVms7f-2e2oI_SA&s", layout="wide")
    
    # Sidebar
    with st.sidebar:
        st.image("https://encrypted-tbn0.gstatic.com/images?q=tbn:ANd9GcSxGdjI58B_NuUd8eAWfRBlVms7f-2e2oI_SA&s", width=150)
        st.title("SisInfoBot Undiksha")
        st.selectbox("Model AI", ["GPT-3.5 Turbo"])
        st.text("Universitas Pendidikan Ganesha")
        st.markdown("---")
        st.markdown("### 🌟 Fitur Unggulan:")
        st.markdown("• Informasi Terkini Prodi")
        st.markdown("• Panduan Akademik")
        st.markdown("• Assiten Virtual Anda")

    # Main content
    st.header("Selamat Datang di SisInfoBot Undiksha! 🤖💻")
    st.info("Halo, Saya adalah asisten virtual khusus untuk mahasiswa Sistem Informasi Undiksha. Tanyakan apa saja seputar program studi, kurikulum, atau informasi yang berkaitan!")
    
    # Display chat history
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = []
    
    for i, message in enumerate(st.session_state.chat_history):
        if message["role"] == "user":
            st.text_area("Anda:", value=message["content"], height=50, disabled=True, key=f"user_{i}")
        else:
            st.text_area("SisInfoBot:", value=message["content"], height=100, disabled=True, key=f"bot_{i}")
    
    # User input
    user_input = st.text_input("Ada yang ingin Anda tanyakan tentang Sistem Informasi Undiksha?")
    
    if user_input:
        st.session_state.chat_history.append({"role": "user", "content": user_input})
        
        chunking_and_retrieval(user_input)
        answer = generation(user_input)
        
        st.session_state.chat_history.append({"role": "assistant", "content": answer})
        
        st.subheader("Jawaban SisInfoBot:")
        with st.container():
            st.markdown(f"""
            <div style="background-color: #f0f2f6; border-radius: 10px; padding: 20px; border-left: 5px solid #4CAF50;">
                <h4 style="color: #4CAF50;">Informasi untuk Anda:</h4>
                <p style="color: #4CAF50; font-size: 16px; line-height: 1.6;">{answer}</p>
            </div>
            """, unsafe_allow_html=True)
        
        # # Add another input field for the next question
        # st.text_input("Ada pertanyaan lain?", key="next_question")

    st.sidebar.markdown("---")
    st.sidebar.markdown("### 📢 Info Terkini:")
    st.sidebar.info("Pendaftaran program magang di Google untuk mahasiswa Sistem Informasi akan dibuka bulan depan! Siapkan CV terbaikmu!")

if __name__ == "__main__":
    main()