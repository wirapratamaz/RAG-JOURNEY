import os
import ssl
import streamlit as st
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain.chains import ConversationalRetrievalChain
from langchain.memory import ConversationBufferMemory
from retriever import retriever

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

def submit_input():
    user_input = st.session_state.user_input
    if user_input:
        st.session_state.chat_history.append({"type": "human", "content": user_input})

        # Get response from the chain
        response = qa_chain.invoke({"question": user_input})

        answer = response["answer"]
        st.session_state.chat_history.append({"type": "assistant", "content": answer})

        # Clear the input field
        st.session_state.user_input = ""

def main():
    st.set_page_config(page_title="EduBot", page_icon="🤖")
    st.title("Hi, I am EduBot for Sistem Information Help")
    st.info(
        "I am here to answer any question you may have about Sistem Information Undiksha. How may I help you today?"
    )

    if "chat_history" not in st.session_state:
        st.session_state.chat_history = []

    # Display chat history
    for message in st.session_state.chat_history:
        if message["type"] == "human":
            with st.chat_message("user"):
                st.markdown(message["content"])
        else:
            with st.chat_message("assistant"):
                st.markdown(message["content"])

    # Get user input with callback
    st.text_input("Start typing...", key="user_input", on_change=submit_input)

if __name__ == "__main__":
    main()