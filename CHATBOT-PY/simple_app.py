import os
import streamlit as st
from dotenv import load_dotenv
import openai
from langchain_openai import ChatOpenAI
import time

# Load environment variables
load_dotenv()

# Set the OpenAI API key
api_key = st.secrets.get("OPENAI_API_KEY", os.getenv("OPENAI_API_KEY"))
if not api_key:
    st.error("OpenAI API key not found. Please add it to your .streamlit/secrets.toml file or .env file.")
    st.stop()

# Initialize OpenAI client
client = openai.Client(api_key=api_key)

def main():
    st.set_page_config(
        page_title="Simple RAG Chatbot",
        page_icon="🤖",
        layout="wide"
    )
    
    # Theme-aware CSS that respects light/dark mode
    st.markdown("""
        <style>
        /* Custom styling for chat elements that respects theme */
        .chat-message {
            padding: 1rem;
            border-radius: 0.5rem;
            margin-bottom: 1rem;
            display: flex;
        }
        
        /* Custom styling for user input area */
        .stTextInput > div > div > input {
            border-radius: 20px;
        }
        </style>
    """, unsafe_allow_html=True)
    
    # Initialize session state for chat history
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = []
        
    # Sidebar
    with st.sidebar:
        st.image("https://encrypted-tbn0.gstatic.com/images?q=tbn:ANd9GcSxGdjI58B_NuUd8eAWfRBlVms7f-2e2oI_SA&s", width=150)
        st.title("Simple RAG Chatbot")
        st.text("A simplified version for testing")
        st.markdown("---")
        
        # Add a note about the full version
        st.info("This is a simplified version of the app for testing deployment.")
        
    # Main content area
    if not st.session_state.chat_history:
        st.header("Welcome to the Simple RAG Chatbot! 🤖")
        st.info("This is a simplified version for testing Streamlit deployment. Ask me anything!")

    # Chat container
    chat_container = st.container()
    with chat_container:
        for message in st.session_state.chat_history:
            with st.chat_message(message["role"]):
                st.markdown(message["content"])

    # Chat input
    user_input = st.chat_input("Type your question here...")
    
    if user_input:
        # Add user message to chat history
        st.session_state.chat_history.append({"role": "user", "content": user_input})
        
        # Display user message
        with st.chat_message("user"):
            st.markdown(user_input)
            
        # Display assistant response with a spinner
        with st.chat_message("assistant"):
            with st.spinner("Thinking..."):
                try:
                    # Generate response using OpenAI
                    response = client.chat.completions.create(
                        model="gpt-3.5-turbo",
                        messages=[
                            {"role": "system", "content": "You are a helpful assistant."},
                            {"role": "user", "content": user_input}
                        ],
                        temperature=0.7,
                        max_tokens=500
                    )
                    
                    answer = response.choices[0].message.content
                    
                    # Use a delay with progress to simulate processing
                    progress_bar = st.progress(0)
                    for i in range(100):
                        time.sleep(0.01)
                        progress_bar.progress(i + 1)
                    
                    # Display the answer
                    st.markdown(answer)
                    
                    # Add assistant response to chat history
                    st.session_state.chat_history.append({"role": "assistant", "content": answer})
                    
                except Exception as e:
                    st.error(f"Error generating response: {e}")
                    st.session_state.chat_history.append({"role": "assistant", "content": f"I encountered an error: {e}"})
        
        # Use st.rerun() to update the UI
        # st.rerun()  # Uncomment if needed

if __name__ == "__main__":
    main() 
