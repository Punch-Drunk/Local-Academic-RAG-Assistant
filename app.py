import streamlit as st
import os
import sys
from pathlib import Path

# Add src directory to path
sys.path.insert(0, str(Path(__file__).parent / "src"))

from query_engine import QueryEngine

# Page config
st.set_page_config(
    page_title="Academic RAG Assistant",
    page_icon="📚",
    layout="wide"
)

# Initialize session state
if "query_engine" not in st.session_state:
    with st.spinner("Initializing query engine..."):
        # Use path relative to project root
        persist_dir = os.path.join(os.path.dirname(__file__), "chroma_db")
        st.session_state.query_engine = QueryEngine(persist_directory=persist_dir)

if "agent" not in st.session_state:
    with st.spinner("Creating agent..."):
        st.session_state.agent = st.session_state.query_engine.create_agent()

if "messages" not in st.session_state:
    st.session_state.messages = []

# Load assistant image
assistant_image_path = os.path.join(os.path.dirname(__file__), "src", "unnamed.jpg")
assistant_image = None
if os.path.exists(assistant_image_path):
    assistant_image = assistant_image_path

# Sidebar
with st.sidebar:
    st.title("📚 Academic RAG Assistant")
    st.markdown("Ask questions about your course materials!")
    
    if assistant_image:
        st.image(assistant_image, width=200, caption="Your Assistant")
    
    st.markdown("---")
    st.markdown("### About")
    st.markdown("This assistant can answer questions based on your course readings and lectures using RAG (Retrieval-Augmented Generation).")
    
    if st.button("Clear Chat History"):
        st.session_state.messages = []
        st.rerun()

# Main content
st.title("Academic RAG Assistant")
st.markdown("Ask me anything about your course materials!")

# Display chat history
for message in st.session_state.messages:
    with st.chat_message(message["role"], avatar=assistant_image if message["role"] == "assistant" else None):
        st.markdown(message["content"])

# Chat input
user_query = st.chat_input("Type your question here...")

if user_query:
    # Add user message to chat history
    st.session_state.messages.append({"role": "user", "content": user_query})
    
    # Display user message
    with st.chat_message("user"):
        st.markdown(user_query)
    
    # Get response from agent
    with st.chat_message("assistant", avatar=assistant_image):
        with st.spinner("Thinking..."):
            try:
                response = st.session_state.query_engine.query(
                    st.session_state.agent,
                    user_query
                )
                st.markdown(response)
                # Add assistant response to chat history
                st.session_state.messages.append({"role": "assistant", "content": response})
            except Exception as e:
                error_message = f"Error: {str(e)}"
                st.error(error_message)
                st.session_state.messages.append({"role": "assistant", "content": error_message})

