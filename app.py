import streamlit as st
import tempfile
import os
from ai.db import db_obj
from ai.query import query  # Import the function from the chat module

st.set_page_config(page_title="AI Powered, Assistant", page_icon="🤖", layout="centered")

# Initialize chat history as a session state
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

if "user_input" not in st.session_state:
    st.session_state.user_input = ""


# Header section
st.title("🤖 Anurag's personal assistant")
st.write("Welcome! Please specify the data source to start the chat.")


for i, message in enumerate(st.session_state.chat_history):
    with st.chat_message(message.type):
        st.markdown(message.content)

if prompt := st.chat_input("what's on your mind?"):
    st.chat_message("user").markdown(prompt)
    success, chat_history, response = query(st.session_state.chat_history, prompt, dummy=False)
    if success:
        st.session_state.chat_history.extend(chat_history)
    else:
        st.error("Got error from AI: retry sending the message")
    
    with st.chat_message("AI"):
        st.markdown(response)


uploaded_file = st.file_uploader("Choose a file")
if uploaded_file is not None:
    filename = uploaded_file.name
    st.write("filename:", filename)
    if uploaded_file:
        st.write("uploading file ...") 
        temp_dir = tempfile.mkdtemp()
        filepath = os.path.join(temp_dir, filename)
        with open(filepath, "wb") as f:
            f.write(uploaded_file.getvalue())
        
        db_obj.add_document_from_file(filepath)
        st.write("file uploaded successfully")
