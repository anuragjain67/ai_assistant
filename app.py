import streamlit as st
from query_controller import process_input,initialize_chain  # Import the function from the chat module

st.set_page_config(page_title="Anurag's personal assistant", page_icon="🤖", layout="centered")

if "rag_chain" not in st.session_state:
    st.session_state.rag_chain = None
    # st.session_state.rag_chain = initialize_chain()  # Initialize chain only once

if "data_name" not in st.session_state:
    st.session_state.data_name = ""

# Initialize chat history as a session state
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

if "user_input" not in st.session_state:
    st.session_state.user_input = ""


# Header section
st.title("🤖 Anurag's personal assistant")
st.write("Welcome! Please specify the data source to start the chat.")


# Input for data_name
if not st.session_state.rag_chain:
    st.session_state.data_name = st.text_input("Enter data source name:", key="data_name_input")

    if st.button("Start Chat"):
        if st.session_state.data_name:
            # Initialize the chain once after data_name is provided
            st.session_state.rag_chain = initialize_chain(st.session_state.data_name)
            st.success(f"Data source '{st.session_state.data_name}' selected. Start chatting!")
            st.rerun()
        else:
            st.error("Please provide a valid data source.")

else:

    for i, message in enumerate(st.session_state.chat_history):
        with st.chat_message(message.type):
            st.markdown(message.content)
    
    if prompt := st.chat_input("what's on your mind?"):
        st.chat_message("user").markdown(prompt)
        success, chat_history, response = process_input(st.session_state.rag_chain, st.session_state.chat_history, prompt, dummy=False)
        if success:
            st.session_state.chat_history.extend(chat_history)
        else:
            st.error("Got error from AI: retry sending the message")
        
        with st.chat_message("AI"):
            st.markdown(response)
