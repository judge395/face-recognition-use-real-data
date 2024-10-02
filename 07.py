# Run this file using : streamlit run 07.py
import os
import uuid
import streamlit as st
from dotenv import load_dotenv
from langchain_groq import ChatGroq
from langchain_core.messages import HumanMessage
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain_core.chat_history import (
    BaseChatMessageHistory,
    InMemoryChatMessageHistory,
)

# Load environment variables
load_dotenv()
os.environ['GROQ_API_KEY'] = os.getenv("GROQ_API_KEY")
groq_api_key = os.getenv("GROQ_API_KEY")

# Initialize the language model
llm = ChatGroq(groq_api_key=groq_api_key, model_name="Llama3-8b-8192")

# Maintain a session id
if 'session_id' not in st.session_state:
    st.session_state.session_id = str(uuid.uuid4())

# Set up message history store
if 'store' not in st.session_state:
    st.session_state.store = {}

def get_session_history(session_id: str) -> BaseChatMessageHistory:
    if session_id not in st.session_state.store:
        st.session_state.store[session_id] = InMemoryChatMessageHistory()
    return st.session_state.store[session_id]

with_message_history = RunnableWithMessageHistory(llm, get_session_history)

def generate_response(prompt, last_message=False):
    conversation_history = get_session_history(st.session_state.session_id)
    
    if last_message:
        return conversation_history

    config = {"configurable": {"session_id": st.session_state.session_id}}

    response = with_message_history.invoke(
        [HumanMessage(content=prompt)],
        config=config,
    )
    return response

# Streamlit UI
st.title("AI Assistant Judge")

if 'history' not in st.session_state:
    st.session_state.history = []

if 'messages' not in st.session_state:
    st.session_state.messages = []

def submit_message():
    user_input = st.session_state.user_input
    if user_input.lower() == "exit":
        st.session_state.history = generate_response(prompt=user_input, last_message=True)
        st.session_state.messages.append({"role": "user", "content": user_input})
        st.session_state.messages.append({"role": "assistant", "content": "Session ended. Goodbye!"})
        st.session_state.user_input = ""
        return
    
    result = generate_response(prompt=user_input)
    st.session_state.messages.append({"role": "user", "content": user_input})
    st.session_state.messages.append({"role": "assistant", "content": result.content})
    st.session_state.user_input = ""

st.text_input("You:", key="user_input", on_change=submit_message)

for message in st.session_state.messages:
    if message['role'] == "user":
        st.write(f"You: {message['content']}")
    else:
        st.write(f"Assistant: {message['content']}")

if st.session_state.history:
    st.write("<-------------- History -------------->")
    for msg in st.session_state.history.messages:
        st.write(f"{msg['role']}: {msg['content']}")
