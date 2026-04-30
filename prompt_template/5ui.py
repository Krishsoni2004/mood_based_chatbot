# streamlit_app.py

import os
import streamlit as st
from dotenv import load_dotenv

from langchain_huggingface import HuggingFaceEndpoint, ChatHuggingFace
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage

# -----------------------
# Load ENV
# -----------------------
load_dotenv()
hf_token = os.getenv("HUGGINGFACEHUB_API_TOKEN")

# -----------------------
# Page Config
# -----------------------
st.set_page_config(page_title="Mood AI Chatbot", page_icon="🤖", layout="centered")

st.title("🤖 Mood AI Chatbot")
st.write("Choose AI personality: Angry 😡 | Funny 😂 | Sad 😢")

# -----------------------
# Sidebar Mode Selection
# -----------------------
mode_choice = st.sidebar.selectbox(
    "Choose AI Mode",
    ["Angry 😡", "Funny 😂", "Sad 😢"]
)

# -----------------------
# Personality Prompt
# -----------------------
if mode_choice == "Angry 😡":
    mode = "You are a very angry AI agent. Respond aggressively and impatiently."
elif mode_choice == "Funny 😂":
    mode = "You are a funny AI agent. Add jokes in every response."
else:
    mode = "You are a sad AI agent. Respond emotionally and sadly."

# -----------------------
# Load Model Once
# -----------------------
@st.cache_resource
def load_model():
    llm = HuggingFaceEndpoint(
        repo_id="Qwen/Qwen2.5-7B-Instruct",
        task="text-generation",
        huggingfacehub_api_token=hf_token,
        max_new_tokens=200
    )
    return ChatHuggingFace(llm=llm)

chat_model = load_model()

# -----------------------
# Session State
# -----------------------
if "messages" not in st.session_state:
    st.session_state.messages = [
        SystemMessage(content=mode)
    ]

# Reset if mode changes
if st.session_state.messages[0].content != mode:
    st.session_state.messages = [
        SystemMessage(content=mode)
    ]

# -----------------------
# Show Chat History
# -----------------------
for msg in st.session_state.messages[1:]:
    if isinstance(msg, HumanMessage):
        with st.chat_message("user"):
            st.write(msg.content)
    elif isinstance(msg, AIMessage):
        with st.chat_message("assistant"):
            st.write(msg.content)

# -----------------------
# User Input
# -----------------------
prompt = st.chat_input("Type your message...")

if prompt:
    # Show user msg
    with st.chat_message("user"):
        st.write(prompt)

    st.session_state.messages.append(HumanMessage(content=prompt))

    # Generate response
    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            response = chat_model.invoke(st.session_state.messages)
            st.write(response.content)

    st.session_state.messages.append(AIMessage(content=response.content))