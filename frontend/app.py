import streamlit as st
import requests
from openai import OpenAI
import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# FastAPI backend URL
FASTAPI_URL = "http://localhost:8000/retrieve" # Change this if your FastAPI runs on a different URL

# OpenAI API Key
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

# Streamlit UI Setup
st.set_page_config(page_title="Quality Manual Chatbot", layout="wide")
st.title("📖 Quality Manual Chatbot")
st.write("Ask me anything about the Quality Manual!")

# Chat History in Streamlit session state
if "messages" not in st.session_state:
    st.session_state.messages = []

# Display chat history
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# Function to get relevant documents from FastAPI
def get_relevant_documents(query):
    payload = {"query": query, "top_k": 5}
    response = requests.post(FASTAPI_URL, json=payload)

    if response.status_code == 200:
        return response.json().get("results", [])
    else:
        st.error("Failed to fetch documents from backend. Please try again.")
        return []

# Function to generate response using OpenAI LLM
def generate_response(query, context):
    prompt = f"""
    You are an AI assistant specialized in the university's Quality Manual.
    Your primary role is to provide accurate information based on the provided context.
    However, if the user's input is a greeting or a casual remark (e.g., "hi there," "hello," "how's it going?"), respond with a friendly introduction and offer assistance.
    Answer the user's question using ONLY the provided context.
    If you don't know the answer, say 'I'm not sure based on the available information.'

    Context:
    {context}

    User Query: {query}

    Response:
    """

    response = client.chat.completions.create(
        model="gpt-4o-mini",  # Replace with your own model later
        messages=[{"role": "system", "content": prompt}],
        max_tokens=300
    )
    return response.choices[0].message.content.strip()

# User Input Section
user_query = st.chat_input("Type your question here...")

if user_query:
    # Display user message
    st.session_state.messages.append({"role": "user", "content": user_query})
    with st.chat_message("user"):
        st.markdown(user_query)

    # Retrieve relevant documents
    retrieved_docs = get_relevant_documents(user_query)

    # Prepare context from retrieved docs
    context = "\n\n".join([doc["content"] for doc in retrieved_docs]) if retrieved_docs else "No relevant documents found."

    # Generate AI response
    ai_response = generate_response(user_query, context)

    # Display AI response
    with st.chat_message("assistant"):
        st.markdown(ai_response)

    # Save response to chat history
    st.session_state.messages.append({"role": "assistant", "content": ai_response})
