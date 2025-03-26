import streamlit as st
import requests
from openai import OpenAI
import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# FastAPI backend URL
FASTAPI_URL = "http://localhost:8000/retrieve"  # Change if necessary

# OpenAI API Key
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

# Streamlit UI Setup
st.set_page_config(page_title="Quality Manual Chatbot", layout="wide")
st.title("📖 Quality Manual Chatbot")
st.write("Ask me anything about the Quality Manual!")

# Initialize chat history and regulations memory in session state
if "messages" not in st.session_state:
    st.session_state.messages = []
if "regulations_flow" not in st.session_state:
    st.session_state.regulations_flow = {}

# Display chat history
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# Function to get the last 5 exchanges as chat memory
def get_memory():
    return "\n".join([f"{m['role'].capitalize()}: {m['content']}" for m in st.session_state.messages[-5:]])

# Function to get relevant documents from FastAPI (for regular conversation)
def get_relevant_documents(query):
    payload = {"query": query, "top_k": 5}
    response = requests.post(FASTAPI_URL, json=payload)
    if response.status_code == 200:
        return response.json().get("results", [])
    else:
        st.error("Failed to fetch documents from backend. Please try again.")
        return []

# Function to generate a response using OpenAI (for regular conversation)
def generate_response(query, context, sources, chat_memory):
    prompt = f"""
    You are an AI assistant specialized in the university's Quality Manual.
    Your primary role is to provide accurate information based on the provided context.
    However, if the user's input is a greeting or a casual remark (e.g., "hi there," "hello," "how's it going?"), respond with a friendly introduction and offer assistance.
    
    Below is the previous conversation history. Use this to maintain context when answering the user's question:
    
    Previous conversation:
    {chat_memory}
    
    Now, answer the user's query based on your knowledge on the Quality Manual of the University of Nottingham.
    If you don't know the answer, you can refer to the retrieved context to generate an answer.
    If you don't know the answer even with the context, say 'I'm not sure based on the available information.'

    Context:
    {context}

    User Query: {query}

    Response:
    """
    response = client.chat.completions.create(
        model="ft:gpt-3.5-turbo-0125:personal::BEXJyNN3",  # Replace as needed
        messages=[{"role": "system", "content": prompt}],
        max_tokens=1000
    )
    response_content = response.choices[0].message.content.strip()
    if sources:
        sources_list = "\n".join([f"- [{src['section_header']}]({src['source_url']})" for src in sources])
        response_content += f"\n\n**Sources:**\n{sources_list}"
    return response_content

# Main user input
user_query = st.chat_input("Type your question here...")

# Toggle for Regulations mode appears below the input box
regulations_mode = st.checkbox("Regulations")

# the if statements are only for 1 query
if user_query:
    if regulations_mode:
        # Regulations mode logic:
        if "regulations_query" not in st.session_state.regulations_flow and "student_type" not in st.session_state.regulations_flow and "regulations_year" not in st.session_state.regulations_flow:
            with st.chat_message("user"):
                st.markdown(user_query)
            st.session_state.messages.append({"role": "user", "content": user_query})
            st.session_state.regulations_flow["regulations_query"] = user_query
            with st.chat_message("assistant"):
                st.markdown("Are you asking with regards to **undergraduate** or **postgraduate** regulations?")
            st.session_state.messages.append({"role": "assistant", "content": "Are you asking with regards to **undergraduate** or **postgraduate** regulations?"})
        elif "regulations_query" in st.session_state.regulations_flow and "student_type" not in st.session_state.regulations_flow and "regulations_year" not in st.session_state.regulations_flow:
            st.session_state.regulations_flow["student_type"] = user_query
            st.session_state.messages.append({"role": "user", "content": user_query})
            with st.chat_message("user"):
                st.markdown(st.session_state.regulations_flow["student_type"])
            with st.chat_message("assistant"):
                st.markdown(f"Which **year** of {st.session_state.regulations_flow["student_type"]} regulations would you like to refer to?")
            st.session_state.messages.append({"role": "assistant", "content": f"Which **year** of {st.session_state.regulations_flow['student_type']} regulations would you like to refer to?"})
        elif "regulations_query" in st.session_state.regulations_flow and "student_type" in st.session_state.regulations_flow and "regulations_year" not in st.session_state.regulations_flow:
            st.session_state.regulations_flow["regulations_year"] = user_query
            st.session_state.messages.append({"role": "user", "content": user_query})
            with st.chat_message("user"):
                st.markdown(st.session_state.regulations_flow["regulations_year"])
        # if all 3 queries are in, and if there is a new query, update the regulations_query
        else:
            st.session_state.regulations_flow["regulations_query"] = user_query
            st.session_state.messages.append({"role": "user", "content": user_query})
            with st.chat_message("user"):
                st.markdown(st.session_state.regulations_flow["regulations_query"])

        memory_data = st.session_state.regulations_flow
        st.write("**Collected Regulations Data:**")
        st.json(memory_data)
        st.session_state.messages.append({
            "role": "assistant",
            "content": f"Collected Regulations Data: {memory_data}"
        })
    else:
        # Reset regulations flow if not in regulations mode
        st.session_state.regulations_flow = {}
        st.session_state.current_regulations_step = 1
        # Regular conversation logic:
        st.session_state.messages.append({"role": "user", "content": user_query})
        with st.chat_message("user"):
            st.markdown(user_query)
        retrieved_docs = get_relevant_documents(user_query)
        context = "\n\n".join([doc["content"] for doc in retrieved_docs]) if retrieved_docs else "No relevant documents found."
        chat_memory = get_memory()
        ai_response = generate_response(user_query, context, retrieved_docs, chat_memory)
        with st.chat_message("assistant"):
            st.markdown(ai_response)
        st.session_state.messages.append({"role": "assistant", "content": ai_response})
