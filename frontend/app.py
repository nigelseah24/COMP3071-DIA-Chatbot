import streamlit as st
from pinecone import Pinecone
from openai import OpenAI
from namespace_classifier import get_regulation_page  # Import our regulation function

# Instead of loading from .env file, access keys like this:
OPENAI_API_KEY = st.secrets["general"]["OPENAI_API_KEY"]
PINECONE_API_KEY = st.secrets["general"]["PINECONE_API_KEY"]

# Initialize Pinecone and OpenAI
pc = Pinecone(api_key=PINECONE_API_KEY)
index = pc.Index("quality-manual")

# OpenAI API Key
client = OpenAI(api_key=OPENAI_API_KEY)

# Streamlit UI Setup
st.set_page_config(page_title="Quality Manual Chatbot", layout="wide")
st.title("📖 Quality Manual Chatbot")
st.write("Ask me anything about the Quality Manual!")

# Display Academic Regulations explanation when checkbox is selected
regulations_mode = st.toggle("Ask regarding Academic Regulations")

if regulations_mode:
    st.subheader("What are Academic Regulations?")
    st.markdown("""
    **Academic Regulations** are a set of formal guidelines that outline the rules and expectations for undergraduate (and postgraduate) study. They ensure consistency and fairness in academic processes. Here are some key examples of what the content covers:
    
    - **Passing Marks:** The regulations specify that the pass mark for a module is **40%**, which is the minimum required to pass.
    - **Module Selection and Credit Limits:** Students must select modules in accordance with their programme's requirements. There are also limits on the number of credits a student can register for in any one semester.
    - **Assessment and Re-assessment Procedures:** Guidelines detail the processes for assessments and the options available if a module is failed, including opportunities for re-assessment.
    - **Degree Classification:** They outline how final marks are calculated and how degree classifications (e.g., First Class, Second Class, etc.) are determined.
    - **Progression Requirements:** The regulations state the criteria a student must meet to advance from one stage of their course to the next.
    """)
    
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

# Function to analyze if a query requires similarity search
def analyze_query_needs_search(query, chat_memory):
    prompt = f"""
    You are an AI assistant specialized in analyzing queries for a university's Quality Manual chatbot.
    
    Your task is to determine if the following user query requires searching a specialized knowledge base 
    (the university's Quality Manual) or if it can be answered directly without needing to look up specific information.
    
    Previous conversation:
    {chat_memory}
    
    User Query: {query}
    
    Analyze the query and respond ONLY with one of these options:
    1. "NEEDS_SEARCH" - if the query is asking for specific information from the Quality Manual or regulations 
       that would require looking up facts, policies, procedures, or other specific content.
    2. "NO_SEARCH" - if the query is a greeting, casual remark, clarification question, or can be answered 
       based on general knowledge about universities without specific manual details.
    
    Response:
    """
    
    response = client.chat.completions.create(
        model="ft:gpt-3.5-turbo-0125:personal::BEXJyNN3",  # Use the same model as your main responses
        messages=[{"role": "system", "content": prompt}],
        max_tokens=10
    )
    
    decision = response.choices[0].message.content.strip()
    return "NEEDS_SEARCH" in decision

# Function to get relevant documents from FastAPI (for regular conversation)
def get_relevant_documents(query, namespace="general"):
    results = retrieve_relevant_vectors(query, namespace=namespace, top_k=5)
    return results

def retrieve_relevant_vectors(query: str, namespace: str, top_k: int = 5):
    # Get the embedding for the query
    embedding_response = client.embeddings.create(
        input=query,
        model="text-embedding-3-small"  # Use the same model as used for upserting
    )
    query_embedding = embedding_response.data[0].embedding

    # Perform similarity search in Pinecone
    search_results = index.query(vector=query_embedding, top_k=top_k, include_metadata=True, namespace=namespace)

    # Format the response
    relevant_docs = []
    for match in search_results["matches"]:
        relevant_docs.append({
            "score": match["score"],
            "content": match["metadata"].get("content", "No content available"),
            "source_url": match["metadata"].get("source_url", "No URL available"),
            "section_header": match["metadata"].get("section_header", "No header available"),
            "intro_paragraph": match["metadata"].get("intro_paragraph", "No intro available")
        })

    return relevant_docs

def generate_response_without_search(query, chat_memory):
    """Generate a response without doing a similarity search"""
    prompt = f"""
    You are an AI assistant specialized in the University of Nottingham's Quality Manual.
    Your primary role is to provide accurate information based on your general knowledge about university procedures.
    This is for queries that don't require specific lookups in the Quality Manual.
    
    Below is the previous conversation history. Use this to maintain context when answering the user's question:
    
    Previous conversation:
    {chat_memory}
    
    User Query: {query}
    
    If the user is asking for specific details from the Quality Manual, inform them that you're not certain about these specific details,
    and you'd need to search the manual for accurate information. Respond conversationally but don't make up specific details about the Quality Manual.
    
    Response:
    """
    
    response = client.chat.completions.create(
        model="ft:gpt-3.5-turbo-0125:personal::BEXJyNN3",  # Replace as needed
        messages=[{"role": "system", "content": prompt}],
        max_tokens=500
    )
    
    return response.choices[0].message.content.strip()

def generate_response(query, context, sources, chat_memory):
    prompt = f"""
    You are an AI assistant specialized in the university's Quality Manual.
    Your primary role is to provide accurate information based on the provided context.
    However, if the user's input is a greeting or a casual remark (e.g., "hi there," "hello," "how's it going?"), respond with a friendly introduction and offer assistance.
    
    Below is the previous conversation history. Use this to maintain context when answering the user's question:
    
    Previous conversation:
    {chat_memory}
    
    Now, answer the user's query based on your knowledge on the Quality Manual of the University of Nottingham. 
    You should also ask the user for clarification if the question is ambiguous or if you need more information to provide a complete answer.
    You should also ask the user follow-up questions regarding the topic if you think it is necessary.
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
    
    # Otherwise, process sources (and remove duplicates)
    if sources:
        seen = set()
        unique_sources = []
        for src in sources:
            src_entry = f"- [{src['section_header']}]({src['source_url']})"
            if src_entry not in seen:
                seen.add(src_entry)
                unique_sources.append(src_entry)
        sources_list = "\n".join(unique_sources)
        response_content += f"\n\n**More information:**\n{sources_list}"

    return response_content


# Main user input
user_query = st.chat_input("Type your question here...")

# Process the input query
if user_query:
    if regulations_mode:
        # Regulations mode logic:
        if ("regulations_query" not in st.session_state.regulations_flow 
            and "student_type" not in st.session_state.regulations_flow 
            and "honour_type" not in st.session_state.regulations_flow
            and "regulations_year" not in st.session_state.regulations_flow):
            # Step 1: Ask for the regulations query
            with st.chat_message("user"):
                st.markdown(user_query)
            st.session_state.messages.append({"role": "user", "content": user_query})
            st.session_state.regulations_flow["regulations_query"] = user_query
            with st.chat_message("assistant"):
                st.markdown("Are you asking with regards to **undergraduate** or **postgraduate** regulations?")
            st.session_state.messages.append({"role": "assistant", "content": "Are you asking with regards to **undergraduate** or **postgraduate** regulations?"})
        
        elif ("regulations_query" in st.session_state.regulations_flow 
              and "student_type" not in st.session_state.regulations_flow 
              and "honour_type" not in st.session_state.regulations_flow  
              and "regulations_year" not in st.session_state.regulations_flow):
            # Step 2: Ask for the program type and validate it.
            student_type_input = user_query.strip().lower()
            with st.chat_message("user"):
                st.markdown(user_query)
            st.session_state.messages.append({"role": "user", "content": user_query})
            if student_type_input not in ["undergraduate", "postgraduate"]:
                with st.chat_message("assistant"):
                    st.markdown("Please enter a valid program type: **undergraduate** or **postgraduate**.")
                st.session_state.messages.append({"role": "assistant", "content": "Invalid program type. Please enter either 'undergraduate' or 'postgraduate'."})
            else:
                st.session_state.regulations_flow["student_type"] = student_type_input
                st.session_state.messages.append({"role": "user", "content": user_query})
                # Only ask about honours for undergraduate queries.
                if student_type_input == "undergraduate":
                    with st.chat_message("assistant"):
                        st.markdown("Are you asking with regards to **Honours** or **Non-Honours** regulations?")
                    st.session_state.messages.append({"role": "assistant", "content": "Are you asking with regards to Honours or Non-Honours regulations?"})
                else:
                    # For postgraduate, skip the honours step and ask for the year.
                    with st.chat_message("assistant"):
                        st.markdown("Which **year** of postgraduate regulations would you like to refer to?")
                    st.session_state.messages.append({"role": "assistant", "content": "Which **year** of postgraduate regulations would you like to refer to?"})
        
        elif ("regulations_query" in st.session_state.regulations_flow 
              and "student_type" in st.session_state.regulations_flow 
              and st.session_state.regulations_flow["student_type"] == "undergraduate"
              and "honour_type" not in st.session_state.regulations_flow  
              and "regulations_year" not in st.session_state.regulations_flow):
            # Step 3 (for undergraduate): Ask for honour type and validate it.
            honour_type_input = user_query.strip().lower()
            with st.chat_message("user"):
                st.markdown(user_query)
            st.session_state.messages.append({"role": "user", "content": user_query})
            if honour_type_input not in ["honours", "non-honours"]:
                with st.chat_message("assistant"):
                    st.markdown("Please enter a valid honour type: **honours** or **non-honours**.")
                st.session_state.messages.append({"role": "assistant", "content": "Invalid honour type. Please enter either 'honours' or 'non-honours'."})
            else:
                st.session_state.regulations_flow["honour_type"] = honour_type_input
                with st.chat_message("assistant"):
                    st.markdown(f"Which **year** of {st.session_state.regulations_flow['student_type']} regulations would you like to refer to?")
                st.session_state.messages.append({"role": "assistant", "content": f"Which **year** of {st.session_state.regulations_flow['student_type']} regulations would you like to refer to?"})
        
        elif ("regulations_query" in st.session_state.regulations_flow 
              and "student_type" in st.session_state.regulations_flow 
              and ((st.session_state.regulations_flow["student_type"] == "undergraduate" and "honour_type" in st.session_state.regulations_flow)
                   or st.session_state.regulations_flow["student_type"] == "postgraduate")
              and "regulations_year" not in st.session_state.regulations_flow):
            # Step 4: Ask for the year and validate it.
            year_input = user_query.strip()
            with st.chat_message("user"):
                st.markdown(year_input)
            st.session_state.messages.append({"role": "user", "content": year_input})
            try:
                reg_year = int(year_input)
            except ValueError:
                with st.chat_message("assistant"):
                    st.markdown("The year you provided is not valid. Please enter a valid year (e.g., 2020).")
                st.session_state.messages.append({"role": "assistant", "content": "Invalid year. Please enter a valid year (e.g., 2020)."})
            else:
                st.session_state.regulations_flow["regulations_year"] = year_input
                # Now that we have all required information, call get_regulation_page.
                program = st.session_state.regulations_flow["student_type"]
                if program == "undergraduate":
                    # For undergraduate, use honour_type to decide which regulation to get.
                    honour_type = st.session_state.regulations_flow.get("honour_type", "honours")
                    if honour_type == "non-honours":
                        regulation_id = get_regulation_page(reg_year, program="undergraduate", student_type="non-honours")
                    else:
                        regulation_id = get_regulation_page(reg_year, program="undergraduate", student_type="honours")
                elif program == "postgraduate":
                    regulation_id = get_regulation_page(reg_year, program="postgraduate")
                else:
                    regulation_id = "unknown program"
                
                retrieved_docs = get_relevant_documents(st.session_state['regulations_flow']['regulations_query'], namespace=regulation_id)
                context = "\n\n".join([doc["content"] for doc in retrieved_docs]) if retrieved_docs else "No relevant documents found."
                chat_memory = get_memory()
                ai_response = generate_response(st.session_state['regulations_flow']['regulations_query'], context, retrieved_docs, chat_memory)
                
                with st.chat_message("assistant"):
                    st.markdown("Searching...")
                st.session_state.messages.append({"role": "assistant", "content": "Searching..."})

                with st.chat_message("assistant"):
                    st.markdown(ai_response)

                st.session_state.messages.append({"role": "assistant", "content": ai_response})
        
        else:
            # If all pieces are already in, update the regulations query and display the regulation page.
            st.session_state.regulations_flow["regulations_query"] = user_query
            st.session_state.messages.append({"role": "user", "content": user_query})
            with st.chat_message("user"):
                st.markdown(st.session_state.regulations_flow["regulations_query"])
            
            with st.chat_message("assistant"):
                st.markdown("Searching...")
            st.session_state.messages.append({"role": "assistant", "content": "Searching..."})
            
            # Re-run the lookup if needed.
            try:
                reg_year = int(st.session_state.regulations_flow.get("regulations_year", 0))
            except ValueError:
                reg_year = 0
            program = st.session_state.regulations_flow.get("student_type", "undergraduate")
            if program == "undergraduate":
                honour_type = st.session_state.regulations_flow.get("honour_type", "honours")
                regulation_id = get_regulation_page(reg_year, program="undergraduate", student_type=honour_type)
            elif program == "postgraduate":
                regulation_id = get_regulation_page(reg_year, program="postgraduate")
            else:
                regulation_id = "unknown program"
            
            retrieved_docs = get_relevant_documents(st.session_state['regulations_flow']['regulations_query'], namespace=regulation_id)
            context = "\n\n".join([doc["content"] for doc in retrieved_docs]) if retrieved_docs else "No relevant documents found."
            chat_memory = get_memory()
            ai_response = generate_response(st.session_state['regulations_flow']['regulations_query'], context, retrieved_docs, chat_memory)

            with st.chat_message("assistant"):
                st.markdown(ai_response)

            st.session_state.messages.append({"role": "assistant", "content": ai_response})
            
    else:
        # Reset regulations flow if not in regulations mode
        st.session_state.regulations_flow = {}
        st.session_state.current_regulations_step = 1
        
        # Regular conversation logic (updated):
        with st.chat_message("user"):
            st.markdown(user_query)
        st.session_state.messages.append({"role": "user", "content": user_query})
        
        # Get chat memory for context
        chat_memory = get_memory()
        
        # First, determine if we need to perform a search
        needs_search = analyze_query_needs_search(user_query, chat_memory)
        
        if needs_search:
            # If search is needed, perform the similarity search
            with st.chat_message("assistant"):
                st.markdown("Looking up information...")
            
            retrieved_docs = get_relevant_documents(user_query, namespace="general")
            context = "\n\n".join([doc["content"] for doc in retrieved_docs]) if retrieved_docs else "No relevant documents found."
            ai_response = generate_response(user_query, context, retrieved_docs, chat_memory)
        else:
            # If search is not needed, generate a direct response
            ai_response = generate_response_without_search(user_query, chat_memory)
        
        # Update the chat with the final response
        # Remove the "Looking up information..." message if it exists
        if len(st.session_state.messages) > 0 and st.session_state.messages[-1]["content"] == "Looking up information...":
            st.session_state.messages.pop()
        
        with st.chat_message("assistant"):
            st.markdown(ai_response)
        
        st.session_state.messages.append({"role": "assistant", "content": ai_response})