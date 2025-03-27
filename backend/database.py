import os
import re
from dotenv import load_dotenv
from pinecone import Pinecone
from langchain.text_splitter import RecursiveCharacterTextSplitter
from openai import OpenAI
import tiktoken


# Load environment variables
load_dotenv()

# Initialize Pinecone and OpenAI
pc = Pinecone(api_key=os.getenv("PINECONE_API_KEY"))
index = pc.Index("quality-manual")

client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

# File path (update as needed)
file_path = "C:\\Users\\PC 5\\Desktop\\COMP3071-DIA-Chatbot\\postgraduate regulations files\\Regulations before September 2006.txt"

# Tokenizer for counting tokens
encoding = tiktoken.get_encoding("cl100k_base")

# Function to count tokens
def count_tokens(text):
    return len(encoding.encode(text))



# Function to extract metadata (URL, Header, Intro, Content)
def extract_metadata(section):
    url_match = re.search(r"URL:\s*(.*)", section)
    header_match = re.search(r"Header:\s*(.*)", section)
    intro_match = re.search(r"Intro:\s*(.*)", section)

    url = url_match.group(1).strip() if url_match else "Unknown URL"
    header = header_match.group(1).strip() if header_match else "Unknown Header"
    intro = intro_match.group(1).strip() if intro_match else "No intro available"

    # Extract content (everything after "Content:")
    content_start = section.find("Content:")
    content = section[content_start + len("Content:"):].strip() if content_start != -1 else ""
    
    print(f"✅ Extracted URL: {url}")
    print(f"✅ Extracted Header: {header}")
    print(f"✅ Extracted Intro: {intro[:100]}...")  # Print first 100 chars for preview
    print(f"✅ Extracted Content Tokens: {count_tokens(content)} tokens")

    return url, header, intro, content

# Initialize RecursiveCharacterTextSplitter
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=1000,  # Maximum number of characters per chunk
    chunk_overlap=200  # Number of characters to overlap between chunks
)

# Function to split large content
def split_large_content(content, max_tokens=8000):
    tokens = count_tokens(content)
    if tokens <= max_tokens:
        return [content]  # Return as a single chunk

    print(f"⚠️ Content too large ({tokens} tokens). Splitting...")

    # Use the text splitter to create documents
    documents = text_splitter.create_documents([content])
    # Extract the page_content from each document
    chunks = [doc.page_content for doc in documents]

    print(f"✅ Split into {len(chunks)} chunks")
    return chunks


def upsert_vectors_to_pinecone(section_content, url, header, intro, namespace):
    content_chunks = split_large_content(section_content)  # Split large content

    for i, chunk in enumerate(content_chunks):
        print(f"\n--- Generating embedding for chunk {i+1}/{len(content_chunks)} ---")

        embedding_response = client.embeddings.create(
            input=chunk,
            model="text-embedding-3-small"  # Adjust model if needed
        )
        embedding = embedding_response.data[0].embedding

        # Create a unique ID for each chunk
        chunk_id = f"{header}_{i+1}"

        # Upsert each chunk to Pinecone
        vector = (
            chunk_id,  # Unique ID for the chunk
            embedding,
            {"content": chunk, "source_url": url, "section_header": header, "intro_paragraph": intro}
        )

        index.upsert(vectors=[vector], namespace = namespace)
        print(f"--- Finished upserting chunk {i+1} of section: {header} into namespace: {namespace} ---")



# Process and store documents
def process_and_store_documents(file_path):
    with open(file_path, "r", encoding="utf-8") as file:
        text_data = file.read()

    url, header, intro, content = extract_metadata(text_data)

    if content.strip():
        upsert_vectors_to_pinecone(content, url, header, intro, namespace="pg-null-sep2006")


if __name__ == "__main__":
    process_and_store_documents(file_path)
    print("\n✅ Data processing and upsertion completed!")


# Function to retrieve relevant vectors from Pinecone
def retrieve_relevant_vectors(query: str, top_k: int = 5):
    # Get the embedding for the query
    embedding_response = client.embeddings.create(
        input=query,
        model="text-embedding-3-small"  # Use the same model as used for upserting
    )
    query_embedding = embedding_response.data[0].embedding

    # Perform similarity search in Pinecone
    search_results = index.query(vector=query_embedding, top_k=top_k, include_metadata=True)

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



### 
# id : name of section + number
# description : small descriptopn
# url
# content vector
