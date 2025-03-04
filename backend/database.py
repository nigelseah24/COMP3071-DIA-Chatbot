import os
import requests
from dotenv import load_dotenv
from pinecone import Pinecone
from langchain.text_splitter import RecursiveCharacterTextSplitter
from openai import OpenAI
from bs4 import BeautifulSoup


# Load environment variables
load_dotenv()

# Initialize Pinecone and OpenAI
pc = Pinecone(api_key=os.getenv("PINECONE_API_KEY"))
index = pc.Index("quality-manual")

client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))


# Jina AI API Key & Reader URL
JINA_API_KEY = os.getenv("JINA_API_KEY")
JINA_READER_URL = "https://r.jina.ai/"

urls = [
    "https://www.nottingham.ac.uk/qualitymanual/prog-and-mod-design-and-approval/changes-to-prog-mod-specs.aspx"
]

def extract_clean_text(url):
    # Step 1: Extract full clean text using Jina AI
    full_url = f"{JINA_READER_URL}{url}"
    headers = {"Authorization": f"Bearer {JINA_API_KEY}"}

    response = requests.get(full_url, headers=headers)
    
    print(f"Status Code: {response.status_code}")  # Check if request was successful
    print(f"Raw Response: {response.text[:500]}")  # Print first 500 chars for debugging
    
    if response.status_code == 200:
        cleaned_text = response.text.strip()
    else:
        print(f"Failed to extract from {url}, Status Code: {response.status_code}")
        return "", ""

    # Step 2: Fetch the raw webpage HTML to extract intro paragraph
    try:
        page_response = requests.get(url, timeout=10)
        if page_response.status_code != 200:
            print(f"Failed to fetch HTML from {url}, Status Code: {page_response.status_code}")
            return "", cleaned_text

        soup = BeautifulSoup(page_response.text, "html.parser")

        div_tag = soup.find('div', class_='sys_one_7030')
        if div_tag:
            h1_tag = div_tag.find('h1')
            if h1_tag:
                header_text = h1_tag.get_text(strip=True)
                print("Header text: " + header_text)
            else:
                print("No <h1> tag found within the specified <div>.")
        else:
            print("No <div> with class 'sys_one_7030' found.")


        # Extract intro paragraph using BeautifulSoup
        intro_paragraph = soup.find("p", class_="introParagraph")
        intro_text = intro_paragraph.get_text(strip=True) if intro_paragraph else "No intro paragraph found"

    except Exception as e:
        print(f"Error extracting intro paragraph from {url}: {str(e)}")
        intro_text = "No intro paragraph found"

    return intro_text, cleaned_text, header_text
    
    
    
# Split text into meaningful chunks
def split_documents(text, chunk_size=1000, chunk_overlap=200):
    splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
    return splitter.split_text(text)


# Upsert vectors to Pinecone
def upsert_vectors_to_pinecone(docs, source_url, intro_text, header_text):
    print(f"\n--- Upserting {len(docs)} chunks to Pinecone ---")

    vectors = []
    
    for i, doc in enumerate(docs):
        embedding_response = client.embeddings.create(
            input=doc,
            model="text-embedding-3-small"  # Specify embedding model
        )
        embedding = embedding_response.data[0].embedding
        
        vectors.append((f"{header_text}-{i}", embedding, {"content": doc, "source_url": source_url, "intro_paragraph": intro_text}))

    index.upsert(vectors=vectors)
    print(f"--- Finished upserting {len(vectors)} vectors ---")
    
    
# Process and store documents
def process_and_store_documents(urls):
    for url in urls:

        intro_text, cleaned_text, header_text = extract_clean_text(url)
        if not cleaned_text:
            print(f"Skipping {url} due to extraction failure.")
            continue

        chunks = split_documents(cleaned_text)
        upsert_vectors_to_pinecone(chunks, url, intro_text, header_text)
        

# Run the process to store documents
if __name__ == "__main__":
    #process_and_store_documents(urls)
    process_and_store_documents(urls)  # Store in a custom namespace


# Function to retrieve relevant vectors from Pinecone
def retrieve_relevant_vectors(query: str, top_k: int = 3):
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
            "content": match["metadata"]["content"],
            "source_url": match["metadata"]["source_url"],
            "intro_paragraph": match["metadata"].get("intro_paragraph", "No intro available")
        })

    return relevant_docs



### 
# id : name of section + number
# description : small descriptopn
# url
# content vector
