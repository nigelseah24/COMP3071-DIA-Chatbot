from fastapi import FastAPI
import os
import requests
from dotenv import load_dotenv
from pinecone import Pinecone
from langchain.text_splitter import RecursiveCharacterTextSplitter
from openai import OpenAI


app = FastAPI()
# Load environment variables
load_dotenv()

# Initialize Pinecone and OpenAI
pc = Pinecone(api_key=os.getenv("PINECONE_API_KEY"))
index = pc.Index("website-chatbot")

client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))


# Jina AI API Key & Reader URL
JINA_API_KEY = "jina_833f960bd0b94f919920f4dabb63af001uiZNzo-5urqa6K9XmLOrk5CgrbV"
JINA_READER_URL = "https://r.jina.ai/"

urls = [
    # 'https://www.nottingham.edu.my/ugstudy/course/nottingham-foundation-programme',
    # "https://www.nottingham.edu.my/ugstudy/course/computer-science-bsc-hons",
    # "https://www.nottingham.edu.my/ugstudy/course/computer-science-with-artificial-intelligence-bsc-hons",
    # "https://www.nottingham.edu.my/pgstudy/course/research/computer-science-mphil-phd",
    # "https://www.nottingham.edu.my/Study/Fees-and-Scholarships/Scholarships/Foundation-undergraduate-scholarships.aspx",
    "https://www.nottingham.edu.my/Study/Make-an-enquiry/Enquire-now.aspx"
    # https://www.nottingham.edu.my/Study/How-to-apply/When-to-apply.aspx
    
    # facilities
    # "https://www.nottingham.edu.my/CurrentStudents/Facilities/Sport/Sport.aspx",
    # "https://www.nottingham.edu.my/CurrentStudents/Facilities/Sport/Swimming-pool.aspx",
    # "https://www.nottingham.edu.my/CurrentStudents/Facilities/Health.aspx",
    # "https://www.nottingham.edu.my/CurrentStudents/Facilities/Prayer.aspx",
    # "https://www.nottingham.edu.my/CurrentStudents/Facilities/amenities.aspx"
    
    # current students
    #
]


@app.get("/")
async def root():
    return {"message": "Hello World"}