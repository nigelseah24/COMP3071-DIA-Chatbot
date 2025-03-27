from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from database import*


app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class QueryRequest(BaseModel):
    query: str
    top_k: int = 5
    namespace: str = "general"

@app.post("/retrieve")
async def root(request: QueryRequest):
    results = retrieve_relevant_vectors(request.query, namespace=request.namespace, top_k=request.top_k)
    return {"results": results}