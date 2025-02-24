from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from sentence_transformers import SentenceTransformer
import faiss
import numpy as np
import json
import requests
from pydantic import BaseModel  # Add this import
import config

app = FastAPI()
model = SentenceTransformer('all-MiniLM-L6-v2')
index = faiss.read_index("faiss_index.bin")
with open("chunks.json", "r") as f:
    chunks = json.load(f)
OPENROUTER_API_KEY = "1"  # Replace with your key

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["POST"],
    allow_headers=["*"],
)

@app.get("/")
async def read_root():
    return {"message": "Welcome to the Document Query API. Use POST /query to search documents."}

# Define Pydantic model
class Query(BaseModel):
    query: str

@app.post("/query")
async def query_documents(query: Query):
    query_embedding = model.encode(query.query).astype('float32').reshape(1, -1)  # Use query.query
    D, I = index.search(query_embedding, k=3)
    relevant_chunks = [chunks[i] for i in I[0]]
    prompt = f"Answer this query based on the documents: {query.query}\nDocuments:\n" + "\n".join(relevant_chunks)
    response = requests.post(
        "https://openrouter.ai/api/v1/chat/completions",
        headers={"Authorization": f"Bearer {config.LLM_API_KEY}"},
        json={"model": "google/gemini-flash-1.5-8b", "messages": [{"role": "user", "content": prompt}]}
    ).json()
    if "choices" in response:
        return {
            "answer": response["choices"][0]["message"]["content"],
            "sources": relevant_chunks  # Return the retrieved chunks
        }
    else:
        return {"error": "Failed to get response from OpenRouter", "details": response}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="localhost", port=8000, reload=True)