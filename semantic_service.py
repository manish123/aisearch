import faiss
import numpy as np
import pandas as pd
from fastapi import FastAPI, Query
from pydantic import BaseModel
from sentence_transformers import SentenceTransformer
from typing import List

app = FastAPI()
model = SentenceTransformer('all-MiniLM-L6-v2')

# Globals for FAISS
index = None
quotes = []

class Quote(BaseModel):
    id: int
    text: str
    author: str
    tags: List[str] = []

@app.on_event("startup")
def load_index():
    global index, quotes
    # Load quotes from CSV
    df = pd.read_csv("quotes.csv")  # Place your CSV in the root directory
    quotes = df.to_dict(orient="records")
    # Compute embeddings
    embeddings = model.encode([q["text"] for q in quotes], show_progress_bar=True)
    # Build FAISS index
    index = faiss.IndexFlatL2(embeddings.shape[1])
    index.add(np.array(embeddings).astype('float32'))

@app.get("/semantic-search/")
def semantic_search(query: str = Query(...), limit: int = 10):
    global index, quotes
    query_vec = model.encode([query]).astype('float32')
    D, I = index.search(query_vec, limit)
    results = [quotes[i] for i in I[0]]
    return results

@app.get("/health")
def health():
    return {"status": "ok"}
