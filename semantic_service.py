import faiss
import numpy as np
import pandas as pd
from fastapi import FastAPI, Query
from pydantic import BaseModel
from sentence_transformers import SentenceTransformer
from typing import List

app = FastAPI()
model = SentenceTransformer('all-MiniLM-L6-v2')

index = None
quotes = []

class Quote(BaseModel):
    id: int
    text: str
    author: str
    category: str
    mood: List[str]

@app.on_event("startup")
def load_index():
    global index, quotes
    # Load your CSV file
    df = pd.read_csv("final_quotes_with_5000_additional.csv")
    # Add an id column if not present
    if "id" not in df.columns:
        df["id"] = range(len(df))
    # Standardize columns
    df = df.rename(columns={
        "Quote": "text",
        "Author": "author",
        "Category": "category",
        "Mood": "mood"
    })
    # Convert mood to list
    df["mood"] = df["mood"].fillna("").apply(lambda x: [m.strip() for m in str(x).split("|") if m.strip()])
    quotes = df[["id", "text", "author", "category", "mood"]].to_dict(orient="records")
    # Compute embeddings
    embeddings = model.encode([q["text"] for q in quotes], show_progress_bar=True)
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
