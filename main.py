from fastapi import FastAPI, Request
from pydantic import BaseModel
from services.loader import load_and_chunk_docs
from services.embedder import embed_chunks
from services.retriever import get_top_k_chunks
from services.generator import generate_answer

app = FastAPI()

class Query(BaseModel):
    question: str

# Load & index docs on startup
chunks = load_and_chunk_docs("data/docs/")
index, chunk_texts = embed_chunks(chunks)

@app.post("/rag")
def rag_pipeline(query: Query):
    top_chunks = get_top_k_chunks(query.question, index, chunk_texts)
    answer = generate_answer(query.question, top_chunks)
    return {"answer": answer}
