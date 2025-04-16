import numpy as np
from together import Together
import os
from dotenv import load_dotenv

load_dotenv()

TOGETHER_API_KEY = os.getenv("TOGETHER_API_KEY")
if not TOGETHER_API_KEY:
    raise ValueError("TOGETHER_API_KEY is not set in the environment variables.")

client = Together(api_key=TOGETHER_API_KEY)

def get_top_k_chunks(query, index, chunks, k=3):
    embed_response = client.embeddings.create(
        model="togethercomputer/m2-bert-80M-8k-retrieval",
        input=query
    )
    if not embed_response.data:
        raise ValueError("Failed to generate embedding for the query.")

    embed = embed_response.data[0].embedding
    D, I = index.search(np.array([embed]).astype("float32"), k)
    return [chunks[i] for i in I[0]]
