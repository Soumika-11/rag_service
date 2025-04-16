import faiss
import numpy as np
from together import Together
import os
from dotenv import load_dotenv
import logging
from tqdm import tqdm

load_dotenv()

logging.basicConfig(level=logging.INFO)

TOGETHER_API_KEY = os.getenv("TOGETHER_API_KEY")
if not TOGETHER_API_KEY:
    raise ValueError("TOGETHER_API_KEY is not set in the environment variables.")

client = Together(api_key=TOGETHER_API_KEY)

def embed_chunks(chunks):
    if not chunks:
        raise ValueError("The chunks list is empty. Ensure that documents are properly loaded and chunked.")

    embeddings = []
    for i, chunk in enumerate(tqdm(chunks, desc="Processing chunks", unit="chunk")):
        logging.info(f"Processing chunk {i + 1}/{len(chunks)}")
        response = client.embeddings.create(
            model="togethercomputer/m2-bert-80M-8k-retrieval",
            input=chunk
        )
        if not response.data:
            logging.error(f"Failed to generate embedding for chunk {i + 1}. Skipping...")
            continue
        embeddings.append(response.data[0].embedding)

    if not embeddings:
        raise ValueError("No embeddings were generated. Check the embedding service or input data.")

    embeddings_np = np.array(embeddings).astype("float32")
    if embeddings_np.ndim != 2:
        raise ValueError("Embeddings array is not two-dimensional. Check the embedding generation process.")

    index = faiss.IndexFlatL2(embeddings_np.shape[1])
    index.add(embeddings_np)
    return index, chunks
