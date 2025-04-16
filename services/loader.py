import os
from pypdf import PdfReader
from tqdm import tqdm
import logging

logging.basicConfig(level=logging.INFO)

def load_and_chunk_docs(directory, chunk_size=300, overlap=50, output_file="rag/chunks.txt"):
    chunks = []
    if not os.path.exists(directory):
        raise ValueError(f"Directory '{directory}' does not exist.")

    files = [file for file in os.listdir(directory) if file.endswith(".pdf")]
    if not files:
        logging.warning("No PDF files found in the directory.")
        return chunks

    with tqdm(total=len(files), desc="Processing files", unit="file") as progress_bar:
        for file in files:
            file_path = os.path.join(directory, file)
            logging.info(f"Processing file: {file_path}")
            try:
                reader = PdfReader(file_path)
                text = " ".join(page.extract_text() for page in reader.pages if page.extract_text())
                if not text.strip():
                    logging.warning(f"No text extracted from file: {file_path}")
                    continue
                chunks += chunk_text(text, chunk_size, overlap)
            except Exception as e:
                logging.error(f"Error processing file {file_path}: {e}")
                continue  # Skip to the next file
            finally:
                progress_bar.update(1)  # Explicitly update the progress bar

    if not chunks:
        logging.warning("No chunks were created. Ensure the directory contains valid PDF files with extractable text.")
    else:
        logging.info(f"Successfully created {len(chunks)} chunks from {len(files)} files.")
        save_chunks(chunks, output_file)

    return chunks

def chunk_text(text, size, overlap):
    words = text.split()
    return [" ".join(words[i:i+size]) for i in range(0, len(words)-size+1, size-overlap)]

def save_chunks(chunks, output_file):
    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    with open(output_file, "w", encoding="utf-8") as f:
        for chunk in chunks:
            f.write(chunk + "\n")
    logging.info(f"Chunks saved to {output_file}")
