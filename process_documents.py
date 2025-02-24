import os
from pdfplumber import open as pdf_open
from docx import Document
from bs4 import BeautifulSoup
from sentence_transformers import SentenceTransformer
import json
import faiss
import numpy as np

# Text extraction function
def extract_text(file_path):
    ext = os.path.splitext(file_path)[1].lower()
    if ext == '.pdf':
        with pdf_open(file_path) as pdf:
            return "".join(page.extract_text() or "" for page in pdf.pages)
    elif ext == '.docx':
        doc = Document(file_path)
        return "\n".join([para.text for para in doc.paragraphs])
    elif ext == '.txt':
        with open(file_path, 'r', encoding='utf-8') as f:
            return f.read()
    elif ext == '.html':
        with open(file_path, 'r', encoding='utf-8') as f:
            soup = BeautifulSoup(f, 'html.parser')
            return soup.get_text()
    else:
        raise ValueError(f"Unsupported file type: {ext}")

# Chunking function
def chunk_text(text, max_tokens=500):
    words = text.split()
    return [" ".join(words[i:i+max_tokens]) for i in range(0, len(words), max_tokens)]

# Process documents
model = SentenceTransformer('all-MiniLM-L6-v2')
directory = "docs/"  # Replace with your directory
chunks = []
embeddings = []
metadata = []  # Store file origin for each chunk

for filename in os.listdir(directory):
    file_path = os.path.join(directory, filename)
    try:
        text = extract_text(file_path)
        file_chunks = chunk_text(text)
        chunks.extend(file_chunks)
        embeddings.extend(model.encode(file_chunks).tolist())
        metadata.extend([{"file": filename, "chunk_idx": i} for i in range(len(file_chunks))])
    except Exception as e:
        print(f"Error processing {filename}: {e}")

# Save to disk
with open("chunks.json", "w") as f:
    json.dump(chunks, f)
with open("metadata.json", "w") as f:
    json.dump(metadata, f)


# Convert embeddings to FAISS-compatible format
embeddings = np.array(embeddings).astype('float32')
dimension = embeddings.shape[1]
index = faiss.IndexFlatL2(dimension)  # L2 distance for similarity
index.add(embeddings)
faiss.write_index(index, "faiss_index.bin")