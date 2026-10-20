from pypdf import PdfReader
from sentence_transformers import SentenceTransformer
import faiss
import numpy as np

from chunking import chunk_text  # reuse your Day 6 code

embedding_model = SentenceTransformer("all-MiniLM-L6-v2")

def build_rag_from_pdf(pdf_file):
    # 1. Read PDF
    reader = PdfReader(pdf_file)
    raw_text = ""
    for page in reader.pages:
        raw_text += page.extract_text() + " "

    # 2. Clean text
    clean_text = " ".join(raw_text.split())

    # 3. Chunk
    chunks = chunk_text(clean_text, max_words=300, overlap_words=50)

    # 4. Create embeddings
    embeddings = embedding_model.encode(chunks)

    # 5. Build FAISS index
    dim = embeddings.shape[1]
    index = faiss.IndexFlatL2(dim)
    index.add(embeddings)

    return index, chunks
