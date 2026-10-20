from sentence_transformers import SentenceTransformer
import faiss
import numpy as np
from chunking import chunk_text
from load_concall import load_and_clean_pdf

# 1. Load embedding model
model = SentenceTransformer("all-MiniLM-L6-v2")

# 2. Create embeddings
clean_text = load_and_clean_pdf("concall.pdf")
chunks=chunk_text(clean_text)
chunk_embeddings = model.encode(chunks)

# 3. Create FAISS index
dimension = chunk_embeddings.shape[1]
index = faiss.IndexFlatL2(dimension)

# 4. Add embeddings to index
index.add(chunk_embeddings)

# 5. Store chunk text
chunk_texts = chunks

# 6. Query
query = "which company?"
query_embedding = model.encode([query])

# 7. Search
k = 3
distances, indices = index.search(query_embedding, k)

# 8. save index and chunk texts
faiss.write_index(index, "vector_store.index")
np.save("chunk_texts.npy", np.array(chunk_texts))
