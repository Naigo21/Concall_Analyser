# ==============================
# Full RAG Pipeline (FREE LLM)
# ==============================

import os
import re
import faiss
import numpy as np
from sentence_transformers import SentenceTransformer
from dotenv import load_dotenv

# =========================================================
# 0. Resolve BASE DIRECTORY
# =========================================================

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
load_dotenv(os.path.join(BASE_DIR, '.env'))

# =========================================================
# 1. Load Embedding Model (Day 7)
# =========================================================

embedding_model = SentenceTransformer("all-MiniLM-L6-v2")

# =========================================================
# 2. Load Vector Store
# =========================================================

index = faiss.read_index(os.path.join(BASE_DIR, "vector_store.index"))
chunk_texts = np.load(os.path.join(BASE_DIR, "chunk_texts.npy"), allow_pickle=True)

# =========================================================
# 3. Load Prompt Template
# =========================================================

def load_prompt(prompt_path):
    with open(prompt_path, "r", encoding="utf-8") as f:
        return f.read()

# =========================================================
# 4. Retrieve Relevant Chunks
# =========================================================

def retrieve_chunks(query, k=4):
    query_embedding = embedding_model.encode([query])
    distances, indices = index.search(query_embedding, k)

    return [chunk_texts[idx] for idx in indices[0]]

# =========================================================
# 5. Build Prompt
# =========================================================

def build_prompt(prompt_template, retrieved_chunks, question=None):
    """Number evidence excerpts and insert the user's question when supplied."""
    context = "\n\n".join(
        f"[Excerpt {position}]\n{str(chunk)}"
        for position, chunk in enumerate(retrieved_chunks, start=1)
    )
    prompt = prompt_template.replace("{retrieved_chunks}", context)
    if question is not None:
        prompt = prompt.replace("{question}", question)
    return prompt
# =========================================================
# 6. Call Groq API using local environment configuration
# =========================================================

from groq import Groq

def call_llm(prompt):
    """Generate a response using the configured Groq key and model."""
    api_key = os.getenv("GROQ_API_KEY")
    if not api_key or api_key == "put_your_new_groq_key_here":
        raise RuntimeError("GROQ_API_KEY is missing. Add a new key to the project .env file.")
    model_name = os.getenv("GROQ_MODEL", "openai/gpt-oss-20b")
    groq_client = Groq(api_key=api_key)
    response = groq_client.chat.completions.create(
        model=model_name,
        messages=[
            {"role": "system", "content": "You are a careful financial analyst. Do not speculate."},
            {"role": "user", "content": prompt},
        ],
        temperature=0.2,
    )
    return response.choices[0].message.content


# =========================================================
# 7. Main RAG Function
# =========================================================

def ask_concall(query, prompt_type="summary", top_k=4):
    retrieved_chunks = retrieve_chunks(query, k=top_k)

    if prompt_type == "summary":
        prompt_path = os.path.join(BASE_DIR, "prompts", "summary_prompt.txt")
    elif prompt_type == "risk":
        prompt_path = os.path.join(BASE_DIR, "prompts", "risk_prompt.txt")
    elif prompt_type == "guidance":
        prompt_path = os.path.join(BASE_DIR, "prompts", "guidance_prompt.txt")
    else:
        raise ValueError("Invalid prompt_type")

    prompt_template = load_prompt(prompt_path)
    final_prompt = build_prompt(prompt_template, retrieved_chunks)

    return call_llm(final_prompt)
# =========================================================
# 7b. Dynamic RAG Function (with custom index and chunks)
# =========================================================

def ask_concall_dynamic(query, index, chunks, prompt_type="qa", top_k=5):
    """Answer any transcript question or summarize the full transcript in batches."""
    query = (query or "").strip()
    if not query:
        return "Please enter a question about the uploaded transcript."
    prompt_dir = os.path.join(BASE_DIR, "prompts")
    if prompt_type == "summary":
        if not chunks:
            return "No transcript text is available to summarize."
        batch_size = 6
        partial_summaries = []
        template = load_prompt(os.path.join(prompt_dir, "summary_prompt.txt"))
        for start in range(0, len(chunks), batch_size):
            batch = [str(chunk) for chunk in chunks[start:start + batch_size]]
            partial_summaries.append(call_llm(build_prompt(template, batch)))
        if len(partial_summaries) == 1:
            return partial_summaries[0]
        reducer = load_prompt(os.path.join(prompt_dir, "summary_reduce_prompt.txt"))
        return call_llm(build_prompt(reducer, partial_summaries))
    prompt_names = {"qa": "qa_prompt.txt", "risk": "risk_prompt.txt", "guidance": "guidance_prompt.txt"}
    if prompt_type not in prompt_names:
        raise ValueError("Invalid prompt type")
    if index.ntotal == 0 or not chunks:
        return "The transcript index is empty, so I cannot answer from transcript evidence."
    k = min(max(1, int(top_k)), int(index.ntotal), len(chunks))
    query_embedding = embedding_model.encode([query])
    distances, indices = index.search(query_embedding, k)
    valid_indices = [int(i) for i in indices[0] if 0 <= int(i) < len(chunks)]
    retrieved = [str(chunks[i]) for i in valid_indices]
    identity_question = re.search(
        r"\b(company|company name|which company|quarter|which quarter|fiscal year|financial year|reporting period|which year|call is this|transcript about)\b",
        query,
        flags=re.IGNORECASE,
    )
    if identity_question and chunks and 0 not in valid_indices:
        retrieved.insert(0, str(chunks[0]))
    template = load_prompt(os.path.join(prompt_dir, prompt_names[prompt_type]))
    return call_llm(build_prompt(template, retrieved, question=query))
# =========================================================
# 8. Test Run
# =========================================================

if __name__ == "__main__":
    response = ask_concall(
        query="What risks did management mention?",
        prompt_type="risk"
    )

    print("\n========== RAG OUTPUT ==========\n")
    print(response)
