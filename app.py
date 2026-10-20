import streamlit as st
import sys
import os

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(BASE_DIR)

from rag.build_rag import build_rag_from_pdf
from rag.rag_pipeline import ask_concall_dynamic

st.set_page_config(page_title="Concall Analyzer", layout="wide")

st.title("📊 Earnings Concall Analyzer (Upload & Ask)")

# -----------------------------
# Session state
# -----------------------------
if "index" not in st.session_state:
    st.session_state.index = None
    st.session_state.chunks = None

# -----------------------------
# Upload PDF
# -----------------------------
uploaded_file = st.file_uploader(
    "Upload earnings concall PDF",
    type=["pdf"]
)

if uploaded_file and st.session_state.index is None:
    with st.spinner("Processing concall..."):
        index, chunks = build_rag_from_pdf(uploaded_file)
        st.session_state.index = index
        st.session_state.chunks = chunks

    st.success("Concall processed successfully!")

    # Auto summary
    st.subheader("📌 Auto Generated Summary")
    summary = ask_concall_dynamic(
        query="Give a complete structured summary of the concall",
        index=index,
        chunks=chunks,
        prompt_type="summary"
    )
    st.write(summary)

# -----------------------------
# Ask questions
# -----------------------------
if st.session_state.index is not None:
    st.divider()
    st.subheader("❓ Ask Questions from this Concall")

    user_query = st.text_input(
        "Your question",
        placeholder="e.g. What risks did management mention?"
    )
    st.caption('Ask any question about the transcript, including company name, quarter, financials, strategy, risks, guidance, or analyst Q&A.')
    if st.button("Ask"):
        with st.spinner("Analyzing..."):
            answer = ask_concall_dynamic(
                query=user_query,
                index=st.session_state.index,
                chunks=st.session_state.chunks,
                prompt_type="qa",
                top_k=5
            )
        st.write(answer)
