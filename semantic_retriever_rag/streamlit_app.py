"""Streamlit demo that compares the three retrievers side-by-side.

Run with:

    streamlit run streamlit_app.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import streamlit as st

ROOT = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from data.corpus import DOCUMENTS, QUERIES, RELEVANT
from bm25_retriever import BM25Retriever
from evaluator import evaluate_all
from transformer_retriever import TransformerRetriever
from word2vec_retriever import Word2VecRetriever


@st.cache_resource(show_spinner="Building retrievers…")
def build_retrievers():
    bm25 = BM25Retriever()
    bm25.build_index(DOCUMENTS)

    w2v = Word2VecRetriever(vector_size=100, window=5, min_count=1)
    w2v.build_index(DOCUMENTS)

    tfm = TransformerRetriever()
    tfm.build_index(DOCUMENTS)

    return bm25, w2v, tfm


def main() -> None:
    st.set_page_config(
        page_title="Semantic Retriever Comparison",
        page_icon="🔎",
        layout="wide",
    )
    st.title("Semantic Retriever — BM25 vs Word2Vec vs Transformer")
    st.markdown(
        "Compare three retrieval approaches on the same employee-handbook "
        "corpus. Pick a sample query or type your own, then look at the "
        "top-k results returned by each method."
    )

    bm25, w2v, tfm = build_retrievers()

    with st.sidebar:
        st.header("Settings")
        top_k = st.slider("top_k", min_value=1, max_value=5, value=3)
        sample = st.selectbox("Sample queries", ["(custom)"] + QUERIES)

    default_query = "" if sample == "(custom)" else sample
    query = st.text_input("Query", value=default_query)

    if query:
        col_bm25, col_w2v, col_tfm = st.columns(3)
        for col, name, retriever in zip(
            (col_bm25, col_w2v, col_tfm),
            ("BM25", "Word2Vec", "Transformer"),
            (bm25, w2v, tfm),
        ):
            with col:
                st.subheader(name)
                results = retriever.retrieve([query], top_k=top_k)[0]
                for rank, doc_id in enumerate(results, start=1):
                    st.markdown(f"**#{rank} — doc {doc_id}**")
                    st.write(DOCUMENTS[doc_id])

    st.divider()
    st.header("Aggregate evaluation on the labelled query set")
    bm25_res = bm25.retrieve(QUERIES, top_k=top_k)
    w2v_res = w2v.retrieve(QUERIES, top_k=top_k)
    tfm_res = tfm.retrieve(QUERIES, top_k=top_k)
    metrics = {
        "BM25": evaluate_all(bm25_res, RELEVANT, k=top_k),
        "Word2Vec": evaluate_all(w2v_res, RELEVANT, k=top_k),
        "Transformer": evaluate_all(tfm_res, RELEVANT, k=top_k),
    }
    st.dataframe(metrics)


if __name__ == "__main__":
    main()
