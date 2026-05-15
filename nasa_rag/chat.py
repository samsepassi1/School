"""Streamlit chat UI for the NASA RAG Mission Intelligence system."""
from __future__ import annotations

import os
import time
from pathlib import Path

import streamlit as st
from dotenv import load_dotenv

from llm_client import LLMClient
from rag_client import KNOWN_MISSIONS, RAGClient, RetrievalResult
from ragas_evaluator import RAGASEvaluator


load_dotenv()


# ---------------------------------------------------------------------------
# Session bootstrapping
# ---------------------------------------------------------------------------

@st.cache_resource(show_spinner=False)
def get_rag_client(chroma_dir: str, collection_name: str, embed_model: str) -> RAGClient:
    return RAGClient(
        chroma_dir=chroma_dir,
        collection_name=collection_name,
        embed_model=embed_model,
    )


@st.cache_resource(show_spinner=False)
def get_evaluator() -> RAGASEvaluator:
    return RAGASEvaluator()


def _init_state() -> None:
    if "messages" not in st.session_state:
        st.session_state.messages = []  # list[dict]: role, content, meta (for assistant)
    if "llm" not in st.session_state:
        st.session_state.llm = LLMClient()


# ---------------------------------------------------------------------------
# UI
# ---------------------------------------------------------------------------

def sidebar_controls() -> dict:
    st.sidebar.title("NASA Mission Intelligence")
    st.sidebar.caption("Retrieval-augmented Q&A over Apollo 11, Apollo 13, and Challenger archives.")

    model = st.sidebar.selectbox(
        "Chat model",
        options=[
            os.getenv("OPENAI_CHAT_MODEL", "gpt-4o-mini"),
            "gpt-4o-mini",
            "gpt-4o",
            "gpt-3.5-turbo",
        ],
        index=0,
        help="OpenAI chat-completions model used to generate the answer.",
    )
    mission = st.sidebar.selectbox(
        "Mission focus",
        options=["All"] + list(KNOWN_MISSIONS),
        index=0,
        help="When set, retrieval filters to chunks from this mission only.",
    )
    top_k = st.sidebar.slider("Top-k retrieved chunks", 1, 10, 4)
    temperature = st.sidebar.slider("LLM temperature", 0.0, 1.0, 0.2, 0.05)
    show_eval = st.sidebar.toggle("Show RAGAS scores", value=True)
    show_context = st.sidebar.toggle("Show retrieved context", value=True)

    st.sidebar.divider()
    if st.sidebar.button("Reset conversation", use_container_width=True):
        st.session_state.messages = []
        st.session_state.llm.reset()
        st.rerun()

    chroma_dir = os.getenv("CHROMA_DIR", "./chroma_db")
    collection_name = os.getenv("CHROMA_COLLECTION", "nasa_missions")
    embed_model = os.getenv("OPENAI_EMBED_MODEL", "text-embedding-3-small")

    rag = get_rag_client(chroma_dir, collection_name, embed_model)
    size = rag.collection_size()
    st.sidebar.markdown(f"**Collection:** `{collection_name}`")
    st.sidebar.markdown(f"**Chunks indexed:** {size}")
    if size == 0:
        st.sidebar.warning("No chunks indexed yet. Run `python embedding_pipeline.py` first.")

    return {
        "model": model,
        "mission": mission,
        "top_k": top_k,
        "temperature": temperature,
        "show_eval": show_eval,
        "show_context": show_context,
        "rag": rag,
    }


def render_message(message: dict) -> None:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])
        meta = message.get("meta")
        if not meta:
            return
        if meta.get("show_context") and meta.get("retrieved"):
            with st.expander("Retrieved context", expanded=False):
                for i, chunk in enumerate(meta["retrieved"], start=1):
                    st.markdown(
                        f"**[{i}] {chunk['mission']} — `{chunk['source']}`"
                        f"** _(score {chunk['score']:.3f})_"
                    )
                    st.code(chunk["text"], language="markdown")
        if meta.get("show_eval") and meta.get("scores") is not None:
            scores = meta["scores"]
            if scores.get("metrics"):
                st.markdown("**Quality metrics**")
                metric_items = list(scores["metrics"].items())
                cols = st.columns(min(len(metric_items), 4))
                for col, (name, value) in zip(cols, metric_items):
                    col.metric(label=name, value=f"{value:.3f}")
            if scores.get("errors"):
                st.warning(" / ".join(scores["errors"]))


def main() -> None:
    st.set_page_config(page_title="NASA Mission Intelligence", page_icon=":rocket:", layout="wide")
    _init_state()
    controls = sidebar_controls()
    rag: RAGClient = controls["rag"]

    st.title("NASA Mission Intelligence")
    st.caption(
        "Ask about Apollo 11, Apollo 13, or Challenger. Answers are grounded in "
        "NASA archive excerpts and cite the source chunks used."
    )

    for message in st.session_state.messages:
        render_message(message)

    if not os.getenv("OPENAI_API_KEY"):
        st.error("OPENAI_API_KEY is not set. Copy .env.example to .env and provide a key.")
        st.stop()

    question = st.chat_input("Ask a question about Apollo 11, Apollo 13, or Challenger…")
    if not question:
        return

    user_msg = {"role": "user", "content": question}
    st.session_state.messages.append(user_msg)
    render_message(user_msg)

    with st.chat_message("assistant"):
        with st.spinner("Retrieving context and generating answer…"):
            t0 = time.time()
            retrieval: RetrievalResult = rag.search(
                question, k=controls["top_k"], mission=controls["mission"]
            )
            context = retrieval.context

            llm: LLMClient = st.session_state.llm
            answer = llm.generate(
                question,
                context,
                model=controls["model"],
                temperature=controls["temperature"],
            )
            elapsed = time.time() - t0

        st.markdown(answer)
        st.caption(f"Generated in {elapsed:.1f}s using {controls['model']} — top {controls['top_k']} chunks")

        scores_payload = None
        if controls["show_eval"]:
            with st.spinner("Scoring with RAGAS…"):
                try:
                    scores = get_evaluator().score(
                        question=question,
                        contexts=retrieval.contexts,
                        answer=answer,
                    )
                    scores_payload = scores.to_dict()
                except Exception as exc:  # noqa: BLE001
                    scores_payload = {"metrics": {}, "errors": [f"evaluator error: {exc}"]}

        retrieved_payload = [
            {
                "mission": c.mission,
                "source": c.source,
                "score": c.score,
                "text": c.text,
            }
            for c in retrieval.chunks
        ]
        meta = {
            "retrieved": retrieved_payload,
            "scores": scores_payload,
            "show_eval": controls["show_eval"],
            "show_context": controls["show_context"],
        }
        assistant_msg = {"role": "assistant", "content": answer, "meta": meta}
        st.session_state.messages.append(assistant_msg)

        if controls["show_context"] and retrieved_payload:
            with st.expander("Retrieved context", expanded=False):
                for i, chunk in enumerate(retrieved_payload, start=1):
                    st.markdown(
                        f"**[{i}] {chunk['mission']} — `{chunk['source']}`"
                        f"** _(score {chunk['score']:.3f})_"
                    )
                    st.code(chunk["text"], language="markdown")

        if controls["show_eval"] and scores_payload and scores_payload.get("metrics"):
            st.markdown("**Quality metrics**")
            metric_items = list(scores_payload["metrics"].items())
            cols = st.columns(min(len(metric_items), 4))
            for col, (name, value) in zip(cols, metric_items):
                col.metric(label=name, value=f"{value:.3f}")
            if scores_payload.get("errors"):
                st.warning(" / ".join(scores_payload["errors"]))


if __name__ == "__main__":
    main()
