# Semantic Retriever — Transformer-backed RAG retriever

This project builds the retrieval half of a Retrieval-Augmented Generation
(RAG) pipeline for an internal employee assistant. Given a natural-language
question, the retriever finds the most relevant company-document passages,
which a downstream LLM would then condition its answer on.

Three retrievers are implemented and benchmarked head-to-head:

| Retriever            | File                          | What it captures                                  |
| -------------------- | ----------------------------- | ------------------------------------------------- |
| BM25                 | `bm25_retriever.py`           | Lexical overlap (TF/IDF + length normalisation)   |
| Word2Vec             | `word2vec_retriever.py`       | Static word-vector semantics (mean-pooled)        |
| Sentence-Transformer | `transformer_retriever.py`    | Contextual transformer embeddings                 |

All three share the same `build_index()` / `retrieve()` API and produce
results in the form `{query_index: [doc_id, ...]}`, so they're directly
interchangeable inside the evaluation harness.

## Files

```
semantic_retriever_rag/
├── data/corpus.py                       # synthetic employee-handbook corpus + ground truth
├── bm25_retriever.py                    # BM25Retriever (from scratch)
├── word2vec_retriever.py                # Word2VecRetriever + hyperparameter grid search
├── transformer_retriever.py             # TransformerRetriever (sentence-transformers)
├── evaluator.py                         # recall_at_k, precision_at_k, mrr (from scratch)
├── streamlit_app.py                     # interactive 3-way comparison demo
├── unified_retrieval_comparison.ipynb   # the deliverable notebook
├── tests/test_metrics.py                # smoke tests for the metrics
└── requirements.txt
```

## Setup

```bash
pip install -r requirements.txt
```

`sentence-transformers` will download the `all-MiniLM-L6-v2` model on first
use (~80 MB).

## Running it

* **Notebook** — open `unified_retrieval_comparison.ipynb` and run all cells.
  It builds the three retrievers, prints per-query top-k hits, evaluates with
  Recall@k / Precision@k / MRR, and shows a PCA + t-SNE projection of the
  embedding space.
* **Streamlit demo** — `streamlit run streamlit_app.py` for an interactive
  side-by-side view (pick a sample query or type your own).
* **Tests** — `python -m pytest tests/` runs the metric smoke tests (no model
  download required).

## How this fits into a RAG pipeline

```
user query ──► [retriever] ──► top-k passages ──┐
                                                ├──► [LLM generator] ──► grounded answer
         user query ─────────────────────────────┘
```

At query time we (1) embed the question with the same transformer used for
the corpus, (2) cosine-rank documents, (3) splice the top-k into the
generator's prompt, and (4) let the LLM answer. The retriever's quality
upper-bounds the answer's quality: if the right document isn't in the top-k,
the generator can't ground its answer in it. This is the pattern behind
**Perplexity** (web search + summariser), **GitHub Copilot @workspace** (repo
chunks + code LLM) and **ChatGPT with file/web search**.

## What the comparison shows

The transformer beats BM25 on queries that paraphrase the document — e.g.
"online course" maps to the document about "tuition or professional
certifications" without sharing any content words. BM25 still wins on queries
that reuse exact vocabulary (and is essentially free), which is why
production systems usually run a hybrid retriever.
