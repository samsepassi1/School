# NASA Mission Intelligence — RAG Chat System

End-to-end retrieval-augmented Q&A over Apollo 11, Apollo 13, and Challenger
archive material. Built around ChromaDB (vector store), OpenAI (embeddings +
chat), RAGAS (real-time evaluation), and Streamlit (chat UI).

```
nasa_rag/
├── data/                       NASA mission .txt sources (Apollo 11/13, Challenger)
├── embedding_pipeline.py       CLI: chunk, embed, persist to ChromaDB
├── rag_client.py               Semantic retrieval + context formatting
├── llm_client.py               OpenAI chat wrapper with NASA-expert system prompt
├── ragas_evaluator.py          Faithfulness / Answer Relevancy / BLEU / ROUGE-L
├── run_evaluation.py           Batch evaluation runner over the eval set
├── chat.py                     Streamlit chat application
├── evaluation_dataset.txt      Human-readable eval questions (rubric-required)
├── test_questions.json         Same eval set in JSON form
├── requirements.txt
└── .env.example
```

## Setup

```bash
cd nasa_rag
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env   # then edit and add your OPENAI_API_KEY
```

## 1. Build the index

```bash
python embedding_pipeline.py \
    --data-dir ./data \
    --chunk-size 800 \
    --chunk-overlap 120 \
    --update-mode skip
```

Useful flags:

| flag | purpose |
|------|---------|
| `--data-dir`         | directory tree of `.txt` files to ingest |
| `--chunk-size`       | max characters per chunk |
| `--chunk-overlap`    | character overlap between consecutive chunks |
| `--update-mode`      | `skip` (default) / `update` / `replace` |
| `--collection-name`  | Chroma collection (default `nasa_missions`) |
| `--chroma-dir`       | Chroma persistence directory (default `./chroma_db`) |
| `--embed-model`      | OpenAI embedding model (default `text-embedding-3-small`) |
| `--stats-only`       | print collection size + per-mission breakdown and exit |

## 2. Launch the chat UI

```bash
streamlit run chat.py
```

Sidebar controls: chat model, mission focus filter, top-k retrieval, LLM
temperature, RAGAS toggle, retrieved-context toggle, reset conversation.

## 3. Run batch evaluation

```bash
python run_evaluation.py --eval-set evaluation_dataset.txt --top-k 4
# or JSON
python run_evaluation.py --eval-set test_questions.json --out report.json
```

Output: per-question metrics + aggregate mean/min/max/stdev.

## Rubric mapping

| Rubric area | Where it lives |
|-------------|----------------|
| Chunking with configurable size / overlap, capped chunk length | `embedding_pipeline.chunk_text` + `--chunk-size`, `--chunk-overlap` |
| OpenAI embeddings + per-chunk metadata (source, mission) | `embedding_pipeline.persist_chunks` |
| `--update-mode skip / update / replace` | `embedding_pipeline.persist_chunks` |
| Persisted ChromaDB collection + `--stats-only` aggregate | `get_chroma_collection`, `print_collection_stats` |
| Semantic retrieval with configurable `k` and mission filter | `RAGClient.search` |
| Clean context construction with separators, sources, dedup, sort by score | `rag_client.format_context` |
| NASA-expert system prompt + conversation history | `llm_client.SYSTEM_PROMPT`, `LLMClient.history` |
| Context-grounded answer generation | `LLMClient.generate` |
| RAGAS Response Relevancy + Faithfulness (+ BLEU, ROUGE-L) | `ragas_evaluator.RAGASEvaluator` |
| (question, context, answer) triple scoring with error handling | `RAGASEvaluator.score` |
| Batch eval with per-question + aggregate metrics | `RAGASEvaluator.evaluate_batch`, `run_evaluation.py` |
| Eval dataset, 5+ mission-relevant Qs across categories | `evaluation_dataset.txt`, `test_questions.json` |

## Notes on data sources

The `data/` tree ships with concise, factual excerpts derived from public
NASA mission summaries and transcripts (Apollo 11 landing comm loop, Apollo
13 crisis transcript, Rogers Commission findings). Drop additional `.txt`
files into the appropriate mission subdirectory and re-run the pipeline with
`--update-mode update` to incorporate them.
