# RAG: how knowledge retrieval works

UDA-Hub uses Retrieval-Augmented Generation to answer Cult Pass support
questions strictly from approved Cult Pass knowledge. The Resolver agent
never invents policy — it cites article ids inline (e.g. `[cp_kb_002]`).

## Pipeline

1. **Source** — `data/external/cultpass_articles.jsonl` ships 16 articles from
   CultPass. Notebook **02** loads them into the core database `Knowledge`
   table (article_id, title, category, tags, body).
2. **Chunking** — articles are short (a few hundred characters), so we embed
   `title + body` as a single chunk per article. No splitting needed.
3. **Embeddings** — OpenAI `text-embedding-3-small` (1536 dims) via
   `langchain_openai.OpenAIEmbeddings`. Configurable through
   `UDA_EMBED_MODEL`.
4. **Index** — `langchain_community.vectorstores.FAISS`, persisted to
   `data/models/uda_hub_kb` so subsequent runs skip the embedding cost.
5. **Retrieval** — at query time, the Retriever agent calls
   `similarity_search_with_score(query, k=4)`. FAISS returns L2 distance over
   the unit-normalised vectors; we convert to a cosine-style 0..1 score:
   `score = 1 - dist^2 / 2`.
6. **Confidence** — the *best* score across the top-k becomes
   `state["retrieval_confidence"]`. The supervisor compares it to
   `settings.confidence_threshold` (default 0.55) to decide between the
   Resolver and Escalation paths.
7. **Grounding** — the Resolver receives the retrieved articles as part of
   its prompt and is instructed to cite the article ids in the answer.

## Offline fallback

The Retriever falls back to a pure-Python keyword overlap scorer
(`agentic/retrieval.keyword_retrieve`) when no `OPENAI_API_KEY` is set or if
FAISS is unavailable. This guarantees the rest of the system stays
exercisable during local development and tests.

## Trade-offs and future work

- One chunk per article suits the current corpus; for longer documents,
  switch to `RecursiveCharacterTextSplitter` (`chunk_size=800,
  chunk_overlap=120`).
- L2-to-cosine conversion assumes normalised embeddings, which OpenAI
  embeddings already are. Switching to a non-normalised model would require
  swapping the score formula for raw inner product.
- Long-term semantic memory currently uses substring search; it can be
  upgraded to embeddings without changing the calling code by replacing
  `SqliteLongTermStore.search`.
