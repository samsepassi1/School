"""NASA mission embedding pipeline.

Reads raw NASA mission text files, splits them into overlapping chunks,
embeds each chunk with an OpenAI embedding model, and persists the result
to a ChromaDB collection.

Usage examples:
    python embedding_pipeline.py --data-dir ./data --chunk-size 800 --chunk-overlap 120
    python embedding_pipeline.py --update-mode replace --collection-name nasa_missions
    python embedding_pipeline.py --stats-only
"""
from __future__ import annotations

import argparse
import hashlib
import os
import re
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Iterator

import chromadb
from chromadb.config import Settings
from dotenv import load_dotenv
from openai import OpenAI


# ---------------------------------------------------------------------------
# Mission inference
# ---------------------------------------------------------------------------

MISSION_PATTERNS: list[tuple[re.Pattern[str], str]] = [
    (re.compile(r"apollo[_\s-]?11", re.IGNORECASE), "Apollo 11"),
    (re.compile(r"apollo[_\s-]?13", re.IGNORECASE), "Apollo 13"),
    (re.compile(r"challenger|sts[_\s-]?51[_\s-]?l", re.IGNORECASE), "Challenger"),
]


def infer_mission(path: Path) -> str:
    """Infer the mission name from a file path. Returns 'Unknown' on miss."""
    haystack = str(path).lower()
    for pattern, mission in MISSION_PATTERNS:
        if pattern.search(haystack):
            return mission
    return "Unknown"


# ---------------------------------------------------------------------------
# Chunking
# ---------------------------------------------------------------------------

@dataclass
class Chunk:
    text: str
    source: str
    mission: str
    chunk_index: int


def _split_paragraphs(text: str) -> list[str]:
    parts = re.split(r"\n\s*\n", text.strip())
    return [p.strip() for p in parts if p.strip()]


def chunk_text(text: str, *, chunk_size: int, chunk_overlap: int) -> list[str]:
    """Greedy character-based chunker with overlap.

    Guarantees no chunk exceeds ``chunk_size`` characters and that consecutive
    chunks share approximately ``chunk_overlap`` characters of trailing
    context.
    """
    if chunk_size <= 0:
        raise ValueError("chunk_size must be > 0")
    if chunk_overlap < 0 or chunk_overlap >= chunk_size:
        raise ValueError("chunk_overlap must satisfy 0 <= overlap < chunk_size")

    text = text.strip()
    if not text:
        return []

    paragraphs = _split_paragraphs(text)
    chunks: list[str] = []
    buffer = ""

    def flush_buffer() -> None:
        nonlocal buffer
        if buffer.strip():
            chunks.append(buffer.strip())
        buffer = ""

    for paragraph in paragraphs:
        # Long paragraphs are sliced directly to respect chunk_size.
        if len(paragraph) > chunk_size:
            flush_buffer()
            start = 0
            while start < len(paragraph):
                end = min(start + chunk_size, len(paragraph))
                chunks.append(paragraph[start:end].strip())
                if end == len(paragraph):
                    break
                start = end - chunk_overlap
            continue

        prospective = (buffer + "\n\n" + paragraph).strip() if buffer else paragraph
        if len(prospective) <= chunk_size:
            buffer = prospective
        else:
            flush_buffer()
            buffer = paragraph

    flush_buffer()

    # Inject overlap between sequential chunks by prepending the tail of the
    # previous chunk. This keeps chunks below chunk_size while still giving the
    # retriever shared context.
    if chunk_overlap == 0 or len(chunks) <= 1:
        return chunks

    overlapped: list[str] = [chunks[0]]
    for i in range(1, len(chunks)):
        prev_tail = overlapped[-1][-chunk_overlap:]
        combined = (prev_tail + " " + chunks[i]).strip()
        if len(combined) > chunk_size:
            combined = combined[-chunk_size:]
        overlapped.append(combined)
    return overlapped


def iter_documents(data_dir: Path) -> Iterator[tuple[Path, str]]:
    """Yield ``(path, text)`` pairs for every .txt file under ``data_dir``."""
    if not data_dir.exists():
        raise FileNotFoundError(f"Data directory not found: {data_dir}")
    for path in sorted(data_dir.rglob("*.txt")):
        try:
            text = path.read_text(encoding="utf-8")
        except UnicodeDecodeError:
            text = path.read_text(encoding="latin-1")
        if text.strip():
            yield path, text


def build_chunks(
    data_dir: Path,
    *,
    chunk_size: int,
    chunk_overlap: int,
) -> list[Chunk]:
    out: list[Chunk] = []
    for path, text in iter_documents(data_dir):
        mission = infer_mission(path)
        pieces = chunk_text(text, chunk_size=chunk_size, chunk_overlap=chunk_overlap)
        for idx, piece in enumerate(pieces):
            out.append(
                Chunk(
                    text=piece,
                    source=str(path.relative_to(data_dir.parent) if path.is_relative_to(data_dir.parent) else path),
                    mission=mission,
                    chunk_index=idx,
                )
            )
    return out


def chunk_id(chunk: Chunk) -> str:
    digest = hashlib.sha1(chunk.text.encode("utf-8")).hexdigest()[:12]
    return f"{chunk.source}::{chunk.chunk_index}::{digest}"


# ---------------------------------------------------------------------------
# Embeddings
# ---------------------------------------------------------------------------

def embed_batch(client: OpenAI, model: str, texts: list[str]) -> list[list[float]]:
    """Embed a batch of texts, retrying on transient errors."""
    last_err: Exception | None = None
    for attempt in range(4):
        try:
            resp = client.embeddings.create(model=model, input=texts)
            return [item.embedding for item in resp.data]
        except Exception as exc:  # noqa: BLE001 - surface as retry/raise
            last_err = exc
            wait = 2 ** attempt
            print(f"[embed] retry {attempt + 1} after {wait}s: {exc}", file=sys.stderr)
            time.sleep(wait)
    assert last_err is not None
    raise last_err


# ---------------------------------------------------------------------------
# ChromaDB persistence
# ---------------------------------------------------------------------------

def get_chroma_collection(chroma_dir: Path, collection_name: str):
    chroma_dir.mkdir(parents=True, exist_ok=True)
    client = chromadb.PersistentClient(
        path=str(chroma_dir),
        settings=Settings(anonymized_telemetry=False, allow_reset=True),
    )
    return client, client.get_or_create_collection(
        name=collection_name,
        metadata={"hnsw:space": "cosine"},
    )


def _batched(items: list, size: int) -> Iterable[list]:
    for i in range(0, len(items), size):
        yield items[i : i + size]


def persist_chunks(
    chunks: list[Chunk],
    *,
    collection,
    openai_client: OpenAI,
    embed_model: str,
    update_mode: str,
    batch_size: int = 64,
) -> dict:
    """Embed and write chunks to ChromaDB respecting ``update_mode``."""
    if update_mode not in {"skip", "update", "replace"}:
        raise ValueError("update-mode must be one of: skip, update, replace")

    if update_mode == "replace":
        existing = collection.get()
        if existing and existing.get("ids"):
            collection.delete(ids=existing["ids"])

    # Determine which IDs already exist (for skip / update accounting).
    all_ids = [chunk_id(c) for c in chunks]
    existing_ids: set[str] = set()
    if update_mode in {"skip", "update"}:
        # Chroma.get with explicit IDs returns only those present.
        present = collection.get(ids=all_ids)
        existing_ids = set(present.get("ids", []))

    to_process: list[tuple[str, Chunk]] = []
    skipped = 0
    for cid, chunk in zip(all_ids, chunks):
        if update_mode == "skip" and cid in existing_ids:
            skipped += 1
            continue
        to_process.append((cid, chunk))

    inserted = 0
    updated = 0
    for batch in _batched(to_process, batch_size):
        ids = [cid for cid, _ in batch]
        docs = [c.text for _, c in batch]
        metadatas = [
            {"source": c.source, "mission": c.mission, "chunk_index": c.chunk_index}
            for _, c in batch
        ]
        embeddings = embed_batch(openai_client, embed_model, docs)

        if update_mode == "update":
            existing_in_batch = [cid for cid in ids if cid in existing_ids]
            new_in_batch = [(i, cid) for i, cid in enumerate(ids) if cid not in existing_ids]
            if existing_in_batch:
                collection.update(
                    ids=existing_in_batch,
                    documents=[docs[i] for i, cid in enumerate(ids) if cid in existing_ids],
                    metadatas=[metadatas[i] for i, cid in enumerate(ids) if cid in existing_ids],
                    embeddings=[embeddings[i] for i, cid in enumerate(ids) if cid in existing_ids],
                )
                updated += len(existing_in_batch)
            if new_in_batch:
                collection.add(
                    ids=[ids[i] for i, _ in new_in_batch],
                    documents=[docs[i] for i, _ in new_in_batch],
                    metadatas=[metadatas[i] for i, _ in new_in_batch],
                    embeddings=[embeddings[i] for i, _ in new_in_batch],
                )
                inserted += len(new_in_batch)
        else:
            collection.add(ids=ids, documents=docs, metadatas=metadatas, embeddings=embeddings)
            inserted += len(ids)

    return {
        "considered": len(chunks),
        "inserted": inserted,
        "updated": updated,
        "skipped": skipped,
    }


# ---------------------------------------------------------------------------
# Stats
# ---------------------------------------------------------------------------

def print_collection_stats(collection) -> None:
    count = collection.count()
    print(f"Collection: {collection.name}")
    print(f"Total chunks: {count}")
    if count == 0:
        return
    sample_size = min(count, 1000)
    sample = collection.get(limit=sample_size, include=["metadatas"])
    missions: dict[str, int] = {}
    sources: set[str] = set()
    for md in sample.get("metadatas", []) or []:
        missions[md.get("mission", "Unknown")] = missions.get(md.get("mission", "Unknown"), 0) + 1
        sources.add(md.get("source", "Unknown"))
    print(f"Distinct source files (in sample of {sample_size}): {len(sources)}")
    print("Chunks per mission (in sample):")
    for mission, n in sorted(missions.items()):
        print(f"  - {mission}: {n}")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="NASA mission embedding pipeline")
    parser.add_argument("--data-dir", type=Path, default=Path(__file__).parent / "data")
    parser.add_argument("--chroma-dir", type=Path, default=None,
                        help="ChromaDB persistence directory (default from CHROMA_DIR env)")
    parser.add_argument("--collection-name", type=str, default=None,
                        help="Chroma collection name (default from CHROMA_COLLECTION env)")
    parser.add_argument("--chunk-size", type=int, default=800,
                        help="Max chunk size in characters")
    parser.add_argument("--chunk-overlap", type=int, default=120,
                        help="Overlap between consecutive chunks in characters")
    parser.add_argument("--embed-model", type=str, default=None,
                        help="OpenAI embedding model (default from OPENAI_EMBED_MODEL)")
    parser.add_argument("--update-mode", choices=["skip", "update", "replace"], default="skip",
                        help="How to handle chunks already present in the collection")
    parser.add_argument("--batch-size", type=int, default=64,
                        help="OpenAI embedding batch size")
    parser.add_argument("--stats-only", action="store_true",
                        help="Print collection stats and exit without ingesting")
    return parser


def main(argv: list[str] | None = None) -> int:
    load_dotenv()
    args = build_arg_parser().parse_args(argv)

    chroma_dir = args.chroma_dir or Path(os.getenv("CHROMA_DIR", "./chroma_db"))
    collection_name = args.collection_name or os.getenv("CHROMA_COLLECTION", "nasa_missions")
    embed_model = args.embed_model or os.getenv("OPENAI_EMBED_MODEL", "text-embedding-3-small")

    _, collection = get_chroma_collection(chroma_dir, collection_name)

    if args.stats_only:
        print_collection_stats(collection)
        return 0

    if not os.getenv("OPENAI_API_KEY"):
        print("ERROR: OPENAI_API_KEY is not set. Copy .env.example -> .env and fill it in.",
              file=sys.stderr)
        return 2

    openai_client = OpenAI()

    print(f"Scanning data dir: {args.data_dir}")
    chunks = build_chunks(
        args.data_dir,
        chunk_size=args.chunk_size,
        chunk_overlap=args.chunk_overlap,
    )
    print(f"Produced {len(chunks)} chunks (chunk_size={args.chunk_size}, "
          f"overlap={args.chunk_overlap}).")
    if not chunks:
        print("No content to ingest.")
        return 0

    print(f"Embedding with {embed_model} and writing to '{collection_name}' "
          f"at {chroma_dir} (update_mode={args.update_mode})...")
    stats = persist_chunks(
        chunks,
        collection=collection,
        openai_client=openai_client,
        embed_model=embed_model,
        update_mode=args.update_mode,
        batch_size=args.batch_size,
    )
    print(
        f"Done. considered={stats['considered']} "
        f"inserted={stats['inserted']} updated={stats['updated']} skipped={stats['skipped']}"
    )
    print_collection_stats(collection)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
