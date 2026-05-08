"""Runtime configuration. All paths resolve relative to the solution/ root so the
project can be moved without breaking imports."""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

try:
    from dotenv import load_dotenv

    load_dotenv()
except Exception:
    pass


SOLUTION_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = SOLUTION_ROOT / "data"
CORE_DIR = DATA_DIR / "core"
EXTERNAL_DIR = DATA_DIR / "external"
MODELS_DIR = DATA_DIR / "models"
for d in (CORE_DIR, EXTERNAL_DIR, MODELS_DIR):
    d.mkdir(parents=True, exist_ok=True)


@dataclass(frozen=True)
class Settings:
    # Databases
    core_db_path: str = os.getenv("UDA_CORE_DB", str(CORE_DIR / "uda_hub.db"))
    external_db_path: str = os.getenv("UDA_EXTERNAL_DB", str(EXTERNAL_DIR / "cultpass.db"))
    checkpoint_path: str = os.getenv("UDA_CHECKPOINT_DB", str(CORE_DIR / "checkpoints.db"))
    longterm_path: str = os.getenv("UDA_LONGTERM_DB", str(CORE_DIR / "longterm.db"))

    # Knowledge-base files
    cultpass_articles_path: str = os.getenv(
        "UDA_CULTPASS_ARTICLES",
        str(EXTERNAL_DIR / "cultpass_articles.jsonl"),
    )
    vectorstore_path: str = os.getenv(
        "UDA_VECTORSTORE", str(MODELS_DIR / "uda_hub_kb")
    )

    # LLM
    llm_model: str = os.getenv("UDA_LLM_MODEL", "gpt-4o-mini")
    embed_model: str = os.getenv("UDA_EMBED_MODEL", "text-embedding-3-small")

    # Routing
    confidence_threshold: float = float(os.getenv("UDA_CONFIDENCE_THRESHOLD", "0.55"))


settings = Settings()
