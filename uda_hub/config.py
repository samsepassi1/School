"""Runtime configuration loaded from environment variables."""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

try:
    from dotenv import load_dotenv

    load_dotenv()
except Exception:
    pass


PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_ROOT / "data"
DATA_DIR.mkdir(parents=True, exist_ok=True)


@dataclass(frozen=True)
class Settings:
    db_path: str = os.getenv("UDA_DB_PATH", str(DATA_DIR / "uda_hub.db"))
    checkpoint_path: str = os.getenv(
        "UDA_CHECKPOINT_PATH", str(DATA_DIR / "uda_checkpoints.db")
    )
    longterm_path: str = os.getenv(
        "UDA_LONGTERM_PATH", str(DATA_DIR / "uda_longterm.db")
    )
    vectorstore_path: str = os.getenv(
        "UDA_VECTORSTORE_PATH", str(DATA_DIR / "uda_faiss")
    )
    llm_model: str = os.getenv("UDA_LLM_MODEL", "gpt-4o-mini")
    embed_model: str = os.getenv("UDA_EMBED_MODEL", "text-embedding-3-small")
    confidence_threshold: float = float(os.getenv("UDA_CONFIDENCE_THRESHOLD", "0.55"))


settings = Settings()
