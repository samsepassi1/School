"""Entry point for ``uv run multimodal-moderation``.

Starts Arize Phoenix, the FastAPI backend on :8000, and the Gradio frontend on
:7860 in parallel. Each process is supervised in its own subprocess so a crash
in one component doesn't take the others down.
"""

from __future__ import annotations

import os
import signal
import subprocess
import sys
import time
from typing import List


def _spawn(name: str, cmd: list[str], env: dict[str, str] | None = None) -> subprocess.Popen:
    print(f"[startup] launching {name}: {' '.join(cmd)}")
    return subprocess.Popen(
        cmd,
        env={**os.environ, **(env or {})},
        stdout=sys.stdout,
        stderr=sys.stderr,
    )


def main() -> None:
    procs: List[subprocess.Popen] = []

    # Phoenix (best-effort: skip if not installed)
    try:
        procs.append(
            _spawn(
                "phoenix",
                [sys.executable, "-m", "phoenix.server.main", "serve"],
            )
        )
    except Exception as exc:
        print(f"[startup] phoenix not started: {exc}")

    # FastAPI backend
    procs.append(
        _spawn(
            "fastapi",
            [
                sys.executable,
                "-m",
                "uvicorn",
                "fastapi_app:app",
                "--host",
                "0.0.0.0",
                "--port",
                "8000",
            ],
        )
    )

    # Gradio frontend
    procs.append(_spawn("gradio", [sys.executable, "gradio_app.py"]))

    def shutdown(*_: object) -> None:
        print("[shutdown] stopping services…")
        for p in procs:
            if p.poll() is None:
                p.send_signal(signal.SIGTERM)
        for p in procs:
            try:
                p.wait(timeout=5)
            except subprocess.TimeoutExpired:
                p.kill()
        sys.exit(0)

    signal.signal(signal.SIGINT, shutdown)
    signal.signal(signal.SIGTERM, shutdown)

    # Block forever, surface child failures.
    try:
        while True:
            for p in procs:
                rc = p.poll()
                if rc is not None and rc != 0:
                    print(f"[startup] child exited with code {rc}")
            time.sleep(2)
    except KeyboardInterrupt:
        shutdown()


if __name__ == "__main__":
    main()
