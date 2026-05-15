"""Run a prompt against Ollama's kimi-k2:cloud model.

Usage:
    python ollama_kimi_k2.py "your prompt here"
    echo "your prompt" | python ollama_kimi_k2.py
    python ollama_kimi_k2.py            # uses a default prompt

Auth:
    Set OLLAMA_API_KEY to use Ollama Cloud (https://ollama.com).
    Otherwise the script talks to a local Ollama daemon at
    OLLAMA_HOST (default http://localhost:11434), which must already
    be signed in via `ollama signin` to access cloud models.
"""

from __future__ import annotations

import os
import sys

import httpx

MODEL = "kimi-k2:cloud"
CLOUD_HOST = "https://ollama.com"
DEFAULT_LOCAL_HOST = "http://localhost:11434"


def resolve_host_and_headers() -> tuple[str, dict[str, str]]:
    api_key = os.environ.get("OLLAMA_API_KEY")
    if api_key:
        host = os.environ.get("OLLAMA_HOST", CLOUD_HOST)
        return host, {"Authorization": f"Bearer {api_key}"}
    host = os.environ.get("OLLAMA_HOST", DEFAULT_LOCAL_HOST)
    return host, {}


def read_prompt() -> str:
    if len(sys.argv) > 1:
        return " ".join(sys.argv[1:])
    if not sys.stdin.isatty():
        piped = sys.stdin.read().strip()
        if piped:
            return piped
    return "In one short paragraph, introduce yourself and the model you are."


def stream_chat(host: str, headers: dict[str, str], prompt: str) -> None:
    url = f"{host.rstrip('/')}/api/chat"
    payload = {
        "model": MODEL,
        "messages": [{"role": "user", "content": prompt}],
        "stream": True,
    }
    with httpx.stream("POST", url, json=payload, headers=headers, timeout=None) as resp:
        resp.raise_for_status()
        for line in resp.iter_lines():
            if not line:
                continue
            import json
            chunk = json.loads(line)
            piece = chunk.get("message", {}).get("content", "")
            if piece:
                print(piece, end="", flush=True)
            if chunk.get("done"):
                print()
                return


def main() -> int:
    host, headers = resolve_host_and_headers()
    prompt = read_prompt()
    print(f"[ollama] host={host} model={MODEL}", file=sys.stderr)
    try:
        stream_chat(host, headers, prompt)
    except httpx.HTTPStatusError as e:
        print(f"\n[error] {e.response.status_code}: {e.response.text}", file=sys.stderr)
        return 1
    except httpx.HTTPError as e:
        print(f"\n[error] request failed: {e}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
