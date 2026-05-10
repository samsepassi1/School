"""Run every Phase 1 test script and the Phase 2 workflow, writing each
script's stdout+stderr to a file under <phase>/outputs/.

Usage:
    OPENAI_API_KEY=sk-... python capture_outputs.py
    # or, with a .env file in this folder:
    python capture_outputs.py

The resulting files satisfy rubric items 5 and 12, which require captured
evidence of every test script execution and the full workflow run.
"""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path
from typing import List, Tuple

try:
    from dotenv import load_dotenv
    load_dotenv(Path(__file__).parent / ".env")
except ImportError:
    pass


ROOT = Path(__file__).parent

PHASE_1_SCRIPTS: List[str] = [
    "direct_prompt_agent.py",
    "augmented_prompt_agent.py",
    "knowledge_augmented_prompt_agent.py",
    "rag_knowledge_prompt_agent.py",
    "evaluation_agent.py",
    "routing_agent.py",
    "action_planning_agent.py",
]

PHASE_2_SCRIPT = "agentic_workflow.py"


def _run(script_dir: Path, script: str, output_dir: Path) -> Tuple[str, int]:
    """Run ``python <script>`` from ``script_dir`` and write combined
    stdout+stderr to ``output_dir/<script-stem>.txt``. Returns (path, rc)."""
    output_dir.mkdir(parents=True, exist_ok=True)
    out_path = output_dir / (Path(script).stem + ".txt")
    print(f"\n>>> Running {script_dir.name}/{script}")
    proc = subprocess.run(
        [sys.executable, script],
        cwd=script_dir,
        capture_output=True,
        text=True,
    )
    body = (
        f"$ python {script}\n"
        f"# cwd: {script_dir}\n"
        f"# returncode: {proc.returncode}\n"
        f"# --- stdout ---\n{proc.stdout}"
        f"\n# --- stderr ---\n{proc.stderr}"
    )
    out_path.write_text(body, encoding="utf-8")
    print(f"    wrote {out_path}  (rc={proc.returncode})")
    return str(out_path), proc.returncode


def main() -> int:
    if not os.getenv("OPENAI_API_KEY"):
        print(
            "ERROR: OPENAI_API_KEY is not set. Put it in your environment or "
            "create a .env file in this directory."
        )
        return 2

    failures: List[str] = []

    phase_1_dir = ROOT / "phase_1"
    phase_1_outputs = phase_1_dir / "outputs"
    for script in PHASE_1_SCRIPTS:
        path, rc = _run(phase_1_dir, script, phase_1_outputs)
        if rc != 0:
            failures.append(path)

    phase_2_dir = ROOT / "phase_2"
    phase_2_outputs = phase_2_dir / "outputs"
    path, rc = _run(phase_2_dir, PHASE_2_SCRIPT, phase_2_outputs)
    if rc != 0:
        failures.append(path)

    print("\n" + "=" * 60)
    if failures:
        print(f"Completed with {len(failures)} non-zero exit(s):")
        for f in failures:
            print(f"  - {f}")
        return 1
    print("All scripts completed successfully.")
    print(f"Phase 1 outputs: {phase_1_outputs}")
    print(f"Phase 2 outputs: {phase_2_outputs}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
