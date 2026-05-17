#!/usr/bin/env python3
"""Run the Arvato notebook, save outputs, export HTML, and zip submission files.

Run this inside the Udacity workspace or a local folder containing the proprietary
Arvato CSV/MD data files. Do not commit the generated data or HTML unless your
course instructions explicitly allow it.
"""
from __future__ import annotations

import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
NOTEBOOK = ROOT / "Identify_Customer_Segments.ipynb"
HTML = ROOT / "Identify_Customer_Segments.html"
ZIP = ROOT / "arvato_customer_segments_submission.zip"
REQUIRED_DATA = [
    "Udacity_AZDIAS_Subset.csv",
    "Udacity_CUSTOMERS_Subset.csv",
    "AZDIAS_Feature_Summary.csv",
]


def run(cmd: list[str]) -> None:
    print("+", " ".join(cmd))
    subprocess.run(cmd, cwd=ROOT, check=True)


def main() -> int:
    missing = [name for name in REQUIRED_DATA if not (ROOT / name).exists()]
    if missing:
        print("Missing required proprietary data files:")
        for name in missing:
            print(f"  - {name}")
        print("\nPlace them in this folder or run this script in the Udacity workspace.")
        return 2

    run([
        sys.executable,
        "-m",
        "jupyter",
        "nbconvert",
        "--to",
        "notebook",
        "--execute",
        "--inplace",
        "--ExecutePreprocessor.timeout=1800",
        str(NOTEBOOK.name),
    ])
    run([sys.executable, "-m", "jupyter", "nbconvert", "--to", "html", str(NOTEBOOK.name)])
    if ZIP.exists():
        ZIP.unlink()
    run(["zip", "-j", str(ZIP), str(NOTEBOOK), str(HTML)])
    print(f"\nCreated submission zip: {ZIP}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
