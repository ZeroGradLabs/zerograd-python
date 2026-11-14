"""Tidy executed notebooks before they are committed: drop per-cell timing metadata and report each file's size.

    python scripts/clean_notebook_outputs.py examples/*.ipynb
"""

import json
import sys
from pathlib import Path

LIMIT_MB = 5  # a notebook bigger than this is reported, so a stray huge output is noticed in review


def clean(path: Path) -> float:
    nb = json.loads(path.read_text())
    for cell in nb["cells"]:
        cell.get("metadata", {}).pop("execution", None)  # nbclient's start/end timestamps: noise in every diff
    nb.get("metadata", {}).pop("widgets", None)
    path.write_text(json.dumps(nb, indent=1, ensure_ascii=False) + "\n")
    return path.stat().st_size / 1e6


if __name__ == "__main__":
    big = []
    for arg in sys.argv[1:]:
        mb = clean(Path(arg))
        print(f"{arg}: {mb:.2f} MB")
        if mb > LIMIT_MB:
            big.append(arg)
    if big:
        print(f"Over {LIMIT_MB} MB: {', '.join(big)}")
