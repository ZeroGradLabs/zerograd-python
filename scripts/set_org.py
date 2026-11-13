"""Point every link at the GitHub org, once it exists: python scripts/set_org.py zerograd-dev

Replaces the __ORG__ placeholder in the README, pyproject, notebooks and docs. Run it once before the first push.
"""

import pathlib
import re
import sys

ROOT = pathlib.Path(__file__).resolve().parents[1]
SUFFIXES = {".md", ".toml", ".ipynb", ".py", ".yml", ".yaml", ".txt"}


def main() -> None:
    if len(sys.argv) != 2 or not re.fullmatch(r"[A-Za-z0-9](?:[A-Za-z0-9-]{0,37}[A-Za-z0-9])?", sys.argv[1]):
        sys.exit("usage: python scripts/set_org.py <github-org>")
    org = sys.argv[1]
    changed = []
    for path in ROOT.rglob("*"):
        if path.suffix not in SUFFIXES or ".git" in path.parts or path == pathlib.Path(__file__).resolve():
            continue
        text = path.read_text(encoding="utf-8")
        if "__ORG__" in text:
            path.write_text(text.replace("__ORG__", org), encoding="utf-8")
            changed.append(path.relative_to(ROOT))
    print(f"__ORG__ -> {org} in {len(changed)} files:", *changed, sep="\n  ")


if __name__ == "__main__":
    main()
