"""Verify every path the task book declares.

WHY A SCRIPT AND NOT A grep ONE-LINER
-------------------------------------
The task book used to print:

    grep -oE "D:\\\\项目\\\\[^ |]+" docs/review_prompt.md | while read -r p; do ...

That command was syntactically valid, passed bash -n, and verified nothing.
Two independent reasons:

  * in an ERE the printed pattern needs two literal backslashes, but the paths
    in the document have one, so it matched almost nothing;
  * [^ |]+ stops at a space, and four of the six declared paths contain one
    ("cpc github v1", "cpc github v2", ...).

A round-7 review ran it and got a single line: MISS ]

So the check lives here, reads the document, and resolves the paths itself.
This is the same reason measure.py exists rather than a grep.

Usage:
    python checkpaths.py [path-to-review_prompt.md]
"""
import re
import sys
from pathlib import Path

# a declared path is anything inside backticks that starts with a drive letter
DECLARED = re.compile("\x60([A-Za-z]:[^\x60]*?)\x60")


def declared_paths(doc: Path):
    text = doc.read_text(encoding="utf-8")
    found = []
    for m in DECLARED.finditer(text):
        p = m.group(1).strip()
        if p and p not in found:
            found.append(p)
    return found


def main():
    doc = Path(sys.argv[1]) if len(sys.argv) > 1 else Path(__file__).with_name("review_prompt.md")
    paths = declared_paths(doc)
    print("document :", doc)
    print("declared :", len(paths))
    missing = 0
    for p in paths:
        ok = Path(p).exists()
        missing += 0 if ok else 1
        print(("  OK   " if ok else "  MISS ") + p)
    print("missing  :", missing)
    return 1 if missing else 0


if __name__ == "__main__":
    raise SystemExit(main())
