"""Report markdown links under docs/ that no longer resolve.

Documentation rots by way of links: a file gets archived, the reference does not
follow, and the next reader lands on nothing.  This is cheap enough to run after
any docs move, and it is referenced by `docs/README.md` so the maintenance rule
has something to point at.
"""

from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
DOCS = ROOT / 'docs'


def check(docs: Path) -> list[tuple[str, str, str]]:
    problems: list[tuple[str, str, str]] = []
    for doc in sorted(docs.rglob('*.md')):
        text = doc.read_text(encoding='utf-8', errors='ignore')
        for match in re.finditer(r'\[[^\]]*\]\(([^)#\s]+\.md)(?:#[^)]*)?\)', text):
            target = match.group(1)
            if target.startswith(('http://', 'https://')):
                continue
            resolved = (doc.parent / target).resolve()
            if not resolved.exists():
                problems.append((str(doc.relative_to(ROOT)), target, 'missing'))
    return problems


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument('--docs', default=str(DOCS))
    args = parser.parse_args()

    problems = check(Path(args.docs))
    if not problems:
        print(f'OK: every markdown link under {Path(args.docs).name}/ resolves')
        return 0
    print(f'{len(problems)} broken link(s):')
    for doc, target, why in problems:
        print(f'  {doc} -> {target}  [{why}]')
    return 1


if __name__ == '__main__':
    raise SystemExit(main())