"""Classify functions by name between a baseline tree and a target tree.

WHY THIS FILE EXISTS
--------------------
The first task book quoted a classification of the upstream functions
(331 / 429 / 297 ...) that came from a script nobody kept.  A later review
noted that 297 = 331 - 11 - 23 is a back-derivation, and that the numbers the
task book called measured were not reproducible from anything in the tree.

So the script lives here now.  Run it, and quote its output - not a
remembered number.

Usage:
    python fnclass.py <baseline-root> <target-root>
"""
import ast
import re
import sys
from pathlib import Path

SCOPE = ("src", "cellpaint_pipeline")
SHIMS = ("_native", "_impl")


def normalise(src: str) -> str:
    """Make the two dispatch styles look the same before comparing."""
    for name in SHIMS:
        src = re.sub(name + r"\((['\"])([A-Za-z_][A-Za-z0-9_]*)\1\)", r"\2", src)
    return src


def functions(root: Path):
    base = root.joinpath(*SCOPE)
    out = {}
    for p in sorted(base.rglob("*.py")):
        if "__pycache__" in str(p):
            continue
        tree = ast.parse(p.read_text(encoding="utf-8"))
        for node in ast.walk(tree):
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                body = ast.get_source_segment(p.read_text(encoding="utf-8"), node) or ""
                out.setdefault(node.name, []).append(normalise(body))
    return out


def main(base_root, target_root):
    a = functions(Path(base_root))
    b = functions(Path(target_root))
    identical = rewritten = 0
    for name, bodies in a.items():
        if name not in b:
            continue
        if bodies == b[name]:
            identical += 1
        else:
            rewritten += 1
    deleted = sum(1 for n in a if n not in b)
    added = sum(1 for n in b if n not in a)
    print("baseline distinct function names :", len(a))
    print("target   distinct function names :", len(b))
    print("identical (name + normalised body):", identical)
    print("rewritten (same name, other body) :", rewritten)
    print("deleted   (only in baseline)      :", deleted)
    print("added     (only in target)        :", added)
    print("check: identical + rewritten + deleted =", identical + rewritten + deleted)


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2])
