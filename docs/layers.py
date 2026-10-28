"""Assert the internal import layering.

WHY THIS EXISTS
---------------
The package was split from two 1,600-line modules into 65.  During that split a
cycle appeared between ``cli`` and ``cli.app`` that upstream never had: the package
__init__ imported the app module while the app module imported the package to reach
its commands.  Nothing stopped it - there was no rule about which way imports may
point.  This file is that rule.

Layers are ordered; an import may only target the same layer or a lower one.

Run it directly, or let the test in tests/test_layering.py run it.
"""
from __future__ import annotations

import ast
import sys
from pathlib import Path

LAYERS: list[tuple[str, ...]] = [
    # runner only builds on errors + ports, so it is a primitive too
    ("errors", "capabilities", "registry", "config", "ports", "runner"),
    ("data_access", "adapters", "cppipe", "profiling_native", "segmentation_native"),
    ("skills", "workflows", "orchestration", "presets", "delivery", "reporting",
     "evaluation", "profile_summaries", "deepprofiler_pipeline", "nanobot_handoff"),
    ("cli", "mcp_tools", "mcp_server", "public_api"),
]


def layer_of(module: str) -> int:
    top = module.split(".")[0]
    for i, group in enumerate(LAYERS):
        if top in group:
            return i
    return len(LAYERS)          # unknown modules sit on top and may import anything


def check(src: Path) -> list[str]:
    violations = []
    for p in sorted(src.rglob("*.py")):
        if "__pycache__" in str(p):
            continue
        rel = p.relative_to(src).with_suffix("").as_posix()
        name = (rel[:-9] if rel.endswith("/__init__") else rel).replace("/", ".")
        here = name.rstrip(".")
        tree = ast.parse(p.read_text(encoding="utf-8", errors="replace"))
        guarded = set()
        for n in ast.walk(tree):
            if isinstance(n, ast.If) and isinstance(n.test, ast.Name) and n.test.id == "TYPE_CHECKING":
                for b in n.body:
                    for ln in range(b.lineno, (b.end_lineno or b.lineno) + 1):
                        guarded.add(ln)
        for n in ast.walk(tree):
            tgt = None
            if isinstance(n, ast.ImportFrom) and n.module and n.module.startswith("cellpaint_pipeline"):
                tgt = n.module
            elif isinstance(n, ast.Import):
                for a in n.names:
                    if a.name.startswith("cellpaint_pipeline"):
                        tgt = a.name
            if not tgt or n.lineno in guarded:
                continue
            target = tgt.replace("cellpaint_pipeline.", "").replace("cellpaint_pipeline", "")
            if not target or target.split(".")[0] == here.split(".")[0]:
                continue
            if layer_of(target) > layer_of(here):
                violations.append("%s (layer %d) imports %s (layer %d)" % (
                    here, layer_of(here), target, layer_of(target)))
    return violations


def main() -> int:
    root = Path(__file__).resolve().parents[1]
    src = root / "src" / "cellpaint_pipeline"
    bad = check(src)
    if bad:
        print("layering violations: %d" % len(bad))
        for b in bad:
            print("   " + b)
        return 1
    print("layering OK - no upward imports")
    return 0


if __name__ == "__main__":
    sys.exit(main())