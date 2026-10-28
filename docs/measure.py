"""Measure every number the review task book quotes.

WHY A SCRIPT
------------
Round 5 of the review found that the commands printed next to the numbers did
not produce those numbers: `grep -rc "^def _native" src` returned 4 where the
table said 1, because a prefix match also catches `_native_result_to_dict`;
`grep -ro "_native(" src` returned 71 where the table said 25, because a
substring match also catches `run_pycytominer_native(`.  The numbers came from
an AST walk; the grep was decoration that happened to be wrong.

So the numbers are produced here, by the same AST walk, and the task book
quotes this script instead of a grep.

Usage:  python measure.py <tree-root>        # e.g. python measure.py .
"""
from __future__ import annotations

import ast
import json
import sys
from pathlib import Path


def measure(root: Path) -> dict:
    src = root / "src" / "cellpaint_pipeline"
    files = sorted(p for p in src.rglob("*.py") if "__pycache__" not in str(p))
    if not files:
        raise SystemExit(f"no python files under {src}")
    sizes = {p: len(p.read_text(encoding="utf-8").splitlines()) for p in files}
    lines = sum(sizes.values())

    functions: list[tuple[int, str, str]] = []
    ifs = 0
    defs = {"_impl": 0, "_native": 0, "_lazy": 0}
    calls = {"_impl": 0, "_native": 0, "_lazy": 0}
    lazy_bindings = 0
    for p in files:
        tree = ast.parse(p.read_text(encoding="utf-8"))
        ifs += sum(1 for n in ast.walk(tree) if isinstance(n, ast.If))
        for node in ast.walk(tree):
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                functions.append((node.end_lineno - node.lineno + 1, p.name, node.name))
                if node.name in defs:
                    defs[node.name] += 1
            if isinstance(node, ast.Call) and isinstance(node.func, ast.Name):
                if node.func.id in calls:
                    calls[node.func.id] += 1
            if isinstance(node, ast.Assign):
                for t in node.targets:
                    if isinstance(t, ast.Name) and t.id.startswith("_lazy_"):
                        lazy_bindings += 1

    # run_pipeline_skill parameters -- search every file: upstream keeps it in
    # skills.py, the refactor moved it to skills/dispatch.py
    params = None
    params_where = None
    for p in files:
        for node in ast.walk(ast.parse(p.read_text(encoding="utf-8"))):
            if isinstance(node, ast.FunctionDef) and node.name == "run_pipeline_skill":
                a = node.args
                params = len(a.posonlyargs) + len(a.args) + len(a.kwonlyargs) + (1 if a.kwarg else 0)
                params_where = f"{p.name}:{node.lineno}"
    longest = max(functions)
    biggest = max(sizes.items(), key=lambda kv: kv[1])
    return {
        "tree": str(root),
        "files": len(files),
        "lines": lines,
        "functions": len(functions),
        "avg_function_lines": round(sum(f[0] for f in functions) / len(functions), 1),
        # NOT the same thing: total module lines divided by function count.  The
        # two differ a lot (20.7 vs 32.9) because every import, class body, comment
        # and blank line lands in the numerator of this one.
        "lines_per_function": round(lines / len(functions), 1),
        "longest_function": {"name": longest[2], "file": longest[1], "lines": longest[0]},
        "largest_file": {"file": biggest[0].name, "lines": biggest[1]},
        "files_over_400_lines": sum(1 for v in sizes.values() if v > 400),
        "if_density_per_kloc": round(ifs * 1000 / lines, 1),
        "dispatch_definitions": defs,
        "dispatch_definition_total": sum(defs.values()),
        "dispatch_call_sites": calls,
        "dispatch_call_total": sum(calls.values()),
        "run_pipeline_skill_params": params,
        "run_pipeline_skill_at": params_where,
    }


if __name__ == "__main__":
    target = Path(sys.argv[1] if len(sys.argv) > 1 else ".").resolve()
    print(json.dumps(measure(target), indent=2, ensure_ascii=False))
