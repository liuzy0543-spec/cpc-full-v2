"""Fail the build if an import points up the layering."""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "docs"))

from layers import check  # noqa: E402


def test_no_upward_imports():
    src = Path(__file__).resolve().parents[1] / "src" / "cellpaint_pipeline"
    violations = check(src)
    assert not violations, "upward imports:\n  " + "\n  ".join(violations)