"""Public CLI wrapper for CellPainting-Claw."""

from __future__ import annotations

from cellpaint_pipeline.cli import main as _legacy_main


def main(argv: list[str] | None = None) -> int:
    return _legacy_main(
        argv=argv,
        prog="cellpainting-claw",
        description="CellPainting-Claw CLI for standardized Cell Painting workflows.",
    )
