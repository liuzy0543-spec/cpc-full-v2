"""Public CellPainting-Claw compatibility layer."""

from __future__ import annotations

from importlib import import_module
from typing import Any

from cellpaint_pipeline import __all__ as _legacy_all
from cellpaint_pipeline import __version__

__all__ = list(_legacy_all)


def __getattr__(name: str) -> Any:
    if name == '__version__':
        return __version__
    module = import_module('cellpaint_pipeline')
    value = getattr(module, name)
    globals()[name] = value
    return value


def __dir__() -> list[str]:
    return sorted(set(globals()) | set(__all__))
