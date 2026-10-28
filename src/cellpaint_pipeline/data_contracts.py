from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class FieldChannels:
    plate: str
    well: str
    site: str
    image_number: str
    dna_path: Path
    rna_path: Path
    er_path: Path
    agp_path: Path
    mito_path: Path
    outline_path: Path | None = None


@dataclass(frozen=True)
class NucleiLocation:
    image_number: str
    object_number: str
    center_x: float
    center_y: float
