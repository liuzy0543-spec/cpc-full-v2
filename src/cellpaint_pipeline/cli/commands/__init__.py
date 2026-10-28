"""Per-domain CLI command groups.

``DOMAIN_ORDER`` is the order in which the groups must register their
sub-commands: it is what ``--help`` prints, so it is part of the
observable surface.
"""
from __future__ import annotations

from cellpaint_pipeline.cli.commands import (
    config,
    transfer,
    profiling,
    segmentation,
    pipeline,
    skills,
    automation,
    workflows,
    deepprofiler,
)

DOMAIN_ORDER = (
    config,
    transfer,
    profiling,
    segmentation,
    pipeline,
    skills,
    automation,
    workflows,
    deepprofiler,
)

__all__ = ['DOMAIN_ORDER', "config", "transfer", "profiling", "segmentation", "pipeline", "skills", "automation", "workflows", "deepprofiler"]
