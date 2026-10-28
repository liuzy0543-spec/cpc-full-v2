# Installation

This page describes the recommended installation path for CellPainting-Claw.

The goal of installation is to obtain a working installation of the main package, the public CLI commands, and the optional OpenClaw-facing MCP support.

## Prerequisites

Before installing the package, make sure you have:

- a local checkout of the `CellPainting-Claw` repository
- Conda available on the target machine
- permission to create a dedicated environment for the toolkit

CellPainting-Claw currently ships with a validated Conda environment file:

- `environment/cellpainting-claw.environment.yml`

That environment targets **Python 3.10** and includes the main toolkit dependencies used by this repository.

## Recommended Installation Path

From the repository root, create and activate the Conda environment:

```bash
conda env create -f environment/cellpainting-claw.environment.yml
conda activate cellpainting-claw
```

Then install the package itself in editable mode:

```bash
pip install -e .
```

This is the recommended installation path for the current project because the validated toolkit depends on a mixed Conda and pip stack.

## Available Components After Installation

After installation, the main things you can use are:

- the Python packages `cellpainting_claw` and `cellpainting_skills`
- the CLI commands `cellpainting-claw` and `cellpainting-skills`
- the optional MCP support used by OpenClaw

## Optional Extras

The project also defines optional extras in `pyproject.toml`. These are mainly useful for a more customized environment rather than the standard installation path.

```bash
pip install -e .[test]
pip install -e .[mcp]
pip install -e .[data-access]
pip install -e .[deepprofiler]
```

In the current repository, the validated Conda environment already includes the main toolkit stack used by the standard installation path.

## Verify The Installation

A minimal post-installation check is to confirm that the public CLI entrypoints are available:

```bash
cellpainting-claw --help
cellpainting-skills --help
```

You can also verify the import surface in Python:

```python
import cellpainting_claw
import cellpainting_skills
```

If both the CLI and imports work, the package layer is installed correctly.

## Recommended Next Step

After installation, the shortest practical next page is:

- [Quick Start](../quick_start/index.ipynb)

From there, you can continue into:

- [Skills](../skills/index.md) for the skill catalog
- [CLI](../cli/index.md) for the command-line usage
- [OpenClaw](../openclaw/index.md) if you want natural-language execution
