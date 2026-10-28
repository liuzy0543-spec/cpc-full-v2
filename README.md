<!-- release-note -->
> **本树是 4 个官方权重方法 + 3 个替身方法的完全体**：19 个 skill + 训练/数据/配置/实验 + 7 个 embedding 方法，已实测全绿。见 [RELEASE.md](RELEASE.md)，一条命令验证：`python verify_all.py`。

# CellPainting-Claw

[Documentation](https://cellpainting-claw.readthedocs.io/en/latest/)

CellPainting-Claw turns a fragmented Cell Painting stack into one documented skill catalog. Instead of asking users to stitch together data-access utilities, CellProfiler steps, pycytominer processing, DeepProfiler preparation, and agent glue by hand, the project exposes the same skills for direct use and for agent-mediated use.

## Public Entry Points

The repository exposes three public entry points.

| Entry point | Use it when | What it gives you |
| --- | --- | --- |
| `cellpainting_skills` | you want to run documented tasks directly | the public skill catalog for data access, profiling, segmentation, crop export, and DeepProfiler |
| `OpenClaw` | you want to use the same skills through natural language | an agent front end that maps prompts onto the same skill catalog |
| `cellpainting_claw` | you need lower-level control | the advanced toolkit package for configuration, direct helpers, and MCP serving |

## Foundation Packages

CellPainting-Claw integrates these package and tool families in workflow order.

| Capability area | Packages or tools | Main capability |
| --- | --- | --- |
| Data access | `boto3`, `quilt3`, `cpgdata` | inspect Cell Painting sources and download dataset slices |
| Measurement extraction | `CellProfiler` | run profiling and segmentation pipelines, write masks, and export object tables |
| Classical profile generation | `pycytominer` | aggregate and normalize Cell Painting features |
| Deep feature extraction | `DeepProfiler` | turn segmentation-guided single-cell inputs into embedding features |

## Public Skill Catalog

The public skill catalog is grouped by concrete user tasks.

### Data Access

| Skill | Main result |
| --- | --- |
| `data-inspect-availability` | inspect configured sources and write an availability summary |
| `data-plan-download` | resolve a download request into a saved download plan |
| `data-download` | download one dataset slice into a local cache |

### Profiling

| Skill | Main result |
| --- | --- |
| `cp-extract-measurements` | write CellProfiler measurement tables |
| `cp-build-single-cell-table` | merge CellProfiler tables into one single-cell measurements table |
| `cyto-aggregate-profiles` | aggregate single-cell measurements into classical profiles |
| `cyto-annotate-profiles` | attach metadata to aggregated profiles |
| `cyto-normalize-profiles` | normalize annotated profiles |
| `cyto-select-profile-features` | write the feature-selected classical profile table |
| `cyto-summarize-classical-profiles` | turn classical profile outputs into readable summaries and PCA views |

### Segmentation

| Skill | Main result |
| --- | --- |
| `cp-prepare-segmentation-inputs` | prepare the load-data table used by segmentation |
| `cp-extract-segmentation-artifacts` | write segmentation masks, labels, outlines, and object tables |
| `cp-generate-segmentation-previews` | write preview PNGs for quick segmentation review |
| `crop-export-single-cell-crops` | export masked or unmasked single-cell crop stacks |

### Deep Features

| Skill | Main result |
| --- | --- |
| `dp-export-deep-feature-inputs` | build DeepProfiler-ready metadata and location files |
| `dp-build-deep-feature-project` | assemble a runnable DeepProfiler project directory |
| `dp-run-deep-feature-model` | run the DeepProfiler model and write raw feature files |
| `dp-collect-deep-features` | collect raw feature files into tabular outputs |
| `dp-summarize-deep-features` | turn DeepProfiler outputs into readable summaries and PCA views |

## Quick Start

Create the environment and install the package:

```bash
cd <repo-root>
conda env create -f environment/cellpainting-claw.environment.yml
conda activate cellpainting-claw
pip install -e .
```

Use the bundled demo config:

```bash
CONFIG=configs/project_config.demo.json
```

Look at the skill catalog and inspect one task:

```bash
cellpainting-skills list
cellpainting-skills describe --skill cp-extract-segmentation-artifacts
```

Run one skill:

```bash
cellpainting-skills run \
  --config "$CONFIG" \
  --skill cp-extract-segmentation-artifacts \
  --output-dir outputs/demo_segmentation
```

Run a follow-up skill from the segmentation result:

```bash
cellpainting-skills run \
  --config "$CONFIG" \
  --skill crop-export-single-cell-crops \
  --workflow-root outputs/demo_segmentation \
  --crop-mode masked \
  --output-dir outputs/demo_crops
```

## Python API Example

```python
from cellpainting_claw import ProjectConfig
import cellpainting_skills as cps

config = ProjectConfig.from_json("configs/project_config.demo.json")
result = cps.run_pipeline_skill(
    config,
    "cp-extract-segmentation-artifacts",
    output_dir="outputs/demo_segmentation",
)
print(result.ok)
print(result.primary_outputs["workflow_root"])
```

## Agent And OpenClaw Integration

OpenClaw is an **optional natural-language entry point** for the same toolkit.

In normal use, OpenClaw connects to the MCP surface and maps natural-language requests onto the documented skills. It does not replace the core library or CLI.

## Documentation

Start here:

- [Read the Docs](https://cellpainting-claw.readthedocs.io/en/latest/)
- `docs/index.md`
- `docs/skills/index.md`
- `docs/cli/index.md`
- `docs/openclaw/index.md`

## Publication Hygiene

Repository-managed docs and config templates are English-only and do not store provider keys.

If a provider key was previously typed into a shell, written into a local config, or pasted into chat, rotate or revoke it before publishing the project.
