# `dp-run-deep-feature-model`

`dp-run-deep-feature-model` is the step that turns a prepared DeepProfiler project into raw learned feature files.

It runs the pretrained DeepProfiler model on each segmented cell and writes the per-field feature bundles that later collection steps turn into tables.

## Purpose

Use this skill when you want:

- learned single-cell morphology features
- the first true biological outputs from the DeepProfiler branch
- raw per-field model outputs before tabular collection

## Main Outcome

After this skill finishes, each segmented cell has a learned morphology representation.

This is the point where the deep-feature branch starts to produce actual per-cell phenotype embeddings.

## Inputs

This skill reads:

- a project config such as `configs/project_config.demo.json`
- a runnable DeepProfiler project from [dp-build-deep-feature-project](dp_build_deep_feature_project.md)
- the DeepProfiler runtime settings from the project config
- an optional output directory

In the demo setup, the model runs on two image fields and profiles four segmented cells across the five standard Cell Painting channels.

## Outputs

The main user-facing results are:

- one learned feature vector for each segmented cell
- raw DeepProfiler feature files for each field
- a DeepProfiler run log

## Agent Demo

This page is based on a real local OpenClaw run against the repository demo project:

- session id: `dprun-local-v3`
- turn timestamp: `2026-05-06 20:03 UTC+8`
- model: `vibe/gpt-5-mini`

### Request

```text
I already built the DeepProfiler project and now want to generate the learned single-cell features for the demo cells. Please use configs/project_config.demo.json, run the deep-feature model for the prepared demo project, and then explain what was produced and what it means biologically.
```

## Structured Trace

```text
user_input:
I already built the DeepProfiler project and now want to generate the learned single-cell features for the demo cells.

planner_step_1:
intent:
Load the documented Cell Painting routing guidance and confirm which public skill runs the prepared DeepProfiler project.
tool_calls:
- read("integrations/openclaw/autodl/workspace/skills/cellpaint-pipeline/SKILL.md")
- exec("cd /root/pipeline/CellPainting-Claw && /root/autodl-tmp/miniconda3_envs/lyx_env/bin/cellpainting-skills describe --skill dp-run-deep-feature-model")
observed_result:
- the routing guidance identified `dp-run-deep-feature-model` as the public step for generating raw DeepProfiler feature files
- the skill description confirmed that this step runs the DeepProfiler model on a prepared project without collecting tables yet

planner_step_2:
intent:
Locate the prepared demo project and check that the expected DeepProfiler project files already exist.
tool_calls:
- exec("cd /root/pipeline/CellPainting-Claw && find demo/workspace/outputs -maxdepth 3 -type f | sort")
- exec("cd /root/pipeline/CellPainting-Claw && sed -n '1,220p' demo/workspace/outputs/agent_demo_dp_project/project_manifest.json")
observed_result:
- the prepared project root was confirmed at `demo/workspace/outputs/agent_demo_dp_project`
- the project already contained the expected DeepProfiler config, metadata, and checkpoint assets

planner_step_3:
intent:
Run the documented model-execution skill against the prepared demo project.
tool_calls:
- exec("cd /root/pipeline/CellPainting-Claw && /root/autodl-tmp/miniconda3_envs/lyx_env/bin/cellpainting-skills run --config /root/pipeline/CellPainting-Claw/configs/project_config.demo.json --skill dp-run-deep-feature-model --project-root /root/pipeline/CellPainting-Claw/demo/workspace/outputs/agent_demo_dp_project --output-dir /root/pipeline/CellPainting-Claw/demo/workspace/outputs/agent_demo_dp_model_run")
observed_result:
- skill_key = dp-run-deep-feature-model
- returncode = 0
- feature_dir = demo/workspace/outputs/agent_demo_dp_project/outputs/cell_painting_cnn_demo/features
- log_path = demo/workspace/outputs/logs/deepprofiler/20260506T120454Z_deepprofiler_profile.log

planner_step_4:
intent:
Inspect the produced feature files so the biological meaning can be summarized concretely.
tool_calls:
- exec("python ... load BR00000001/A01/1.npz and BR00000001/A02/1.npz, inspect array keys and feature shapes")
observed_result:
- there were 2 feature files, one for each field
- each field contained 2 single cells
- each cell was represented by a 672-dimensional learned feature vector
- total cells represented = 4

final_answer_snapshot:
The demo DeepProfiler run completed successfully and produced raw learned feature files for four segmented cells across two image fields.
```

## Observed Result

A real successful run for this skill wrote:

- `outputs/cell_painting_cnn_demo/features/BR00000001/A01/1.npz`
- `outputs/cell_painting_cnn_demo/features/BR00000001/A02/1.npz`
- `demo/workspace/outputs/agent_demo_dp_model_run/pipeline_skill_manifest.json`
- `demo/workspace/outputs/logs/deepprofiler/20260506T120454Z_deepprofiler_profile.log`

The recorded output summary showed:

- `2` image fields
- `4` total cells
- `2` cells in well `A01`
- `2` cells in well `A02`
- `672` learned features per cell
- run `returncode = 0`

## Recorded Reply

```text
I ran the DeepProfiler model on the prepared demo project using the stable skill `dp-run-deep-feature-model`.

Main result:
- the run completed successfully
- the prepared project now contains learned single-cell feature files for two fields
- four segmented cells were represented in total

Produced outputs:
- `outputs/cell_painting_cnn_demo/features/BR00000001/A01/1.npz`
- `outputs/cell_painting_cnn_demo/features/BR00000001/A02/1.npz`
- `demo/workspace/outputs/agent_demo_dp_model_run/pipeline_skill_manifest.json`
- `demo/workspace/outputs/logs/deepprofiler/20260506T120454Z_deepprofiler_profile.log`

What this means biologically:
- each segmented cell now has a learned single-cell morphology embedding
- each embedding is a 672-dimensional fingerprint built from the Cell Painting channels together
- these outputs can now be collected into single-cell and well-level tables in the next step
```

## Next Skills

- [dp-collect-deep-features](dp_collect_deep_features.md)
