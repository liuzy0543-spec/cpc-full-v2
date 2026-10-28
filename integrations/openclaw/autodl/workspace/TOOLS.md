# Tools

Important local paths:

- repo root: `$REPO_ROOT`
- example config: `$REPO_ROOT/configs/project_config.example.json`
- Python interpreter: `$PYTHON_BIN`

Preferred CLI form in this AutoDL runtime:

```bash
/root/autodl-tmp/miniconda3_envs/lyx_env/bin/cellpainting-skills <command> ...
```

Fallback form if the executable path changes:

```bash
cd $REPO_ROOT
PYTHONPATH=$REPO_ROOT/src $PYTHON_BIN -m cellpainting_skills <command> ...
```

Preferred discovery commands:

```bash
/root/autodl-tmp/miniconda3_envs/lyx_env/bin/cellpainting-skills list
/root/autodl-tmp/miniconda3_envs/lyx_env/bin/cellpainting-skills describe --skill <skill-key>
```

Preferred execution commands:

```bash
cd $REPO_ROOT
/root/autodl-tmp/miniconda3_envs/lyx_env/bin/cellpainting-skills run \
  --config $REPO_ROOT/configs/project_config.example.json \
  --skill <skill-key> \
  --output-dir <output-dir>
```

Fallback execution form if the executable path changes:

```bash
cd $REPO_ROOT
PYTHONPATH=$REPO_ROOT/src $PYTHON_BIN -m cellpainting_skills run \
  --config $REPO_ROOT/configs/project_config.example.json \
  --skill <skill-key> \
  --output-dir <output-dir>
```

Optional MCP mode:

```bash
$REPO_ROOT/integrations/openclaw/start_cellpaint_mcp_http.sh
```

Preferred segmentation skill keys:

- `cp-prepare-segmentation-inputs`
- `cp-extract-segmentation-artifacts`
- `cp-generate-segmentation-previews`

Preferred classical profiling skill keys:

- `cp-extract-measurements`
- `cp-build-single-cell-table`
- `cyto-aggregate-profiles`
- `cyto-annotate-profiles`
- `cyto-normalize-profiles`
- `cyto-select-profile-features`
- `cyto-summarize-classical-profiles`

Avoid defaulting to older compatibility aliases such as `run-segmentation`, `run-classical-profiling`, or `run-deepprofiler` when a current fine-grained skill exists.

Run skill commands from `$REPO_ROOT`, not from the OpenClaw workspace subdirectory.
