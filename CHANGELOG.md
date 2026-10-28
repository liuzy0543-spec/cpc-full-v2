# Changelog

## Unreleased - modular refactor line

### Added

- `demo/backend/profiling_backend/scripts/07_run_official_cellprofiler.py`.  `cp-extract-measurements`
  shells out to this script, and upstream does not ship it, so the skill always took
  its FileNotFoundError branch and reused the bundled measurement tables while
  returning `ok: true` with `mode: bundled-demo-outputs` buried in `details`.  Those tables hold
  four rows, so the classical branch had never run at dataset scale and the
  degradation was invisible from outside.

  The script follows `segmentation_backend/scripts/03_run_mask_export.py`: it absolutises
  the load-data table, resizes each illumination array to its raw image's shape,
  and runs the analysis pipeline.  The demo's `load_data_with_illum.csv` carries only
  the 19 `Orig*` columns, so when the `Illum*` columns are absent they are built
  from the illumination library that ships with the segmentation backend.

  Measured on the shipped demo: `Image.csv` 96 B -> 114,431 B, `Cells.csv` 535 B ->
  206,278 B, `Nuclei.csv` 159 B -> 179,205 B, and `Cytoplasm.csv` appears where it
  previously did not exist.  `details.mode` is now `None`.


### Changed

- `cellpaint_pipeline/skills.py` (1646 lines) and `cellpaint_pipeline/cli.py`
  (1604 lines) become the `skills/` and `cli/` packages, plus four new modules:
  `ports.py` (subprocess / environment / path ports), `registry.py` (one source
  of truth for the automation entry points), `capabilities.py` (the capability
  catalogue, which removes the reporting -> workflow dependency) and `errors.py`.
  Line counts are `len(text.splitlines())` on `src/cellpaint_pipeline/`.
- `run_pipeline_skill` accepts `**legacy_kwargs` instead of 36 explicit keyword
  parameters: `(config, skill_key, *, inputs=None, **legacy_kwargs)`.  Every
  historical keyword still works, the seven ignored ones still emit
  `DeprecationWarning`, and supplying the same input twice is still rejected.
  One observable difference: an unknown keyword now raises
  `TypeError: Unexpected keyword argument for run_pipeline_skill: <name>`
  instead of the interpreter's default message.
- `cellpaint_pipeline.skills` resolves its re-exports on first access (PEP 562)
  instead of importing the catalogue, the registry and every runner eagerly.
  Importing the package is roughly eight times cheaper, and importing a leaf
  such as `cellpaint_pipeline.skills.definitions` no longer pulls the package.
- Coupling is now below upstream on every measure, computed on the **runtime**
  import graph (a `TYPE_CHECKING` import creates no runtime dependency, so it is
  not an edge).  Seventeen modules that only annotated with `ProjectConfig` and
  never touched it import it under `TYPE_CHECKING` now.

  | | upstream | this tree |
  |---|---|---|
  | average fan-out (edges / all modules) | 2.51 | **2.06** |
  | maximum fan-out | 12 | **11** |
  | maximum fan-in | 25 | **21** |

  The edge count itself rises, 88 -> 134, because the module count rises 35 -> 65;
  edges grow more slowly than modules, which is why both averages fall.  Quoting
  the raw edge count as "coupling went up" reads the wrong number.

  `config` keeps the highest fan-in (21) and has `Ce = 0`: it is a stable leaf,
  which is the shape the Stable Dependencies Principle asks for.

- `docs/layers.py` states the intended layering and fails on any import that points
  up it; `tests/test_layering.py` runs it in CI.  No such rule existed before, which
  is how the `cli` <-> `cli.app` cycle got introduced during the split.


### Removed

- Namespace narrowing.  Splitting the two single-file modules drops the 31
  module-level names they leaked by accident.  `skills` no longer exposes
  `Any`, `Callable`, `Path`, `ProjectConfig`, `asdict`, `dataclass`, `datetime`,
  `field`, `is_dataclass`, `json`, `replace`, `timezone`; `evaluation` no longer
  exposes `PCA`, `StandardScaler`, `np`, `pairwise_distances`, `pd`, `plt`;
  `runner` no longer exposes `datetime`, `deque`, `os`, `re`, `subprocess`,
  `timezone`; `cli` no longer exposes `Path`, `argparse`, `import_module`,
  `json`; `adapters.deepprofiler_project` no longer exposes `os`.  All of them
  were implementation imports rather than API, and none was reachable through a
  `patch()` seam that any test relies on.  The two that did belong to the
  interface - `skills.ExecutionResult` and `cli.ProjectConfig` - are kept.

### Fixed

- `configs/*.json`: `data_access.data_cache_root` and `index_cache_root` are
  resolved relative to `workspace_root`, but the shipped values were written
  relative to the config directory, so both resolved to
  `<repo>/demo/demo/workspace/cache/...`, a path that cannot exist.
- Re-running the demo after renaming its folder left the recorded paths
  pointing at the old name; the shipped artefacts are re-pointed.
- `docs/measure.py`: `avg_function_lines` was computed as *total module lines
  divided by function count*, which counts every import, class body, comment and
  blank line in the numerator.  It reported 32.9 where the mean function body is
  20.7.  The metric now measures what its name says, and the old figure is kept
  under the honest name `lines_per_function`.


## 0.1.0 - 2026-03-24

### Added

- Stable release helper scripts:
  - `scripts/run_release_smoke_test.sh`
  - `scripts/build_release_bundle.sh`
- Release documentation:
  - `docs/release_quickstart.md`
  - `docs/first_run_guide.md`
  - `RELEASE_NOTES.md`
- Portable distribution config template:
  - `configs/project_config.portable.example.json`
- Env-driven OpenClaw provider setup for AutoDL and Docker tracks.

### Changed

- Rewrote release-facing documentation in English.
- Standardized OpenClaw guidance around TUI-first usage.
- Updated Docker integration to use generic OpenAI-compatible provider environment variables instead of provider-specific hardcoding.
- Updated release packaging to exclude local runtime state and secret-bearing files.

### Removed

- Secret-bearing OpenClaw backup configs from the repository-managed tree.
- Local temporary build artifacts, caches, and leftover agent runtime state.

### Validated

- Release smoke test passes on the current validated environment.
- Source release bundle generation completes successfully.
