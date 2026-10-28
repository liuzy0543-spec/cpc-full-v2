# Release Notes

`CellPainting-Claw` is now packaged as a cleaner release candidate oriented around three goals:

- a stable Python and CLI entry surface
- a documented release workflow
- secret-safe agent integration templates

## What This Release Is Good For

- wrapping the validated Cell Painting workflow as a reusable library
- running profiling, segmentation, and DeepProfiler branches through stable entrypoints
- exposing the workflow to MCP and OpenClaw-style agent systems
- preparing a source bundle that can be shared or published more safely

## Main Improvements in This Release

- English-only release-facing docs
- release smoke-test wrapper
- source release bundle builder
- portable project config template
- env-driven OpenClaw provider setup for both AutoDL and Docker
- cleaner ignore rules for local runtime state and secrets

## Recommended First Commands

```bash
cd <repo-root>
./scripts/run_release_smoke_test.sh
./scripts/build_release_bundle.sh
```

## Important Operational Note

If an API key was previously exposed in local configs, shell history, or chat logs, rotate it before sharing the repository or any derived image.
