# `data-inspect-availability`

`data-inspect-availability` is the inspection step that checks which Cell Painting data sources are available before any download begins.

It tells you which dataset the project is currently pointed at and which discovery routes are ready to use.

## Purpose

Use this skill when you want:

- see which dataset and source the current run is using
- check whether downloadable data can be found before starting a download

## Main Outcome

This skill does not download any data. It tells you what this run is currently pointed at and whether the configured discovery routes are working.

In the demo setup, it confirms:

- the default dataset
- the default source
- whether Gallery discovery works
- whether Quilt discovery works
- whether CPGData inventory discovery works

## Inputs

This skill reads:

- a project config such as `configs/project_config.demo.json`
- the data-access settings inside that config
- an optional output directory

In the demo config, those settings already define the default dataset, default source, and the configured discovery routes.

## Outputs

This skill writes:

- `data_access_summary.json`
  The main inspection result. It records the configured dataset, source, and discovery results.
- `pipeline_skill_manifest.json`
  The run record for this skill invocation.

## Agent Demo

This page is based on a real local OpenClaw turn against the repository demo config:

- recorded on `2026-05-06 13:31 UTC`
- model: `vibe/gpt-5-mini`

### Request

```text
Before I download anything, I want to confirm which Cell Painting dataset this demo is pointing to and whether the configured source is actually reachable.
```

## Structured Trace

```text
user_input:
Before I download anything, I want to confirm which Cell Painting dataset this demo is pointing to and whether the configured source is actually reachable.

recorded_turn:
- the saved OpenClaw turn answered this request directly
- the reply confirmed the configured dataset and source
- the reply reported that discovery and planning routes were reachable in the current environment

final_answer_snapshot:
The demo points to cpg0016-jump / source_4, and the configured source is reachable for discovery and download planning.
```

## Observed Result

The recorded turn confirmed that the demo config is currently pointed at:

- dataset `cpg0016-jump`
- source `source_4`

The recorded reply also confirmed that the current environment could reach:

- Gallery / S3 listing
- CPGData inventory discovery
- the configured Quilt adapter path
- download planning for the configured dataset/source pair

The reply explicitly mentioned these installed access tools:

- `boto3`
- `botocore`
- `cpgdata`
- `quilt3`
- `aws` CLI

## Recorded Reply

```text
I confirmed the demo is configured to point to:

- Dataset: `cpg0016-jump`
- Source: `source_4`

So the configured default target is:

- `cpg0016-jump/source_4/`

Is the configured source reachable?
Yes — the configured source is reachable through the data-access layer I checked.

What I verified
- the Cell Painting Gallery route is working
- dataset `cpg0016-jump` is present
- source `source_4` is present under that dataset
- a download plan can be created successfully for that dataset/source pair

Access routes
- Gallery / S3 listing: working
- cpgdata inventory route: working
- Quilt adapter: available and did not error in the summary check

Environment readiness
- `boto3`
- `botocore`
- `cpgdata`
- `quilt3`
- `aws` CLI

Bottom line
- the demo points to `cpg0016-jump / source_4`
- the configured source is reachable
- the environment can discover and plan downloads from it successfully
```

## Next Skills

- [data-plan-download](data_plan_download.md)
- [data-download](data_download.md)
