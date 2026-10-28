---
orphan: true
---

# CPPIPE Customization Design

This document outlines a practical design for making CellPainting-Claw friendlier to users who need to work with **custom CellProfiler `.cppipe` pipelines**.

The goal is **not** to push all users toward manual CellProfiler GUI construction. The goal is to support three usage levels:

1. **standard templates** for most users
2. **parameterized customization** for users who want controlled variation
3. **bring-your-own `.cppipe`** for advanced users

## Problem Statement

Right now, the repository already includes validated `.cppipe` files for the bundled demo and for backend-oriented execution. That is useful, but it does not yet define a user-facing customization story.

If a user wants to change the CellProfiler processing setup, the current options are still relatively raw:

- edit a `.cppipe` manually
- build a new pipeline inside the CellProfiler GUI
- replace backend assets outside the main public interface

That is workable for expert users, but it is not a good default experience.

## Design Goals

A better design should make it possible to:

- reuse validated default `.cppipe` files
- expose a small set of safe, common configuration choices
- let advanced users provide their own `.cppipe`
- keep the Python API, CLI, and skills layer aligned with the same configuration model
- avoid forcing all customization through manual GUI editing

## Three User Modes

### Mode 1: Standard Templates

This is the default path.

The library ships with a small set of named `.cppipe` templates, for example:

- `profiling_standard`
- `segmentation_mask_export`
- `illumination_then_profiling`
- `deepprofiler_export`

Users choose a template by name and run it through the normal toolkit surfaces.

This mode should cover the majority of users.

### Mode 2: Parameterized Template Customization

This is the recommended middle layer.

Instead of editing a `.cppipe` directly, users select a template and provide a small set of supported options, such as:

- whether illumination correction is enabled
- whether mask export is enabled
- whether single-cell crops are masked or unmasked
- crop size
- whether preview PNG generation is enabled
- whether DeepProfiler export mode is enabled

In this mode, the library still owns the template structure, but users can safely vary a known set of parameters.

This is the best balance between flexibility and reproducibility.

### Mode 3: Bring Your Own `.cppipe`

This is the advanced path.

Users can provide a custom `.cppipe` path directly. The toolkit should then use that pipeline file in place of the default template.

This mode is useful when:

- a lab has its own validated CellProfiler pipeline
- segmentation or measurement logic differs substantially from the defaults
- the user already understands CellProfiler and wants full control

This mode should be supported, but it should not be the primary story for most users.

## Proposed Configuration Model

A future project config could include a new block like this:

```json
{
  "cellprofiler": {
    "profiling_pipeline_template": "profiling_standard",
    "segmentation_pipeline_template": "segmentation_mask_export",
    "custom_profiling_cppipe_path": "",
    "custom_segmentation_cppipe_path": "",
    "template_options": {
      "illumination_correction": true,
      "mask_export": true,
      "single_cell_mode": "masked",
      "crop_size": 128,
      "generate_previews": true,
      "deepprofiler_export": false
    }
  }
}
```

Interpretation:

- `profiling_pipeline_template` and `segmentation_pipeline_template` select library-owned defaults
- `custom_*_cppipe_path` overrides the template if present
- `template_options` only affects templates that declare support for those options

## Proposed Validation Rules

The toolkit should validate these rules early:

- if `custom_*_cppipe_path` is set, the file must exist
- users should not set both an unsupported template option and a template that cannot honor it
- invalid combinations should fail fast with a clear config error
- custom `.cppipe` files should be treated as advanced input and not silently patched unless explicitly supported

## Python API Surface

A future Python-side helper layer could expose functions like:

- `list_cppipe_templates()`
- `describe_cppipe_template(template_key)`
- `resolve_cppipe_selection(config, kind="profiling" | "segmentation")`
- `validate_cppipe_configuration(config)`

Purpose:

- let users inspect what templates exist
- show which options each template supports
- resolve the actual `.cppipe` path to run
- catch errors before a long execution starts

## CLI Surface

A user-facing CLI could expose commands such as:

- `cellpainting-claw list-cppipe-templates`
- `cellpainting-claw describe-cppipe-template --template segmentation_mask_export`
- `cellpainting-claw validate-cppipe-config --config ...`
- `cellpainting-claw show-cppipe-selection --config ...`

This would make customization much easier without forcing users to dig through backend directories.

## Skills Layer

The skills layer should not expose raw `.cppipe` complexity directly.

Instead, the skills should continue to expose named tasks, while the `.cppipe` selection is handled through config.

That means:

- users choose a skill such as `run-segmentation`
- the config determines which `.cppipe` template or custom `.cppipe` is actually used

This keeps skills stable while still allowing advanced customization underneath.

## Recommended First Templates

If this feature is implemented incrementally, the first templates should probably be:

- one standard profiling template
- one standard segmentation mask-export template
- one DeepProfiler export template

That is enough to cover the current main capability groups of the toolkit without overdesigning the first version.

## Documentation Strategy

If this is implemented, the documentation should add one small section such as:

- `Using Standard CPPIPE Templates`
- `Customizing A Template`
- `Using Your Own .cppipe`

The documentation should make one thing very clear:

- **most users should start from templates**
- **advanced users can bring their own `.cppipe`**

That is a much better user story than expecting everyone to build a new CellProfiler GUI pipeline from scratch.

## Recommendation

The recommended project direction is:

1. define a small stable template catalog
2. add a constrained set of template options
3. support custom `.cppipe` paths as an advanced override
4. expose inspection and validation through Python and CLI

This would make CellPainting-Claw much easier to customize while preserving the reproducibility benefits of validated pipeline files.
