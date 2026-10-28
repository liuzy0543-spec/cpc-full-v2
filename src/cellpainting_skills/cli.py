"""Focused CLI wrapper for CellPainting-Skills."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from cellpaint_pipeline.config import ProjectConfig
from cellpaint_pipeline.data_access import build_data_request, load_download_plan
from cellpaint_pipeline.skills import (
    available_pipeline_skills,
    get_pipeline_skill_definition,
    pipeline_skill_definition_to_dict,
    pipeline_skill_result_to_dict,
    run_pipeline_skill,
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="cellpainting-skills",
        description="Focused skill catalog and runner for CellPainting-Claw.",
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    list_parser = subparsers.add_parser("list", help="List available Cell Painting skills.")
    list_parser.add_argument("--include-advanced", action="store_true", help="Also show advanced direct-control skill names.")
    list_parser.add_argument("--include-legacy", action="store_true", help="Also show legacy compatibility skill names.")

    describe_parser = subparsers.add_parser("describe", help="Describe one named Cell Painting skill.")
    describe_parser.add_argument("--skill", required=True, help="Named pipeline skill.")

    run_parser = subparsers.add_parser("run", help="Run one named Cell Painting skill.")
    run_parser.add_argument("--config", required=True, help="Path to project config JSON.")
    run_parser.add_argument("--skill", required=True, help="Named pipeline skill.")
    run_parser.add_argument("--output-dir", default=None, help="Optional output directory for the skill run.")
    run_parser.add_argument("--plan-path", default=None, help="Optional path to a previously saved download plan JSON.")
    run_parser.add_argument("--workflow-root", default=None, help="Optional workflow root used by segmentation or DeepProfiler bridge skills.")
    run_parser.add_argument("--export-root", default=None, help="Optional DeepProfiler export root used by project-building skills.")
    run_parser.add_argument("--project-root", default=None, help="Optional DeepProfiler project root used by profiling or feature-collection skills.")
    run_parser.add_argument("--image-csv-path", default=None, help="Optional explicit Image.csv path.")
    run_parser.add_argument("--nuclei-csv-path", default=None, help="Optional explicit Nuclei.csv path.")
    run_parser.add_argument("--load-data-csv-path", default=None, help="Optional explicit load-data CSV path.")
    run_parser.add_argument("--manifest-path", default=None, help="Optional explicit manifest CSV path for validation-like skills.")
    run_parser.add_argument("--object-table-path", default=None, help="Optional explicit CellProfiler object table CSV path.")
    run_parser.add_argument("--single-cell-path", default=None, help="Optional explicit single-cell table path for pycytominer skills.")
    run_parser.add_argument("--aggregated-path", default=None, help="Optional explicit aggregated profile table path for pycytominer annotation or downstream stages.")
    run_parser.add_argument("--annotated-path", default=None, help="Optional explicit annotated profile table path for pycytominer normalization or downstream stages.")
    run_parser.add_argument("--normalized-path", default=None, help="Optional explicit normalized profile table path for pycytominer feature selection.")
    run_parser.add_argument("--feature-selected-path", default=None, help="Optional explicit feature-selected profile table path for summarize-classical-profiles.")
    run_parser.add_argument("--single-cell-parquet-path", default=None, help="Optional explicit DeepProfiler single-cell parquet path for summarize-deepprofiler-profiles.")
    run_parser.add_argument("--well-aggregated-parquet-path", default=None, help="Optional explicit DeepProfiler well-level parquet path for summarize-deepprofiler-profiles.")
    run_parser.add_argument("--object-table", default=None, help="Optional CellProfiler object table name override, for example Cells or Cytoplasm.")
    run_parser.add_argument("--crop-mode", default=None, choices=["masked", "unmasked"], help="Optional crop mode override for export-single-cell-crops.")
    run_parser.add_argument("--workers", type=int, default=0, help="Optional worker override for crop-export skills.")
    run_parser.add_argument("--chunk-size", type=int, default=64, help="Optional chunk size override for preview-like skills.")
    run_parser.add_argument("--gpu", default=None, help="Optional GPU identifier override for DeepProfiler profile runs.")
    run_parser.add_argument("--experiment-name", default=None, help="Optional DeepProfiler experiment name override.")
    run_parser.add_argument("--config-filename", default=None, help="Optional DeepProfiler project config filename override.")
    run_parser.add_argument("--metadata-filename", default=None, help="Optional DeepProfiler metadata filename override.")
    run_parser.add_argument("--request-mode", default="gallery-source", choices=["gallery-source", "gallery-prefix"], help="How to interpret optional request inputs when no plan-path is supplied.")
    run_parser.add_argument("--dataset-id", default=None, help="Optional dataset id override for gallery-source requests.")
    run_parser.add_argument("--source-id", default=None, help="Optional source id override for gallery-source requests.")
    run_parser.add_argument("--prefix", default=None, help="Raw gallery prefix used by gallery-prefix requests.")
    run_parser.add_argument("--subprefix", default="", help="Optional nested subprefix under a dataset/source root.")
    run_parser.add_argument("--bucket", default=None, help="Optional bucket override for the data request.")
    run_parser.add_argument("--include-substring", action="append", default=None, help="Only keep object keys containing this substring. May be repeated.")
    run_parser.add_argument("--exclude-substring", action="append", default=None, help="Skip object keys containing this substring. May be repeated.")
    run_parser.add_argument("--max-files", type=int, default=None, help="Optional max file count for the data request.")
    run_parser.add_argument("--overwrite", action="store_true", help="Overwrite existing local files during download execution.")
    run_parser.add_argument("--dry-run", action="store_true", help="Mark the embedded data request as dry-run.")

    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)

    if args.command == "list":
        payload = [
            pipeline_skill_definition_to_dict(get_pipeline_skill_definition(skill_key))
            for skill_key in available_pipeline_skills(include_advanced=args.include_advanced, include_legacy=args.include_legacy)
        ]
        print(json.dumps(payload, indent=2, ensure_ascii=False))
        return 0

    if args.command == "describe":
        payload = pipeline_skill_definition_to_dict(get_pipeline_skill_definition(args.skill))
        print(json.dumps(payload, indent=2, ensure_ascii=False))
        return 0

    config = ProjectConfig.from_json(args.config)
    download_plan = load_download_plan(Path(args.plan_path).expanduser().resolve()) if args.plan_path else None
    data_request = None
    if download_plan is None:
        data_request = build_data_request(
            mode=args.request_mode,
            dataset_id=args.dataset_id,
            source_id=args.source_id,
            prefix=args.prefix,
            subprefix=args.subprefix,
            bucket=args.bucket,
            include_substrings=args.include_substring or [],
            exclude_substrings=args.exclude_substring or [],
            max_files=args.max_files,
            overwrite=args.overwrite,
            dry_run=args.dry_run,
        )

    result = run_pipeline_skill(
        config,
        args.skill,
        output_dir=Path(args.output_dir).expanduser().resolve() if args.output_dir else None,
        data_request=data_request,
        download_plan=download_plan,
        workflow_root=Path(args.workflow_root).expanduser().resolve() if args.workflow_root else None,
        export_root=Path(args.export_root).expanduser().resolve() if args.export_root else None,
        project_root=Path(args.project_root).expanduser().resolve() if args.project_root else None,
        image_csv_path=Path(args.image_csv_path).expanduser().resolve() if args.image_csv_path else None,
        nuclei_csv_path=Path(args.nuclei_csv_path).expanduser().resolve() if args.nuclei_csv_path else None,
        load_data_csv_path=Path(args.load_data_csv_path).expanduser().resolve() if args.load_data_csv_path else None,
        manifest_path=Path(args.manifest_path).expanduser().resolve() if args.manifest_path else None,
        object_table_path=Path(args.object_table_path).expanduser().resolve() if args.object_table_path else None,
        single_cell_path=Path(args.single_cell_path).expanduser().resolve() if args.single_cell_path else None,
        aggregated_path=Path(args.aggregated_path).expanduser().resolve() if args.aggregated_path else None,
        annotated_path=Path(args.annotated_path).expanduser().resolve() if args.annotated_path else None,
        normalized_path=Path(args.normalized_path).expanduser().resolve() if args.normalized_path else None,
        feature_selected_path=Path(args.feature_selected_path).expanduser().resolve() if args.feature_selected_path else None,
        single_cell_parquet_path=Path(args.single_cell_parquet_path).expanduser().resolve() if args.single_cell_parquet_path else None,
        well_aggregated_parquet_path=Path(args.well_aggregated_parquet_path).expanduser().resolve() if args.well_aggregated_parquet_path else None,
        object_table=args.object_table,
        crop_mode=args.crop_mode,
        workers=args.workers,
        chunk_size=args.chunk_size,
        gpu=args.gpu,
        experiment_name=args.experiment_name,
        config_filename=args.config_filename,
        metadata_filename=args.metadata_filename,
        overwrite=args.overwrite,
    )
    print(json.dumps(pipeline_skill_result_to_dict(result), indent=2, ensure_ascii=False))
    return 0 if result.ok else 1
