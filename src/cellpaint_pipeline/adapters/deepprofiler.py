from __future__ import annotations

import csv
import json
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING


if TYPE_CHECKING:
    from cellpaint_pipeline.config import ProjectConfig


IMAGE_CHANNELS = {
    "dna": ("PathName_OrigDNA", "FileName_OrigDNA"),
    "rna": ("PathName_OrigRNA", "FileName_OrigRNA"),
    "er": ("PathName_OrigER", "FileName_OrigER"),
    "agp": ("PathName_OrigAGP", "FileName_OrigAGP"),
    "mito": ("PathName_OrigMito", "FileName_OrigMito"),
}


@dataclass(frozen=True)
class DeepProfilerExportResult:
    export_root: Path
    manifest_path: Path
    field_metadata_path: Path
    locations_root: Path
    field_count: int
    location_file_count: int
    total_nuclei: int
    source_image_csv: Path
    source_nuclei_csv: Path
    source_load_data_csv: Path
    source_label: str


def _load_image_rows(image_csv_path: Path) -> dict[str, dict[str, str]]:
    with image_csv_path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        return {row["ImageNumber"]: row for row in reader}


def _load_nuclei_locations(nuclei_csv_path: Path) -> dict[str, list[dict[str, str]]]:
    by_image_number: dict[str, list[dict[str, str]]] = defaultdict(list)
    with nuclei_csv_path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            by_image_number[row["ImageNumber"]].append(
                {
                    "ImageNumber": row["ImageNumber"],
                    "ObjectNumber": row["ObjectNumber"],
                    "Location_Center_X": row["Location_Center_X"],
                    "Location_Center_Y": row["Location_Center_Y"],
                }
            )
    return by_image_number


def export_deepprofiler_input(
    config: ProjectConfig,
    *,
    output_dir: Path | None = None,
    image_csv_path: Path | None = None,
    nuclei_csv_path: Path | None = None,
    load_data_csv_path: Path | None = None,
    source_label: str | None = None,
) -> DeepProfilerExportResult:
    resolved_image_csv_path, resolved_nuclei_csv_path, resolved_load_data_csv_path = _resolve_source_paths(
        config,
        image_csv_path=image_csv_path,
        nuclei_csv_path=nuclei_csv_path,
        load_data_csv_path=load_data_csv_path,
    )
    resolved_source_label = source_label or 'segmentation-backend-default'

    for required in [resolved_image_csv_path, resolved_nuclei_csv_path, resolved_load_data_csv_path]:
        if not required.exists():
            raise FileNotFoundError(f"Required file not found: {required}")

    export_root = (output_dir or config.deepprofiler_export_root).resolve()
    images_dir = export_root / "images"
    locations_dir = export_root / "locations"
    images_dir.mkdir(parents=True, exist_ok=True)
    locations_dir.mkdir(parents=True, exist_ok=True)

    image_rows = _load_image_rows(resolved_image_csv_path)
    nuclei_by_image = _load_nuclei_locations(resolved_nuclei_csv_path)

    # DeepProfiler resolves the paths in field_metadata.csv relative to the
    # project's own images/ directory, so they must be absolute here.
    bases: list[Path] = [Path.cwd()]
    for anchor in (config.workspace_root, config.segmentation_backend_root):
        for parent in list(anchor.parents)[:3]:
            if parent not in bases:
                bases.append(parent)

    def _absolute(value: str) -> str:
        candidate = Path(value)
        if candidate.is_absolute():
            return str(candidate)
        for base in bases:
            resolved = base / candidate
            if resolved.exists():
                return str(resolved)
        return str(bases[0] / candidate)

    field_metadata_path = images_dir / "field_metadata.csv"
    manifest_path = export_root / "manifest.json"
    field_count = 0
    location_file_count = 0
    total_nuclei = 0

    with field_metadata_path.open("w", encoding="utf-8", newline="") as handle:
        fieldnames = [
            "ImageNumber",
            "Metadata_Plate",
            "Metadata_Well",
            "Metadata_Site",
            "dna_path",
            "rna_path",
            "er_path",
            "agp_path",
            "mito_path",
            "outline_path",
            "nuclei_locations_csv",
        ]
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()

        for image_number, row in sorted(image_rows.items(), key=lambda item: int(item[0])):
            plate = row["Metadata_Plate"]
            well = row["Metadata_Well"]
            site = str(row["Metadata_Site"])

            location_subdir = locations_dir / plate / well
            location_subdir.mkdir(parents=True, exist_ok=True)
            location_csv_path = location_subdir / f"site_{site}.csv"

            nuclei_rows = nuclei_by_image.get(image_number, [])
            with location_csv_path.open("w", encoding="utf-8", newline="") as loc_handle:
                loc_writer = csv.DictWriter(
                    loc_handle,
                    fieldnames=["ImageNumber", "ObjectNumber", "Location_Center_X", "Location_Center_Y"],
                )
                loc_writer.writeheader()
                for location_row in nuclei_rows:
                    loc_writer.writerow(location_row)

            channel_paths = {}
            for channel_name, (path_key, file_key) in IMAGE_CHANNELS.items():
                channel_paths[f"{channel_name}_path"] = _absolute(
                    str(Path(row[path_key]) / row[file_key])
                )

            outline_path = ""
            if row.get("PathName_CellOutlines") and row.get("FileName_CellOutlines"):
                outline_path = _absolute(
                    str(Path(row["PathName_CellOutlines"]) / row["FileName_CellOutlines"])
                )

            writer.writerow(
                {
                    "ImageNumber": image_number,
                    "Metadata_Plate": plate,
                    "Metadata_Well": well,
                    "Metadata_Site": site,
                    "outline_path": outline_path,
                    "nuclei_locations_csv": str(location_csv_path),
                    **channel_paths,
                }
            )
            field_count += 1
            location_file_count += 1
            total_nuclei += len(nuclei_rows)

    manifest = {
        "project_name": config.project_name,
        "source_label": resolved_source_label,
        "source_load_data_csv": str(resolved_load_data_csv_path),
        "source_image_csv": str(resolved_image_csv_path),
        "source_nuclei_csv": str(resolved_nuclei_csv_path),
        "field_metadata_csv": str(field_metadata_path),
        "locations_root": str(locations_dir),
        "channels": list(IMAGE_CHANNELS),
        "field_count": field_count,
        "location_file_count": location_file_count,
        "total_nuclei": total_nuclei,
        "notes": [
            "This export packages field-level metadata and per-field nuclei locations.",
            "Raw image paths follow the current source provenance; no image copies are made.",
            "This export can now target either the default segmentation backend outputs or a workflow-local CellProfiler result directory.",
        ],
    }
    manifest_path.write_text(json.dumps(manifest, indent=2) + chr(10), encoding="utf-8")
    return DeepProfilerExportResult(
        export_root=export_root,
        manifest_path=manifest_path,
        field_metadata_path=field_metadata_path,
        locations_root=locations_dir,
        field_count=field_count,
        location_file_count=location_file_count,
        total_nuclei=total_nuclei,
        source_image_csv=resolved_image_csv_path,
        source_nuclei_csv=resolved_nuclei_csv_path,
        source_load_data_csv=resolved_load_data_csv_path,
        source_label=resolved_source_label,
    )


def infer_deepprofiler_sources_from_workflow_root(workflow_root: Path) -> dict[str, Path]:
    resolved_workflow_root = workflow_root.expanduser().resolve()
    return {
        'image_csv_path': resolved_workflow_root / 'cellprofiler_masks' / 'Image.csv',
        'nuclei_csv_path': resolved_workflow_root / 'cellprofiler_masks' / 'Nuclei.csv',
        'load_data_csv_path': resolved_workflow_root / 'load_data_for_segmentation.csv',
    }


def _resolve_source_paths(
    config: ProjectConfig,
    *,
    image_csv_path: Path | None,
    nuclei_csv_path: Path | None,
    load_data_csv_path: Path | None,
) -> tuple[Path, Path, Path]:
    segmentation_root = config.segmentation_backend_root
    data_root = segmentation_root / 'data'
    mask_output_root = segmentation_root / 'outputs' / 'cellprofiler_masks'

    resolved_image_csv_path = image_csv_path.resolve() if image_csv_path is not None else (mask_output_root / 'Image.csv')
    resolved_nuclei_csv_path = nuclei_csv_path.resolve() if nuclei_csv_path is not None else (mask_output_root / 'Nuclei.csv')
    resolved_load_data_csv_path = load_data_csv_path.resolve() if load_data_csv_path is not None else (data_root / 'load_data_for_segmentation.csv')
    return resolved_image_csv_path, resolved_nuclei_csv_path, resolved_load_data_csv_path
