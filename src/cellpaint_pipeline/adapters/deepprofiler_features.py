from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any


if TYPE_CHECKING:
    from cellpaint_pipeline.config import ProjectConfig


DEFAULT_SINGLE_CELL_PARQUET_NAME = "deepprofiler_single_cell.parquet"
DEFAULT_SINGLE_CELL_CSV_NAME = "deepprofiler_single_cell.csv.gz"
DEFAULT_WELL_AGGREGATED_PARQUET_NAME = "deepprofiler_well_aggregated.parquet"
DEFAULT_WELL_AGGREGATED_CSV_NAME = "deepprofiler_well_aggregated.csv.gz"
DEFAULT_FIELD_SUMMARY_NAME = "deepprofiler_field_summary.csv"
DEFAULT_MANIFEST_NAME = "deepprofiler_feature_manifest.json"


@dataclass(frozen=True)
class DeepProfilerFeatureCollectionResult:
    project_root: Path
    feature_dir: Path
    output_dir: Path
    manifest_path: Path
    single_cell_parquet_path: Path
    single_cell_csv_gz_path: Path
    well_aggregated_parquet_path: Path
    well_aggregated_csv_gz_path: Path
    field_summary_path: Path
    experiment_name: str
    field_file_count: int
    cell_count: int
    feature_count: int
    metadata_column_count: int
    well_count: int


def collect_deepprofiler_features(
    config: ProjectConfig,
    *,
    project_root: Path,
    output_dir: Path | None = None,
    experiment_name: str | None = None,
) -> DeepProfilerFeatureCollectionResult:
    try:
        import pandas as pd
    except ImportError as exc:
        raise RuntimeError("pandas and pyarrow are required in the active CellPainting-Claw runtime for DeepProfiler feature collection.") from exc

    resolved_project_root = project_root.expanduser().resolve()
    project_manifest_path = resolved_project_root / "project_manifest.json"
    project_manifest = _read_json_if_exists(project_manifest_path)

    resolved_experiment_name = str(
        experiment_name
        or project_manifest.get("experiment_name")
        or config.deepprofiler_runtime.get("experiment_name")
        or "imagenet_pretrained_cellpainting"
    )
    feature_dir = resolved_project_root / "outputs" / resolved_experiment_name / "features"
    if not feature_dir.exists():
        raise FileNotFoundError(f"DeepProfiler feature directory not found: {feature_dir}")

    npz_paths = sorted(feature_dir.glob("*/*/*.npz"))
    if not npz_paths:
        raise FileNotFoundError(f"No DeepProfiler .npz feature files found under: {feature_dir}")

    if output_dir is None:
        resolved_output_dir = (config.default_output_root / "deepprofiler_features").resolve()
    else:
        resolved_output_dir = output_dir.expanduser().resolve()
    resolved_output_dir.mkdir(parents=True, exist_ok=True)

    single_cell_frames = []
    field_summary_rows: list[dict[str, Any]] = []
    feature_columns: list[str] | None = None

    for npz_path in npz_paths:
        loaded = _load_feature_npz(npz_path)
        actual_feature_count = loaded["feature_count"]
        if feature_columns is None:
            feature_columns = [
                f"Cells_DeepProfiler_{index:04d}"
                for index in range(1, actual_feature_count + 1)
            ]
        else:
            expected_feature_count = len(feature_columns)
            if expected_feature_count != actual_feature_count:
                raise ValueError(
                    f"Feature width mismatch in {npz_path}: expected {expected_feature_count}, got {actual_feature_count}"
                )

        metadata = dict(loaded["metadata"])
        metadata["Metadata_DeepProfiler_Experiment"] = resolved_experiment_name
        metadata["Metadata_DeepProfiler_FeatureFile"] = str(npz_path)
        cell_count = loaded["cell_count"]
        locations = loaded["locations"]

        cell_frame = pd.DataFrame({
            **{key: [value] * cell_count for key, value in metadata.items()},
            "Metadata_DeepProfiler_CellIndex": list(range(1, cell_count + 1)),
            "Metadata_Nuclei_Location_Center_X": locations[:, 0],
            "Metadata_Nuclei_Location_Center_Y": locations[:, 1],
        })
        feature_frame = pd.DataFrame(loaded["features"], columns=feature_columns)
        single_cell_frames.append(pd.concat([cell_frame, feature_frame], axis=1))

        field_summary_rows.append({
            "Metadata_Plate": metadata.get("Metadata_Plate", ""),
            "Metadata_Well": metadata.get("Metadata_Well", ""),
            "Metadata_Site": metadata.get("Metadata_Site", ""),
            "Metadata_DeepProfiler_Experiment": resolved_experiment_name,
            "Metadata_DeepProfiler_FeatureFile": str(npz_path),
            "Metadata_DeepProfiler_CellCount": cell_count,
            "Metadata_DeepProfiler_FeatureCount": actual_feature_count,
        })

    assert feature_columns is not None
    single_cell_df = pd.concat(single_cell_frames, ignore_index=True)
    metadata_columns = [column for column in single_cell_df.columns if column.startswith("Metadata_")]
    metadata_columns_ordered = sorted(metadata_columns)
    single_cell_df = single_cell_df[metadata_columns_ordered + feature_columns]

    grouped = single_cell_df.groupby(["Metadata_Plate", "Metadata_Well"], sort=True, dropna=False)
    aggregated_features_df = grouped[feature_columns].mean().reset_index()
    aggregated_counts_df = grouped.agg(
        Metadata_DeepProfiler_CellCount=("Metadata_DeepProfiler_CellIndex", "size"),
        Metadata_DeepProfiler_FieldCount=("Metadata_Site", "nunique"),
    ).reset_index()
    well_aggregated_df = aggregated_counts_df.merge(
        aggregated_features_df,
        on=["Metadata_Plate", "Metadata_Well"],
        how="left",
        validate="one_to_one",
    )
    well_aggregated_df.insert(2, "Metadata_DeepProfiler_Experiment", resolved_experiment_name)

    single_cell_parquet_path = resolved_output_dir / DEFAULT_SINGLE_CELL_PARQUET_NAME
    single_cell_csv_gz_path = resolved_output_dir / DEFAULT_SINGLE_CELL_CSV_NAME
    well_aggregated_parquet_path = resolved_output_dir / DEFAULT_WELL_AGGREGATED_PARQUET_NAME
    well_aggregated_csv_gz_path = resolved_output_dir / DEFAULT_WELL_AGGREGATED_CSV_NAME
    field_summary_path = resolved_output_dir / DEFAULT_FIELD_SUMMARY_NAME
    manifest_path = resolved_output_dir / DEFAULT_MANIFEST_NAME

    single_cell_df.to_parquet(single_cell_parquet_path, index=False)
    single_cell_df.to_csv(single_cell_csv_gz_path, index=False, compression="gzip")
    well_aggregated_df.to_parquet(well_aggregated_parquet_path, index=False)
    well_aggregated_df.to_csv(well_aggregated_csv_gz_path, index=False, compression="gzip")
    pd.DataFrame(field_summary_rows).to_csv(field_summary_path, index=False)

    manifest = {
        "implementation": "cellpaint_pipeline.deepprofiler_feature_collection",
        "project_root": str(resolved_project_root),
        "project_manifest_path": str(project_manifest_path) if project_manifest_path.exists() else None,
        "feature_dir": str(feature_dir),
        "output_dir": str(resolved_output_dir),
        "experiment_name": resolved_experiment_name,
        "single_cell_parquet_path": str(single_cell_parquet_path),
        "single_cell_csv_gz_path": str(single_cell_csv_gz_path),
        "well_aggregated_parquet_path": str(well_aggregated_parquet_path),
        "well_aggregated_csv_gz_path": str(well_aggregated_csv_gz_path),
        "field_summary_path": str(field_summary_path),
        "field_file_count": len(npz_paths),
        "cell_count": int(len(single_cell_df)),
        "feature_count": len(feature_columns),
        "metadata_column_count": len(metadata_columns_ordered),
        "well_count": int(len(well_aggregated_df)),
        "notes": [
            "Single-cell output is pycytominer-friendly: metadata columns are normalized to the Metadata_ prefix.",
            "Feature columns are exported as Cells_DeepProfiler_####.",
            "Well-level output is a mean aggregation over all DeepProfiler cells within each (plate, well).",
        ],
    }
    manifest_path.write_text(json.dumps(manifest, indent=2) + chr(10), encoding='utf-8')

    return DeepProfilerFeatureCollectionResult(
        project_root=resolved_project_root,
        feature_dir=feature_dir,
        output_dir=resolved_output_dir,
        manifest_path=manifest_path,
        single_cell_parquet_path=single_cell_parquet_path,
        single_cell_csv_gz_path=single_cell_csv_gz_path,
        well_aggregated_parquet_path=well_aggregated_parquet_path,
        well_aggregated_csv_gz_path=well_aggregated_csv_gz_path,
        field_summary_path=field_summary_path,
        experiment_name=resolved_experiment_name,
        field_file_count=len(npz_paths),
        cell_count=int(len(single_cell_df)),
        feature_count=len(feature_columns),
        metadata_column_count=len(metadata_columns_ordered),
        well_count=int(len(well_aggregated_df)),
    )


def _load_feature_npz(npz_path: Path) -> dict[str, Any]:
    import numpy as np

    with np.load(npz_path, allow_pickle=True) as payload:
        required = {"features", "metadata", "locations"}
        present = set(payload.files)
        missing = sorted(required - present)
        if missing:
            raise ValueError(f"DeepProfiler feature file is missing required arrays {missing}: {npz_path}")

        features = np.asarray(payload["features"])
        locations = np.asarray(payload["locations"])
        metadata = _normalize_metadata_dict(payload["metadata"].item())

    if features.ndim != 2:
        raise ValueError(f"Expected 2D features array in {npz_path}, got shape {features.shape}")
    if locations.ndim != 2 or locations.shape[1] != 2:
        raise ValueError(f"Expected (N, 2) locations array in {npz_path}, got shape {locations.shape}")
    if int(features.shape[0]) != int(locations.shape[0]):
        raise ValueError(
            f"Feature/location row mismatch in {npz_path}: {features.shape[0]} feature rows vs {locations.shape[0]} locations"
        )

    return {
        "features": features,
        "locations": locations,
        "metadata": metadata,
        "cell_count": int(features.shape[0]),
        "feature_count": int(features.shape[1]),
    }


def _normalize_metadata_dict(metadata: dict[str, Any]) -> dict[str, Any]:
    normalized: dict[str, Any] = {}
    for key, value in metadata.items():
        column_name = key if key.startswith("Metadata_") else f"Metadata_{key}"
        normalized[column_name] = _coerce_scalar(value)

    for required_column in ["Metadata_Plate", "Metadata_Well", "Metadata_Site"]:
        if required_column not in normalized:
            raise ValueError(f"DeepProfiler metadata is missing required field: {required_column}")

    normalized["Metadata_Plate"] = str(normalized["Metadata_Plate"])
    normalized["Metadata_Well"] = str(normalized["Metadata_Well"])
    normalized["Metadata_Site"] = _normalize_site_token(normalized["Metadata_Site"])
    return normalized


def _coerce_scalar(value: Any) -> Any:
    if isinstance(value, bytes):
        return value.decode("utf-8")
    if isinstance(value, Path):
        return str(value)
    if hasattr(value, "item"):
        try:
            return value.item()
        except Exception:
            return value
    return value


def _normalize_site_token(value: Any) -> str:
    token = str(value).strip()
    if not token:
        return token
    try:
        return str(int(float(token)))
    except ValueError:
        return token


def _read_json_if_exists(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))
