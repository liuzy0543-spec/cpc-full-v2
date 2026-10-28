from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any


if TYPE_CHECKING:
    from cellpaint_pipeline.config import ProjectConfig


@dataclass(frozen=True)
class ClassicalProfileSummaryResult:
    input_profile_path: Path
    output_dir: Path
    summary_path: Path
    well_metadata_summary_path: Path
    top_variable_features_path: Path
    pca_coordinates_path: Path
    pca_plot_path: Path | None
    row_count: int
    feature_count: int
    metadata_column_count: int
    top_feature_count: int
    pca_explained_variance_ratio: tuple[float, ...]


@dataclass(frozen=True)
class DeepProfilerProfileSummaryResult:
    single_cell_path: Path
    well_aggregated_path: Path
    output_dir: Path
    summary_path: Path
    well_metadata_summary_path: Path
    top_variable_features_path: Path
    pca_coordinates_path: Path
    pca_plot_path: Path | None
    cell_count: int
    well_count: int
    feature_count: int
    metadata_column_count: int
    top_feature_count: int
    pca_explained_variance_ratio: tuple[float, ...]


def summarize_classical_profiles(
    config: ProjectConfig,
    *,
    output_dir: Path,
    feature_selected_path: Path | None = None,
    manifest_path: Path | None = None,
    top_n: int = 50,
) -> ClassicalProfileSummaryResult:
    try:
        import numpy as np
        import pandas as pd
    except ImportError as exc:
        raise RuntimeError(
            'pandas, numpy, and pyarrow are required in the active CellPainting-Claw runtime for summarize-classical-profiles.'
        ) from exc

    backend_payload = config.load_profiling_backend_payload()
    default_feature_selected_path = config.resolve_profiling_backend_path(
        backend_payload['paths']['feature_selected_output_parquet']
    )
    resolved_feature_selected_path = (
        feature_selected_path.expanduser().resolve()
        if feature_selected_path is not None
        else _resolve_classical_profile_path_from_manifest(manifest_path) or default_feature_selected_path
    )
    if not resolved_feature_selected_path.exists():
        raise FileNotFoundError(f'Classical profile table not found: {resolved_feature_selected_path}')

    resolved_output_dir = output_dir.expanduser().resolve()
    resolved_output_dir.mkdir(parents=True, exist_ok=True)

    feature_selected_df = _read_table(resolved_feature_selected_path)
    metadata_columns = [column for column in feature_selected_df.columns if column.startswith('Metadata_')]
    feature_columns = [column for column in feature_selected_df.columns if column not in metadata_columns]
    if not feature_columns:
        raise ValueError(
            f'Classical profile summary expected at least one non-metadata feature column: {resolved_feature_selected_path}'
        )

    numeric_feature_df = feature_selected_df[feature_columns].apply(pd.to_numeric, errors='coerce')
    top_variable_df = _build_top_variable_features(pd, numeric_feature_df, top_n=top_n)
    pca_coordinates_df, explained_variance_ratio = _build_pca_coordinates(
        pd,
        np,
        feature_selected_df,
        metadata_columns=metadata_columns,
        numeric_feature_df=numeric_feature_df,
    )

    summary_path = resolved_output_dir / 'profile_summary.json'
    well_metadata_summary_path = resolved_output_dir / 'well_metadata_summary.csv'
    top_variable_features_path = resolved_output_dir / 'top_variable_features.csv'
    pca_coordinates_path = resolved_output_dir / 'pca_coordinates.csv'

    metadata_summary_df = _build_metadata_summary(feature_selected_df, metadata_columns)
    metadata_summary_df.to_csv(well_metadata_summary_path, index=False)
    top_variable_df.to_csv(top_variable_features_path, index=False)
    pca_coordinates_df.to_csv(pca_coordinates_path, index=False)
    pca_plot_path = _write_pca_plot(
        dataframe=pca_coordinates_df,
        output_path=resolved_output_dir / 'pca_plot.png',
        title='Classical profile PCA',
    )

    summary_payload = {
        'implementation': 'cellpaint_pipeline.profile_summaries.classical',
        'input_profile_path': str(resolved_feature_selected_path),
        'output_dir': str(resolved_output_dir),
        'row_count': int(len(feature_selected_df)),
        'feature_count': int(len(feature_columns)),
        'metadata_column_count': int(len(metadata_columns)),
        'metadata_columns': list(metadata_columns),
        'top_feature_count': int(len(top_variable_df)),
        'top_variable_feature_names': top_variable_df['feature_name'].head(10).tolist(),
        'pca_explained_variance_ratio': list(explained_variance_ratio),
        'pca_plot_path': str(pca_plot_path) if pca_plot_path is not None else None,
    }
    summary_path.write_text(json.dumps(summary_payload, indent=2, ensure_ascii=False) + '\n', encoding='utf-8')

    return ClassicalProfileSummaryResult(
        input_profile_path=resolved_feature_selected_path,
        output_dir=resolved_output_dir,
        summary_path=summary_path,
        well_metadata_summary_path=well_metadata_summary_path,
        top_variable_features_path=top_variable_features_path,
        pca_coordinates_path=pca_coordinates_path,
        pca_plot_path=pca_plot_path,
        row_count=int(len(feature_selected_df)),
        feature_count=int(len(feature_columns)),
        metadata_column_count=int(len(metadata_columns)),
        top_feature_count=int(len(top_variable_df)),
        pca_explained_variance_ratio=tuple(float(item) for item in explained_variance_ratio),
    )


def summarize_deepprofiler_profiles(
    *,
    output_dir: Path,
    single_cell_parquet_path: Path | None = None,
    well_aggregated_parquet_path: Path | None = None,
    manifest_path: Path | None = None,
    top_n: int = 50,
) -> DeepProfilerProfileSummaryResult:
    try:
        import numpy as np
        import pandas as pd
    except ImportError as exc:
        raise RuntimeError(
            'pandas, numpy, and pyarrow are required in the active CellPainting-Claw runtime for summarize-deepprofiler-profiles.'
        ) from exc

    resolved_single_cell_path, resolved_well_aggregated_path = _resolve_deepprofiler_profile_paths(
        single_cell_parquet_path=single_cell_parquet_path,
        well_aggregated_parquet_path=well_aggregated_parquet_path,
        manifest_path=manifest_path,
    )
    if not resolved_single_cell_path.exists():
        raise FileNotFoundError(f'DeepProfiler single-cell table not found: {resolved_single_cell_path}')
    if not resolved_well_aggregated_path.exists():
        raise FileNotFoundError(f'DeepProfiler well-level table not found: {resolved_well_aggregated_path}')

    resolved_output_dir = output_dir.expanduser().resolve()
    resolved_output_dir.mkdir(parents=True, exist_ok=True)

    single_cell_df = _read_table(resolved_single_cell_path)
    well_aggregated_df = _read_table(resolved_well_aggregated_path)
    metadata_columns = [column for column in well_aggregated_df.columns if column.startswith('Metadata_')]
    feature_columns = [column for column in well_aggregated_df.columns if column not in metadata_columns]
    if not feature_columns:
        raise ValueError(
            f'DeepProfiler profile summary expected at least one non-metadata feature column: {resolved_well_aggregated_path}'
        )

    numeric_feature_df = well_aggregated_df[feature_columns].apply(pd.to_numeric, errors='coerce')
    top_variable_df = _build_top_variable_features(pd, numeric_feature_df, top_n=top_n)
    pca_coordinates_df, explained_variance_ratio = _build_pca_coordinates(
        pd,
        np,
        well_aggregated_df,
        metadata_columns=metadata_columns,
        numeric_feature_df=numeric_feature_df,
    )

    summary_path = resolved_output_dir / 'profile_summary.json'
    well_metadata_summary_path = resolved_output_dir / 'well_metadata_summary.csv'
    top_variable_features_path = resolved_output_dir / 'top_variable_features.csv'
    pca_coordinates_path = resolved_output_dir / 'pca_coordinates.csv'

    metadata_summary_df = _build_metadata_summary(well_aggregated_df, metadata_columns)
    metadata_summary_df.to_csv(well_metadata_summary_path, index=False)
    top_variable_df.to_csv(top_variable_features_path, index=False)
    pca_coordinates_df.to_csv(pca_coordinates_path, index=False)
    pca_plot_path = _write_pca_plot(
        dataframe=pca_coordinates_df,
        output_path=resolved_output_dir / 'pca_plot.png',
        title='DeepProfiler well-level PCA',
    )

    summary_payload = {
        'implementation': 'cellpaint_pipeline.profile_summaries.deepprofiler',
        'single_cell_path': str(resolved_single_cell_path),
        'well_aggregated_path': str(resolved_well_aggregated_path),
        'output_dir': str(resolved_output_dir),
        'cell_count': int(len(single_cell_df)),
        'well_count': int(len(well_aggregated_df)),
        'feature_count': int(len(feature_columns)),
        'metadata_column_count': int(len(metadata_columns)),
        'metadata_columns': list(metadata_columns),
        'top_feature_count': int(len(top_variable_df)),
        'top_variable_feature_names': top_variable_df['feature_name'].head(10).tolist(),
        'pca_explained_variance_ratio': list(explained_variance_ratio),
        'pca_plot_path': str(pca_plot_path) if pca_plot_path is not None else None,
    }
    summary_path.write_text(json.dumps(summary_payload, indent=2, ensure_ascii=False) + '\n', encoding='utf-8')

    return DeepProfilerProfileSummaryResult(
        single_cell_path=resolved_single_cell_path,
        well_aggregated_path=resolved_well_aggregated_path,
        output_dir=resolved_output_dir,
        summary_path=summary_path,
        well_metadata_summary_path=well_metadata_summary_path,
        top_variable_features_path=top_variable_features_path,
        pca_coordinates_path=pca_coordinates_path,
        pca_plot_path=pca_plot_path,
        cell_count=int(len(single_cell_df)),
        well_count=int(len(well_aggregated_df)),
        feature_count=int(len(feature_columns)),
        metadata_column_count=int(len(metadata_columns)),
        top_feature_count=int(len(top_variable_df)),
        pca_explained_variance_ratio=tuple(float(item) for item in explained_variance_ratio),
    )


def _resolve_classical_profile_path_from_manifest(manifest_path: Path | None) -> Path | None:
    if manifest_path is None:
        return None
    payload = _read_json(manifest_path.expanduser().resolve())
    feature_selected_value = _find_nested_value(payload, {'feature_selected_path'})
    if feature_selected_value:
        return Path(str(feature_selected_value)).expanduser().resolve()
    return None


def _resolve_deepprofiler_profile_paths(
    *,
    single_cell_parquet_path: Path | None,
    well_aggregated_parquet_path: Path | None,
    manifest_path: Path | None,
) -> tuple[Path, Path]:
    resolved_single_cell_path = (
        single_cell_parquet_path.expanduser().resolve() if single_cell_parquet_path is not None else None
    )
    resolved_well_aggregated_path = (
        well_aggregated_parquet_path.expanduser().resolve() if well_aggregated_parquet_path is not None else None
    )
    if resolved_single_cell_path is not None and resolved_well_aggregated_path is not None:
        return resolved_single_cell_path, resolved_well_aggregated_path

    if manifest_path is None:
        raise ValueError(
            'summarize-deepprofiler-profiles requires either explicit DeepProfiler table paths or a manifest_path from run-deepprofiler.'
        )

    payload = _read_json(manifest_path.expanduser().resolve())
    direct_single_cell_value = _find_nested_value(payload, {'single_cell_parquet_path'})
    direct_well_value = _find_nested_value(payload, {'well_aggregated_parquet_path'})
    if direct_single_cell_value and direct_well_value:
        return (
            Path(str(direct_single_cell_value)).expanduser().resolve(),
            Path(str(direct_well_value)).expanduser().resolve(),
        )

    collection_manifest_value = _find_nested_value(payload, {'collection_manifest_path', 'feature_manifest_path'})
    if collection_manifest_value:
        collection_payload = _read_json(Path(str(collection_manifest_value)).expanduser().resolve())
        nested_single_cell_value = _find_nested_value(collection_payload, {'single_cell_parquet_path'})
        nested_well_value = _find_nested_value(collection_payload, {'well_aggregated_parquet_path'})
        if nested_single_cell_value and nested_well_value:
            return (
                Path(str(nested_single_cell_value)).expanduser().resolve(),
                Path(str(nested_well_value)).expanduser().resolve(),
            )

    raise ValueError(
        'Could not resolve DeepProfiler table paths from the provided manifest. '
        'Pass explicit single_cell_parquet_path and well_aggregated_parquet_path instead.'
    )


def _read_table(path: Path):
    try:
        import pandas as pd
    except ImportError as exc:
        raise RuntimeError('pandas is required in the active CellPainting-Claw runtime for profile summary.') from exc

    suffixes = [suffix.lower() for suffix in path.suffixes]
    if suffixes[-1:] == ['.parquet']:
        return pd.read_parquet(path)
    if suffixes[-2:] == ['.csv', '.gz'] or suffixes[-1:] == ['.csv']:
        return pd.read_csv(path)
    raise ValueError(f'Unsupported tabular input format for profile summary: {path}')


def _build_metadata_summary(dataframe, metadata_columns: list[str]):
    if not metadata_columns:
        return dataframe.iloc[:, 0:0].copy()
    preferred_columns = [
        column
        for column in [
            'Metadata_Plate',
            'Metadata_Well',
            'Metadata_Site',
            'Metadata_Treatment',
            'Metadata_ControlType',
            'Metadata_Batch',
            'Metadata_DeepProfiler_Experiment',
        ]
        if column in metadata_columns
    ]
    ordered_columns = preferred_columns + [column for column in metadata_columns if column not in preferred_columns]
    return dataframe[ordered_columns].drop_duplicates().reset_index(drop=True)


def _build_top_variable_features(pd, numeric_feature_df, *, top_n: int):
    variance_series = numeric_feature_df.fillna(0.0).var(axis=0, ddof=1).sort_values(ascending=False)
    top_variance_series = variance_series.head(max(int(top_n), 1))
    return pd.DataFrame({
        'feature_name': top_variance_series.index.tolist(),
        'variance': [float(value) for value in top_variance_series.tolist()],
    })


def _build_pca_coordinates(pd, np, dataframe, *, metadata_columns: list[str], numeric_feature_df):
    feature_matrix = numeric_feature_df.fillna(0.0).to_numpy(dtype=float)
    row_count = int(feature_matrix.shape[0])
    feature_count = int(feature_matrix.shape[1])
    if row_count == 0:
        raise ValueError('Profile summary cannot be computed on an empty table.')

    if feature_count == 0:
        coordinate_payload = {column: dataframe[column].tolist() for column in metadata_columns}
        coordinate_payload['PC1'] = [0.0] * row_count
        coordinate_payload['PC2'] = [0.0] * row_count
        return pd.DataFrame(coordinate_payload), (0.0, 0.0)

    centered = feature_matrix - feature_matrix.mean(axis=0, keepdims=True)
    if row_count == 1:
        pc1 = centered[:, 0].astype(float)
        pc2 = np.zeros(1, dtype=float)
        coordinate_payload = {column: dataframe[column].tolist() for column in metadata_columns}
        coordinate_payload['PC1'] = pc1.tolist()
        coordinate_payload['PC2'] = pc2.tolist()
        return pd.DataFrame(coordinate_payload), (1.0, 0.0)

    singular_vectors, singular_values, _ = np.linalg.svd(centered, full_matrices=False)
    component_count = min(2, singular_values.shape[0])
    projected = singular_vectors[:, :component_count] * singular_values[:component_count]
    if component_count == 1:
        projected = np.column_stack([projected[:, 0], np.zeros(row_count, dtype=float)])

    explained_variance = (singular_values ** 2) / max(row_count - 1, 1)
    explained_variance_sum = float(explained_variance.sum())
    if explained_variance_sum > 0:
        explained_variance_ratio = tuple(
            float(value) for value in (explained_variance[:2] / explained_variance_sum).tolist()
        )
    else:
        explained_variance_ratio = (0.0, 0.0)
    if len(explained_variance_ratio) == 1:
        explained_variance_ratio = (explained_variance_ratio[0], 0.0)

    coordinate_payload = {column: dataframe[column].tolist() for column in metadata_columns}
    coordinate_payload['PC1'] = projected[:, 0].astype(float).tolist()
    coordinate_payload['PC2'] = projected[:, 1].astype(float).tolist()
    return pd.DataFrame(coordinate_payload), explained_variance_ratio


def _write_pca_plot(*, dataframe, output_path: Path, title: str) -> Path | None:
    try:
        import matplotlib

        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
    except ImportError:
        return None

    color_column = next(
        (
            column
            for column in ['Metadata_ControlType', 'Metadata_Treatment', 'Metadata_Batch', 'Metadata_Plate']
            if column in dataframe.columns
        ),
        None,
    )

    figure, axis = plt.subplots(figsize=(6, 5))
    if color_column is None:
        axis.scatter(dataframe['PC1'], dataframe['PC2'], s=24, alpha=0.8)
    else:
        grouped = dataframe.groupby(color_column, dropna=False, sort=True)
        for label, group in grouped:
            axis.scatter(group['PC1'], group['PC2'], s=24, alpha=0.8, label=str(label))
        if len(grouped) <= 8:
            axis.legend(loc='best', fontsize=8)
    axis.set_title(title)
    axis.set_xlabel('PC1')
    axis.set_ylabel('PC2')
    axis.axhline(0.0, color='#cccccc', linewidth=0.8)
    axis.axvline(0.0, color='#cccccc', linewidth=0.8)
    axis.grid(alpha=0.2)
    figure.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output_path, dpi=150)
    plt.close(figure)
    return output_path


def _read_json(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding='utf-8'))
    if not isinstance(payload, dict):
        raise ValueError(f'Expected a JSON object at: {path}')
    return payload


def _find_nested_value(payload: Any, candidate_keys: set[str]) -> Any:
    if isinstance(payload, dict):
        for key, value in payload.items():
            if key in candidate_keys and value not in {None, ''}:
                return value
            nested = _find_nested_value(value, candidate_keys)
            if nested not in {None, ''}:
                return nested
        return None
    if isinstance(payload, list):
        for item in payload:
            nested = _find_nested_value(item, candidate_keys)
            if nested not in {None, ''}:
                return nested
    return None
