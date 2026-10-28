from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import TYPE_CHECKING, Any

from cellpaint_pipeline.config import ProjectConfig

if TYPE_CHECKING:  # pragma: no cover - typing only
    import pandas as pd

SAMPLE_ID_COL = "Metadata_PlateWell"

_MISSING_DEPENDENCY_MESSAGE = (
    'The native evaluation step requires pandas, numpy, scikit-learn and '
    'matplotlib in the active CellPainting-Claw runtime.'
)


def _load_runtime() -> tuple[Any, Any, Any, Any, Any, Any]:
    """Import the scientific stack on demand.

    Every other module in this package imports pandas / numpy / matplotlib
    inside the function that needs them and reports a missing dependency with a
    ``RuntimeError``.  Doing the same here means ``import
    cellpaint_pipeline.evaluation`` succeeds in a runtime that only wants the
    config helpers, and it also keeps matplotlib's backend selection under this
    module's own control instead of depending on import order.
    """
    try:
        import matplotlib
        import numpy as np
        import pandas as pd
        from sklearn.decomposition import PCA
        from sklearn.metrics import pairwise_distances
        from sklearn.preprocessing import StandardScaler
    except ImportError as exc:  # pragma: no cover - environment dependent
        raise RuntimeError(_MISSING_DEPENDENCY_MESSAGE) from exc

    # Headless rendering must be selected before pyplot is imported, otherwise
    # an earlier import of this module could leave an interactive backend
    # active.  ``use`` is a no-op-safe call when the backend is already set.
    try:
        matplotlib.use('Agg')
    except Exception:  # pragma: no cover - backend already fixed
        pass
    import matplotlib.pyplot as plt

    return plt, np, pd, PCA, pairwise_distances, StandardScaler


@dataclass(frozen=True)
class EvaluationPaths:
    feature_selected_parquet: Path
    annotated_parquet: Path
    output_dir: Path


@dataclass(frozen=True)
class NativeEvaluationResult:
    output_dir: Path
    n_wells: int
    n_feature_columns: int
    sample_id_column: str
    run_manifest_path: Path


def resolve_evaluation_paths(
    config: ProjectConfig,
    output_dir: Path | None = None,
    feature_selected_path: Path | None = None,
    annotated_path: Path | None = None,
) -> EvaluationPaths:
    backend_payload = config.load_profiling_backend_payload()
    paths_payload = backend_payload["paths"]
    feature_selected = feature_selected_path.resolve() if feature_selected_path is not None else config.resolve_profiling_backend_path(paths_payload["feature_selected_output_parquet"])
    annotated = annotated_path.resolve() if annotated_path is not None else config.resolve_profiling_backend_path(paths_payload["annotated_output_parquet"])
    resolved_output_dir = output_dir.resolve() if output_dir is not None else config.resolve_profiling_backend_path("outputs/evaluation")
    return EvaluationPaths(
        feature_selected_parquet=feature_selected,
        annotated_parquet=annotated,
        output_dir=resolved_output_dir,
    )


def run_native_evaluation(
    config: ProjectConfig,
    output_dir: Path | None = None,
    feature_selected_path: Path | None = None,
    annotated_path: Path | None = None,
) -> NativeEvaluationResult:
    _, _, pd, _, _, _ = _load_runtime()
    paths = resolve_evaluation_paths(
        config,
        output_dir=output_dir,
        feature_selected_path=feature_selected_path,
        annotated_path=annotated_path,
    )
    paths.output_dir.mkdir(parents=True, exist_ok=True)

    feature_selected = pd.read_parquet(paths.feature_selected_parquet)
    annotated = pd.read_parquet(paths.annotated_parquet)
    joined = build_joined_table(feature_selected=feature_selected, annotated=annotated)
    feature_cols = get_feature_columns(feature_selected)
    scaled_features = get_scaled_feature_df(joined=joined, feature_cols=feature_cols)

    pca_df = save_pca(joined=joined, scaled_features=scaled_features, output_dir=paths.output_dir)
    corr_pairs = save_correlation(joined=joined, feature_cols=feature_cols, output_dir=paths.output_dir)
    deviation, dmso_summary = save_control_deviation(
        joined=joined,
        scaled_features=scaled_features,
        output_dir=paths.output_dir,
    )
    save_nearest_neighbors(joined=joined, scaled_features=scaled_features, output_dir=paths.output_dir)
    write_summary(
        pca_df=pca_df,
        corr_pairs=corr_pairs,
        deviation=deviation,
        dmso_summary=dmso_summary,
        output_dir=paths.output_dir,
        feature_selected_path=paths.feature_selected_parquet,
    )

    manifest = {
        "implementation": "cellpaint_pipeline.native_evaluation",
        "generated_utc": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "input_feature_selected": str(paths.feature_selected_parquet),
        "input_annotated": str(paths.annotated_parquet),
        "output_dir": str(paths.output_dir),
        "n_wells": int(len(joined)),
        "n_feature_columns": int(len(feature_cols)),
        "sample_id_column": SAMPLE_ID_COL,
    }
    run_manifest_path = paths.output_dir / "run_manifest.json"
    run_manifest_path.write_text(json.dumps(manifest, indent=2) + chr(10), encoding="utf-8")
    return NativeEvaluationResult(
        output_dir=paths.output_dir,
        n_wells=int(len(joined)),
        n_feature_columns=int(len(feature_cols)),
        sample_id_column=SAMPLE_ID_COL,
        run_manifest_path=run_manifest_path,
    )


def get_feature_columns(df: pd.DataFrame) -> list[str]:
    return [col for col in df.columns if not col.startswith("Metadata_")]


def build_joined_table(feature_selected: pd.DataFrame, annotated: pd.DataFrame) -> pd.DataFrame:
    _, _, pd, _, _, _ = _load_runtime()
    meta_cols = [col for col in annotated.columns if col.startswith("Metadata_")]
    meta = annotated[meta_cols].copy()
    overlap_meta = [col for col in meta.columns if col in feature_selected.columns]
    if overlap_meta:
        meta = meta.drop(columns=overlap_meta)
    joined = pd.concat([feature_selected.copy(), meta], axis=1)
    joined[SAMPLE_ID_COL] = joined["Metadata_Plate"].astype(str) + "__" + joined["Metadata_Well"].astype(str)
    return joined


def get_scaled_feature_df(joined: pd.DataFrame, feature_cols: list[str]) -> pd.DataFrame:
    _, np, pd, _, _, StandardScaler = _load_runtime()
    scaler = StandardScaler()
    scaled = scaler.fit_transform(joined[feature_cols].to_numpy(dtype=float))
    return pd.DataFrame(scaled, columns=feature_cols, index=joined.index)


def save_pca(joined: pd.DataFrame, scaled_features: pd.DataFrame, output_dir: Path) -> pd.DataFrame:
    plt, np, pd, PCA, _, _ = _load_runtime()
    x_values = scaled_features.to_numpy(dtype=float)
    pca = PCA(n_components=min(5, x_values.shape[0], x_values.shape[1]), random_state=0)
    scores = pca.fit_transform(x_values)

    pca_df = joined[[
        SAMPLE_ID_COL,
        "Metadata_Plate",
        "Metadata_Well",
        "Metadata_Treatment",
        "Metadata_ControlType",
        "Metadata_Object_Count",
    ]].copy()
    pca_df["PC1"] = scores[:, 0]
    pca_df["PC2"] = scores[:, 1] if scores.shape[1] > 1 else 0.0
    pca_df.to_csv(output_dir / "pca_scores.csv", index=False)

    fig, ax = plt.subplots(figsize=(9, 7))
    colors = {"negative_control": "#1f77b4", "treatment": "#d62728"}
    for control_type, subset in pca_df.groupby("Metadata_ControlType", dropna=False):
        ax.scatter(
            subset["PC1"],
            subset["PC2"],
            s=70,
            label=str(control_type),
            color=colors.get(str(control_type), "#7f7f7f"),
            alpha=0.9,
        )
    for row in pca_df.itertuples(index=False):
        ax.text(row.PC1, row.PC2, row.Metadata_PlateWell, fontsize=6, ha="left", va="bottom")
    variance = pca.explained_variance_ratio_ * 100
    ax.set_xlabel(f"PC1 ({variance[0]:.1f}% var)")
    ax.set_ylabel(f"PC2 ({variance[1]:.1f}% var)" if len(variance) > 1 else "PC2")
    ax.set_title("Well-Level Profile PCA")
    ax.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(output_dir / "pca_plot.png", dpi=200)
    plt.close(fig)

    pd.DataFrame(
        {
            "component": [f"PC{i+1}" for i in range(len(variance))],
            "explained_variance_ratio": variance / 100.0,
            "explained_variance_percent": variance,
        }
    ).to_csv(output_dir / "pca_variance.csv", index=False)
    return pca_df


def save_correlation(joined: pd.DataFrame, feature_cols: list[str], output_dir: Path) -> pd.DataFrame:
    plt, _, pd, _, _, _ = _load_runtime()
    profile_matrix = joined.set_index(SAMPLE_ID_COL)[feature_cols].T.corr(method="pearson")
    profile_matrix.to_csv(output_dir / "well_correlation_matrix.csv")

    fig, ax = plt.subplots(figsize=(10, 9))
    im = ax.imshow(profile_matrix.to_numpy(), cmap="coolwarm", vmin=-1, vmax=1)
    sample_ids = profile_matrix.index.tolist()
    ax.set_xticks(range(len(sample_ids)))
    ax.set_yticks(range(len(sample_ids)))
    ax.set_xticklabels(sample_ids, rotation=90, fontsize=6)
    ax.set_yticklabels(sample_ids, fontsize=6)
    ax.set_title("Pearson Correlation Between Well Profiles")
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    fig.tight_layout()
    fig.savefig(output_dir / "well_correlation_heatmap.png", dpi=200)
    plt.close(fig)

    pairs = []
    for i, sample_a in enumerate(sample_ids):
        for j in range(i + 1, len(sample_ids)):
            sample_b = sample_ids[j]
            plate_a, well_a = sample_a.split("__", 1)
            plate_b, well_b = sample_b.split("__", 1)
            pairs.append(
                {
                    "sample_a": sample_a,
                    "plate_a": plate_a,
                    "well_a": well_a,
                    "sample_b": sample_b,
                    "plate_b": plate_b,
                    "well_b": well_b,
                    "pearson_correlation": float(profile_matrix.iloc[i, j]),
                }
            )
    pair_df = pd.DataFrame(pairs).sort_values("pearson_correlation", ascending=False).reset_index(drop=True)
    pair_df.to_csv(output_dir / "well_correlation_pairs.csv", index=False)
    return pair_df


def save_control_deviation(
    joined: pd.DataFrame,
    scaled_features: pd.DataFrame,
    output_dir: Path,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    _, np, pd, _, pairwise_distances, _ = _load_runtime()
    control_mask = joined["Metadata_ControlType"].eq("negative_control")
    controls = joined.loc[control_mask].copy()
    treatments = joined.loc[~control_mask].copy()

    scaled_controls = scaled_features.loc[controls.index]
    scaled_treatments = scaled_features.loc[treatments.index]
    control_center = scaled_controls.mean(axis=0)
    control_center_df = pd.DataFrame([control_center], index=["DMSO_CENTROID"])

    treatment_profiles = scaled_treatments.copy()
    treatment_profiles.index = treatments[SAMPLE_ID_COL].to_numpy()
    control_profiles = scaled_controls.copy()
    control_profiles.index = controls[SAMPLE_ID_COL].to_numpy()

    euclidean = pairwise_distances(treatment_profiles, control_center_df, metric="euclidean").ravel()
    cosine = pairwise_distances(treatment_profiles, control_center_df, metric="cosine").ravel()
    treatment_corr_to_centroid = treatment_profiles.T.corrwith(control_center, method="pearson")

    summary = treatments[[
        SAMPLE_ID_COL,
        "Metadata_Plate",
        "Metadata_Well",
        "Metadata_Treatment",
        "Metadata_Object_Count",
    ]].copy()
    summary["EuclideanDistanceToDMSOCentroid"] = euclidean
    summary["CosineDistanceToDMSOCentroid"] = cosine
    summary["PearsonCorrelationToDMSOCentroid"] = treatment_corr_to_centroid.values
    summary = summary.sort_values("EuclideanDistanceToDMSOCentroid", ascending=False).reset_index(drop=True)
    summary.to_csv(output_dir / "treatment_vs_dmso_centroid.csv", index=False)

    control_corr = control_profiles.T.corr(method="pearson")
    control_pair_values = control_corr.to_numpy()
    upper = control_pair_values[np.triu_indices_from(control_pair_values, k=1)]
    dmso_summary = pd.DataFrame(
        {
            "metric": [
                "dmso_replicate_mean_correlation",
                "dmso_replicate_min_correlation",
                "dmso_replicate_max_correlation",
            ],
            "value": [
                float(np.nanmean(upper)) if upper.size else np.nan,
                float(np.nanmin(upper)) if upper.size else np.nan,
                float(np.nanmax(upper)) if upper.size else np.nan,
            ],
        }
    )
    dmso_summary.to_csv(output_dir / "dmso_replicate_summary.csv", index=False)
    return summary, dmso_summary


def save_nearest_neighbors(joined: pd.DataFrame, scaled_features: pd.DataFrame, output_dir: Path) -> pd.DataFrame:
    _, np, pd, _, pairwise_distances, _ = _load_runtime()
    profiles = scaled_features.copy()
    profiles.index = joined[SAMPLE_ID_COL].to_numpy()
    dists = pairwise_distances(profiles, metric="cosine")
    sample_ids = profiles.index.to_list()
    joined_indexed = joined.set_index(SAMPLE_ID_COL)
    rows = []
    for i, sample_id in enumerate(sample_ids):
        order = np.argsort(dists[i])
        order = [idx for idx in order if idx != i][:3]
        base = joined_indexed.loc[sample_id]
        for rank, idx in enumerate(order, start=1):
            neighbor_id = sample_ids[idx]
            neighbor = joined_indexed.loc[neighbor_id]
            rows.append(
                {
                    SAMPLE_ID_COL: sample_id,
                    "Metadata_Plate": base["Metadata_Plate"],
                    "Metadata_Well": base["Metadata_Well"],
                    "Metadata_Treatment": base["Metadata_Treatment"],
                    "NeighborRank": rank,
                    "NeighborSample": neighbor_id,
                    "NeighborPlate": neighbor["Metadata_Plate"],
                    "NeighborWell": neighbor["Metadata_Well"],
                    "NeighborTreatment": neighbor["Metadata_Treatment"],
                    "NeighborControlType": neighbor["Metadata_ControlType"],
                    "CosineDistance": float(dists[i, idx]),
                }
            )
    nn_df = pd.DataFrame(rows)
    nn_df.to_csv(output_dir / "nearest_neighbors.csv", index=False)
    return nn_df


def write_summary(
    pca_df: pd.DataFrame,
    corr_pairs: pd.DataFrame,
    deviation: pd.DataFrame,
    dmso_summary: pd.DataFrame,
    output_dir: Path,
    feature_selected_path: Path,
) -> None:
    _, np, _, _, _, _ = _load_runtime()
    controls = pca_df[pca_df["Metadata_ControlType"] == "negative_control"]
    control_pc1_mean = float(controls["PC1"].mean()) if not controls.empty else np.nan
    control_pc2_mean = float(controls["PC2"].mean()) if not controls.empty else np.nan
    top_pair = corr_pairs.iloc[0]
    bottom_pair = corr_pairs.iloc[-1]
    dmso_mean_corr = float(
        dmso_summary.loc[
            dmso_summary["metric"] == "dmso_replicate_mean_correlation",
            "value",
        ].iloc[0]
    )
    updated_utc = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC")

    lines = [
        "# Profile Evaluation Summary",
        "",
        f"Updated: {updated_utc}",
        f"Input profiles: `{feature_selected_path}`",
        "",
        "## Scope",
        "",
        "This evaluation uses downstream well-level pycytominer profiles, consistent with the paper's downstream-analysis framing.",
        "It does not reproduce the paper's external supervised cell-injury model, because the current local subset does not contain the required labeled training setup.",
        "This version uses `Metadata_Plate + Metadata_Well` as the unique sample identifier to avoid mixing same-position wells across plates.",
        "",
        "## Dataset Used",
        "",
        f"- Wells: {len(pca_df)}",
        f"- Treatments: {int((pca_df['Metadata_ControlType'] == 'treatment').sum())}",
        f"- Negative controls: {int((pca_df['Metadata_ControlType'] == 'negative_control').sum())}",
        "",
        "## PCA Snapshot",
        "",
        f"- DMSO centroid in PCA space: PC1={control_pc1_mean:.3f}, PC2={control_pc2_mean:.3f}",
        f"- Most shifted treatment from DMSO centroid: {deviation.iloc[0][SAMPLE_ID_COL]} ({deviation.iloc[0]['Metadata_Treatment']})",
        f"- Largest Euclidean distance to DMSO centroid: {deviation.iloc[0]['EuclideanDistanceToDMSOCentroid']:.3f}",
        "",
        "## Correlation Snapshot",
        "",
        f"- Mean DMSO-DMSO correlation: {dmso_mean_corr:.3f}",
        f"- Highest pairwise correlation overall: {top_pair['sample_a']} vs {top_pair['sample_b']} = {top_pair['pearson_correlation']:.3f}",
        f"- Lowest pairwise correlation overall: {bottom_pair['sample_a']} vs {bottom_pair['sample_b']} = {bottom_pair['pearson_correlation']:.3f}",
        "",
        "## Main Output Files",
        "",
        "- `pca_plot.png`",
        "- `pca_scores.csv`",
        "- `well_correlation_heatmap.png`",
        "- `well_correlation_pairs.csv`",
        "- `treatment_vs_dmso_centroid.csv`",
        "- `nearest_neighbors.csv`",
    ]
    (output_dir / "evaluation_summary.md").write_text(chr(10).join(lines) + chr(10), encoding="utf-8")
