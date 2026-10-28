from __future__ import annotations

"""Run the official CellProfiler profiling pipeline and write the measurement tables.

WHY THIS FILE EXISTS
--------------------
``cp-extract-measurements`` shells out to `scripts/07_run_official_cellprofiler.py`
in this backend.  That script is not part of the public demo checkout upstream, so
the skill always fell through to its FileNotFoundError branch and reused the bundled
measurement tables instead - reporting `mode: bundled-demo-outputs` and `ok: true`.
Those tables hold four rows, so the classical branch was never exercised at dataset
scale and the degradation was invisible from the outside.

This is the missing producer.  Built to the same shape as
`segmentation_backend/scripts/03_run_mask_export.py`.

The analysis pipeline needs `Illum*` images.  The demo's
`load_data_with_illum.csv` carries only the 19 `Orig*` columns, so when the
`Illum*` columns are absent they are built here from the illumination library that
ships with the segmentation backend, resized to each raw image's shape the same way
the segmentation script does it.
"""
import argparse
import json
import shutil
import subprocess
import sys
from pathlib import Path

ILLUM_TO_ORIG = {
    "IllumMito": "OrigMito",
    "IllumAGP": "OrigAGP",
    "IllumRNA": "OrigRNA",
    "IllumER": "OrigER",
    "IllumDNA": "OrigDNA",
    "IllumBrightfield": "OrigBrightfield",
    "IllumHighZBF": "OrigHighZBF",
    "IllumLowZBF": "OrigLowZBF",
}


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run the demo profiling analysis pipeline with CellProfiler.",
    )
    parser.add_argument("--config", required=True, help="Path to pipeline_config.json.")
    parser.add_argument("--reuse-load-data", action="store_true", help="Accepted for compatibility; load-data is always reused.")
    parser.add_argument("--reuse-pipeline", action="store_true", help="Accepted for compatibility; pipeline is always reused.")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    config_path = Path(args.config).expanduser().resolve()
    if not config_path.exists():
        raise FileNotFoundError(f"Config not found: {config_path}")

    with config_path.open("r", encoding="utf-8") as handle:
        payload = json.load(handle)

    backend_root = config_path.parent.parent
    paths = payload["paths"]
    repo_root = _find_repo_root(config_path)

    load_data_path = _resolve_backend_path(backend_root, paths["load_data_with_illum_csv"])
    pipeline_path = _resolve_backend_path(backend_root, paths["cellprofiler_pipeline_analysis"])
    output_dir = _resolve_backend_path(backend_root, paths["cellprofiler_output_dir"])

    if not load_data_path.exists():
        raise FileNotFoundError(f"Profiling load-data CSV not found: {load_data_path}")
    if not pipeline_path.exists():
        raise FileNotFoundError(f"Profiling analysis pipeline not found: {pipeline_path}")

    output_dir.mkdir(parents=True, exist_ok=True)

    absolute_load_data_path = output_dir / "load_data_with_illum.absolute.csv"
    absolutize_load_data(load_data_path, absolute_load_data_path,
                         repo_root=repo_root, backend_root=backend_root)

    cellprofiler_executable = _resolve_cellprofiler_executable()
    command = [
        str(cellprofiler_executable),
        "-c",
        "-r",
        "-p",
        str(pipeline_path),
        "-o",
        str(output_dir),
        "-i",
        str(repo_root),
        "--data-file",
        str(absolute_load_data_path),
    ]
    print("[profiling_demo] running:", " ".join(command))
    return int(subprocess.run(
        command,
        check=False,
        # CellProfiler resolves its own resources with os.path.relpath(cwd);
        # a cwd on a different drive than the install raises ValueError, so the
        # child is anchored on the executable's drive.
        cwd=str(Path(cellprofiler_executable).anchor),
    ).returncode)


def absolutize_load_data(source_path: Path, destination_path: Path, *,
                         repo_root: Path, backend_root: Path) -> None:
    import numpy as np
    import pandas as pd
    from PIL import Image

    df = pd.read_csv(source_path)
    runtime_illum_dir = destination_path.parent / "illumination_runtime"
    runtime_illum_dir.mkdir(parents=True, exist_ok=True)

    for column in df.columns:
        if column.startswith("PathName_"):
            df[column] = df[column].map(lambda value: str(_resolve_repo_relative_path(repo_root, value)))

    illum_search_dirs = _illumination_search_dirs(backend_root, repo_root)
    plate = str(df["Metadata_Plate"].iloc[0]) if "Metadata_Plate" in df.columns and len(df) else ""

    for illum_name, orig_name in ILLUM_TO_ORIG.items():
        raw_file_column = f"FileName_{orig_name}"
        raw_path_column = f"PathName_{orig_name}"
        if raw_file_column not in df.columns:
            continue

        file_column = f"FileName_{illum_name}"
        path_column = f"PathName_{illum_name}"

        adjusted_paths: list[str] = []
        adjusted_names: list[str] = []
        for row in df.itertuples(index=False):
            raw_path = Path(getattr(row, raw_path_column)) / str(getattr(row, raw_file_column))
            with Image.open(raw_path) as raw_image:
                target_shape = tuple(int(value) for value in raw_image.size[::-1])

            illum_path = _locate_illumination(row, file_column, path_column,
                                              illum_name, plate, illum_search_dirs)
            if illum_path is None:
                break
            adjusted_array = _match_shape(np.load(illum_path), target_shape)
            adjusted_name = f"{illum_path.stem}_{target_shape[0]}x{target_shape[1]}.npy"
            np.save(runtime_illum_dir / adjusted_name, adjusted_array)
            adjusted_paths.append(str(runtime_illum_dir))
            adjusted_names.append(adjusted_name)

        if len(adjusted_names) != len(df):
            # No illumination available for this channel: leave the columns absent and
            # let CellProfiler report it, rather than silently substituting something.
            df = df.drop(columns=[c for c in (file_column, path_column) if c in df.columns])
            continue

        df[path_column] = adjusted_paths
        df[file_column] = adjusted_names

    df.to_csv(destination_path, index=False)


def _locate_illumination(row, file_column: str, path_column: str,
                         illum_name: str, plate: str, search_dirs: list[Path]) -> Path | None:
    """Prefer the path named by the table; otherwise look in the illumination library."""
    if file_column in row._fields and path_column in row._fields:
        named = Path(str(getattr(row, path_column))) / str(getattr(row, file_column))
        if named.exists():
            return named
    for directory in search_dirs:
        for candidate in (directory / f"{plate}_{illum_name}.npy",
                          directory / f"{illum_name}.npy"):
            if candidate.exists():
                return candidate
    return None


def _illumination_search_dirs(backend_root: Path, repo_root: Path) -> list[Path]:
    backend_dir = backend_root.parent
    return [
        backend_root / "data" / "illumination",
        backend_dir / "segmentation_backend" / "data" / "illumination",
        repo_root / "demo" / "backend" / "segmentation_backend" / "data" / "illumination",
    ]


def _match_shape(array, target_shape: tuple[int, int]):
    current_shape = tuple(int(value) for value in array.shape[:2])
    target_height, target_width = target_shape
    height, width = current_shape
    if (height, width) == (target_height, target_width):
        return array

    start_y = max(0, (height - target_height) // 2)
    start_x = max(0, (width - target_width) // 2)
    cropped = array[start_y:start_y + target_height, start_x:start_x + target_width]
    if cropped.shape[:2] == (target_height, target_width):
        return cropped

    result = array[:target_height, :target_width]
    if result.shape[:2] != (target_height, target_width):
        raise ValueError(
            f"Could not match illumination array from shape {current_shape} to target shape {(target_height, target_width)}."
        )
    return result


def _resolve_cellprofiler_executable() -> Path:
    import os

    override = os.environ.get("CELLPROFILER_EXECUTABLE")
    if override:
        candidate = Path(override).expanduser()
        if candidate.exists():
            return candidate
    sibling = Path(sys.executable).resolve().with_name("cellprofiler")
    if sibling.exists():
        return sibling
    resolved = shutil.which("cellprofiler")
    if resolved:
        return Path(resolved)
    for candidate in (
        Path(r"C:\CellProfiler\CellProfiler.exe"),
        Path(os.environ.get("ProgramFiles", r"C:\Program Files")) / "CellProfiler" / "CellProfiler.exe",
    ):
        if candidate.exists():
            return candidate
    resolved = None
    if resolved is None:
        raise FileNotFoundError("Could not find CellProfiler executable in the current runtime.")
    return Path(resolved).resolve()


def _find_repo_root(config_path: Path) -> Path:
    for candidate in [config_path, *config_path.parents]:
        if (candidate / "src" / "cellpaint_pipeline").exists():
            return candidate
    for parent in config_path.parents:
        if parent.name == "CellPainting-Claw":
            return parent
    raise FileNotFoundError(f"Could not determine repository root from {config_path}")


def _resolve_backend_path(backend_root: Path, value: str) -> Path:
    candidate = Path(value).expanduser()
    if candidate.is_absolute():
        return candidate.resolve()
    return (backend_root / candidate).resolve()


def _resolve_repo_relative_path(repo_root: Path, value: str) -> Path:
    candidate = Path(str(value)).expanduser()
    if candidate.is_absolute():
        return candidate.resolve()
    return (repo_root / candidate).resolve()


if __name__ == "__main__":
    raise SystemExit(main())
