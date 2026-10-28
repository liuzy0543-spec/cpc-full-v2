from __future__ import annotations

import json
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any


class ConfigContractError(ValueError):
    """Raised when a project config cannot be parsed into a stable contract."""


@dataclass(frozen=True)
class DataAccessConfig:
    gallery_bucket: str
    gallery_region: str
    gallery_endpoint_url: str | None
    use_unsigned: bool
    default_dataset_id: str | None
    default_source_id: str | None
    quilt_registry: str | None
    cpgdata_inventory_bucket: str
    cpgdata_index_prefix: str
    cpgdata_inventory_prefix: str
    data_cache_root: Path
    index_cache_root: Path
    preferred_packages: tuple[str, ...]

    def as_dict(self) -> dict[str, Any]:
        return {
            "gallery_bucket": self.gallery_bucket,
            "gallery_region": self.gallery_region,
            "gallery_endpoint_url": self.gallery_endpoint_url,
            "use_unsigned": self.use_unsigned,
            "default_dataset_id": self.default_dataset_id,
            "default_source_id": self.default_source_id,
            "quilt_registry": self.quilt_registry,
            "cpgdata_inventory_bucket": self.cpgdata_inventory_bucket,
            "cpgdata_index_prefix": self.cpgdata_index_prefix,
            "cpgdata_inventory_prefix": self.cpgdata_inventory_prefix,
            "data_cache_root": str(self.data_cache_root),
            "index_cache_root": str(self.index_cache_root),
            "preferred_packages": list(self.preferred_packages),
        }


@dataclass(frozen=True)
class CellProfilerConfig:
    profiling_template: str
    segmentation_template: str
    custom_profiling_cppipe_path: Path | None
    custom_segmentation_cppipe_path: Path | None

    def as_dict(self) -> dict[str, Any]:
        return {
            "profiling_template": self.profiling_template,
            "segmentation_template": self.segmentation_template,
            "custom_profiling_cppipe_path": (
                str(self.custom_profiling_cppipe_path) if self.custom_profiling_cppipe_path is not None else None
            ),
            "custom_segmentation_cppipe_path": (
                str(self.custom_segmentation_cppipe_path) if self.custom_segmentation_cppipe_path is not None else None
            ),
        }


@dataclass(frozen=True)
class ProjectConfig:
    project_name: str
    python_executable: str
    profiling_backend_root: Path
    profiling_backend_config: Path
    segmentation_backend_root: Path
    segmentation_backend_config: Path
    workspace_root: Path
    default_output_root: Path
    deepprofiler_export_root: Path
    deepprofiler_project_root: Path
    mask_export_runtime: dict[str, int | str | bool]
    deepprofiler_runtime: dict[str, Any]
    data_access: DataAccessConfig
    cellprofiler: CellProfilerConfig

    @classmethod
    def from_json(cls, path: str | Path) -> "ProjectConfig":
        config_path = Path(path).expanduser().resolve()
        if not config_path.exists():
            raise ConfigContractError(f'Config file not found: {config_path}')

        try:
            with config_path.open("r", encoding="utf-8") as handle:
                payload = json.load(handle)
        except json.JSONDecodeError as exc:
            raise ConfigContractError(
                f'Failed to parse config JSON at {config_path}: {exc.msg} (line {exc.lineno}, column {exc.colno}).'
            ) from exc
        except OSError as exc:
            raise ConfigContractError(f'Failed to read config file {config_path}: {exc}') from exc

        required_fields = [
            "project_name",
            "profiling_backend_root",
            "segmentation_backend_root",
            "workspace_root",
            "default_output_root",
            "deepprofiler_export_root",
        ]
        missing_fields = [name for name in required_fields if name not in payload or payload[name] in (None, '')]
        if missing_fields:
            missing = ', '.join(missing_fields)
            # Spell the contract out here instead of pointing at a document.
            # The message used to say "see docs/config_contract.md", and that
            # file does not exist -- a dead pointer in the one place a user is
            # already stuck.
            raise ConfigContractError(
                f'Missing required config field(s) in {config_path}: {missing}. '
                f'The top-level contract requires all of: '
                f'{", ".join(required_fields)}.'
            )

        config_dir = config_path.parent
        python_executable = payload.get("python_executable") or sys.executable
        profiling_backend_root = _resolve_from_base(config_dir, payload["profiling_backend_root"], config_dir / payload["profiling_backend_root"])
        segmentation_backend_root = _resolve_from_base(config_dir, payload["segmentation_backend_root"], config_dir / payload["segmentation_backend_root"])
        workspace_root = _resolve_from_base(config_dir, payload["workspace_root"], config_dir / payload["workspace_root"])
        default_output_root = _resolve_from_base(config_dir, payload["default_output_root"], config_dir / payload["default_output_root"])
        deepprofiler_export_root = _resolve_from_base(config_dir, payload["deepprofiler_export_root"], config_dir / payload["deepprofiler_export_root"])
        deepprofiler_project_root = _resolve_from_base(
            config_dir,
            payload.get("deepprofiler_project_root"),
            default_output_root / "deepprofiler_project",
        )

        for label, resolved_path in [
            ("profiling_backend_root", profiling_backend_root),
            ("segmentation_backend_root", segmentation_backend_root),
            ("workspace_root", workspace_root),
        ]:
            if not resolved_path.exists():
                raise ConfigContractError(
                    f'Configured {label} does not exist: {resolved_path}. '
                    f'Check {config_path} or create the expected workspace path first.'
                )

        profiling_backend_config = _resolve_under_root(
            profiling_backend_root,
            payload.get("profiling_backend_config", "configs/pipeline_config.json"),
        )
        if not profiling_backend_config.exists():
            raise ConfigContractError(
                f'Configured profiling_backend_config does not exist: {profiling_backend_config}. '
                f'It is resolved relative to profiling_backend_root unless given as an absolute path.'
            )

        segmentation_backend_config = _resolve_under_root(
            segmentation_backend_root,
            payload.get("segmentation_backend_config", "configs/segmentation_config.json"),
        )
        if not segmentation_backend_config.exists():
            raise ConfigContractError(
                f'Configured segmentation_backend_config does not exist: {segmentation_backend_config}. '
                f'It is resolved relative to segmentation_backend_root unless given as an absolute path.'
            )

        data_access = _build_data_access_config(dict(payload.get("data_access", {})), workspace_root)
        cellprofiler = _build_cellprofiler_config(dict(payload.get("cellprofiler", {})), config_dir)

        return cls(
            project_name=payload["project_name"],
            python_executable=python_executable,
            profiling_backend_root=profiling_backend_root,
            profiling_backend_config=profiling_backend_config,
            segmentation_backend_root=segmentation_backend_root,
            segmentation_backend_config=segmentation_backend_config,
            workspace_root=workspace_root,
            default_output_root=default_output_root,
            deepprofiler_export_root=deepprofiler_export_root,
            deepprofiler_project_root=deepprofiler_project_root,
            mask_export_runtime=dict(payload.get("mask_export_runtime", {})),
            deepprofiler_runtime=dict(payload.get("deepprofiler_runtime", {})),
            data_access=data_access,
            cellprofiler=cellprofiler,
        )

    @property
    def log_root(self) -> Path:
        return self.default_output_root / "logs"

    def load_profiling_backend_payload(self) -> dict:
        with self.profiling_backend_config.open("r", encoding="utf-8") as handle:
            return json.load(handle)

    def load_segmentation_backend_payload(self) -> dict:
        with self.segmentation_backend_config.open("r", encoding="utf-8") as handle:
            return json.load(handle)

    def resolve_profiling_backend_path(self, value: str | Path) -> Path:
        return _resolve_under_root(self.profiling_backend_root, str(value))

    def resolve_segmentation_backend_path(self, value: str | Path) -> Path:
        return _resolve_under_root(self.segmentation_backend_root, str(value))

    def as_dict(self) -> dict[str, Any]:
        return {
            "project_name": self.project_name,
            "python_executable": self.python_executable,
            "profiling_backend_root": str(self.profiling_backend_root),
            "profiling_backend_config": str(self.profiling_backend_config),
            "segmentation_backend_root": str(self.segmentation_backend_root),
            "segmentation_backend_config": str(self.segmentation_backend_config),
            "workspace_root": str(self.workspace_root),
            "default_output_root": str(self.default_output_root),
            "deepprofiler_export_root": str(self.deepprofiler_export_root),
            "deepprofiler_project_root": str(self.deepprofiler_project_root),
            "mask_export_runtime": self.mask_export_runtime,
            "deepprofiler_runtime": self.deepprofiler_runtime,
            "data_access": self.data_access.as_dict(),
            "cellprofiler": self.cellprofiler.as_dict(),
            "log_root": str(self.log_root),
        }

PROJECT_CONFIG_FIELD_GUIDE: tuple[dict[str, Any], ...] = (
    {
        'name': 'project_name',
        'level': 'required',
        'default': None,
        'description': 'Logical name for the project and reporting outputs.',
    },
    {
        'name': 'profiling_backend_root',
        'level': 'required',
        'default': None,
        'description': 'Root directory of the validated profiling backend workspace.',
    },
    {
        'name': 'segmentation_backend_root',
        'level': 'required',
        'default': None,
        'description': 'Root directory of the validated segmentation backend workspace.',
    },
    {
        'name': 'workspace_root',
        'level': 'required',
        'default': None,
        'description': 'Shared workspace base used to resolve relative cache and output paths.',
    },
    {
        'name': 'default_output_root',
        'level': 'required',
        'default': None,
        'description': 'Default root under which library outputs, deliveries, and reports are written.',
    },
    {
        'name': 'deepprofiler_export_root',
        'level': 'required',
        'default': None,
        'description': 'Default export root used when building DeepProfiler-ready inputs.',
    },
    {
        'name': 'python_executable',
        'level': 'recommended',
        'default': 'sys.executable',
        'description': 'Interpreter used for subprocess-backed tasks such as DeepProfiler profile execution.',
    },
    {
        'name': 'profiling_backend_config',
        'level': 'recommended',
        'default': 'configs/pipeline_config.json under profiling_backend_root',
        'description': 'Config file resolved under the profiling backend root unless overridden with an absolute path.',
    },
    {
        'name': 'segmentation_backend_config',
        'level': 'recommended',
        'default': 'configs/segmentation_config.json under segmentation_backend_root',
        'description': 'Config file resolved under the segmentation backend root unless overridden with an absolute path.',
    },
    {
        'name': 'deepprofiler_project_root',
        'level': 'recommended',
        'default': 'default_output_root/deepprofiler_project',
        'description': 'Default location for generated DeepProfiler project directories.',
    },
    {
        'name': 'mask_export_runtime',
        'level': 'advanced',
        'default': '{}',
        'description': 'Runtime overrides for sharding and worker allocation during mask-export style workflows.',
    },
    {
        'name': 'deepprofiler_runtime',
        'level': 'advanced',
        'default': '{}',
        'description': 'DeepProfiler-specific runtime knobs such as experiment name, batch size, and checkpoint settings.',
    },
    {
        'name': 'data_access',
        'level': 'recommended',
        'default': '{} with library defaults',
        'description': 'Nested data-access configuration block covering gallery, Quilt, and cpgdata defaults.',
    },
    {
        'name': 'cellprofiler',
        'level': 'recommended',
        'default': '{} with library defaults',
        'description': 'Nested CellProfiler pipeline-selection block covering bundled template choice and optional custom .cppipe overrides.',
    },
)

DATA_ACCESS_CONFIG_FIELD_GUIDE: tuple[dict[str, Any], ...] = (
    {
        'name': 'gallery_bucket',
        'level': 'recommended',
        'default': 'cellpainting-gallery',
        'description': 'Default Cell Painting Gallery bucket used for listing and download planning.',
    },
    {
        'name': 'gallery_region',
        'level': 'recommended',
        'default': 'us-east-1',
        'description': 'Region used for unsigned gallery access.',
    },
    {
        'name': 'gallery_endpoint_url',
        'level': 'advanced',
        'default': None,
        'description': 'Optional endpoint override for gallery-compatible object storage.',
    },
    {
        'name': 'use_unsigned',
        'level': 'recommended',
        'default': True,
        'description': 'Whether to use unsigned access for the gallery bucket.',
    },
    {
        'name': 'default_dataset_id',
        'level': 'recommended',
        'default': None,
        'description': 'Default dataset used when higher-level planning omits an explicit dataset id.',
    },
    {
        'name': 'default_source_id',
        'level': 'recommended',
        'default': None,
        'description': 'Default source used when higher-level planning omits an explicit source id.',
    },
    {
        'name': 'quilt_registry',
        'level': 'recommended',
        'default': 's3://cellpainting-gallery',
        'description': 'Registry used for Quilt package discovery.',
    },
    {
        'name': 'cpgdata_inventory_bucket',
        'level': 'recommended',
        'default': 'cellpainting-gallery-inventory',
        'description': 'Bucket used for cpgdata inventory access.',
    },
    {
        'name': 'cpgdata_index_prefix',
        'level': 'recommended',
        'default': 'cellpainting-gallery/index',
        'description': 'Prefix used when syncing cpgdata index files.',
    },
    {
        'name': 'cpgdata_inventory_prefix',
        'level': 'recommended',
        'default': 'cellpainting-gallery/whole_bucket/',
        'description': 'Prefix used when browsing or syncing cpgdata inventory files.',
    },
    {
        'name': 'data_cache_root',
        'level': 'recommended',
        'default': 'workspace_root/cellpainting-claw/cache/data',
        'description': 'Default cache root for downloaded gallery data.',
    },
    {
        'name': 'index_cache_root',
        'level': 'recommended',
        'default': 'workspace_root/cellpainting-claw/cache/index',
        'description': 'Default cache root for listing snapshots and access-package metadata.',
    },
    {
        'name': 'preferred_packages',
        'level': 'advanced',
        'default': ['boto3', 'cpgdata', 'quilt3'],
        'description': 'Preferred adapter order for access-package discovery or future auto-selection logic.',
    },
)


CELLPROFILER_CONFIG_FIELD_GUIDE: tuple[dict[str, Any], ...] = (
    {
        'name': 'profiling_template',
        'level': 'recommended',
        'default': 'profiling-analysis',
        'description': 'Bundled profiling-side .cppipe template key used when no custom profiling .cppipe path is provided.',
    },
    {
        'name': 'segmentation_template',
        'level': 'recommended',
        'default': 'segmentation-base',
        'description': 'Bundled segmentation-side .cppipe template key used when no custom segmentation .cppipe path is provided.',
    },
    {
        'name': 'custom_profiling_cppipe_path',
        'level': 'advanced',
        'default': None,
        'description': 'Optional custom profiling .cppipe path. When set, it overrides profiling_template.',
    },
    {
        'name': 'custom_segmentation_cppipe_path',
        'level': 'advanced',
        'default': None,
        'description': 'Optional custom segmentation .cppipe path. When set, it overrides segmentation_template.',
    },
)


def project_config_field_guide() -> list[dict[str, Any]]:
    return [dict(item) for item in PROJECT_CONFIG_FIELD_GUIDE]


def data_access_config_field_guide() -> list[dict[str, Any]]:
    return [dict(item) for item in DATA_ACCESS_CONFIG_FIELD_GUIDE]


def cellprofiler_config_field_guide() -> list[dict[str, Any]]:
    return [dict(item) for item in CELLPROFILER_CONFIG_FIELD_GUIDE]


def project_config_contract_summary() -> dict[str, Any]:
    return {
        'required_project_fields': [item['name'] for item in PROJECT_CONFIG_FIELD_GUIDE if item['level'] == 'required'],
        'recommended_project_fields': [item['name'] for item in PROJECT_CONFIG_FIELD_GUIDE if item['level'] == 'recommended'],
        'advanced_project_fields': [item['name'] for item in PROJECT_CONFIG_FIELD_GUIDE if item['level'] == 'advanced'],
        'project_fields': project_config_field_guide(),
        'data_access_fields': data_access_config_field_guide(),
        'cellprofiler_fields': cellprofiler_config_field_guide(),
        'notes': [
            'ProjectConfig.from_json resolves relative top-level paths from the config file location and fills several backend-related defaults automatically.',
            'The data_access block is optional, but providing default_dataset_id and default_source_id is recommended for reproducible planning flows.',
            'The cellprofiler block is optional. Bundled template defaults are used unless custom .cppipe override paths are provided.',
        ],
    }




def _build_data_access_config(payload: dict[str, Any], workspace_root: Path) -> DataAccessConfig:
    return DataAccessConfig(
        gallery_bucket=str(payload.get("gallery_bucket", "cellpainting-gallery")),
        gallery_region=str(payload.get("gallery_region", "us-east-1")),
        gallery_endpoint_url=(str(payload["gallery_endpoint_url"]).strip() if payload.get("gallery_endpoint_url") else None),
        use_unsigned=bool(payload.get("use_unsigned", True)),
        default_dataset_id=(str(payload["default_dataset_id"]) if payload.get("default_dataset_id") else None),
        default_source_id=(str(payload["default_source_id"]) if payload.get("default_source_id") else None),
        quilt_registry=(str(payload["quilt_registry"]).strip() if payload.get("quilt_registry") else "s3://cellpainting-gallery"),
        cpgdata_inventory_bucket=str(payload.get("cpgdata_inventory_bucket", "cellpainting-gallery-inventory")),
        cpgdata_index_prefix=str(payload.get("cpgdata_index_prefix", "cellpainting-gallery/index")),
        cpgdata_inventory_prefix=str(payload.get("cpgdata_inventory_prefix", "cellpainting-gallery/whole_bucket/")),
        data_cache_root=_resolve_from_base(
            workspace_root,
            payload.get("data_cache_root"),
            workspace_root / "cellpainting-claw" / "cache" / "data",
        ),
        index_cache_root=_resolve_from_base(
            workspace_root,
            payload.get("index_cache_root"),
            workspace_root / "cellpainting-claw" / "cache" / "index",
        ),
        preferred_packages=tuple(str(item) for item in payload.get("preferred_packages", ["boto3", "cpgdata", "quilt3"])),
    )


def _build_cellprofiler_config(payload: dict[str, Any], config_root: Path) -> CellProfilerConfig:
    return CellProfilerConfig(
        profiling_template=str(payload.get("profiling_template", "profiling-analysis")),
        segmentation_template=str(payload.get("segmentation_template", "segmentation-base")),
        custom_profiling_cppipe_path=_resolve_optional_from_base(
            config_root,
            payload.get("custom_profiling_cppipe_path"),
        ),
        custom_segmentation_cppipe_path=_resolve_optional_from_base(
            config_root,
            payload.get("custom_segmentation_cppipe_path"),
        ),
    )


def _resolve_under_root(root: Path, value: str) -> Path:
    path = Path(value).expanduser()
    if path.is_absolute():
        return path.resolve()
    return (root / path).resolve()


def _resolve_from_base(base: Path, value: str | Path | None, default: Path) -> Path:
    if value is None:
        return default.expanduser().resolve()
    path = Path(value).expanduser()
    if path.is_absolute():
        return path.resolve()
    return (base / path).resolve()


def _resolve_optional_from_base(base: Path, value: str | Path | None) -> Path | None:
    if value in (None, ''):
        return None
    path = Path(value).expanduser()
    if path.is_absolute():
        return path.resolve()
    return (base / path).resolve()
