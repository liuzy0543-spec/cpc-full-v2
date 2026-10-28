from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any


if TYPE_CHECKING:
    from cellpaint_pipeline.config import ProjectConfig


@dataclass(frozen=True)
class QuiltPackageListResult:
    registry: str
    package_names: list[str]
    limit: int | None


@dataclass(frozen=True)
class QuiltPackageBrowseResult:
    registry: str
    package_name: str
    requested_top_hash: str | None
    resolved_top_hash: str | None
    top_level_keys: list[str]
    top_level_key_count: int


@dataclass(frozen=True)
class CPGDataPrefixListResult:
    bucket: str
    prefix: str
    recursive: bool
    limit: int | None
    entries: list[str]


@dataclass(frozen=True)
class CPGDataSyncResult:
    mode: str
    bucket: str
    prefix: str
    output_dir: Path
    revision: int | None
    include: str | None
    exclude: str | None
    no_progress: bool


def list_quilt_packages(
    config: ProjectConfig,
    *,
    registry: str | None = None,
    limit: int | None = None,
) -> QuiltPackageListResult:
    quilt3 = _require_quilt3()
    resolved_registry = registry or config.data_access.quilt_registry
    if not resolved_registry:
        raise ValueError('registry is required when data_access.quilt_registry is not configured.')
    package_names = sorted(str(name) for name in quilt3.list_packages(registry=resolved_registry))
    if limit is not None:
        package_names = package_names[:limit]
    return QuiltPackageListResult(
        registry=resolved_registry,
        package_names=package_names,
        limit=limit,
    )


def quilt_package_list_result_to_dict(result: QuiltPackageListResult) -> dict[str, Any]:
    return {
        'implementation': 'cellpaint_pipeline.data_access.access_packages',
        'adapter': 'quilt3',
        'registry': result.registry,
        'limit': result.limit,
        'package_names': result.package_names,
        'package_count': len(result.package_names),
    }


def browse_quilt_package(
    config: ProjectConfig,
    *,
    package_name: str,
    registry: str | None = None,
    top_hash: str | None = None,
    max_keys: int | None = 200,
) -> QuiltPackageBrowseResult:
    quilt3 = _require_quilt3()
    resolved_registry = registry or config.data_access.quilt_registry
    if not resolved_registry:
        raise ValueError('registry is required when data_access.quilt_registry is not configured.')
    package = quilt3.Package.browse(package_name, registry=resolved_registry, top_hash=top_hash)
    keys = sorted(str(key) for key in package.keys())
    if max_keys is not None:
        keys = keys[:max_keys]
    return QuiltPackageBrowseResult(
        registry=resolved_registry,
        package_name=package_name,
        requested_top_hash=top_hash,
        resolved_top_hash=str(package.top_hash) if getattr(package, 'top_hash', None) else None,
        top_level_keys=keys,
        top_level_key_count=len(keys),
    )


def quilt_package_browse_result_to_dict(result: QuiltPackageBrowseResult) -> dict[str, Any]:
    return {
        'implementation': 'cellpaint_pipeline.data_access.access_packages',
        'adapter': 'quilt3',
        'registry': result.registry,
        'package_name': result.package_name,
        'requested_top_hash': result.requested_top_hash,
        'resolved_top_hash': result.resolved_top_hash,
        'top_level_keys': result.top_level_keys,
        'top_level_key_count': result.top_level_key_count,
    }


def list_cpgdata_prefixes(
    config: ProjectConfig,
    *,
    bucket: str | None = None,
    prefix: str | None = None,
    recursive: bool = False,
    limit: int | None = None,
) -> CPGDataPrefixListResult:
    cpg_utils = _require_cpgdata_utils()
    resolved_bucket = bucket or config.data_access.cpgdata_inventory_bucket
    resolved_prefix = prefix or config.data_access.cpgdata_inventory_prefix
    entries = [str(item) for item in cpg_utils.ls_s3_prefix(resolved_bucket, resolved_prefix, recursive=recursive)]
    if limit is not None:
        entries = entries[:limit]
    return CPGDataPrefixListResult(
        bucket=resolved_bucket,
        prefix=resolved_prefix,
        recursive=recursive,
        limit=limit,
        entries=entries,
    )


def cpgdata_prefix_list_result_to_dict(result: CPGDataPrefixListResult) -> dict[str, Any]:
    return {
        'implementation': 'cellpaint_pipeline.data_access.access_packages',
        'adapter': 'cpgdata',
        'bucket': result.bucket,
        'prefix': result.prefix,
        'recursive': result.recursive,
        'limit': result.limit,
        'entries': result.entries,
        'entry_count': len(result.entries),
    }


def sync_cpgdata_index(
    config: ProjectConfig,
    *,
    output_dir: Path | None = None,
    bucket: str | None = None,
    prefix: str | None = None,
    include: str | None = None,
    exclude: str | None = None,
    no_progress: bool = True,
) -> CPGDataSyncResult:
    cpg_utils = _require_cpgdata_utils()
    resolved_bucket = bucket or config.data_access.cpgdata_inventory_bucket
    resolved_prefix = prefix or config.data_access.cpgdata_index_prefix
    resolved_output_dir = (
        output_dir.expanduser().resolve()
        if output_dir is not None
        else config.data_access.index_cache_root / 'cpgdata_index'
    )
    resolved_output_dir.mkdir(parents=True, exist_ok=True)
    cpg_utils.sync_s3_prefix(
        resolved_bucket,
        resolved_prefix,
        resolved_output_dir,
        include=include,
        exclude=exclude,
        no_progress=no_progress,
    )
    return CPGDataSyncResult(
        mode='index',
        bucket=resolved_bucket,
        prefix=resolved_prefix,
        output_dir=resolved_output_dir,
        revision=None,
        include=include,
        exclude=exclude,
        no_progress=no_progress,
    )


def sync_cpgdata_inventory(
    config: ProjectConfig,
    *,
    output_dir: Path | None = None,
    bucket: str | None = None,
    prefix: str | None = None,
    revision: int = 0,
) -> CPGDataSyncResult:
    cpg_utils = _require_cpgdata_utils()
    resolved_bucket = bucket or config.data_access.cpgdata_inventory_bucket
    resolved_prefix = prefix or config.data_access.cpgdata_inventory_prefix
    resolved_output_dir = (
        output_dir.expanduser().resolve()
        if output_dir is not None
        else config.data_access.index_cache_root / 'cpgdata_inventory'
    )
    resolved_output_dir.mkdir(parents=True, exist_ok=True)
    cpg_utils.sync_inventory(resolved_bucket, resolved_prefix, resolved_output_dir, revision=revision)
    return CPGDataSyncResult(
        mode='inventory',
        bucket=resolved_bucket,
        prefix=resolved_prefix,
        output_dir=resolved_output_dir,
        revision=revision,
        include=None,
        exclude=None,
        no_progress=False,
    )


def cpgdata_sync_result_to_dict(result: CPGDataSyncResult) -> dict[str, Any]:
    return {
        'implementation': 'cellpaint_pipeline.data_access.access_packages',
        'adapter': 'cpgdata',
        'mode': result.mode,
        'bucket': result.bucket,
        'prefix': result.prefix,
        'output_dir': str(result.output_dir),
        'revision': result.revision,
        'include': result.include,
        'exclude': result.exclude,
        'no_progress': result.no_progress,
    }


def _require_quilt3():
    try:
        import quilt3
    except ImportError as exc:
        raise RuntimeError('quilt3 is required for quilt-backed data access. Install the data-access extra first.') from exc
    return quilt3


def _require_cpgdata_utils():
    try:
        from cpgdata import utils as cpg_utils
    except ImportError as exc:
        raise RuntimeError('cpgdata is required for cpgdata-backed data access. Install the data-access extra first.') from exc
    return cpg_utils
