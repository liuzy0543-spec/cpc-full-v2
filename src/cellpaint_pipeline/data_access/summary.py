from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

from cellpaint_pipeline.data_access.access_packages import (
    CPGDataPrefixListResult,
    QuiltPackageListResult,
    cpgdata_prefix_list_result_to_dict,
    list_cpgdata_prefixes,
    list_quilt_packages,
    quilt_package_list_result_to_dict,
)
from cellpaint_pipeline.data_access.gallery import (
    DataAccessStatus,
    GalleryCatalogResult,
    build_data_access_status,
    data_access_status_to_dict,
    gallery_catalog_result_to_dict,
    list_gallery_datasets,
    list_gallery_sources,
)


if TYPE_CHECKING:
    from cellpaint_pipeline.config import ProjectConfig


@dataclass(frozen=True)
class DataAccessSummaryResult:
    status: DataAccessStatus
    resolved_dataset_id: str | None
    gallery_bucket: str
    gallery_max_keys: int
    dataset_listing: GalleryCatalogResult | None
    source_listing: GalleryCatalogResult | None
    quilt_registry: str | None
    quilt_limit: int | None
    quilt_packages: QuiltPackageListResult | None
    cpgdata_bucket: str
    cpgdata_prefix: str
    cpgdata_recursive: bool
    cpgdata_limit: int | None
    cpgdata_prefixes: CPGDataPrefixListResult | None
    include_gallery: bool
    include_quilt: bool
    include_cpgdata: bool
    errors: dict[str, str]
    notes: list[str]
    ok: bool


def summarize_data_access(
    config: ProjectConfig,
    *,
    dataset_id: str | None = None,
    gallery_bucket: str | None = None,
    gallery_max_keys: int = 1000,
    registry: str | None = None,
    quilt_limit: int | None = None,
    cpgdata_bucket: str | None = None,
    cpgdata_prefix: str | None = None,
    cpgdata_recursive: bool = False,
    cpgdata_limit: int | None = None,
    include_gallery: bool = True,
    include_quilt: bool = True,
    include_cpgdata: bool = True,
) -> DataAccessSummaryResult:
    status = build_data_access_status(config)
    resolved_dataset_id = dataset_id or config.data_access.default_dataset_id
    resolved_gallery_bucket = gallery_bucket or config.data_access.gallery_bucket
    resolved_registry = registry or config.data_access.quilt_registry
    resolved_cpgdata_bucket = cpgdata_bucket or config.data_access.cpgdata_inventory_bucket
    resolved_cpgdata_prefix = cpgdata_prefix or config.data_access.cpgdata_inventory_prefix

    dataset_listing = None
    source_listing = None
    quilt_packages = None
    cpgdata_prefixes = None
    errors: dict[str, str] = {}
    notes: list[str] = []

    if include_gallery:
        try:
            dataset_listing = list_gallery_datasets(
                config,
                max_keys=gallery_max_keys,
                bucket=resolved_gallery_bucket,
            )
        except Exception as exc:
            errors['gallery_datasets'] = str(exc)
        if resolved_dataset_id:
            try:
                source_listing = list_gallery_sources(
                    config,
                    dataset_id=resolved_dataset_id,
                    max_keys=gallery_max_keys,
                    bucket=resolved_gallery_bucket,
                )
            except Exception as exc:
                errors['gallery_sources'] = str(exc)
        else:
            notes.append('No dataset_id resolved; skipped gallery source listing.')

    if include_quilt:
        if resolved_registry:
            try:
                quilt_packages = list_quilt_packages(
                    config,
                    registry=resolved_registry,
                    limit=quilt_limit,
                )
            except Exception as exc:
                errors['quilt_packages'] = str(exc)
        else:
            notes.append('No Quilt registry resolved; skipped Quilt package listing.')

    if include_cpgdata:
        try:
            cpgdata_prefixes = list_cpgdata_prefixes(
                config,
                bucket=resolved_cpgdata_bucket,
                prefix=resolved_cpgdata_prefix,
                recursive=cpgdata_recursive,
                limit=cpgdata_limit,
            )
        except Exception as exc:
            errors['cpgdata_prefixes'] = str(exc)

    return DataAccessSummaryResult(
        status=status,
        resolved_dataset_id=resolved_dataset_id,
        gallery_bucket=resolved_gallery_bucket,
        gallery_max_keys=gallery_max_keys,
        dataset_listing=dataset_listing,
        source_listing=source_listing,
        quilt_registry=resolved_registry,
        quilt_limit=quilt_limit,
        quilt_packages=quilt_packages,
        cpgdata_bucket=resolved_cpgdata_bucket,
        cpgdata_prefix=resolved_cpgdata_prefix,
        cpgdata_recursive=cpgdata_recursive,
        cpgdata_limit=cpgdata_limit,
        cpgdata_prefixes=cpgdata_prefixes,
        include_gallery=include_gallery,
        include_quilt=include_quilt,
        include_cpgdata=include_cpgdata,
        errors=errors,
        notes=notes,
        # An inspection reports what is reachable; a missing optional adapter is
        # a finding, not a failure of the inspection.  Config-level problems
        # (missing required packages) still fail through ``status.ok``.
        ok=status.ok,
    )


def data_access_summary_to_dict(result: DataAccessSummaryResult) -> dict[str, Any]:
    return {
        'implementation': 'cellpaint_pipeline.data_access.summary',
        'ok': result.ok,
        'requested': {
            'dataset_id': result.resolved_dataset_id,
            'gallery_bucket': result.gallery_bucket,
            'gallery_max_keys': result.gallery_max_keys,
            'quilt_registry': result.quilt_registry,
            'quilt_limit': result.quilt_limit,
            'cpgdata_bucket': result.cpgdata_bucket,
            'cpgdata_prefix': result.cpgdata_prefix,
            'cpgdata_recursive': result.cpgdata_recursive,
            'cpgdata_limit': result.cpgdata_limit,
            'include_gallery': result.include_gallery,
            'include_quilt': result.include_quilt,
            'include_cpgdata': result.include_cpgdata,
        },
        'status': data_access_status_to_dict(result.status),
        'gallery': {
            'enabled': result.include_gallery,
            'bucket': result.gallery_bucket,
            'resolved_dataset_id': result.resolved_dataset_id,
            'max_keys': result.gallery_max_keys,
            'datasets': gallery_catalog_result_to_dict(result.dataset_listing) if result.dataset_listing else None,
            'sources': gallery_catalog_result_to_dict(result.source_listing) if result.source_listing else None,
            'errors': {
                'datasets': result.errors.get('gallery_datasets'),
                'sources': result.errors.get('gallery_sources'),
            },
        },
        'quilt': {
            'enabled': result.include_quilt,
            'registry': result.quilt_registry,
            'limit': result.quilt_limit,
            'packages': quilt_package_list_result_to_dict(result.quilt_packages) if result.quilt_packages else None,
            'error': result.errors.get('quilt_packages'),
        },
        'cpgdata': {
            'enabled': result.include_cpgdata,
            'bucket': result.cpgdata_bucket,
            'prefix': result.cpgdata_prefix,
            'recursive': result.cpgdata_recursive,
            'limit': result.cpgdata_limit,
            'prefixes': cpgdata_prefix_list_result_to_dict(result.cpgdata_prefixes) if result.cpgdata_prefixes else None,
            'error': result.errors.get('cpgdata_prefixes'),
        },
        'errors': result.errors,
        # Which optional adapters could not be queried, so a caller can see the
        # summary is partial without parsing the free-form error strings.
        'degraded_adapters': sorted(result.errors),
        'notes': result.notes,
    }
