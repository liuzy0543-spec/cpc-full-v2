from __future__ import annotations

import importlib.metadata
import importlib.util
import json
import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any

from cellpaint_pipeline.config import DataAccessConfig


if TYPE_CHECKING:
    from cellpaint_pipeline.config import ProjectConfig


@dataclass(frozen=True)
class PackageAvailability:
    name: str
    import_name: str
    required: bool
    available: bool
    version: str | None
    purpose: str


@dataclass(frozen=True)
class ExecutableAvailability:
    name: str
    required: bool
    available: bool
    path: str | None
    purpose: str


@dataclass(frozen=True)
class DataAccessStatus:
    config: DataAccessConfig
    package_statuses: list[PackageAvailability]
    executable_statuses: list[ExecutableAvailability]
    ok: bool


@dataclass(frozen=True)
class GalleryListResult:
    bucket: str
    prefix: str
    delimiter: str
    max_keys: int
    common_prefixes: list[str]
    object_keys: list[str]
    is_truncated: bool


@dataclass(frozen=True)
class GalleryCatalogResult:
    bucket: str
    level: str
    parent_prefix: str
    max_keys: int
    entries: list[str]
    raw_prefixes: list[str]
    is_truncated: bool


@dataclass(frozen=True)
class GalleryCacheResult:
    output_path: Path
    bucket: str
    prefix: str
    delimiter: str
    max_keys: int
    common_prefix_count: int
    object_count: int


@dataclass(frozen=True)
class GalleryDownloadResult:
    output_dir: Path
    manifest_path: Path
    bucket: str
    prefix: str
    matched_object_count: int
    downloaded_count: int
    skipped_existing_count: int
    dry_run: bool
    overwrite: bool
    max_files: int | None
    include_substrings: tuple[str, ...]
    exclude_substrings: tuple[str, ...]
    object_keys: list[str]


def build_data_access_status(config: ProjectConfig) -> DataAccessStatus:
    package_statuses = [
        _module_status('boto3', 'boto3', True, 'Anonymous S3 listing and download for Cell Painting Gallery.'),
        _module_status('botocore', 'botocore', True, 'Unsigned S3 client configuration and low-level request support.'),
        _module_status('cpgdata', 'cpgdata', False, 'Pre-generated Cell Painting Gallery index sync and filtered downloads.'),
        _module_status('quilt3', 'quilt3', False, 'Optional Quilt-based Cell Painting Gallery browsing.'),
        _module_status('pycytominer', 'pycytominer', False, 'Optional downstream table and cell-location integration.'),
    ]
    executable_statuses = [
        _executable_status('aws', False, 'Optional AWS CLI browsing and dry-run sync support.'),
    ]
    ok = all(item.available for item in package_statuses if item.required)
    return DataAccessStatus(
        config=config.data_access,
        package_statuses=package_statuses,
        executable_statuses=executable_statuses,
        ok=ok,
    )


def data_access_status_to_dict(status: DataAccessStatus) -> dict[str, Any]:
    return {
        'implementation': 'cellpaint_pipeline.data_access',
        'ok': status.ok,
        'config': status.config.as_dict(),
        'packages': [
            {
                'name': item.name,
                'import_name': item.import_name,
                'required': item.required,
                'available': item.available,
                'version': item.version,
                'purpose': item.purpose,
            }
            for item in status.package_statuses
        ],
        'executables': [
            {
                'name': item.name,
                'required': item.required,
                'available': item.available,
                'path': item.path,
                'purpose': item.purpose,
            }
            for item in status.executable_statuses
        ],
        'required_package_count': sum(1 for item in status.package_statuses if item.required),
        'available_required_package_count': sum(1 for item in status.package_statuses if item.required and item.available),
    }


def list_gallery_prefixes(
    config: ProjectConfig,
    *,
    prefix: str = '',
    delimiter: str = '/',
    max_keys: int = 1000,
    bucket: str | None = None,
) -> GalleryListResult:
    bucket_name = bucket or config.data_access.gallery_bucket
    client = _build_s3_client(config)

    request_kwargs: dict[str, Any] = {
        'Bucket': bucket_name,
        'MaxKeys': max_keys,
    }
    if prefix:
        request_kwargs['Prefix'] = prefix
    if delimiter:
        request_kwargs['Delimiter'] = delimiter

    response = client.list_objects_v2(**request_kwargs)
    return GalleryListResult(
        bucket=bucket_name,
        prefix=prefix,
        delimiter=delimiter,
        max_keys=max_keys,
        common_prefixes=[item['Prefix'] for item in response.get('CommonPrefixes', [])],
        object_keys=[item['Key'] for item in response.get('Contents', [])],
        is_truncated=bool(response.get('IsTruncated', False)),
    )


def gallery_list_result_to_dict(result: GalleryListResult) -> dict[str, Any]:
    return {
        'implementation': 'cellpaint_pipeline.data_access.gallery',
        'bucket': result.bucket,
        'prefix': result.prefix,
        'delimiter': result.delimiter,
        'max_keys': result.max_keys,
        'common_prefixes': result.common_prefixes,
        'common_prefix_count': len(result.common_prefixes),
        'object_keys': result.object_keys,
        'object_count': len(result.object_keys),
        'is_truncated': result.is_truncated,
    }


def list_gallery_datasets(
    config: ProjectConfig,
    *,
    max_keys: int = 1000,
    bucket: str | None = None,
) -> GalleryCatalogResult:
    result = list_gallery_prefixes(
        config,
        prefix='',
        delimiter='/',
        max_keys=max_keys,
        bucket=bucket,
    )
    return GalleryCatalogResult(
        bucket=result.bucket,
        level='dataset',
        parent_prefix='',
        max_keys=max_keys,
        entries=[_prefix_basename(prefix) for prefix in result.common_prefixes],
        raw_prefixes=result.common_prefixes,
        is_truncated=result.is_truncated,
    )


def list_gallery_sources(
    config: ProjectConfig,
    *,
    dataset_id: str | None = None,
    max_keys: int = 1000,
    bucket: str | None = None,
) -> GalleryCatalogResult:
    resolved_dataset_id = dataset_id or config.data_access.default_dataset_id
    if not resolved_dataset_id:
        raise ValueError('dataset_id is required when data_access.default_dataset_id is not configured.')

    parent_prefix = build_gallery_dataset_prefix(resolved_dataset_id)
    result = list_gallery_prefixes(
        config,
        prefix=parent_prefix,
        delimiter='/',
        max_keys=max_keys,
        bucket=bucket,
    )
    return GalleryCatalogResult(
        bucket=result.bucket,
        level='source',
        parent_prefix=parent_prefix,
        max_keys=max_keys,
        entries=[_prefix_basename(prefix) for prefix in result.common_prefixes],
        raw_prefixes=result.common_prefixes,
        is_truncated=result.is_truncated,
    )


def build_gallery_dataset_prefix(dataset_id: str) -> str:
    dataset = dataset_id.strip().strip('/')
    if not dataset:
        raise ValueError('dataset_id must not be empty.')
    return f'{dataset}/'


def build_gallery_source_prefix(dataset_id: str, source_id: str) -> str:
    source = source_id.strip().strip('/')
    if not source:
        raise ValueError('source_id must not be empty.')
    return f'{build_gallery_dataset_prefix(dataset_id)}{source}/'


def gallery_catalog_result_to_dict(result: GalleryCatalogResult) -> dict[str, Any]:
    return {
        'implementation': 'cellpaint_pipeline.data_access.gallery',
        'bucket': result.bucket,
        'level': result.level,
        'parent_prefix': result.parent_prefix,
        'max_keys': result.max_keys,
        'entries': result.entries,
        'entry_count': len(result.entries),
        'raw_prefixes': result.raw_prefixes,
        'is_truncated': result.is_truncated,
    }


def cache_gallery_listing(
    config: ProjectConfig,
    *,
    prefix: str = '',
    delimiter: str = '/',
    max_keys: int = 1000,
    bucket: str | None = None,
    output_path: Path | None = None,
) -> GalleryCacheResult:
    result = list_gallery_prefixes(
        config,
        prefix=prefix,
        delimiter=delimiter,
        max_keys=max_keys,
        bucket=bucket,
    )
    resolved_output_path = (
        output_path.expanduser().resolve()
        if output_path is not None
        else config.data_access.index_cache_root / _default_listing_filename(result.bucket, prefix, delimiter)
    )
    resolved_output_path.parent.mkdir(parents=True, exist_ok=True)
    payload = gallery_list_result_to_dict(result)
    payload['cache_path'] = str(resolved_output_path)
    with resolved_output_path.open('w', encoding='utf-8') as handle:
        json.dump(payload, handle, indent=2, ensure_ascii=False)
        handle.write('\n')
    return GalleryCacheResult(
        output_path=resolved_output_path,
        bucket=result.bucket,
        prefix=result.prefix,
        delimiter=result.delimiter,
        max_keys=result.max_keys,
        common_prefix_count=len(result.common_prefixes),
        object_count=len(result.object_keys),
    )


def gallery_cache_result_to_dict(result: GalleryCacheResult) -> dict[str, Any]:
    return {
        'implementation': 'cellpaint_pipeline.data_access.gallery',
        'output_path': str(result.output_path),
        'bucket': result.bucket,
        'prefix': result.prefix,
        'delimiter': result.delimiter,
        'max_keys': result.max_keys,
        'common_prefix_count': result.common_prefix_count,
        'object_count': result.object_count,
    }


def download_gallery_prefix(
    config: ProjectConfig,
    *,
    prefix: str,
    bucket: str | None = None,
    output_dir: Path | None = None,
    manifest_path: Path | None = None,
    include_substrings: list[str] | tuple[str, ...] | None = None,
    exclude_substrings: list[str] | tuple[str, ...] | None = None,
    max_files: int | None = None,
    overwrite: bool = False,
    dry_run: bool = False,
) -> GalleryDownloadResult:
    normalized_prefix = prefix.strip()
    if not normalized_prefix:
        raise ValueError('prefix must not be empty for gallery downloads.')

    bucket_name = bucket or config.data_access.gallery_bucket
    include_tokens = tuple(include_substrings or ())
    exclude_tokens = tuple(exclude_substrings or ())
    resolved_output_dir = (
        output_dir.expanduser().resolve()
        if output_dir is not None
        else config.data_access.data_cache_root / _default_download_dirname(normalized_prefix)
    )
    resolved_manifest_path = (
        manifest_path.expanduser().resolve()
        if manifest_path is not None
        else resolved_output_dir / 'download_manifest.json'
    )
    resolved_output_dir.mkdir(parents=True, exist_ok=True)
    resolved_manifest_path.parent.mkdir(parents=True, exist_ok=True)

    objects = _list_gallery_objects(
        config,
        prefix=normalized_prefix,
        bucket=bucket_name,
        include_substrings=include_tokens,
        exclude_substrings=exclude_tokens,
        max_files=max_files,
    )
    object_keys = [item['Key'] for item in objects]

    downloaded_count = 0
    skipped_existing_count = 0
    if not dry_run and object_keys:
        client = _build_s3_client(config)
        for key in object_keys:
            relative_path = _relative_object_key(normalized_prefix, key)
            destination_path = resolved_output_dir / relative_path
            destination_path.parent.mkdir(parents=True, exist_ok=True)
            if destination_path.exists() and not overwrite:
                skipped_existing_count += 1
                continue
            _download_gallery_object(client, bucket_name, key, destination_path)
            downloaded_count += 1

    result = GalleryDownloadResult(
        output_dir=resolved_output_dir,
        manifest_path=resolved_manifest_path,
        bucket=bucket_name,
        prefix=normalized_prefix,
        matched_object_count=len(object_keys),
        downloaded_count=downloaded_count,
        skipped_existing_count=skipped_existing_count,
        dry_run=dry_run,
        overwrite=overwrite,
        max_files=max_files,
        include_substrings=include_tokens,
        exclude_substrings=exclude_tokens,
        object_keys=object_keys,
    )
    with resolved_manifest_path.open('w', encoding='utf-8') as handle:
        json.dump(gallery_download_result_to_dict(result), handle, indent=2, ensure_ascii=False)
        handle.write('\n')
    return result


def download_gallery_source(
    config: ProjectConfig,
    *,
    dataset_id: str | None = None,
    source_id: str | None = None,
    subprefix: str = '',
    bucket: str | None = None,
    output_dir: Path | None = None,
    manifest_path: Path | None = None,
    include_substrings: list[str] | tuple[str, ...] | None = None,
    exclude_substrings: list[str] | tuple[str, ...] | None = None,
    max_files: int | None = None,
    overwrite: bool = False,
    dry_run: bool = False,
) -> GalleryDownloadResult:
    resolved_dataset_id = dataset_id or config.data_access.default_dataset_id
    if not resolved_dataset_id:
        raise ValueError('dataset_id is required when data_access.default_dataset_id is not configured.')
    resolved_source_id = source_id or config.data_access.default_source_id
    if not resolved_source_id:
        raise ValueError('source_id is required when data_access.default_source_id is not configured.')

    prefix = build_gallery_source_prefix(resolved_dataset_id, resolved_source_id)
    cleaned_subprefix = subprefix.strip().strip('/')
    if cleaned_subprefix:
        prefix = f'{prefix}{cleaned_subprefix}/'

    return download_gallery_prefix(
        config,
        prefix=prefix,
        bucket=bucket,
        output_dir=output_dir,
        manifest_path=manifest_path,
        include_substrings=include_substrings,
        exclude_substrings=exclude_substrings,
        max_files=max_files,
        overwrite=overwrite,
        dry_run=dry_run,
    )


def gallery_download_result_to_dict(result: GalleryDownloadResult) -> dict[str, Any]:
    return {
        'implementation': 'cellpaint_pipeline.data_access.gallery',
        'output_dir': str(result.output_dir),
        'manifest_path': str(result.manifest_path),
        'bucket': result.bucket,
        'prefix': result.prefix,
        'matched_object_count': result.matched_object_count,
        'downloaded_count': result.downloaded_count,
        'skipped_existing_count': result.skipped_existing_count,
        'dry_run': result.dry_run,
        'overwrite': result.overwrite,
        'max_files': result.max_files,
        'include_substrings': list(result.include_substrings),
        'exclude_substrings': list(result.exclude_substrings),
        'object_keys': result.object_keys,
    }


def _module_status(name: str, import_name: str, required: bool, purpose: str) -> PackageAvailability:
    available = importlib.util.find_spec(import_name) is not None
    version = None
    if available:
        try:
            version = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            version = None
    return PackageAvailability(
        name=name,
        import_name=import_name,
        required=required,
        available=available,
        version=version,
        purpose=purpose,
    )


def _executable_status(name: str, required: bool, purpose: str) -> ExecutableAvailability:
    path = shutil.which(name)
    return ExecutableAvailability(
        name=name,
        required=required,
        available=path is not None,
        path=path,
        purpose=purpose,
    )


def _build_s3_client(config: ProjectConfig):
    boto3, unsigned, botocore_config = _require_boto3_stack()
    client_kwargs: dict[str, Any] = {
        'service_name': 's3',
        'region_name': config.data_access.gallery_region,
    }
    if config.data_access.gallery_endpoint_url:
        client_kwargs['endpoint_url'] = config.data_access.gallery_endpoint_url
    if config.data_access.use_unsigned:
        client_kwargs['config'] = botocore_config(signature_version=unsigned)
    return boto3.session.Session().client(**client_kwargs)


def _require_boto3_stack() -> tuple[Any, Any, Any]:
    try:
        import boto3
        from botocore import UNSIGNED
        from botocore.client import Config as BotocoreConfig
    except ImportError as exc:
        raise RuntimeError(
            'boto3/botocore are required for S3-backed data access. Install the optional data-access extra first.'
        ) from exc
    return boto3, UNSIGNED, BotocoreConfig


def _list_gallery_objects(
    config: ProjectConfig,
    *,
    prefix: str,
    bucket: str,
    include_substrings: tuple[str, ...],
    exclude_substrings: tuple[str, ...],
    max_files: int | None,
) -> list[dict[str, Any]]:
    client = _build_s3_client(config)
    paginator = client.get_paginator('list_objects_v2')
    matched: list[dict[str, Any]] = []
    for page in paginator.paginate(Bucket=bucket, Prefix=prefix):
        for item in page.get('Contents', []):
            key = str(item.get('Key', ''))
            if not key or key.endswith('/') or key == prefix:
                continue
            if include_substrings and not any(token in key for token in include_substrings):
                continue
            if exclude_substrings and any(token in key for token in exclude_substrings):
                continue
            matched.append({'Key': key, 'Size': int(item.get('Size', 0))})
            if max_files is not None and len(matched) >= max_files:
                return matched
    return matched


def _download_gallery_object(client, bucket: str, key: str, output_path: Path) -> None:
    client.download_file(bucket, key, str(output_path))


def _prefix_basename(prefix: str) -> str:
    return prefix.rstrip('/').split('/')[-1]


def _default_listing_filename(bucket: str, prefix: str, delimiter: str) -> str:
    prefix_token = prefix.strip('/') or '__root__'
    safe_prefix = prefix_token.replace('/', '__')
    delimiter_token = delimiter if delimiter else '__flat__'
    safe_delimiter = delimiter_token.replace('/', '_slash_')
    return f'{bucket}__{safe_prefix}__{safe_delimiter}.json'


def _default_download_dirname(prefix: str) -> str:
    return prefix.strip('/').replace('/', '__') or '__root__'


def _relative_object_key(prefix: str, key: str) -> Path:
    if prefix and key.startswith(prefix):
        relative_key = key[len(prefix):]
    else:
        relative_key = key
    if not relative_key:
        raise ValueError(f'Could not derive relative key for prefix={prefix!r} and key={key!r}')
    return Path(relative_key)
