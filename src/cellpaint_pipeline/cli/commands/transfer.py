"""CLI commands for the transfer domain."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from cellpaint_pipeline.config import ProjectConfig
from cellpaint_pipeline.cli.helpers import _impl



def _cmd_plan_data_access(args: argparse.Namespace) -> int:
            config = ProjectConfig.from_json(args.config)
            request = _impl("build_data_request")(
                mode=args.mode,
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
                output_dir=Path(args.output_dir).expanduser().resolve() if args.output_dir else None,
                manifest_path=Path(args.manifest_path).expanduser().resolve() if args.manifest_path else None,
            )
            plan = _impl("build_download_plan")(
                config,
                request,
                summary_max_keys=args.summary_max_keys,
                validate_with_summary=not args.skip_summary_check,
            )
            payload = _impl("data_download_plan_to_dict")(plan)
            if args.output_path:
                output_path = _impl("write_download_plan")(plan, Path(args.output_path).expanduser().resolve())
                payload['plan_path'] = str(output_path)
            print(json.dumps(payload, indent=2, ensure_ascii=False))
            return 0 if plan.ok else 1



def _cmd_execute_download_plan(args: argparse.Namespace) -> int:
            config = ProjectConfig.from_json(args.config)
            plan = _impl("load_download_plan")(Path(args.plan_path).expanduser().resolve())
            result = _impl("execute_download_plan")(config, plan)
            print(json.dumps(_impl("data_download_execution_result_to_dict")(result), indent=2, ensure_ascii=False))
            return 0 if result.ok else 1



def _cmd_list_gallery_prefixes(args: argparse.Namespace) -> int:
            config = ProjectConfig.from_json(args.config)
            result = _impl("list_gallery_prefixes")(
                config,
                prefix=args.prefix,
                delimiter=args.delimiter,
                max_keys=args.max_keys,
                bucket=args.bucket,
            )
            print(json.dumps(_impl("gallery_list_result_to_dict")(result), indent=2, ensure_ascii=False))
            return 0



def _cmd_list_gallery_datasets(args: argparse.Namespace) -> int:
            config = ProjectConfig.from_json(args.config)
            result = _impl("list_gallery_datasets")(
                config,
                max_keys=args.max_keys,
                bucket=args.bucket,
            )
            print(json.dumps(_impl("gallery_catalog_result_to_dict")(result), indent=2, ensure_ascii=False))
            return 0



def _cmd_list_gallery_sources(args: argparse.Namespace) -> int:
            config = ProjectConfig.from_json(args.config)
            result = _impl("list_gallery_sources")(
                config,
                dataset_id=args.dataset_id,
                max_keys=args.max_keys,
                bucket=args.bucket,
            )
            print(json.dumps(_impl("gallery_catalog_result_to_dict")(result), indent=2, ensure_ascii=False))
            return 0



def _cmd_cache_gallery_prefixes(args: argparse.Namespace) -> int:
            config = ProjectConfig.from_json(args.config)
            result = _impl("cache_gallery_listing")(
                config,
                prefix=args.prefix,
                delimiter=args.delimiter,
                max_keys=args.max_keys,
                bucket=args.bucket,
                output_path=Path(args.output_path).expanduser().resolve() if args.output_path else None,
            )
            print(json.dumps(_impl("gallery_cache_result_to_dict")(result), indent=2, ensure_ascii=False))
            return 0



def _cmd_download_gallery_prefix(args: argparse.Namespace) -> int:
            config = ProjectConfig.from_json(args.config)
            result = _impl("download_gallery_prefix")(
                config,
                prefix=args.prefix,
                bucket=args.bucket,
                output_dir=Path(args.output_dir).expanduser().resolve() if args.output_dir else None,
                manifest_path=Path(args.manifest_path).expanduser().resolve() if args.manifest_path else None,
                include_substrings=args.include_substring or [],
                exclude_substrings=args.exclude_substring or [],
                max_files=args.max_files,
                overwrite=args.overwrite,
                dry_run=args.dry_run,
            )
            print(json.dumps(_impl("gallery_download_result_to_dict")(result), indent=2, ensure_ascii=False))
            return 0



def _cmd_download_gallery_source(args: argparse.Namespace) -> int:
            config = ProjectConfig.from_json(args.config)
            result = _impl("download_gallery_source")(
                config,
                dataset_id=args.dataset_id,
                source_id=args.source_id,
                subprefix=args.subprefix,
                bucket=args.bucket,
                output_dir=Path(args.output_dir).expanduser().resolve() if args.output_dir else None,
                manifest_path=Path(args.manifest_path).expanduser().resolve() if args.manifest_path else None,
                include_substrings=args.include_substring or [],
                exclude_substrings=args.exclude_substring or [],
                max_files=args.max_files,
                overwrite=args.overwrite,
                dry_run=args.dry_run,
            )
            print(json.dumps(_impl("gallery_download_result_to_dict")(result), indent=2, ensure_ascii=False))
            return 0



def _cmd_list_quilt_packages(args: argparse.Namespace) -> int:
            config = ProjectConfig.from_json(args.config)
            result = _impl("list_quilt_packages")(config, registry=args.registry, limit=args.limit)
            print(json.dumps(_impl("quilt_package_list_result_to_dict")(result), indent=2, ensure_ascii=False))
            return 0



def _cmd_browse_quilt_package(args: argparse.Namespace) -> int:
            config = ProjectConfig.from_json(args.config)
            result = _impl("browse_quilt_package")(
                config,
                package_name=args.package_name,
                registry=args.registry,
                top_hash=args.top_hash,
                max_keys=args.max_keys,
            )
            print(json.dumps(_impl("quilt_package_browse_result_to_dict")(result), indent=2, ensure_ascii=False))
            return 0



def _cmd_list_cpgdata_prefixes(args: argparse.Namespace) -> int:
            config = ProjectConfig.from_json(args.config)
            result = _impl("list_cpgdata_prefixes")(
                config,
                bucket=args.bucket,
                prefix=args.prefix,
                recursive=args.recursive,
                limit=args.limit,
            )
            print(json.dumps(_impl("cpgdata_prefix_list_result_to_dict")(result), indent=2, ensure_ascii=False))
            return 0



def _cmd_sync_cpgdata_index(args: argparse.Namespace) -> int:
            config = ProjectConfig.from_json(args.config)
            result = _impl("sync_cpgdata_index")(
                config,
                output_dir=Path(args.output_dir).expanduser().resolve() if args.output_dir else None,
                bucket=args.bucket,
                prefix=args.prefix,
                include=args.include,
                exclude=args.exclude,
                no_progress=not args.show_progress,
            )
            print(json.dumps(_impl("cpgdata_sync_result_to_dict")(result), indent=2, ensure_ascii=False))
            return 0



def _cmd_sync_cpgdata_inventory(args: argparse.Namespace) -> int:
            config = ProjectConfig.from_json(args.config)
            result = _impl("sync_cpgdata_inventory")(
                config,
                output_dir=Path(args.output_dir).expanduser().resolve() if args.output_dir else None,
                bucket=args.bucket,
                prefix=args.prefix,
                revision=args.revision,
            )
            print(json.dumps(_impl("cpgdata_sync_result_to_dict")(result), indent=2, ensure_ascii=False))
            return 0
def register(subparsers: argparse._SubParsersAction) -> None:
    """Attach this domain's sub-commands, in their original order."""
    plan_data_access_parser = subparsers.add_parser('plan-data-access', help='Build a reusable download plan for gallery-backed data requests.')
    plan_data_access_parser.add_argument('--config', required=True, help='Path to project config JSON.')
    plan_data_access_parser.add_argument('--mode', default='gallery-source', choices=['gallery-source', 'gallery-prefix'], help='Plan a source-based or raw-prefix gallery download.')
    plan_data_access_parser.add_argument('--dataset-id', default=None, help='Optional dataset id override for gallery-source mode.')
    plan_data_access_parser.add_argument('--source-id', default=None, help='Optional source id override for gallery-source mode.')
    plan_data_access_parser.add_argument('--prefix', default=None, help='Required for gallery-prefix mode.')
    plan_data_access_parser.add_argument('--subprefix', default='', help='Optional nested prefix under a dataset/source root for gallery-source mode.')
    plan_data_access_parser.add_argument('--bucket', default=None, help='Optional bucket override. Defaults to the configured gallery bucket.')
    plan_data_access_parser.add_argument('--include-substring', action='append', default=None, help='Only keep object keys containing this substring. May be repeated.')
    plan_data_access_parser.add_argument('--exclude-substring', action='append', default=None, help='Skip object keys containing this substring. May be repeated.')
    plan_data_access_parser.add_argument('--max-files', type=int, default=None, help='Optional maximum number of matched objects to process.')
    plan_data_access_parser.add_argument('--overwrite', action='store_true', help='Overwrite existing local files instead of skipping them during execution.')
    plan_data_access_parser.add_argument('--dry-run', action='store_true', help='Plan for dry-run execution.')
    plan_data_access_parser.add_argument('--output-dir', default=None, help='Optional local output directory override for the planned download.')
    plan_data_access_parser.add_argument('--manifest-path', default=None, help='Optional manifest path override for the planned download.')
    plan_data_access_parser.add_argument('--summary-max-keys', type=int, default=1000, help='Maximum number of gallery prefixes to inspect during summary-based validation.')
    plan_data_access_parser.add_argument('--skip-summary-check', action='store_true', help='Skip gallery summary validation while building the plan.')
    plan_data_access_parser.add_argument('--output-path', default=None, help='Optional JSON output path for persisting the planned download.')

    execute_download_plan_parser = subparsers.add_parser('execute-download-plan', help='Execute a previously generated download plan JSON file.')
    execute_download_plan_parser.add_argument('--config', required=True, help='Path to project config JSON.')
    execute_download_plan_parser.add_argument('--plan-path', required=True, help='Path to a JSON file produced by plan-data-access.')

    list_gallery_parser = subparsers.add_parser('list-gallery-prefixes', help='List prefixes and objects from the configured Cell Painting Gallery bucket.')
    list_gallery_parser.add_argument('--config', required=True, help='Path to project config JSON.')
    list_gallery_parser.add_argument('--prefix', default='', help='Optional S3 prefix to inspect.')
    list_gallery_parser.add_argument('--delimiter', default='/', help='Delimiter passed to list_objects_v2. Use an empty string for flat object listing.')
    list_gallery_parser.add_argument('--max-keys', type=int, default=1000, help='Maximum number of keys returned by the S3 listing request.')
    list_gallery_parser.add_argument('--bucket', default=None, help='Optional bucket override. Defaults to the configured gallery bucket.')

    list_gallery_datasets_parser = subparsers.add_parser('list-gallery-datasets', help='List dataset-level prefixes from the configured Cell Painting Gallery bucket.')
    list_gallery_datasets_parser.add_argument('--config', required=True, help='Path to project config JSON.')
    list_gallery_datasets_parser.add_argument('--max-keys', type=int, default=1000, help='Maximum number of dataset prefixes returned by the S3 listing request.')
    list_gallery_datasets_parser.add_argument('--bucket', default=None, help='Optional bucket override. Defaults to the configured gallery bucket.')

    list_gallery_sources_parser = subparsers.add_parser('list-gallery-sources', help='List source-level prefixes for a dataset in the configured Cell Painting Gallery bucket.')
    list_gallery_sources_parser.add_argument('--config', required=True, help='Path to project config JSON.')
    list_gallery_sources_parser.add_argument('--dataset-id', default=None, help='Optional dataset id override. Defaults to data_access.default_dataset_id.')
    list_gallery_sources_parser.add_argument('--max-keys', type=int, default=1000, help='Maximum number of source prefixes returned by the S3 listing request.')
    list_gallery_sources_parser.add_argument('--bucket', default=None, help='Optional bucket override. Defaults to the configured gallery bucket.')

    cache_gallery_parser = subparsers.add_parser('cache-gallery-prefixes', help='Write a JSON cache snapshot for a Cell Painting Gallery prefix listing.')
    cache_gallery_parser.add_argument('--config', required=True, help='Path to project config JSON.')
    cache_gallery_parser.add_argument('--prefix', default='', help='Optional S3 prefix to inspect before caching.')
    cache_gallery_parser.add_argument('--delimiter', default='/', help='Delimiter passed to list_objects_v2. Use an empty string for flat object listing.')
    cache_gallery_parser.add_argument('--max-keys', type=int, default=1000, help='Maximum number of keys returned by the S3 listing request.')
    cache_gallery_parser.add_argument('--bucket', default=None, help='Optional bucket override. Defaults to the configured gallery bucket.')
    cache_gallery_parser.add_argument('--output-path', default=None, help='Optional JSON output path. Defaults to data_access.index_cache_root.')

    download_gallery_prefix_parser = subparsers.add_parser('download-gallery-prefix', help='Download objects under a Cell Painting Gallery prefix into the local data cache.')
    download_gallery_prefix_parser.add_argument('--config', required=True, help='Path to project config JSON.')
    download_gallery_prefix_parser.add_argument('--prefix', required=True, help='S3 prefix to download recursively.')
    download_gallery_prefix_parser.add_argument('--bucket', default=None, help='Optional bucket override. Defaults to the configured gallery bucket.')
    download_gallery_prefix_parser.add_argument('--output-dir', default=None, help='Optional local output directory. Defaults to data_access.data_cache_root.')
    download_gallery_prefix_parser.add_argument('--manifest-path', default=None, help='Optional JSON manifest path. Defaults to download_manifest.json under the output directory.')
    download_gallery_prefix_parser.add_argument('--include-substring', action='append', default=None, help='Only keep object keys containing this substring. May be repeated.')
    download_gallery_prefix_parser.add_argument('--exclude-substring', action='append', default=None, help='Skip object keys containing this substring. May be repeated.')
    download_gallery_prefix_parser.add_argument('--max-files', type=int, default=None, help='Optional maximum number of matched objects to process.')
    download_gallery_prefix_parser.add_argument('--overwrite', action='store_true', help='Overwrite existing local files instead of skipping them.')
    download_gallery_prefix_parser.add_argument('--dry-run', action='store_true', help='Resolve matches and write the manifest without downloading files.')

    download_gallery_source_parser = subparsers.add_parser('download-gallery-source', help='Download objects for a dataset/source pair from the Cell Painting Gallery into the local data cache.')
    download_gallery_source_parser.add_argument('--config', required=True, help='Path to project config JSON.')
    download_gallery_source_parser.add_argument('--dataset-id', default=None, help='Optional dataset id override. Defaults to data_access.default_dataset_id.')
    download_gallery_source_parser.add_argument('--source-id', default=None, help='Optional source id override. Defaults to data_access.default_source_id.')
    download_gallery_source_parser.add_argument('--subprefix', default='', help='Optional nested prefix under the dataset/source root, for example images or workspace.')
    download_gallery_source_parser.add_argument('--bucket', default=None, help='Optional bucket override. Defaults to the configured gallery bucket.')
    download_gallery_source_parser.add_argument('--output-dir', default=None, help='Optional local output directory. Defaults to data_access.data_cache_root.')
    download_gallery_source_parser.add_argument('--manifest-path', default=None, help='Optional JSON manifest path. Defaults to download_manifest.json under the output directory.')
    download_gallery_source_parser.add_argument('--include-substring', action='append', default=None, help='Only keep object keys containing this substring. May be repeated.')
    download_gallery_source_parser.add_argument('--exclude-substring', action='append', default=None, help='Skip object keys containing this substring. May be repeated.')
    download_gallery_source_parser.add_argument('--max-files', type=int, default=None, help='Optional maximum number of matched objects to process.')
    download_gallery_source_parser.add_argument('--overwrite', action='store_true', help='Overwrite existing local files instead of skipping them.')
    download_gallery_source_parser.add_argument('--dry-run', action='store_true', help='Resolve matches and write the manifest without downloading files.')

    list_quilt_packages_parser = subparsers.add_parser('list-quilt-packages', help='List package names from the configured Quilt registry.')
    list_quilt_packages_parser.add_argument('--config', required=True, help='Path to project config JSON.')
    list_quilt_packages_parser.add_argument('--registry', default=None, help='Optional Quilt registry override. Defaults to data_access.quilt_registry.')
    list_quilt_packages_parser.add_argument('--limit', type=int, default=None, help='Optional maximum number of package names to return.')

    browse_quilt_package_parser = subparsers.add_parser('browse-quilt-package', help='Browse one Quilt package and return its top-level keys.')
    browse_quilt_package_parser.add_argument('--config', required=True, help='Path to project config JSON.')
    browse_quilt_package_parser.add_argument('--package-name', required=True, help='Quilt package name to browse.')
    browse_quilt_package_parser.add_argument('--registry', default=None, help='Optional Quilt registry override. Defaults to data_access.quilt_registry.')
    browse_quilt_package_parser.add_argument('--top-hash', default=None, help='Optional Quilt top hash override.')
    browse_quilt_package_parser.add_argument('--max-keys', type=int, default=200, help='Optional maximum number of top-level keys to return.')

    list_cpgdata_prefixes_parser = subparsers.add_parser('list-cpgdata-prefixes', help='List inventory/index-style prefixes using cpgdata utilities and AWS CLI.')
    list_cpgdata_prefixes_parser.add_argument('--config', required=True, help='Path to project config JSON.')
    list_cpgdata_prefixes_parser.add_argument('--bucket', default=None, help='Optional bucket override. Defaults to data_access.cpgdata_inventory_bucket.')
    list_cpgdata_prefixes_parser.add_argument('--prefix', default=None, help='Optional prefix override. Defaults to data_access.cpgdata_inventory_prefix.')
    list_cpgdata_prefixes_parser.add_argument('--recursive', action='store_true', help='Recursively list matching entries.')
    list_cpgdata_prefixes_parser.add_argument('--limit', type=int, default=None, help='Optional maximum number of entries to return.')

    sync_cpgdata_index_parser = subparsers.add_parser('sync-cpgdata-index', help='Sync cpgdata index files from the inventory bucket into the local cache.')
    sync_cpgdata_index_parser.add_argument('--config', required=True, help='Path to project config JSON.')
    sync_cpgdata_index_parser.add_argument('--output-dir', default=None, help='Optional local output directory. Defaults to data_access.index_cache_root/cpgdata_index.')
    sync_cpgdata_index_parser.add_argument('--bucket', default=None, help='Optional bucket override. Defaults to data_access.cpgdata_inventory_bucket.')
    sync_cpgdata_index_parser.add_argument('--prefix', default=None, help='Optional prefix override. Defaults to data_access.cpgdata_index_prefix.')
    sync_cpgdata_index_parser.add_argument('--include', default=None, help='Optional include glob passed to cpgdata sync_s3_prefix.')
    sync_cpgdata_index_parser.add_argument('--exclude', default=None, help='Optional exclude glob passed to cpgdata sync_s3_prefix.')
    sync_cpgdata_index_parser.add_argument('--show-progress', action='store_true', help='Show AWS CLI progress instead of using no-progress mode.')

    sync_cpgdata_inventory_parser = subparsers.add_parser('sync-cpgdata-inventory', help='Sync cpgdata inventory revision files into the local cache.')
    sync_cpgdata_inventory_parser.add_argument('--config', required=True, help='Path to project config JSON.')
    sync_cpgdata_inventory_parser.add_argument('--output-dir', default=None, help='Optional local output directory. Defaults to data_access.index_cache_root/cpgdata_inventory.')
    sync_cpgdata_inventory_parser.add_argument('--bucket', default=None, help='Optional bucket override. Defaults to data_access.cpgdata_inventory_bucket.')
    sync_cpgdata_inventory_parser.add_argument('--prefix', default=None, help='Optional prefix override. Defaults to data_access.cpgdata_inventory_prefix.')
    sync_cpgdata_inventory_parser.add_argument('--revision', type=int, default=0, help='Inventory revision offset. 0 means latest revision.')


HANDLERS = {
    'plan-data-access': _cmd_plan_data_access,
    'execute-download-plan': _cmd_execute_download_plan,
    'list-gallery-prefixes': _cmd_list_gallery_prefixes,
    'list-gallery-datasets': _cmd_list_gallery_datasets,
    'list-gallery-sources': _cmd_list_gallery_sources,
    'cache-gallery-prefixes': _cmd_cache_gallery_prefixes,
    'download-gallery-prefix': _cmd_download_gallery_prefix,
    'download-gallery-source': _cmd_download_gallery_source,
    'list-quilt-packages': _cmd_list_quilt_packages,
    'browse-quilt-package': _cmd_browse_quilt_package,
    'list-cpgdata-prefixes': _cmd_list_cpgdata_prefixes,
    'sync-cpgdata-index': _cmd_sync_cpgdata_index,
    'sync-cpgdata-inventory': _cmd_sync_cpgdata_inventory,
}
