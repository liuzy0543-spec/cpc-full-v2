"""CLI commands for the config domain."""

from __future__ import annotations

import argparse
import json
from cellpaint_pipeline.config import ProjectConfig
from cellpaint_pipeline.cli.helpers import _impl



def _cmd_show_config(args: argparse.Namespace) -> int:
            config = ProjectConfig.from_json(args.config)
            print(json.dumps(config.as_dict(), indent=2, ensure_ascii=False))
            return 0



def _cmd_show_data_access(args: argparse.Namespace) -> int:
            config = ProjectConfig.from_json(args.config)
            print(json.dumps({
                'implementation': 'cellpaint_pipeline.data_access',
                'config': config.data_access.as_dict(),
            }, indent=2, ensure_ascii=False))
            return 0



def _cmd_list_cppipe_templates(args: argparse.Namespace) -> int:
            config = ProjectConfig.from_json(args.config) if args.config else None
            payload = [
                _impl("cppipe_template_definition_to_dict")(_impl("get_cppipe_template")(key), config=config)
                for key in _impl("available_cppipe_templates")(kind=args.kind)
            ]
            print(json.dumps(payload, indent=2, ensure_ascii=False))
            return 0



def _cmd_describe_cppipe_template(args: argparse.Namespace) -> int:
            config = ProjectConfig.from_json(args.config) if args.config else None
            payload = _impl("cppipe_template_definition_to_dict")(_impl("get_cppipe_template")(args.template), config=config)
            print(json.dumps(payload, indent=2, ensure_ascii=False))
            return 0



def _cmd_show_cppipe_selection(args: argparse.Namespace) -> int:
            config = ProjectConfig.from_json(args.config)
            kinds = ['profiling', 'segmentation'] if args.kind == 'all' else [args.kind]
            payload = [
                _impl("resolved_cppipe_selection_to_dict")(_impl("resolve_cppipe_selection")(config, kind))
                for kind in kinds
            ]
            print(json.dumps(payload, indent=2, ensure_ascii=False))
            return 0



def _cmd_validate_cppipe_config(args: argparse.Namespace) -> int:
            config = ProjectConfig.from_json(args.config)
            result = _impl("validate_cppipe_configuration")(config)
            print(json.dumps(_impl("cppipe_validation_result_to_dict")(result), indent=2, ensure_ascii=False))
            return 0 if result.ok else 1



def _cmd_check_data_access(args: argparse.Namespace) -> int:
            config = ProjectConfig.from_json(args.config)
            status = _impl("build_data_access_status")(config)
            print(json.dumps(_impl("data_access_status_to_dict")(status), indent=2, ensure_ascii=False))
            return 0 if status.ok or not args.strict else 1



def _cmd_summarize_data_access(args: argparse.Namespace) -> int:
            config = ProjectConfig.from_json(args.config)
            result = _impl("summarize_data_access")(
                config,
                dataset_id=args.dataset_id,
                gallery_bucket=args.gallery_bucket,
                gallery_max_keys=args.gallery_max_keys,
                registry=args.registry,
                quilt_limit=args.quilt_limit,
                cpgdata_bucket=args.cpgdata_bucket,
                cpgdata_prefix=args.cpgdata_prefix,
                cpgdata_recursive=args.cpgdata_recursive,
                cpgdata_limit=args.cpgdata_limit,
                include_gallery=not args.skip_gallery,
                include_quilt=not args.skip_quilt,
                include_cpgdata=not args.skip_cpgdata,
            )
            print(json.dumps(_impl("data_access_summary_to_dict")(result), indent=2, ensure_ascii=False))
            return 0 if result.ok else 1
def register(subparsers: argparse._SubParsersAction) -> None:
    """Attach this domain's sub-commands, in their original order."""
    show_parser = subparsers.add_parser('show-config', help='Print the resolved project config.')
    show_parser.add_argument('--config', required=True, help='Path to project config JSON.')

    data_access_show_parser = subparsers.add_parser('show-data-access', help='Print the resolved data-access configuration.')
    data_access_show_parser.add_argument('--config', required=True, help='Path to project config JSON.')

    list_cppipe_templates_parser = subparsers.add_parser('list-cppipe-templates', help='List bundled .cppipe template keys exposed by the library.')
    list_cppipe_templates_parser.add_argument('--kind', choices=['profiling', 'segmentation'], default=None, help='Optionally limit the list to profiling or segmentation templates.')
    list_cppipe_templates_parser.add_argument('--config', default=None, help='Optional project config used to resolve bundled template paths.')

    describe_cppipe_template_parser = subparsers.add_parser('describe-cppipe-template', help='Describe one bundled .cppipe template and optionally resolve its path under a project config.')
    describe_cppipe_template_parser.add_argument('--template', required=True, help='Bundled .cppipe template key.')
    describe_cppipe_template_parser.add_argument('--config', default=None, help='Optional project config used to resolve the template path.')

    show_cppipe_selection_parser = subparsers.add_parser('show-cppipe-selection', help='Show the effective profiling or segmentation .cppipe selection under a project config.')
    show_cppipe_selection_parser.add_argument('--config', required=True, help='Path to project config JSON.')
    show_cppipe_selection_parser.add_argument('--kind', choices=['profiling', 'segmentation', 'all'], default='all', help='Show one side or both effective selections.')

    validate_cppipe_config_parser = subparsers.add_parser('validate-cppipe-config', help='Validate the configured .cppipe template and custom-path selection.')
    validate_cppipe_config_parser.add_argument('--config', required=True, help='Path to project config JSON.')

    data_access_check_parser = subparsers.add_parser('check-data-access', help='Report data-access package and executable availability.')
    data_access_check_parser.add_argument('--config', required=True, help='Path to project config JSON.')
    data_access_check_parser.add_argument('--strict', action='store_true', help='Return a non-zero exit code if required data-access packages are missing.')

    summarize_data_access_parser = subparsers.add_parser('summarize-data-access', help='Build a unified data-access summary across gallery, Quilt, and cpgdata adapters.')
    summarize_data_access_parser.add_argument('--config', required=True, help='Path to project config JSON.')
    summarize_data_access_parser.add_argument('--dataset-id', default=None, help='Optional dataset id override for gallery source listing.')
    summarize_data_access_parser.add_argument('--gallery-bucket', default=None, help='Optional gallery bucket override. Defaults to the configured gallery bucket.')
    summarize_data_access_parser.add_argument('--gallery-max-keys', type=int, default=1000, help='Maximum number of prefixes returned for gallery dataset/source listing calls.')
    summarize_data_access_parser.add_argument('--registry', default=None, help='Optional Quilt registry override. Defaults to data_access.quilt_registry.')
    summarize_data_access_parser.add_argument('--quilt-limit', type=int, default=None, help='Optional maximum number of Quilt packages to return.')
    summarize_data_access_parser.add_argument('--cpgdata-bucket', default=None, help='Optional cpgdata inventory bucket override. Defaults to data_access.cpgdata_inventory_bucket.')
    summarize_data_access_parser.add_argument('--cpgdata-prefix', default=None, help='Optional cpgdata prefix override. Defaults to data_access.cpgdata_inventory_prefix.')
    summarize_data_access_parser.add_argument('--cpgdata-recursive', action='store_true', help='Recursively list cpgdata prefix entries.')
    summarize_data_access_parser.add_argument('--cpgdata-limit', type=int, default=None, help='Optional maximum number of cpgdata entries to return.')
    summarize_data_access_parser.add_argument('--skip-gallery', action='store_true', help='Skip gallery dataset/source discovery in the combined summary.')
    summarize_data_access_parser.add_argument('--skip-quilt', action='store_true', help='Skip Quilt package discovery in the combined summary.')
    summarize_data_access_parser.add_argument('--skip-cpgdata', action='store_true', help='Skip cpgdata prefix discovery in the combined summary.')


HANDLERS = {
    'show-config': _cmd_show_config,
    'show-data-access': _cmd_show_data_access,
    'list-cppipe-templates': _cmd_list_cppipe_templates,
    'describe-cppipe-template': _cmd_describe_cppipe_template,
    'show-cppipe-selection': _cmd_show_cppipe_selection,
    'validate-cppipe-config': _cmd_validate_cppipe_config,
    'check-data-access': _cmd_check_data_access,
    'summarize-data-access': _cmd_summarize_data_access,
}
