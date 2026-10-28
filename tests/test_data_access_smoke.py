from __future__ import annotations

import io
import json
import sys
import tempfile
import unittest
from contextlib import redirect_stdout
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / 'src'
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

import cellpaint_pipeline as package_api
from cellpaint_pipeline.cli import main
from cellpaint_pipeline.config import ProjectConfig
from cellpaint_pipeline.data_access import (
    CPGDataPrefixListResult,
    CPGDataSyncResult,
    DataAccessSummaryResult,
    DataDownloadExecutionResult,
    DataDownloadPlan,
    DataDownloadStep,
    DataDownloadStepResult,
    DataRequest,
    QuiltPackageBrowseResult,
    QuiltPackageListResult,
    browse_quilt_package,
    build_data_access_status,
    build_data_request,
    build_download_plan,
    cache_gallery_listing,
    cpgdata_prefix_list_result_to_dict,
    cpgdata_sync_result_to_dict,
    data_access_status_to_dict,
    data_access_summary_to_dict,
    data_download_execution_result_to_dict,
    data_download_plan_to_dict,
    download_gallery_prefix,
    download_gallery_source,
    gallery_cache_result_to_dict,
    gallery_catalog_result_to_dict,
    gallery_download_result_to_dict,
    gallery_list_result_to_dict,
    list_cpgdata_prefixes,
    load_download_plan,
    list_gallery_datasets,
    list_gallery_sources,
    list_quilt_packages,
    quilt_package_browse_result_to_dict,
    quilt_package_list_result_to_dict,
    summarize_data_access,
    sync_cpgdata_index,
    sync_cpgdata_inventory,
    execute_download_plan,
    write_download_plan,
)
from cellpaint_pipeline.data_access.gallery import (
    DataAccessStatus,
    ExecutableAvailability,
    GalleryCacheResult,
    GalleryCatalogResult,
    GalleryDownloadResult,
    GalleryListResult,
    PackageAvailability,
)

CONFIG_PATH = PROJECT_ROOT / 'configs' / 'project_config.example.json'


class DataAccessSmokeTests(unittest.TestCase):
    def test_project_config_loads_data_access_block(self) -> None:
        config = ProjectConfig.from_json(CONFIG_PATH)
        self.assertEqual(config.data_access.gallery_bucket, 'cellpainting-gallery')
        self.assertEqual(config.data_access.gallery_region, 'us-east-1')
        self.assertTrue(config.data_access.use_unsigned)
        self.assertEqual(config.data_access.default_dataset_id, 'cpg0016-jump')
        self.assertEqual(config.data_access.default_source_id, 'source_4')
        self.assertEqual(config.data_access.quilt_registry, 's3://cellpainting-gallery')
        self.assertEqual(config.data_access.cpgdata_inventory_bucket, 'cellpainting-gallery-inventory')
        self.assertEqual(config.data_access.cpgdata_index_prefix, 'cellpainting-gallery/index')
        self.assertEqual(config.data_access.cpgdata_inventory_prefix, 'cellpainting-gallery/whole_bucket/')
        self.assertIn('boto3', config.data_access.preferred_packages)

    def test_package_root_exports_data_access_helpers(self) -> None:
        self.assertTrue(callable(package_api.build_data_access_status))
        self.assertTrue(callable(package_api.list_gallery_prefixes))
        self.assertTrue(callable(package_api.list_gallery_datasets))
        self.assertTrue(callable(package_api.list_gallery_sources))
        self.assertTrue(callable(package_api.cache_gallery_listing))
        self.assertTrue(callable(package_api.download_gallery_prefix))
        self.assertTrue(callable(package_api.download_gallery_source))
        self.assertTrue(callable(package_api.list_quilt_packages))
        self.assertTrue(callable(package_api.browse_quilt_package))
        self.assertTrue(callable(package_api.list_cpgdata_prefixes))
        self.assertTrue(callable(package_api.summarize_data_access))
        self.assertTrue(callable(package_api.build_download_plan))
        self.assertTrue(callable(package_api.execute_download_plan))
        self.assertTrue(callable(package_api.sync_cpgdata_index))
        self.assertTrue(callable(package_api.sync_cpgdata_inventory))

    def test_cli_show_data_access_command(self) -> None:
        stdout = io.StringIO()
        with redirect_stdout(stdout):
            returncode = main(['show-data-access', '--config', str(CONFIG_PATH)])
        self.assertEqual(returncode, 0)
        payload = json.loads(stdout.getvalue())
        self.assertEqual(payload['implementation'], 'cellpaint_pipeline.data_access')
        self.assertEqual(payload['config']['gallery_bucket'], 'cellpainting-gallery')
        self.assertEqual(payload['config']['default_dataset_id'], 'cpg0016-jump')
        self.assertEqual(payload['config']['quilt_registry'], 's3://cellpainting-gallery')

    @patch('cellpaint_pipeline.cli.build_data_access_status')
    def test_cli_check_data_access_command(self, build_status_mock) -> None:
        config = ProjectConfig.from_json(CONFIG_PATH)
        build_status_mock.return_value = DataAccessStatus(
            config=config.data_access,
            package_statuses=[
                PackageAvailability(
                    name='boto3',
                    import_name='boto3',
                    required=True,
                    available=True,
                    version='1.0.0',
                    purpose='test',
                ),
            ],
            executable_statuses=[
                ExecutableAvailability(
                    name='aws',
                    required=False,
                    available=False,
                    path=None,
                    purpose='test',
                ),
            ],
            ok=True,
        )
        stdout = io.StringIO()
        with redirect_stdout(stdout):
            returncode = main(['check-data-access', '--config', str(CONFIG_PATH), '--strict'])
        self.assertEqual(returncode, 0)
        payload = json.loads(stdout.getvalue())
        self.assertTrue(payload['ok'])
        self.assertEqual(payload['available_required_package_count'], 1)
        build_status_mock.assert_called_once()

    @patch('cellpaint_pipeline.cli.list_gallery_prefixes')
    def test_cli_list_gallery_prefixes_command(self, list_gallery_mock) -> None:
        list_gallery_mock.return_value = GalleryListResult(
            bucket='cellpainting-gallery',
            prefix='cpg0016-jump/',
            delimiter='/',
            max_keys=25,
            common_prefixes=['cpg0016-jump/source_4/'],
            object_keys=[],
            is_truncated=False,
        )
        stdout = io.StringIO()
        with redirect_stdout(stdout):
            returncode = main(['list-gallery-prefixes', '--config', str(CONFIG_PATH), '--prefix', 'cpg0016-jump/', '--max-keys', '25'])
        self.assertEqual(returncode, 0)
        payload = json.loads(stdout.getvalue())
        self.assertEqual(payload['bucket'], 'cellpainting-gallery')
        self.assertEqual(payload['common_prefix_count'], 1)
        list_gallery_mock.assert_called_once()

    @patch('cellpaint_pipeline.cli.summarize_data_access')
    def test_cli_summarize_data_access_command(self, summarize_mock) -> None:
        config = ProjectConfig.from_json(CONFIG_PATH)
        summarize_mock.return_value = DataAccessSummaryResult(
            status=DataAccessStatus(
                config=config.data_access,
                package_statuses=[
                    PackageAvailability('boto3', 'boto3', True, True, '1.0.0', 'required'),
                ],
                executable_statuses=[],
                ok=True,
            ),
            resolved_dataset_id='cpg0016-jump',
            gallery_bucket='cellpainting-gallery',
            gallery_max_keys=25,
            dataset_listing=GalleryCatalogResult(
                bucket='cellpainting-gallery',
                level='dataset',
                parent_prefix='',
                max_keys=25,
                entries=['cpg0016-jump'],
                raw_prefixes=['cpg0016-jump/'],
                is_truncated=False,
            ),
            source_listing=GalleryCatalogResult(
                bucket='cellpainting-gallery',
                level='source',
                parent_prefix='cpg0016-jump/',
                max_keys=25,
                entries=['source_4'],
                raw_prefixes=['cpg0016-jump/source_4/'],
                is_truncated=False,
            ),
            quilt_registry='s3://cellpainting-gallery',
            quilt_limit=5,
            quilt_packages=QuiltPackageListResult(
                registry='s3://cellpainting-gallery',
                package_names=['pkg_a'],
                limit=5,
            ),
            cpgdata_bucket='cellpainting-gallery-inventory',
            cpgdata_prefix='cellpainting-gallery/whole_bucket/',
            cpgdata_recursive=False,
            cpgdata_limit=2,
            cpgdata_prefixes=CPGDataPrefixListResult(
                bucket='cellpainting-gallery-inventory',
                prefix='cellpainting-gallery/whole_bucket/',
                recursive=False,
                limit=2,
                entries=['rev1/', 'rev2/'],
            ),
            include_gallery=True,
            include_quilt=True,
            include_cpgdata=True,
            errors={},
            notes=[],
            ok=True,
        )
        stdout = io.StringIO()
        with redirect_stdout(stdout):
            returncode = main(['summarize-data-access', '--config', str(CONFIG_PATH), '--gallery-max-keys', '25', '--quilt-limit', '5', '--cpgdata-limit', '2'])
        self.assertEqual(returncode, 0)
        payload = json.loads(stdout.getvalue())
        self.assertTrue(payload['ok'])
        self.assertEqual(payload['gallery']['datasets']['entry_count'], 1)
        self.assertEqual(payload['quilt']['packages']['package_count'], 1)
        summarize_mock.assert_called_once()

    @patch('cellpaint_pipeline.data_access.gallery.list_gallery_prefixes')
    def test_list_gallery_datasets_extracts_dataset_ids(self, list_gallery_mock) -> None:
        config = ProjectConfig.from_json(CONFIG_PATH)
        list_gallery_mock.return_value = GalleryListResult(
            bucket='cellpainting-gallery',
            prefix='',
            delimiter='/',
            max_keys=100,
            common_prefixes=['cpg0016-jump/', 'cpg0019-moshkov-deepprofiler/'],
            object_keys=[],
            is_truncated=False,
        )
        result = list_gallery_datasets(config, max_keys=100)
        self.assertEqual(result.level, 'dataset')
        self.assertEqual(result.entries, ['cpg0016-jump', 'cpg0019-moshkov-deepprofiler'])
        list_gallery_mock.assert_called_once()

    @patch('cellpaint_pipeline.data_access.gallery.list_gallery_prefixes')
    def test_list_gallery_sources_uses_default_dataset(self, list_gallery_mock) -> None:
        config = ProjectConfig.from_json(CONFIG_PATH)
        list_gallery_mock.return_value = GalleryListResult(
            bucket='cellpainting-gallery',
            prefix='cpg0016-jump/',
            delimiter='/',
            max_keys=100,
            common_prefixes=['cpg0016-jump/source_4/', 'cpg0016-jump/source_all/'],
            object_keys=[],
            is_truncated=False,
        )
        result = list_gallery_sources(config, max_keys=100)
        self.assertEqual(result.level, 'source')
        self.assertEqual(result.parent_prefix, 'cpg0016-jump/')
        self.assertEqual(result.entries, ['source_4', 'source_all'])
        list_gallery_mock.assert_called_once()

    @patch('cellpaint_pipeline.data_access.summary.list_cpgdata_prefixes')
    @patch('cellpaint_pipeline.data_access.summary.list_quilt_packages')
    @patch('cellpaint_pipeline.data_access.summary.list_gallery_sources')
    @patch('cellpaint_pipeline.data_access.summary.list_gallery_datasets')
    @patch('cellpaint_pipeline.data_access.summary.build_data_access_status')
    def test_summarize_data_access_collects_sections(
        self,
        build_status_mock,
        list_datasets_mock,
        list_sources_mock,
        list_quilt_mock,
        list_cpgdata_mock,
    ) -> None:
        config = ProjectConfig.from_json(CONFIG_PATH)
        build_status_mock.return_value = DataAccessStatus(
            config=config.data_access,
            package_statuses=[
                PackageAvailability('boto3', 'boto3', True, True, '1.0.0', 'required'),
            ],
            executable_statuses=[],
            ok=True,
        )
        list_datasets_mock.return_value = GalleryCatalogResult(
            bucket='cellpainting-gallery',
            level='dataset',
            parent_prefix='',
            max_keys=25,
            entries=['cpg0016-jump'],
            raw_prefixes=['cpg0016-jump/'],
            is_truncated=False,
        )
        list_sources_mock.return_value = GalleryCatalogResult(
            bucket='cellpainting-gallery',
            level='source',
            parent_prefix='cpg0016-jump/',
            max_keys=25,
            entries=['source_4'],
            raw_prefixes=['cpg0016-jump/source_4/'],
            is_truncated=False,
        )
        list_quilt_mock.return_value = QuiltPackageListResult(
            registry='s3://cellpainting-gallery',
            package_names=['pkg_a', 'pkg_b'],
            limit=2,
        )
        list_cpgdata_mock.return_value = CPGDataPrefixListResult(
            bucket='cellpainting-gallery-inventory',
            prefix='cellpainting-gallery/whole_bucket/',
            recursive=False,
            limit=2,
            entries=['rev1/', 'rev2/'],
        )

        result = summarize_data_access(
            config,
            dataset_id='cpg0016-jump',
            gallery_max_keys=25,
            quilt_limit=2,
            cpgdata_limit=2,
        )

        self.assertTrue(result.ok)
        self.assertEqual(result.resolved_dataset_id, 'cpg0016-jump')
        self.assertEqual(result.dataset_listing.entries, ['cpg0016-jump'])
        self.assertEqual(result.source_listing.entries, ['source_4'])
        self.assertEqual(result.quilt_packages.package_names, ['pkg_a', 'pkg_b'])
        self.assertEqual(result.cpgdata_prefixes.entries, ['rev1/', 'rev2/'])

    def test_data_access_summary_to_dict(self) -> None:
        config = ProjectConfig.from_json(CONFIG_PATH)
        result = DataAccessSummaryResult(
            status=DataAccessStatus(
                config=config.data_access,
                package_statuses=[
                    PackageAvailability('boto3', 'boto3', True, True, '1.0.0', 'required'),
                ],
                executable_statuses=[],
                ok=True,
            ),
            resolved_dataset_id='cpg0016-jump',
            gallery_bucket='cellpainting-gallery',
            gallery_max_keys=25,
            dataset_listing=GalleryCatalogResult(
                bucket='cellpainting-gallery',
                level='dataset',
                parent_prefix='',
                max_keys=25,
                entries=['cpg0016-jump'],
                raw_prefixes=['cpg0016-jump/'],
                is_truncated=False,
            ),
            source_listing=GalleryCatalogResult(
                bucket='cellpainting-gallery',
                level='source',
                parent_prefix='cpg0016-jump/',
                max_keys=25,
                entries=['source_4'],
                raw_prefixes=['cpg0016-jump/source_4/'],
                is_truncated=False,
            ),
            quilt_registry='s3://cellpainting-gallery',
            quilt_limit=2,
            quilt_packages=QuiltPackageListResult(
                registry='s3://cellpainting-gallery',
                package_names=['pkg_a', 'pkg_b'],
                limit=2,
            ),
            cpgdata_bucket='cellpainting-gallery-inventory',
            cpgdata_prefix='cellpainting-gallery/whole_bucket/',
            cpgdata_recursive=False,
            cpgdata_limit=2,
            cpgdata_prefixes=CPGDataPrefixListResult(
                bucket='cellpainting-gallery-inventory',
                prefix='cellpainting-gallery/whole_bucket/',
                recursive=False,
                limit=2,
                entries=['rev1/', 'rev2/'],
            ),
            include_gallery=True,
            include_quilt=True,
            include_cpgdata=True,
            errors={},
            notes=[],
            ok=True,
        )
        payload = data_access_summary_to_dict(result)
        self.assertTrue(payload['ok'])
        self.assertEqual(payload['gallery']['datasets']['entry_count'], 1)
        self.assertEqual(payload['quilt']['packages']['package_count'], 2)
        self.assertEqual(payload['cpgdata']['prefixes']['entry_count'], 2)

    @patch('cellpaint_pipeline.data_access.gallery.list_gallery_prefixes')
    def test_cache_gallery_listing_writes_json(self, list_gallery_mock) -> None:
        config = ProjectConfig.from_json(CONFIG_PATH)
        list_gallery_mock.return_value = GalleryListResult(
            bucket='cellpainting-gallery',
            prefix='cpg0016-jump/',
            delimiter='/',
            max_keys=50,
            common_prefixes=['cpg0016-jump/source_4/'],
            object_keys=['cpg0016-jump/README.md'],
            is_truncated=False,
        )
        with tempfile.TemporaryDirectory() as tmpdir:
            output_path = Path(tmpdir) / 'listing.json'
            result = cache_gallery_listing(config, prefix='cpg0016-jump/', output_path=output_path)
            self.assertTrue(output_path.exists())
            payload = json.loads(output_path.read_text(encoding='utf-8'))
            self.assertEqual(payload['bucket'], 'cellpainting-gallery')
            self.assertEqual(payload['common_prefix_count'], 1)
            self.assertEqual(payload['object_count'], 1)
            self.assertEqual(result.output_path, output_path)

    @patch('cellpaint_pipeline.data_access.gallery._list_gallery_objects')
    def test_download_gallery_prefix_dry_run_writes_manifest(self, list_objects_mock) -> None:
        config = ProjectConfig.from_json(CONFIG_PATH)
        list_objects_mock.return_value = [
            {'Key': 'cpg0016-jump/source_4/images/file1.tiff', 'Size': 10},
            {'Key': 'cpg0016-jump/source_4/images/file2.tiff', 'Size': 20},
        ]
        with tempfile.TemporaryDirectory() as tmpdir:
            output_dir = Path(tmpdir) / 'download'
            result = download_gallery_prefix(config, prefix='cpg0016-jump/source_4/images/', output_dir=output_dir, dry_run=True, max_files=2)
            self.assertEqual(result.matched_object_count, 2)
            self.assertEqual(result.downloaded_count, 0)
            self.assertTrue(result.manifest_path.exists())
            payload = json.loads(result.manifest_path.read_text(encoding='utf-8'))
            self.assertTrue(payload['dry_run'])
            self.assertEqual(payload['matched_object_count'], 2)

    @patch('cellpaint_pipeline.data_access.gallery._download_gallery_object')
    @patch('cellpaint_pipeline.data_access.gallery._list_gallery_objects')
    def test_download_gallery_prefix_downloads_files(self, list_objects_mock, download_object_mock) -> None:
        config = ProjectConfig.from_json(CONFIG_PATH)
        list_objects_mock.return_value = [
            {'Key': 'cpg0016-jump/source_4/images/f1.tiff', 'Size': 10},
            {'Key': 'cpg0016-jump/source_4/images/nested/f2.tiff', 'Size': 20},
        ]

        def _fake_download(_client, _bucket, key, output_path):
            output_path.write_text(f'downloaded {key}', encoding='utf-8')

        download_object_mock.side_effect = _fake_download
        with tempfile.TemporaryDirectory() as tmpdir:
            output_dir = Path(tmpdir) / 'download'
            result = download_gallery_prefix(config, prefix='cpg0016-jump/source_4/images/', output_dir=output_dir, dry_run=False)
            self.assertEqual(result.downloaded_count, 2)
            self.assertTrue((output_dir / 'f1.tiff').exists())
            self.assertTrue((output_dir / 'nested' / 'f2.tiff').exists())

    @patch('cellpaint_pipeline.data_access.gallery.download_gallery_prefix')
    def test_download_gallery_source_uses_defaults(self, download_prefix_mock) -> None:
        config = ProjectConfig.from_json(CONFIG_PATH)
        with tempfile.TemporaryDirectory() as tmpdir:
            output_dir = Path(tmpdir) / 'download'
            manifest_path = output_dir / 'download_manifest.json'
            expected = GalleryDownloadResult(
                output_dir=output_dir,
                manifest_path=manifest_path,
                bucket='cellpainting-gallery',
                prefix='cpg0016-jump/source_4/images/',
                matched_object_count=1,
                downloaded_count=0,
                skipped_existing_count=0,
                dry_run=True,
                overwrite=False,
                max_files=1,
                include_substrings=(),
                exclude_substrings=(),
                object_keys=['cpg0016-jump/source_4/images/f1.tiff'],
            )
            download_prefix_mock.return_value = expected
            result = download_gallery_source(config, subprefix='images', output_dir=output_dir, dry_run=True, max_files=1)
            self.assertEqual(result, expected)
            self.assertEqual(download_prefix_mock.call_args.kwargs['prefix'], 'cpg0016-jump/source_4/images/')

    @patch('cellpaint_pipeline.data_access.planning.summarize_data_access')
    def test_build_download_plan_source_uses_defaults(self, summarize_mock) -> None:
        config = ProjectConfig.from_json(CONFIG_PATH)
        summarize_mock.return_value = DataAccessSummaryResult(
            status=DataAccessStatus(config=config.data_access, package_statuses=[], executable_statuses=[], ok=True),
            resolved_dataset_id='cpg0016-jump',
            gallery_bucket='cellpainting-gallery',
            gallery_max_keys=1000,
            dataset_listing=GalleryCatalogResult(
                bucket='cellpainting-gallery',
                level='dataset',
                parent_prefix='',
                max_keys=1000,
                entries=['cpg0016-jump'],
                raw_prefixes=['cpg0016-jump/'],
                is_truncated=False,
            ),
            source_listing=GalleryCatalogResult(
                bucket='cellpainting-gallery',
                level='source',
                parent_prefix='cpg0016-jump/',
                max_keys=1000,
                entries=['source_4'],
                raw_prefixes=['cpg0016-jump/source_4/'],
                is_truncated=False,
            ),
            quilt_registry=None,
            quilt_limit=None,
            quilt_packages=None,
            cpgdata_bucket='cellpainting-gallery-inventory',
            cpgdata_prefix='cellpainting-gallery/whole_bucket/',
            cpgdata_recursive=False,
            cpgdata_limit=None,
            cpgdata_prefixes=None,
            include_gallery=True,
            include_quilt=False,
            include_cpgdata=False,
            errors={},
            notes=[],
            ok=True,
        )
        request = build_data_request(subprefix='images', max_files=3, dry_run=True)
        plan = build_download_plan(config, request, summary_max_keys=50)
        self.assertTrue(plan.ok)
        self.assertTrue(plan.summary_used)
        self.assertEqual(plan.resolved_dataset_id, 'cpg0016-jump')
        self.assertEqual(plan.resolved_source_id, 'source_4')
        self.assertEqual(plan.resolved_prefix, 'cpg0016-jump/source_4/images/')
        self.assertEqual(plan.steps[0].max_files, 3)

    def test_build_download_plan_prefix_mode(self) -> None:
        config = ProjectConfig.from_json(CONFIG_PATH)
        request = build_data_request(mode='gallery-prefix', prefix='cpg0016-jump/source_4/workspace', dry_run=True)
        plan = build_download_plan(config, request, validate_with_summary=False)
        self.assertTrue(plan.ok)
        self.assertEqual(plan.resolved_prefix, 'cpg0016-jump/source_4/workspace/')
        self.assertEqual(plan.steps[0].mode, 'gallery-prefix')

    def test_write_and_load_download_plan_roundtrip(self) -> None:
        config = ProjectConfig.from_json(CONFIG_PATH)
        request = DataRequest(
            mode='gallery-prefix',
            dataset_id=None,
            source_id=None,
            prefix='cpg0016-jump/source_4/workspace/',
            subprefix='',
            bucket='cellpainting-gallery',
            include_substrings=('load_data',),
            exclude_substrings=(),
            max_files=1,
            overwrite=False,
            dry_run=True,
            output_dir=None,
            manifest_path=None,
        )
        step = DataDownloadStep(
            step_key='download-1',
            adapter='gallery',
            mode='gallery-prefix',
            bucket='cellpainting-gallery',
            prefix='cpg0016-jump/source_4/workspace/',
            dataset_id=None,
            source_id=None,
            subprefix='',
            include_substrings=('load_data',),
            exclude_substrings=(),
            max_files=1,
            overwrite=False,
            dry_run=True,
            output_dir=config.data_access.data_cache_root / 'plan_test',
            manifest_path=config.data_access.data_cache_root / 'plan_test' / 'download_manifest.json',
        )
        plan = DataDownloadPlan(
            request=request,
            resolved_dataset_id=None,
            resolved_source_id=None,
            resolved_prefix='cpg0016-jump/source_4/workspace/',
            steps=[step],
            notes=['Prefix-mode request bypasses dataset/source validation.'],
            errors=[],
            summary_used=False,
            ok=True,
        )
        with tempfile.TemporaryDirectory() as tmpdir:
            output_path = Path(tmpdir) / 'plan.json'
            write_download_plan(plan, output_path)
            loaded = load_download_plan(output_path)
            self.assertEqual(data_download_plan_to_dict(loaded), data_download_plan_to_dict(plan))

    @patch('cellpaint_pipeline.data_access.planning.download_gallery_source')
    def test_execute_download_plan_runs_gallery_source(self, download_source_mock) -> None:
        config = ProjectConfig.from_json(CONFIG_PATH)
        step = DataDownloadStep(
            step_key='download-1',
            adapter='gallery',
            mode='gallery-source',
            bucket='cellpainting-gallery',
            prefix='cpg0016-jump/source_4/images/',
            dataset_id='cpg0016-jump',
            source_id='source_4',
            subprefix='images',
            include_substrings=(),
            exclude_substrings=(),
            max_files=2,
            overwrite=False,
            dry_run=True,
            output_dir=Path('/tmp/plan_exec'),
            manifest_path=Path('/tmp/plan_exec/download_manifest.json'),
        )
        plan = DataDownloadPlan(
            request=build_data_request(subprefix='images', max_files=2, dry_run=True),
            resolved_dataset_id='cpg0016-jump',
            resolved_source_id='source_4',
            resolved_prefix='cpg0016-jump/source_4/images/',
            steps=[step],
            notes=[],
            errors=[],
            summary_used=False,
            ok=True,
        )
        download_source_mock.return_value = GalleryDownloadResult(
            output_dir=Path('/tmp/plan_exec'),
            manifest_path=Path('/tmp/plan_exec/download_manifest.json'),
            bucket='cellpainting-gallery',
            prefix='cpg0016-jump/source_4/images/',
            matched_object_count=2,
            downloaded_count=0,
            skipped_existing_count=0,
            dry_run=True,
            overwrite=False,
            max_files=2,
            include_substrings=(),
            exclude_substrings=(),
            object_keys=['a', 'b'],
        )
        result = execute_download_plan(config, plan)
        self.assertTrue(result.ok)
        self.assertEqual(result.step_results[0].download_result.matched_object_count, 2)
        download_source_mock.assert_called_once()

    def test_data_download_execution_result_to_dict(self) -> None:
        step = DataDownloadStep(
            step_key='download-1',
            adapter='gallery',
            mode='gallery-prefix',
            bucket='cellpainting-gallery',
            prefix='cpg0016-jump/source_4/workspace/',
            dataset_id=None,
            source_id=None,
            subprefix='',
            include_substrings=(),
            exclude_substrings=(),
            max_files=1,
            overwrite=False,
            dry_run=True,
            output_dir=Path('/tmp/plan_exec'),
            manifest_path=Path('/tmp/plan_exec/download_manifest.json'),
        )
        plan = DataDownloadPlan(
            request=build_data_request(mode='gallery-prefix', prefix='cpg0016-jump/source_4/workspace/', dry_run=True),
            resolved_dataset_id=None,
            resolved_source_id=None,
            resolved_prefix='cpg0016-jump/source_4/workspace/',
            steps=[step],
            notes=[],
            errors=[],
            summary_used=False,
            ok=True,
        )
        execution = DataDownloadExecutionResult(
            plan=plan,
            step_results=[
                DataDownloadStepResult(
                    step=step,
                    download_result=GalleryDownloadResult(
                        output_dir=Path('/tmp/plan_exec'),
                        manifest_path=Path('/tmp/plan_exec/download_manifest.json'),
                        bucket='cellpainting-gallery',
                        prefix='cpg0016-jump/source_4/workspace/',
                        matched_object_count=1,
                        downloaded_count=0,
                        skipped_existing_count=0,
                        dry_run=True,
                        overwrite=False,
                        max_files=1,
                        include_substrings=(),
                        exclude_substrings=(),
                        object_keys=['a'],
                    ),
                    ok=True,
                )
            ],
            ok=True,
        )
        payload = data_download_execution_result_to_dict(execution)
        self.assertTrue(payload['ok'])
        self.assertEqual(payload['step_result_count'], 1)
        self.assertEqual(payload['step_results'][0]['download_result']['matched_object_count'], 1)

    @patch('cellpaint_pipeline.data_access.access_packages._require_quilt3')
    def test_list_quilt_packages_uses_default_registry(self, require_quilt3_mock) -> None:
        config = ProjectConfig.from_json(CONFIG_PATH)
        require_quilt3_mock.return_value = SimpleNamespace(list_packages=lambda registry: ['pkg_b', 'pkg_a'])
        result = list_quilt_packages(config, limit=1)
        self.assertEqual(result.registry, 's3://cellpainting-gallery')
        self.assertEqual(result.package_names, ['pkg_a'])

    @patch('cellpaint_pipeline.data_access.access_packages._require_quilt3')
    def test_browse_quilt_package_collects_keys(self, require_quilt3_mock) -> None:
        config = ProjectConfig.from_json(CONFIG_PATH)
        fake_package = SimpleNamespace(keys=lambda: ['images', 'metadata'], top_hash='abc123')
        fake_package_cls = SimpleNamespace(browse=lambda name, registry=None, top_hash=None: fake_package)
        require_quilt3_mock.return_value = SimpleNamespace(Package=fake_package_cls)
        result = browse_quilt_package(config, package_name='example', max_keys=10)
        self.assertEqual(result.package_name, 'example')
        self.assertEqual(result.top_level_keys, ['images', 'metadata'])
        self.assertEqual(result.resolved_top_hash, 'abc123')

    @patch('cellpaint_pipeline.data_access.access_packages._require_cpgdata_utils')
    def test_list_cpgdata_prefixes_uses_defaults(self, require_utils_mock) -> None:
        config = ProjectConfig.from_json(CONFIG_PATH)
        require_utils_mock.return_value = SimpleNamespace(ls_s3_prefix=lambda bucket, prefix, recursive=False: ['rev1/', 'rev2/'])
        result = list_cpgdata_prefixes(config, limit=1)
        self.assertEqual(result.bucket, 'cellpainting-gallery-inventory')
        self.assertEqual(result.prefix, 'cellpainting-gallery/whole_bucket/')
        self.assertEqual(result.entries, ['rev1/'])

    @patch('cellpaint_pipeline.data_access.access_packages._require_cpgdata_utils')
    def test_sync_cpgdata_index_uses_defaults(self, require_utils_mock) -> None:
        config = ProjectConfig.from_json(CONFIG_PATH)
        sync_calls = []
        require_utils_mock.return_value = SimpleNamespace(sync_s3_prefix=lambda *args, **kwargs: sync_calls.append((args, kwargs)))
        with tempfile.TemporaryDirectory() as tmpdir:
            output_dir = Path(tmpdir) / 'index'
            result = sync_cpgdata_index(config, output_dir=output_dir, include='*.csv', exclude='*.tmp')
            self.assertEqual(result.mode, 'index')
            self.assertEqual(result.output_dir, output_dir.resolve())
            self.assertEqual(len(sync_calls), 1)

    @patch('cellpaint_pipeline.data_access.access_packages._require_cpgdata_utils')
    def test_sync_cpgdata_inventory_uses_defaults(self, require_utils_mock) -> None:
        config = ProjectConfig.from_json(CONFIG_PATH)
        sync_calls = []
        require_utils_mock.return_value = SimpleNamespace(sync_inventory=lambda *args, **kwargs: sync_calls.append((args, kwargs)))
        with tempfile.TemporaryDirectory() as tmpdir:
            output_dir = Path(tmpdir) / 'inventory'
            result = sync_cpgdata_inventory(config, output_dir=output_dir, revision=2)
            self.assertEqual(result.mode, 'inventory')
            self.assertEqual(result.revision, 2)
            self.assertEqual(result.output_dir, output_dir.resolve())
            self.assertEqual(len(sync_calls), 1)

    @patch('cellpaint_pipeline.cli.list_quilt_packages')
    def test_cli_list_quilt_packages_command(self, list_quilt_packages_mock) -> None:
        list_quilt_packages_mock.return_value = QuiltPackageListResult(
            registry='s3://cellpainting-gallery',
            package_names=['pkg_a', 'pkg_b'],
            limit=2,
        )
        stdout = io.StringIO()
        with redirect_stdout(stdout):
            returncode = main(['list-quilt-packages', '--config', str(CONFIG_PATH), '--limit', '2'])
        self.assertEqual(returncode, 0)
        payload = json.loads(stdout.getvalue())
        self.assertEqual(payload['package_count'], 2)
        list_quilt_packages_mock.assert_called_once()

    @patch('cellpaint_pipeline.cli.browse_quilt_package')
    def test_cli_browse_quilt_package_command(self, browse_quilt_package_mock) -> None:
        browse_quilt_package_mock.return_value = QuiltPackageBrowseResult(
            registry='s3://cellpainting-gallery',
            package_name='pkg_a',
            requested_top_hash=None,
            resolved_top_hash='abc123',
            top_level_keys=['images', 'metadata'],
            top_level_key_count=2,
        )
        stdout = io.StringIO()
        with redirect_stdout(stdout):
            returncode = main(['browse-quilt-package', '--config', str(CONFIG_PATH), '--package-name', 'pkg_a'])
        self.assertEqual(returncode, 0)
        payload = json.loads(stdout.getvalue())
        self.assertEqual(payload['package_name'], 'pkg_a')
        self.assertEqual(payload['top_level_key_count'], 2)
        browse_quilt_package_mock.assert_called_once()

    @patch('cellpaint_pipeline.cli.list_cpgdata_prefixes')
    def test_cli_list_cpgdata_prefixes_command(self, list_cpgdata_prefixes_mock) -> None:
        list_cpgdata_prefixes_mock.return_value = CPGDataPrefixListResult(
            bucket='cellpainting-gallery-inventory',
            prefix='cellpainting-gallery/whole_bucket/',
            recursive=False,
            limit=2,
            entries=['rev1/', 'rev2/'],
        )
        stdout = io.StringIO()
        with redirect_stdout(stdout):
            returncode = main(['list-cpgdata-prefixes', '--config', str(CONFIG_PATH), '--limit', '2'])
        self.assertEqual(returncode, 0)
        payload = json.loads(stdout.getvalue())
        self.assertEqual(payload['entry_count'], 2)
        list_cpgdata_prefixes_mock.assert_called_once()

    @patch('cellpaint_pipeline.cli.sync_cpgdata_index')
    def test_cli_sync_cpgdata_index_command(self, sync_cpgdata_index_mock) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            output_dir = Path(tmpdir) / 'index'
            sync_cpgdata_index_mock.return_value = CPGDataSyncResult(
                mode='index',
                bucket='cellpainting-gallery-inventory',
                prefix='cellpainting-gallery/index',
                output_dir=output_dir,
                revision=None,
                include='*.csv',
                exclude=None,
                no_progress=True,
            )
            stdout = io.StringIO()
            with redirect_stdout(stdout):
                returncode = main(['sync-cpgdata-index', '--config', str(CONFIG_PATH), '--output-dir', str(output_dir)])
            self.assertEqual(returncode, 0)
            payload = json.loads(stdout.getvalue())
            self.assertEqual(payload['mode'], 'index')
            sync_cpgdata_index_mock.assert_called_once()

    @patch('cellpaint_pipeline.cli.sync_cpgdata_inventory')
    def test_cli_sync_cpgdata_inventory_command(self, sync_cpgdata_inventory_mock) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            output_dir = Path(tmpdir) / 'inventory'
            sync_cpgdata_inventory_mock.return_value = CPGDataSyncResult(
                mode='inventory',
                bucket='cellpainting-gallery-inventory',
                prefix='cellpainting-gallery/whole_bucket/',
                output_dir=output_dir,
                revision=0,
                include=None,
                exclude=None,
                no_progress=False,
            )
            stdout = io.StringIO()
            with redirect_stdout(stdout):
                returncode = main(['sync-cpgdata-inventory', '--config', str(CONFIG_PATH), '--output-dir', str(output_dir)])
            self.assertEqual(returncode, 0)
            payload = json.loads(stdout.getvalue())
            self.assertEqual(payload['mode'], 'inventory')
            sync_cpgdata_inventory_mock.assert_called_once()

    @patch('cellpaint_pipeline.cli.build_download_plan')
    def test_cli_plan_data_access_command(self, build_download_plan_mock) -> None:
        plan = DataDownloadPlan(
            request=build_data_request(subprefix='images', max_files=2, dry_run=True),
            resolved_dataset_id='cpg0016-jump',
            resolved_source_id='source_4',
            resolved_prefix='cpg0016-jump/source_4/images/',
            steps=[
                DataDownloadStep(
                    step_key='download-1',
                    adapter='gallery',
                    mode='gallery-source',
                    bucket='cellpainting-gallery',
                    prefix='cpg0016-jump/source_4/images/',
                    dataset_id='cpg0016-jump',
                    source_id='source_4',
                    subprefix='images',
                    include_substrings=(),
                    exclude_substrings=(),
                    max_files=2,
                    overwrite=False,
                    dry_run=True,
                    output_dir=Path('/tmp/plan_cli'),
                    manifest_path=Path('/tmp/plan_cli/download_manifest.json'),
                )
            ],
            notes=[],
            errors=[],
            summary_used=True,
            ok=True,
        )
        build_download_plan_mock.return_value = plan
        stdout = io.StringIO()
        with redirect_stdout(stdout):
            returncode = main(['plan-data-access', '--config', str(CONFIG_PATH), '--subprefix', 'images', '--max-files', '2', '--dry-run'])
        self.assertEqual(returncode, 0)
        payload = json.loads(stdout.getvalue())
        self.assertEqual(payload['resolved_prefix'], 'cpg0016-jump/source_4/images/')
        self.assertEqual(payload['step_count'], 1)
        build_download_plan_mock.assert_called_once()

    @patch('cellpaint_pipeline.cli.execute_download_plan')
    @patch('cellpaint_pipeline.cli.load_download_plan')
    def test_cli_execute_download_plan_command(self, load_download_plan_mock, execute_download_plan_mock) -> None:
        plan = DataDownloadPlan(
            request=build_data_request(mode='gallery-prefix', prefix='cpg0016-jump/source_4/workspace/', dry_run=True),
            resolved_dataset_id=None,
            resolved_source_id=None,
            resolved_prefix='cpg0016-jump/source_4/workspace/',
            steps=[
                DataDownloadStep(
                    step_key='download-1',
                    adapter='gallery',
                    mode='gallery-prefix',
                    bucket='cellpainting-gallery',
                    prefix='cpg0016-jump/source_4/workspace/',
                    dataset_id=None,
                    source_id=None,
                    subprefix='',
                    include_substrings=(),
                    exclude_substrings=(),
                    max_files=1,
                    overwrite=False,
                    dry_run=True,
                    output_dir=Path('/tmp/plan_cli'),
                    manifest_path=Path('/tmp/plan_cli/download_manifest.json'),
                )
            ],
            notes=[],
            errors=[],
            summary_used=False,
            ok=True,
        )
        load_download_plan_mock.return_value = plan
        execute_download_plan_mock.return_value = DataDownloadExecutionResult(
            plan=plan,
            step_results=[
                DataDownloadStepResult(
                    step=plan.steps[0],
                    download_result=GalleryDownloadResult(
                        output_dir=Path('/tmp/plan_cli'),
                        manifest_path=Path('/tmp/plan_cli/download_manifest.json'),
                        bucket='cellpainting-gallery',
                        prefix='cpg0016-jump/source_4/workspace/',
                        matched_object_count=1,
                        downloaded_count=0,
                        skipped_existing_count=0,
                        dry_run=True,
                        overwrite=False,
                        max_files=1,
                        include_substrings=(),
                        exclude_substrings=(),
                        object_keys=['a'],
                    ),
                    ok=True,
                )
            ],
            ok=True,
        )
        stdout = io.StringIO()
        with redirect_stdout(stdout):
            returncode = main(['execute-download-plan', '--config', str(CONFIG_PATH), '--plan-path', '/tmp/fake_plan.json'])
        self.assertEqual(returncode, 0)
        payload = json.loads(stdout.getvalue())
        self.assertEqual(payload['step_result_count'], 1)
        load_download_plan_mock.assert_called_once()
        execute_download_plan_mock.assert_called_once()

    @patch('cellpaint_pipeline.cli.list_gallery_datasets')
    def test_cli_list_gallery_datasets_command(self, list_datasets_mock) -> None:
        list_datasets_mock.return_value = GalleryCatalogResult(
            bucket='cellpainting-gallery',
            level='dataset',
            parent_prefix='',
            max_keys=100,
            entries=['cpg0016-jump', 'cpg0019-moshkov-deepprofiler'],
            raw_prefixes=['cpg0016-jump/', 'cpg0019-moshkov-deepprofiler/'],
            is_truncated=False,
        )
        stdout = io.StringIO()
        with redirect_stdout(stdout):
            returncode = main(['list-gallery-datasets', '--config', str(CONFIG_PATH)])
        self.assertEqual(returncode, 0)
        payload = json.loads(stdout.getvalue())
        self.assertEqual(payload['entry_count'], 2)
        list_datasets_mock.assert_called_once()

    @patch('cellpaint_pipeline.cli.list_gallery_sources')
    def test_cli_list_gallery_sources_command(self, list_sources_mock) -> None:
        list_sources_mock.return_value = GalleryCatalogResult(
            bucket='cellpainting-gallery',
            level='source',
            parent_prefix='cpg0016-jump/',
            max_keys=100,
            entries=['source_4', 'source_all'],
            raw_prefixes=['cpg0016-jump/source_4/', 'cpg0016-jump/source_all/'],
            is_truncated=False,
        )
        stdout = io.StringIO()
        with redirect_stdout(stdout):
            returncode = main(['list-gallery-sources', '--config', str(CONFIG_PATH), '--dataset-id', 'cpg0016-jump'])
        self.assertEqual(returncode, 0)
        payload = json.loads(stdout.getvalue())
        self.assertEqual(payload['entry_count'], 2)
        list_sources_mock.assert_called_once()

    @patch('cellpaint_pipeline.cli.cache_gallery_listing')
    def test_cli_cache_gallery_prefixes_command(self, cache_listing_mock) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            output_path = Path(tmpdir) / 'cached_listing.json'
            cache_listing_mock.return_value = GalleryCacheResult(
                output_path=output_path,
                bucket='cellpainting-gallery',
                prefix='cpg0016-jump/',
                delimiter='/',
                max_keys=100,
                common_prefix_count=1,
                object_count=0,
            )
            stdout = io.StringIO()
            with redirect_stdout(stdout):
                returncode = main(['cache-gallery-prefixes', '--config', str(CONFIG_PATH), '--prefix', 'cpg0016-jump/', '--output-path', str(output_path)])
            self.assertEqual(returncode, 0)
            payload = json.loads(stdout.getvalue())
            self.assertEqual(payload['output_path'], str(output_path))
            cache_listing_mock.assert_called_once()

    @patch('cellpaint_pipeline.cli.download_gallery_prefix')
    def test_cli_download_gallery_prefix_command(self, download_prefix_mock) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            output_dir = Path(tmpdir) / 'download'
            manifest_path = output_dir / 'download_manifest.json'
            download_prefix_mock.return_value = GalleryDownloadResult(
                output_dir=output_dir,
                manifest_path=manifest_path,
                bucket='cellpainting-gallery',
                prefix='cpg0016-jump/source_4/images/',
                matched_object_count=2,
                downloaded_count=2,
                skipped_existing_count=0,
                dry_run=False,
                overwrite=False,
                max_files=2,
                include_substrings=(),
                exclude_substrings=(),
                object_keys=['a', 'b'],
            )
            stdout = io.StringIO()
            with redirect_stdout(stdout):
                returncode = main(['download-gallery-prefix', '--config', str(CONFIG_PATH), '--prefix', 'cpg0016-jump/source_4/images/', '--output-dir', str(output_dir), '--max-files', '2'])
            self.assertEqual(returncode, 0)
            payload = json.loads(stdout.getvalue())
            self.assertEqual(payload['matched_object_count'], 2)
            download_prefix_mock.assert_called_once()

    @patch('cellpaint_pipeline.cli.download_gallery_source')
    def test_cli_download_gallery_source_command(self, download_source_mock) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            output_dir = Path(tmpdir) / 'download'
            manifest_path = output_dir / 'download_manifest.json'
            download_source_mock.return_value = GalleryDownloadResult(
                output_dir=output_dir,
                manifest_path=manifest_path,
                bucket='cellpainting-gallery',
                prefix='cpg0016-jump/source_4/workspace/',
                matched_object_count=1,
                downloaded_count=0,
                skipped_existing_count=0,
                dry_run=True,
                overwrite=False,
                max_files=1,
                include_substrings=('load_data',),
                exclude_substrings=(),
                object_keys=['cpg0016-jump/source_4/workspace/load_data.csv'],
            )
            stdout = io.StringIO()
            with redirect_stdout(stdout):
                returncode = main(['download-gallery-source', '--config', str(CONFIG_PATH), '--dataset-id', 'cpg0016-jump', '--source-id', 'source_4', '--subprefix', 'workspace', '--dry-run', '--max-files', '1'])
            self.assertEqual(returncode, 0)
            payload = json.loads(stdout.getvalue())
            self.assertTrue(payload['dry_run'])
            download_source_mock.assert_called_once()

    def test_data_access_status_to_dict_counts_required_packages(self) -> None:
        config = ProjectConfig.from_json(CONFIG_PATH)
        status = DataAccessStatus(
            config=config.data_access,
            package_statuses=[
                PackageAvailability('boto3', 'boto3', True, True, '1.0.0', 'required'),
                PackageAvailability('botocore', 'botocore', True, False, None, 'required'),
                PackageAvailability('cpgdata', 'cpgdata', False, True, '0.6.0', 'optional'),
            ],
            executable_statuses=[],
            ok=False,
        )
        payload = data_access_status_to_dict(status)
        self.assertEqual(payload['required_package_count'], 2)
        self.assertEqual(payload['available_required_package_count'], 1)

    def test_gallery_list_result_to_dict_counts_items(self) -> None:
        payload = gallery_list_result_to_dict(
            GalleryListResult(
                bucket='cellpainting-gallery',
                prefix='cpg0016-jump/',
                delimiter='/',
                max_keys=10,
                common_prefixes=['a/', 'b/'],
                object_keys=['obj1', 'obj2', 'obj3'],
                is_truncated=True,
            )
        )
        self.assertEqual(payload['common_prefix_count'], 2)
        self.assertEqual(payload['object_count'], 3)
        self.assertTrue(payload['is_truncated'])

    def test_gallery_catalog_result_to_dict_counts_entries(self) -> None:
        payload = gallery_catalog_result_to_dict(
            GalleryCatalogResult(
                bucket='cellpainting-gallery',
                level='dataset',
                parent_prefix='',
                max_keys=10,
                entries=['cpg0016-jump', 'cpg0019-moshkov-deepprofiler'],
                raw_prefixes=['cpg0016-jump/', 'cpg0019-moshkov-deepprofiler/'],
                is_truncated=False,
            )
        )
        self.assertEqual(payload['entry_count'], 2)
        self.assertEqual(payload['entries'][0], 'cpg0016-jump')

    def test_gallery_cache_result_to_dict(self) -> None:
        payload = gallery_cache_result_to_dict(
            GalleryCacheResult(
                output_path=Path('/tmp/listing.json'),
                bucket='cellpainting-gallery',
                prefix='cpg0016-jump/',
                delimiter='/',
                max_keys=25,
                common_prefix_count=3,
                object_count=1,
            )
        )
        self.assertEqual(payload['output_path'], '/tmp/listing.json')
        self.assertEqual(payload['object_count'], 1)

    def test_gallery_download_result_to_dict(self) -> None:
        payload = gallery_download_result_to_dict(
            GalleryDownloadResult(
                output_dir=Path('/tmp/download'),
                manifest_path=Path('/tmp/download/download_manifest.json'),
                bucket='cellpainting-gallery',
                prefix='cpg0016-jump/source_4/images/',
                matched_object_count=2,
                downloaded_count=1,
                skipped_existing_count=1,
                dry_run=False,
                overwrite=False,
                max_files=2,
                include_substrings=('png',),
                exclude_substrings=('thumb',),
                object_keys=['a', 'b'],
            )
        )
        self.assertEqual(payload['matched_object_count'], 2)
        self.assertEqual(payload['include_substrings'], ['png'])

    def test_quilt_package_list_result_to_dict(self) -> None:
        payload = quilt_package_list_result_to_dict(
            QuiltPackageListResult(
                registry='s3://cellpainting-gallery',
                package_names=['pkg_a', 'pkg_b'],
                limit=2,
            )
        )
        self.assertEqual(payload['package_count'], 2)
        self.assertEqual(payload['adapter'], 'quilt3')

    def test_quilt_package_browse_result_to_dict(self) -> None:
        payload = quilt_package_browse_result_to_dict(
            QuiltPackageBrowseResult(
                registry='s3://cellpainting-gallery',
                package_name='pkg_a',
                requested_top_hash=None,
                resolved_top_hash='abc123',
                top_level_keys=['images'],
                top_level_key_count=1,
            )
        )
        self.assertEqual(payload['package_name'], 'pkg_a')
        self.assertEqual(payload['top_level_key_count'], 1)

    def test_cpgdata_prefix_list_result_to_dict(self) -> None:
        payload = cpgdata_prefix_list_result_to_dict(
            CPGDataPrefixListResult(
                bucket='cellpainting-gallery-inventory',
                prefix='cellpainting-gallery/whole_bucket/',
                recursive=False,
                limit=2,
                entries=['rev1/', 'rev2/'],
            )
        )
        self.assertEqual(payload['entry_count'], 2)
        self.assertEqual(payload['adapter'], 'cpgdata')

    def test_cpgdata_sync_result_to_dict(self) -> None:
        payload = cpgdata_sync_result_to_dict(
            CPGDataSyncResult(
                mode='index',
                bucket='cellpainting-gallery-inventory',
                prefix='cellpainting-gallery/index',
                output_dir=Path('/tmp/cpg_index'),
                revision=None,
                include='*.csv',
                exclude='*.tmp',
                no_progress=True,
            )
        )
        self.assertEqual(payload['mode'], 'index')
        self.assertEqual(payload['output_dir'], '/tmp/cpg_index')


if __name__ == '__main__':
    unittest.main()
