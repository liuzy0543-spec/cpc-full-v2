from __future__ import annotations

import json
import shutil
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import cellpaint_pipeline as package_api
from cellpaint_pipeline.cli import main
from cellpaint_pipeline.config import ProjectConfig
from cellpaint_pipeline.delivery import (
    SuiteExecutionError,
    SuiteRunResult,
    available_profiling_suites,
    available_segmentation_suites,
    run_deepprofiler_full_stack,
    run_profiling_suite,
    run_segmentation_bundle,
    run_segmentation_suite,
    run_smoke_test,
)
from cellpaint_pipeline.workflows.orchestration import WorkflowExecutionError, available_workflows


PROJECT_ROOT = Path(__file__).resolve().parents[1]
CONFIG_PATH = PROJECT_ROOT / 'configs' / 'project_config.example.json'
WORKFLOW_ROOT = PROJECT_ROOT / 'outputs' / 'workflows' / 'mask_export_native_postprocessing_runtimecfg'
DEEPPROFILER_WORKFLOW_ROOT = PROJECT_ROOT / 'outputs' / 'workflows' / 'deepprofiler_export_runtimecfg'
DEEPPROFILER_PROJECT_ROOT = PROJECT_ROOT / 'exports' / 'deepprofiler_project_validation_runnable_v3'


class DeliverySmokeTests(unittest.TestCase):
    def _build_minimal_profiling_project_config(self, tmpdir: str) -> Path:
        import pandas as pd

        tmpdir_path = Path(tmpdir)
        project_root = tmpdir_path / 'profiling_fixture_project'
        profiling_root = project_root / 'profiling_backend'
        segmentation_root = project_root / 'segmentation_backend_stub'
        exports_root = project_root / 'exports' / 'deepprofiler'
        output_root = project_root / 'outputs'

        for directory in [profiling_root / 'configs', profiling_root / 'data' / 'metadata', profiling_root / 'outputs' / 'cellprofiler', segmentation_root / 'configs', exports_root, output_root]:
            directory.mkdir(parents=True, exist_ok=True)

        source_pipeline_config = json.loads((PROJECT_ROOT / 'configs' / 'project_config.example.json').read_text(encoding='utf-8'))
        source_profiling_config = json.loads((Path(source_pipeline_config['profiling_backend_root']) / 'configs' / 'pipeline_config.json').read_text(encoding='utf-8'))

        shard_root = Path(source_pipeline_config['profiling_backend_root']) / 'outputs' / 'sharded' / 'cellprofiler' / 'shard_02'
        image_df = pd.read_csv(shard_root / 'Image.csv')
        cells_df = pd.read_csv(shard_root / 'Cells.csv')
        selected_pairs = image_df[['Metadata_Plate', 'Metadata_Well']].drop_duplicates()

        manifest_df = pd.read_csv(Path(source_pipeline_config['profiling_backend_root']) / 'data' / 'metadata' / 'image_manifest.csv')
        plate_map_df = pd.read_csv(Path(source_pipeline_config['profiling_backend_root']) / 'data' / 'metadata' / 'plate_map.csv')

        manifest_subset = manifest_df.merge(selected_pairs, on=['Metadata_Plate', 'Metadata_Well'], how='inner')
        plate_map_subset = plate_map_df.merge(selected_pairs, on=['Metadata_Plate', 'Metadata_Well'], how='inner')

        manifest_subset.to_csv(profiling_root / 'data' / 'metadata' / 'image_manifest.csv', index=False)
        plate_map_subset.to_csv(profiling_root / 'data' / 'metadata' / 'plate_map.csv', index=False)
        image_df.to_csv(profiling_root / 'outputs' / 'cellprofiler' / 'Image.csv', index=False)
        cells_df.to_csv(profiling_root / 'outputs' / 'cellprofiler' / 'Cells.csv', index=False)

        fixture_pipeline_config = dict(source_profiling_config)
        fixture_pipeline_config['paths'] = dict(source_profiling_config['paths'])
        fixture_pipeline_config['paths']['raw_images_dir'] = str(Path(source_pipeline_config['profiling_backend_root']) / 'data' / 'raw_core')
        fixture_pipeline_config['paths']['manifest_csv'] = 'data/metadata/image_manifest.csv'
        fixture_pipeline_config['paths']['plate_map_csv'] = 'data/metadata/plate_map.csv'
        fixture_pipeline_config['paths']['cellprofiler_output_dir'] = 'outputs/cellprofiler'
        fixture_pipeline_config['paths']['image_table_csv'] = 'outputs/cellprofiler/Image.csv'
        fixture_pipeline_config['paths']['cells_table_csv'] = 'outputs/cellprofiler/Cells.csv'
        fixture_pipeline_config['paths']['single_cell_output_csv_gz'] = 'outputs/pycytominer/single_cell.csv.gz'
        fixture_pipeline_config['paths']['aggregated_output_parquet'] = 'outputs/pycytominer/well_aggregated.parquet'
        fixture_pipeline_config['paths']['annotated_output_parquet'] = 'outputs/pycytominer/well_annotated.parquet'
        fixture_pipeline_config['paths']['normalized_output_parquet'] = 'outputs/pycytominer/well_normalized.parquet'
        fixture_pipeline_config['paths']['feature_selected_output_parquet'] = 'outputs/pycytominer/well_feature_selected.parquet'
        fixture_pipeline_config['pycytominer'] = dict(source_profiling_config['pycytominer'])
        fixture_pipeline_config['pycytominer']['normalize'] = dict(source_profiling_config['pycytominer']['normalize'])
        fixture_pipeline_config['pycytominer']['normalize']['method'] = 'standardize'
        fixture_pipeline_config['pycytominer']['normalize']['samples'] = 'all'
        fixture_pipeline_config['pycytominer']['feature_select'] = dict(source_profiling_config['pycytominer']['feature_select'])
        fixture_pipeline_config['pycytominer']['feature_select']['operations'] = ['drop_na_columns']
        (profiling_root / 'configs' / 'pipeline_config.json').write_text(json.dumps(fixture_pipeline_config, indent=2) + '\n', encoding='utf-8')

        segmentation_stub_config = {
            'project_name': 'segmentation_stub',
            'paths': {
                'source_load_data_csv': 'data/source_load_data.csv',
                'load_data_csv': 'data/load_data_for_segmentation.csv',
                'base_pipeline': str(Path(source_pipeline_config['segmentation_backend_root']) / 'cellprofiler' / 'CPJUMP1_analysis_without_batchfile_406.cppipe'),
                'mask_export_pipeline': 'cellprofiler/CPJUMP1_analysis_mask_export.cppipe',
                'cellprofiler_output_dir': 'outputs/cellprofiler_masks',
                'sample_previews_dir': 'outputs/sample_previews_png',
                'masked_crops_dir': 'outputs/single_cell_crops_masked',
                'masked_manifest_csv': 'outputs/single_cell_crops_masked/single_cell_manifest.csv',
                'unmasked_crops_dir': 'outputs/single_cell_crops_unmasked',
                'unmasked_manifest_csv': 'outputs/single_cell_crops_unmasked/single_cell_manifest.csv',
            },
            'subset': {'plates': [], 'wells': [], 'sites': []},
            'crop_extraction': {
                'crop_size_pixels': 128,
                'channel_order': ['OrigMito', 'OrigAGP', 'OrigRNA', 'OrigER', 'OrigDNA', 'OrigBrightfield', 'OrigHighZBF', 'OrigLowZBF'],
            },
        }
        (segmentation_root / 'configs' / 'segmentation_config.json').write_text(json.dumps(segmentation_stub_config, indent=2) + '\n', encoding='utf-8')
        (segmentation_root / 'data' / 'source_load_data.csv').parent.mkdir(parents=True, exist_ok=True)
        (segmentation_root / 'data' / 'source_load_data.csv').write_text('Metadata_Plate,Metadata_Well,Metadata_Site\n', encoding='utf-8')

        project_config = dict(source_pipeline_config)
        project_config['project_name'] = 'delivery_smoke_fixture'
        project_config['profiling_backend_root'] = str(profiling_root)
        project_config['profiling_backend_config'] = 'configs/pipeline_config.json'
        project_config['segmentation_backend_root'] = str(segmentation_root)
        project_config['segmentation_backend_config'] = 'configs/segmentation_config.json'
        project_config['workspace_root'] = str(project_root)
        project_config['default_output_root'] = str(output_root)
        project_config['deepprofiler_export_root'] = str(exports_root)
        project_config['deepprofiler_project_root'] = str(project_root / 'exports' / 'deepprofiler_project')

        project_config_path = project_root / 'project_config.json'
        project_config_path.write_text(json.dumps(project_config, indent=2) + '\n', encoding='utf-8')
        return project_config_path

    def _build_minimal_segmentation_project_config(self, tmpdir: str) -> Path:
        import pandas as pd

        tmpdir_path = Path(tmpdir)
        project_root = tmpdir_path / 'segmentation_fixture_project'
        profiling_root = project_root / 'profiling_backend_stub'
        segmentation_root = project_root / 'segmentation_backend'
        exports_root = project_root / 'exports' / 'deepprofiler'
        output_root = project_root / 'outputs'

        for directory in [profiling_root / 'configs', profiling_root / 'data' / 'metadata', profiling_root / 'outputs', segmentation_root / 'configs', segmentation_root / 'data', segmentation_root / 'outputs' / 'cellprofiler_masks' / 'labels', output_root, exports_root]:
            directory.mkdir(parents=True, exist_ok=True)

        source_pipeline_config = json.loads((PROJECT_ROOT / 'configs' / 'project_config.example.json').read_text(encoding='utf-8'))
        source_segmentation_root = Path(source_pipeline_config['segmentation_backend_root'])
        source_segmentation_config = json.loads((source_segmentation_root / 'configs' / 'segmentation_config.json').read_text(encoding='utf-8'))
        source_load_data = pd.read_csv(source_segmentation_root / 'data' / 'load_data_for_segmentation.csv')
        load_data_subset = source_load_data.head(1).copy()
        plate = str(load_data_subset.iloc[0]['Metadata_Plate'])
        well = str(load_data_subset.iloc[0]['Metadata_Well'])
        site = int(load_data_subset.iloc[0]['Metadata_Site'])

        source_mask_root = source_segmentation_root / 'outputs' / 'cellprofiler_masks'
        image_df = pd.read_csv(source_mask_root / 'Image.csv')
        image_subset = image_df[(image_df['Metadata_Plate'] == plate) & (image_df['Metadata_Well'] == well) & (image_df['Metadata_Site'] == site)].copy()
        image_number = int(image_subset.iloc[0]['ImageNumber'])
        cells_subset = pd.read_csv(source_mask_root / 'Cells.csv')
        cells_subset = cells_subset[cells_subset['ImageNumber'] == image_number].copy()
        nuclei_subset = pd.read_csv(source_mask_root / 'Nuclei.csv')
        nuclei_subset = nuclei_subset[nuclei_subset['ImageNumber'] == image_number].copy()
        cytoplasm_subset = pd.read_csv(source_mask_root / 'Cytoplasm.csv')
        cytoplasm_subset = cytoplasm_subset[cytoplasm_subset['ImageNumber'] == image_number].copy()
        experiment_df = pd.read_csv(source_mask_root / 'Experiment.csv')

        load_data_subset.to_csv(segmentation_root / 'data' / 'source_load_data.csv', index=False)
        load_data_subset.to_csv(segmentation_root / 'data' / 'load_data_for_segmentation.csv', index=False)
        image_subset.to_csv(segmentation_root / 'outputs' / 'cellprofiler_masks' / 'Image.csv', index=False)
        cells_subset.to_csv(segmentation_root / 'outputs' / 'cellprofiler_masks' / 'Cells.csv', index=False)
        nuclei_subset.to_csv(segmentation_root / 'outputs' / 'cellprofiler_masks' / 'Nuclei.csv', index=False)
        cytoplasm_subset.to_csv(segmentation_root / 'outputs' / 'cellprofiler_masks' / 'Cytoplasm.csv', index=False)
        experiment_df.to_csv(segmentation_root / 'outputs' / 'cellprofiler_masks' / 'Experiment.csv', index=False)

        for suffix in ['cell_labels', 'nuclei_labels']:
            source_label = source_mask_root / 'labels' / f'{plate}_{well}_s{site}--{suffix}.tiff'
            target_label = segmentation_root / 'outputs' / 'cellprofiler_masks' / 'labels' / source_label.name
            shutil.copy2(source_label, target_label)

        fixture_segmentation_config = dict(source_segmentation_config)
        fixture_segmentation_config['paths'] = dict(source_segmentation_config['paths'])
        fixture_segmentation_config['paths']['source_load_data_csv'] = 'data/source_load_data.csv'
        fixture_segmentation_config['paths']['load_data_csv'] = 'data/load_data_for_segmentation.csv'
        fixture_segmentation_config['paths']['base_pipeline'] = str(Path(source_pipeline_config['profiling_backend_root']) / 'cellprofiler' / 'CPJUMP1_analysis_without_batchfile_406.cppipe')
        fixture_segmentation_config['paths']['mask_export_pipeline'] = 'cellprofiler/CPJUMP1_analysis_mask_export.cppipe'
        fixture_segmentation_config['paths']['cellprofiler_output_dir'] = 'outputs/cellprofiler_masks'
        fixture_segmentation_config['paths']['sample_previews_dir'] = 'outputs/sample_previews_png'
        fixture_segmentation_config['paths']['masked_crops_dir'] = 'outputs/single_cell_crops_masked'
        fixture_segmentation_config['paths']['masked_manifest_csv'] = 'outputs/single_cell_crops_masked/single_cell_manifest.csv'
        fixture_segmentation_config['paths']['unmasked_crops_dir'] = 'outputs/single_cell_crops_unmasked'
        fixture_segmentation_config['paths']['unmasked_manifest_csv'] = 'outputs/single_cell_crops_unmasked/single_cell_manifest.csv'
        fixture_segmentation_config['subset'] = {'plates': [plate], 'wells': [well], 'sites': [site]}
        (segmentation_root / 'configs' / 'segmentation_config.json').write_text(json.dumps(fixture_segmentation_config, indent=2) + '\n', encoding='utf-8')

        profiling_stub_config = {
            'project_name': 'profiling_stub',
            'paths': {
                'raw_images_dir': str(Path(source_pipeline_config['profiling_backend_root']) / 'data' / 'raw_core'),
                'manifest_csv': 'data/metadata/image_manifest.csv',
                'plate_map_csv': 'data/metadata/plate_map.csv',
                'cellprofiler_output_dir': 'outputs/cellprofiler',
                'image_table_csv': 'outputs/cellprofiler/Image.csv',
                'cells_table_csv': 'outputs/cellprofiler/Cells.csv',
                'nuclei_table_csv': 'outputs/cellprofiler/Nuclei.csv',
                'cytoplasm_table_csv': 'outputs/cellprofiler/Cytoplasm.csv',
                'single_cell_output_csv_gz': 'outputs/pycytominer/single_cell.csv.gz',
                'aggregated_output_parquet': 'outputs/pycytominer/well_aggregated.parquet',
                'annotated_output_parquet': 'outputs/pycytominer/well_annotated.parquet',
                'normalized_output_parquet': 'outputs/pycytominer/well_normalized.parquet',
                'feature_selected_output_parquet': 'outputs/pycytominer/well_feature_selected.parquet',
            },
            'raw_images': {'layout': 'generic', 'valid_extensions': ['.tif'], 'filename_regex': '^$'},
            'metadata': {'required_manifest_columns': ['Metadata_Plate', 'Metadata_Well'], 'required_plate_map_columns': ['Metadata_Plate', 'Metadata_Well']},
            'cellprofiler': {'image_join_key': 'ImageNumber', 'default_object_table': 'Cells', 'metadata_columns_from_image': ['ImageNumber', 'Metadata_Plate', 'Metadata_Well', 'Metadata_Site']},
            'pycytominer': {'aggregate': {'strata': ['Metadata_Plate', 'Metadata_Well'], 'operation': 'median', 'compute_object_count': True, 'object_feature': 'Metadata_ObjectNumber'}, 'annotate': {'join_on_columns': ['Metadata_Plate', 'Metadata_Well']}, 'normalize': {'method': 'mad_robustize', 'samples': "Metadata_ControlType == 'negative_control'"}, 'feature_select': {'operations': ['variance_threshold'], 'na_cutoff': 0.05, 'corr_threshold': 0.9}},
        }
        (profiling_root / 'configs' / 'pipeline_config.json').write_text(json.dumps(profiling_stub_config, indent=2) + '\n', encoding='utf-8')
        (profiling_root / 'data' / 'metadata' / 'image_manifest.csv').write_text('Metadata_Plate,Metadata_Well\n', encoding='utf-8')
        (profiling_root / 'data' / 'metadata' / 'plate_map.csv').write_text('Metadata_Plate,Metadata_Well\n', encoding='utf-8')

        project_config = dict(source_pipeline_config)
        project_config['project_name'] = 'delivery_smoke_segmentation_fixture'
        project_config['profiling_backend_root'] = str(profiling_root)
        project_config['profiling_backend_config'] = 'configs/pipeline_config.json'
        project_config['segmentation_backend_root'] = str(segmentation_root)
        project_config['segmentation_backend_config'] = 'configs/segmentation_config.json'
        project_config['workspace_root'] = str(project_root)
        project_config['default_output_root'] = str(output_root)
        project_config['deepprofiler_export_root'] = str(exports_root)
        project_config['deepprofiler_project_root'] = str(project_root / 'exports' / 'deepprofiler_project')

        project_config_path = project_root / 'project_config.json'
        project_config_path.write_text(json.dumps(project_config, indent=2) + '\n', encoding='utf-8')
        return project_config_path

    def test_package_api_exports_high_level_entrypoints(self) -> None:
        self.assertTrue(callable(package_api.run_segmentation_bundle))
        self.assertTrue(callable(package_api.run_deepprofiler_full_stack))
        self.assertTrue(callable(package_api.run_full_pipeline))

    def test_available_suite_aliases(self) -> None:
        self.assertIn('native', available_profiling_suites())
        self.assertIn('mask-export', available_segmentation_suites())
        self.assertIn('deepprofiler-full', available_segmentation_suites())

    def test_available_workflow_aliases(self) -> None:
        self.assertIn('segmentation-and-deepprofiler-full-stack', available_workflows())

    def test_run_smoke_test_writes_report(self) -> None:
        config = ProjectConfig.from_json(CONFIG_PATH)
        with tempfile.TemporaryDirectory() as tmpdir:
            report_path = Path(tmpdir) / 'smoke_test_report.json'
            result = run_smoke_test(config, output_path=report_path)
            self.assertTrue(result.ok)
            self.assertTrue(report_path.exists())
            payload = json.loads(report_path.read_text(encoding='utf-8'))
            self.assertTrue(payload['ok'])
            self.assertTrue(payload['checks']['profiling_backend_root_exists'])
            self.assertTrue(payload['checks']['validation_report_ok'])

    def test_cli_smoke_test_command(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            report_path = Path(tmpdir) / 'cli_smoke_test_report.json'
            returncode = main([
                'smoke-test',
                '--config',
                str(CONFIG_PATH),
                '--output-path',
                str(report_path),
            ])
            self.assertEqual(returncode, 0)
            self.assertTrue(report_path.exists())

    def test_run_profiling_suite_real_minimal_fixture(self) -> None:
        config = None
        with tempfile.TemporaryDirectory() as tmpdir:
            config = ProjectConfig.from_json(self._build_minimal_profiling_project_config(tmpdir))
            output_dir = Path(tmpdir) / 'profiling_real_run'
            result = run_profiling_suite(config, 'native', output_dir=output_dir)
            self.assertEqual(result.suite_key, 'native')
            self.assertEqual(result.workflow_key, 'post-cellprofiler-native-profiling-with-native-eval')
            self.assertTrue(result.manifest_path.exists())
            payload = json.loads(result.manifest_path.read_text(encoding='utf-8'))
            self.assertEqual(payload['workflow_key'], 'post-cellprofiler-native-profiling-with-native-eval')
            self.assertEqual(len(payload['steps']), 4)
            self.assertEqual(payload['steps'][1]['label'], 'export_cellprofiler_to_singlecell')
            self.assertTrue((output_dir / 'single_cell.csv.gz').exists())
            self.assertTrue((output_dir / 'pycytominer' / 'well_feature_selected.parquet').exists())
            self.assertTrue((output_dir / 'evaluation' / 'run_manifest.json').exists())

    @patch('cellpaint_pipeline.delivery.run_workflow')
    def test_run_profiling_suite_wraps_workflow_error(self, run_workflow_mock) -> None:
        config = ProjectConfig.from_json(CONFIG_PATH)
        with tempfile.TemporaryDirectory() as tmpdir:
            output_dir = Path(tmpdir) / 'profiling_suite_failure'
            run_workflow_mock.side_effect = WorkflowExecutionError(
                'post-cellprofiler-native-profiling-with-native-eval',
                output_dir,
                step_label='validate_inputs',
                reason='Profiling input validation failed.',
            )
            with self.assertRaises(SuiteExecutionError) as ctx:
                run_profiling_suite(config, 'native', output_dir=output_dir)
        message = str(ctx.exception)
        self.assertIn("Profiling suite 'native' failed.", message)
        self.assertIn('Workflow: post-cellprofiler-native-profiling-with-native-eval', message)
        self.assertIn(f'Output dir: {output_dir}', message)
        self.assertIn('Profiling input validation failed.', message)

    def test_run_segmentation_suite_real_minimal_fixture(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            config = ProjectConfig.from_json(self._build_minimal_segmentation_project_config(tmpdir))
            output_dir = Path(tmpdir) / 'segmentation_real_run'
            result = run_segmentation_suite(config, 'native-post-cellprofiler', output_dir=output_dir)
            self.assertEqual(result.suite_key, 'native-post-cellprofiler')
            self.assertEqual(result.workflow_key, 'post-cellprofiler-native-segmentation-suite')
            self.assertTrue(result.manifest_path.exists())
            payload = json.loads(result.manifest_path.read_text(encoding='utf-8'))
            self.assertEqual(payload['workflow_key'], 'post-cellprofiler-native-segmentation-suite')
            self.assertEqual(len(payload['steps']), 8)
            self.assertEqual(payload['steps'][2]['label'], 'extract_single_cell_crops_masked')
            self.assertTrue((output_dir / 'segmentation_native' / 'sample_previews_png').exists())
            self.assertTrue((output_dir / 'segmentation_native' / 'masked' / 'single_cell_manifest.csv').exists())
            self.assertTrue((output_dir / 'segmentation_native' / 'unmasked' / 'single_cell_manifest.csv').exists())
            self.assertTrue((output_dir / 'segmentation_suite_summary.json').exists())

    @patch('cellpaint_pipeline.delivery.run_segmentation_suite')
    def test_run_segmentation_bundle_dispatches(self, run_segmentation_suite_mock) -> None:
        config = ProjectConfig.from_json(CONFIG_PATH)
        expected = SuiteRunResult(
            suite_key='mask-export',
            workflow_key='mask-export-script-with-native-postprocessing',
            output_dir=PROJECT_ROOT / 'outputs' / 'tmp_seg_bundle',
            manifest_path=PROJECT_ROOT / 'outputs' / 'tmp_seg_bundle' / 'workflow_manifest.json',
            step_count=5,
        )
        run_segmentation_suite_mock.return_value = expected
        result = run_segmentation_bundle(
            config,
            output_dir=expected.output_dir,
            extra_args=['--reuse-pipeline'],
        )
        self.assertEqual(result, expected)
        run_segmentation_suite_mock.assert_called_once_with(
            config,
            'mask-export',
            output_dir=expected.output_dir,
            extra_args=['--reuse-pipeline'],
        )

    @patch('cellpaint_pipeline.delivery.run_segmentation_suite')
    def test_run_deepprofiler_full_stack_dispatches(self, run_segmentation_suite_mock) -> None:
        config = ProjectConfig.from_json(CONFIG_PATH)
        expected = SuiteRunResult(
            suite_key='deepprofiler-full',
            workflow_key='segmentation-and-deepprofiler-full-stack',
            output_dir=PROJECT_ROOT / 'outputs' / 'tmp_dp_full',
            manifest_path=PROJECT_ROOT / 'outputs' / 'tmp_dp_full' / 'workflow_manifest.json',
            step_count=8,
        )
        run_segmentation_suite_mock.return_value = expected
        result = run_deepprofiler_full_stack(
            config,
            output_dir=expected.output_dir,
            extra_args=['--reuse-load-data'],
        )
        self.assertEqual(result, expected)
        run_segmentation_suite_mock.assert_called_once_with(
            config,
            'deepprofiler-full',
            output_dir=expected.output_dir,
            extra_args=['--reuse-load-data'],
        )

    @patch('cellpaint_pipeline.cli.run_segmentation_bundle')
    def test_cli_run_segmentation_bundle_command(self, run_segmentation_bundle_mock) -> None:
        output_dir = PROJECT_ROOT / 'outputs' / 'tests_cli_seg_bundle'
        run_segmentation_bundle_mock.return_value = SuiteRunResult(
            suite_key='mask-export',
            workflow_key='mask-export-script-with-native-postprocessing',
            output_dir=output_dir,
            manifest_path=output_dir / 'workflow_manifest.json',
            step_count=9,
        )
        returncode = main([
            'run-segmentation-bundle',
            '--config',
            str(CONFIG_PATH),
            '--output-dir',
            str(output_dir),
        ])
        self.assertEqual(returncode, 0)
        run_segmentation_bundle_mock.assert_called_once()

    @patch('cellpaint_pipeline.cli.run_deepprofiler_full_stack')
    def test_cli_run_deepprofiler_full_stack_command(self, run_deepprofiler_full_stack_mock) -> None:
        output_dir = PROJECT_ROOT / 'outputs' / 'tests_cli_dp_full'
        run_deepprofiler_full_stack_mock.return_value = SuiteRunResult(
            suite_key='deepprofiler-full',
            workflow_key='segmentation-and-deepprofiler-full-stack',
            output_dir=output_dir,
            manifest_path=output_dir / 'workflow_manifest.json',
            step_count=13,
        )
        returncode = main([
            'run-deepprofiler-full-stack',
            '--config',
            str(CONFIG_PATH),
            '--output-dir',
            str(output_dir),
        ])
        self.assertEqual(returncode, 0)
        run_deepprofiler_full_stack_mock.assert_called_once()

    def test_cli_deepprofiler_export_from_workflow_root(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            export_root = Path(tmpdir) / 'deepprofiler_export'
            returncode = main([
                'export-deepprofiler-input',
                '--config',
                str(CONFIG_PATH),
                '--workflow-root',
                str(WORKFLOW_ROOT),
                '--output-dir',
                str(export_root),
                '--source-label',
                'workflow-local-mask-export',
            ])
            self.assertEqual(returncode, 0)
            manifest_path = export_root / 'manifest.json'
            self.assertTrue(manifest_path.exists())
            payload = json.loads(manifest_path.read_text(encoding='utf-8'))
            self.assertEqual(payload['field_count'], 36)
            self.assertEqual(payload['location_file_count'], 36)
            self.assertEqual(payload['source_label'], 'workflow-local-mask-export')

    def test_cli_build_deepprofiler_project_from_workflow_root(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            project_root = Path(tmpdir) / 'deepprofiler_project'
            returncode = main([
                'build-deepprofiler-project',
                '--config',
                str(CONFIG_PATH),
                '--workflow-root',
                str(DEEPPROFILER_WORKFLOW_ROOT),
                '--output-dir',
                str(project_root),
            ])
            self.assertEqual(returncode, 0)
            manifest_path = project_root / 'project_manifest.json'
            metadata_path = project_root / 'inputs' / 'metadata' / 'index.csv'
            config_path = project_root / 'inputs' / 'config' / 'profile_config.json'
            self.assertTrue(manifest_path.exists())
            self.assertTrue(metadata_path.exists())
            self.assertTrue(config_path.exists())
            payload = json.loads(manifest_path.read_text(encoding='utf-8'))
            self.assertEqual(payload['field_count'], 36)
            self.assertEqual(payload['location_file_count'], 36)
            self.assertEqual(payload['experiment_name'], 'cell_painting_cnn')
            self.assertEqual(payload['checkpoint_filename'], 'Cell_Painting_CNN_v1.hdf5')
            config_payload = json.loads(config_path.read_text(encoding='utf-8'))
            self.assertEqual(config_payload['profile']['checkpoint'], 'Cell_Painting_CNN_v1.hdf5')
            self.assertEqual(config_payload['profile']['feature_layer'], 'block6a_activation')
            self.assertNotIn('use_pretrained_input_size', config_payload['profile'])
            self.assertEqual(config_payload['train']['model']['crop_generator'], 'crop_generator')

    def test_cli_collect_deepprofiler_features(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            output_dir = Path(tmpdir) / 'deepprofiler_collected'
            returncode = main([
                'collect-deepprofiler-features',
                '--config',
                str(CONFIG_PATH),
                '--project-root',
                str(DEEPPROFILER_PROJECT_ROOT),
                '--output-dir',
                str(output_dir),
            ])
            self.assertEqual(returncode, 0)
            manifest_path = output_dir / 'deepprofiler_feature_manifest.json'
            self.assertTrue(manifest_path.exists())
            payload = json.loads(manifest_path.read_text(encoding='utf-8'))
            self.assertEqual(payload['field_file_count'], 36)
            self.assertEqual(payload['cell_count'], 2774)
            self.assertEqual(payload['feature_count'], 672)
            self.assertEqual(payload['well_count'], 36)
            self.assertTrue((output_dir / 'deepprofiler_single_cell.parquet').exists())
            self.assertTrue((output_dir / 'deepprofiler_well_aggregated.parquet').exists())


if __name__ == '__main__':
    unittest.main()
