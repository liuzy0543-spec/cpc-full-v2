from __future__ import annotations

import json
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / 'src'
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

import cellpaint_pipeline as package_api
from cellpaint_pipeline.adapters.deepprofiler import DeepProfilerExportResult
from cellpaint_pipeline.adapters.deepprofiler_features import DeepProfilerFeatureCollectionResult
from cellpaint_pipeline.adapters.deepprofiler_project import DeepProfilerProfileResult, DeepProfilerProjectResult
from cellpaint_pipeline.cli import main
from cellpaint_pipeline.config import ProjectConfig
from cellpaint_pipeline.deepprofiler_pipeline import DeepProfilerPipelineResult, run_deepprofiler_pipeline

CONFIG_PATH = PROJECT_ROOT / 'configs' / 'project_config.example.json'
WORKFLOW_ROOT = PROJECT_ROOT / 'outputs' / 'workflows' / 'mask_export_native_postprocessing_runtimecfg'


class DeepProfilerPipelineSmokeTests(unittest.TestCase):
    def test_package_api_exports_deepprofiler_pipeline(self) -> None:
        self.assertTrue(callable(package_api.run_deepprofiler_pipeline))
        self.assertTrue(callable(package_api.deepprofiler_pipeline_result_to_dict))

    @patch('cellpaint_pipeline.deepprofiler_pipeline.collect_deepprofiler_features')
    @patch('cellpaint_pipeline.deepprofiler_pipeline.run_deepprofiler_profile')
    @patch('cellpaint_pipeline.deepprofiler_pipeline.build_deepprofiler_project')
    @patch('cellpaint_pipeline.deepprofiler_pipeline.export_deepprofiler_input')
    @patch('cellpaint_pipeline.deepprofiler_pipeline.infer_deepprofiler_sources_from_workflow_root')
    def test_run_deepprofiler_pipeline_from_workflow_root_dispatches(
        self,
        infer_sources_mock,
        export_mock,
        build_project_mock,
        run_profile_mock,
        collect_mock,
    ) -> None:
        config = ProjectConfig.from_json(CONFIG_PATH)
        with tempfile.TemporaryDirectory() as tmpdir:
            tmpdir_path = Path(tmpdir)
            output_dir = tmpdir_path / 'deepprofiler_pipeline'
            infer_sources_mock.return_value = {
                'image_csv_path': tmpdir_path / 'Image.csv',
                'nuclei_csv_path': tmpdir_path / 'Nuclei.csv',
                'load_data_csv_path': tmpdir_path / 'load_data_for_segmentation.csv',
            }
            export_mock.return_value = DeepProfilerExportResult(
                export_root=output_dir / 'deepprofiler_export',
                manifest_path=output_dir / 'deepprofiler_export' / 'manifest.json',
                field_metadata_path=output_dir / 'deepprofiler_export' / 'images' / 'field_metadata.csv',
                locations_root=output_dir / 'deepprofiler_export' / 'locations',
                field_count=36,
                location_file_count=36,
                total_nuclei=2774,
                source_image_csv=tmpdir_path / 'Image.csv',
                source_nuclei_csv=tmpdir_path / 'Nuclei.csv',
                source_load_data_csv=tmpdir_path / 'load_data_for_segmentation.csv',
                source_label='workflow-local-mask-export',
            )
            build_project_mock.return_value = DeepProfilerProjectResult(
                project_root=output_dir / 'deepprofiler_project',
                manifest_path=output_dir / 'deepprofiler_project' / 'project_manifest.json',
                config_path=output_dir / 'deepprofiler_project' / 'inputs' / 'config' / 'profile_config.json',
                metadata_path=output_dir / 'deepprofiler_project' / 'inputs' / 'metadata' / 'index.csv',
                locations_root=output_dir / 'deepprofiler_project' / 'inputs' / 'locations',
                field_count=36,
                location_file_count=36,
                image_width=1080,
                image_height=1080,
                image_bits=16,
                image_format='tiff',
                experiment_name='cell_painting_cnn',
                label_field='Metadata_Well',
                control_value='A01',
                export_root=output_dir / 'deepprofiler_export',
                source_label='workflow-local-mask-export',
                checkpoint_filename='Cell_Painting_CNN_v1.hdf5',
            )
            run_profile_mock.return_value = DeepProfilerProfileResult(
                project_root=output_dir / 'deepprofiler_project',
                manifest_path=output_dir / 'deepprofiler_project' / 'project_manifest.json',
                config_path=output_dir / 'deepprofiler_project' / 'inputs' / 'config' / 'profile_config.json',
                metadata_path=output_dir / 'deepprofiler_project' / 'inputs' / 'metadata' / 'index.csv',
                experiment_name='cell_painting_cnn',
                feature_dir=output_dir / 'deepprofiler_project' / 'outputs' / 'cell_painting_cnn' / 'features',
                checkpoint_dir=output_dir / 'deepprofiler_project' / 'outputs' / 'cell_painting_cnn' / 'checkpoint',
                checkpoint_path=output_dir / 'deepprofiler_project' / 'outputs' / 'cell_painting_cnn' / 'checkpoint' / 'Cell_Painting_CNN_v1.hdf5',
                log_path=output_dir / 'logs' / 'deepprofiler.log',
                command=['deepprofiler', 'profile'],
                returncode=0,
            )
            collect_mock.return_value = DeepProfilerFeatureCollectionResult(
                project_root=output_dir / 'deepprofiler_project',
                feature_dir=output_dir / 'deepprofiler_project' / 'outputs' / 'cell_painting_cnn' / 'features',
                output_dir=output_dir / 'deepprofiler_tables',
                manifest_path=output_dir / 'deepprofiler_tables' / 'deepprofiler_feature_manifest.json',
                single_cell_parquet_path=output_dir / 'deepprofiler_tables' / 'deepprofiler_single_cell.parquet',
                single_cell_csv_gz_path=output_dir / 'deepprofiler_tables' / 'deepprofiler_single_cell.csv.gz',
                well_aggregated_parquet_path=output_dir / 'deepprofiler_tables' / 'deepprofiler_well_aggregated.parquet',
                well_aggregated_csv_gz_path=output_dir / 'deepprofiler_tables' / 'deepprofiler_well_aggregated.csv.gz',
                field_summary_path=output_dir / 'deepprofiler_tables' / 'deepprofiler_field_summary.csv',
                experiment_name='cell_painting_cnn',
                field_file_count=36,
                cell_count=2774,
                feature_count=672,
                metadata_column_count=69,
                well_count=36,
            )

            result = run_deepprofiler_pipeline(
                config,
                workflow_root=WORKFLOW_ROOT,
                output_dir=output_dir,
            )

            self.assertTrue(result.ok)
            self.assertTrue(result.manifest_path.exists())
            payload = json.loads(result.manifest_path.read_text(encoding='utf-8'))
            self.assertEqual(payload['feature_count'], 672)
            infer_sources_mock.assert_called_once_with(WORKFLOW_ROOT.resolve())
            export_mock.assert_called_once()
            _, export_kwargs = export_mock.call_args
            self.assertEqual(export_kwargs['output_dir'], output_dir / 'deepprofiler_export')
            self.assertEqual(export_kwargs['source_label'], 'workflow-local-mask-export')
            build_project_mock.assert_called_once()
            run_profile_mock.assert_called_once()
            collect_mock.assert_called_once()

    @patch('cellpaint_pipeline.cli.run_deepprofiler_pipeline')
    def test_cli_run_deepprofiler_pipeline_command(self, run_pipeline_mock) -> None:
        output_dir = PROJECT_ROOT / 'outputs' / 'tests_cli_dp_pipeline'
        run_pipeline_mock.return_value = DeepProfilerPipelineResult(
            output_dir=output_dir,
            manifest_path=output_dir / 'deepprofiler_pipeline_manifest.json',
            export_root=output_dir / 'deepprofiler_export',
            export_manifest_path=output_dir / 'deepprofiler_export' / 'manifest.json',
            project_root=output_dir / 'deepprofiler_project',
            project_manifest_path=output_dir / 'deepprofiler_project' / 'project_manifest.json',
            feature_dir=output_dir / 'deepprofiler_project' / 'outputs' / 'cell_painting_cnn' / 'features',
            collection_output_dir=output_dir / 'deepprofiler_tables',
            collection_manifest_path=output_dir / 'deepprofiler_tables' / 'deepprofiler_feature_manifest.json',
            experiment_name='cell_painting_cnn',
            source_label='workflow-local-mask-export',
            field_count=36,
            location_file_count=36,
            total_nuclei=2774,
            field_file_count=36,
            cell_count=2774,
            feature_count=672,
            well_count=36,
            profile_returncode=0,
            log_path=output_dir / 'logs' / 'deepprofiler.log',
            ok=True,
        )
        returncode = main([
            'run-deepprofiler-pipeline',
            '--config',
            str(CONFIG_PATH),
            '--workflow-root',
            str(WORKFLOW_ROOT),
            '--output-dir',
            str(output_dir),
        ])
        self.assertEqual(returncode, 0)
        run_pipeline_mock.assert_called_once()
        _, kwargs = run_pipeline_mock.call_args
        self.assertEqual(kwargs['workflow_root'], WORKFLOW_ROOT.resolve())
        self.assertEqual(kwargs['output_dir'], output_dir.resolve())
