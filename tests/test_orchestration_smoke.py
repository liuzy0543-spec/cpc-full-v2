from __future__ import annotations

import json
from types import SimpleNamespace
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
from cellpaint_pipeline.cli import main
from cellpaint_pipeline.config import ProjectConfig
from cellpaint_pipeline.data_access import build_data_request, build_download_plan, write_download_plan
from cellpaint_pipeline.delivery import SuiteRunResult
from cellpaint_pipeline.orchestration import EndToEndPipelineResult, resolve_segmentation_suite, run_end_to_end_pipeline
from cellpaint_pipeline.workflows.orchestration import WorkflowExecutionError, run_workflow

CONFIG_PATH = PROJECT_ROOT / 'configs' / 'project_config.example.json'


class OrchestrationSmokeTests(unittest.TestCase):
    def test_package_api_exports_orchestration_entrypoints(self) -> None:
        self.assertTrue(callable(package_api.run_end_to_end_pipeline))
        self.assertTrue(callable(package_api.available_deepprofiler_modes))
        self.assertTrue(callable(package_api.resolve_segmentation_suite))

    def test_resolve_segmentation_suite_for_deepprofiler_modes(self) -> None:
        self.assertEqual(resolve_segmentation_suite('mask-export', deepprofiler_mode='off'), 'mask-export')
        self.assertEqual(resolve_segmentation_suite('mask-export', deepprofiler_mode='export'), 'deepprofiler-export')
        self.assertEqual(resolve_segmentation_suite('mask-export', deepprofiler_mode='full'), 'deepprofiler-full')

    @patch('cellpaint_pipeline.orchestration.build_validation_report_payload')
    @patch('cellpaint_pipeline.orchestration.run_segmentation_suite')
    @patch('cellpaint_pipeline.orchestration.run_profiling_suite')
    def test_run_end_to_end_pipeline_dispatches(
        self,
        run_profiling_suite_mock,
        run_segmentation_suite_mock,
        build_validation_report_payload_mock,
    ) -> None:
        config = ProjectConfig.from_json(CONFIG_PATH)
        with tempfile.TemporaryDirectory() as tmpdir:
            output_dir = Path(tmpdir) / 'end_to_end'
            run_profiling_suite_mock.return_value = SuiteRunResult(
                suite_key='native',
                workflow_key='post-cellprofiler-native-profiling-with-native-eval',
                output_dir=output_dir / 'profiling',
                manifest_path=output_dir / 'profiling' / 'workflow_manifest.json',
                step_count=4,
            )
            run_segmentation_suite_mock.return_value = SuiteRunResult(
                suite_key='deepprofiler-full',
                workflow_key='segmentation-and-deepprofiler-full-stack',
                output_dir=output_dir / 'segmentation',
                manifest_path=output_dir / 'segmentation' / 'workflow_manifest.json',
                step_count=8,
            )
            build_validation_report_payload_mock.return_value = {
                'ok': True,
                'artifact_counts': {'total': 0, 'ok': 0, 'missing': 0, 'failed': 0, 'unknown': 0},
                'workflow_manifest_count': 0,
            }

            result = run_end_to_end_pipeline(
                config,
                output_dir=output_dir,
                deepprofiler_mode='full',
            )

            self.assertTrue(result.ok)
            self.assertEqual(result.segmentation_suite, 'deepprofiler-full')
            self.assertTrue(result.manifest_path.exists())
            self.assertTrue(result.run_report_path.exists())
            payload = json.loads(result.manifest_path.read_text(encoding='utf-8'))
            self.assertEqual(payload['segmentation_suite'], 'deepprofiler-full')
            self.assertEqual(payload['stage_count'], 3)
            self.assertIn('run_report_path', payload)
            run_profiling_suite_mock.assert_called_once_with(
                config,
                'native',
                output_dir=output_dir / 'profiling',
            )
            run_segmentation_suite_mock.assert_called_once_with(
                config,
                'deepprofiler-full',
                output_dir=output_dir / 'segmentation',
            )

    def test_run_workflow_reports_validation_context(self) -> None:
        config = ProjectConfig.from_json(CONFIG_PATH)
        fake_validation = SimpleNamespace(
            ok=False,
            manifest_path=Path('/tmp/fake_manifest.csv'),
            plate_map_path=Path('/tmp/fake_plate_map.csv'),
            problems=['No raw image files found under: /tmp/fake_raw'],
        )
        with tempfile.TemporaryDirectory() as tmpdir:
            workflow_root = Path(tmpdir) / 'workflow_validation_failure'
            with patch('cellpaint_pipeline.workflows.orchestration.run_profiling_native', return_value=fake_validation):
                with self.assertRaises(WorkflowExecutionError) as ctx:
                    run_workflow(
                        config,
                        'post-cellprofiler-native-profiling-with-native-eval',
                        export_output_dir=workflow_root,
                    )
        message = str(ctx.exception)
        self.assertIn("Workflow 'post-cellprofiler-native-profiling-with-native-eval' failed.", message)
        self.assertIn('Step: validate_inputs', message)
        self.assertIn('Manifest path: /tmp/fake_manifest.csv', message)
        self.assertIn('No raw image files found under: /tmp/fake_raw', message)

    def test_run_end_to_end_pipeline_real_lightweight_plan_path(self) -> None:
        config = ProjectConfig.from_json(CONFIG_PATH)
        with tempfile.TemporaryDirectory() as tmpdir:
            output_dir = Path(tmpdir) / 'real_end_to_end'
            request = build_data_request(mode='gallery-prefix', prefix='cpg0016-jump/source_4/workspace/', dry_run=True)
            plan = build_download_plan(config, request, validate_with_summary=False)

            self.assertTrue(plan.ok)
            self.assertEqual(plan.resolved_prefix, 'cpg0016-jump/source_4/workspace/')

            result = run_end_to_end_pipeline(
                config,
                output_dir=output_dir,
                download_plan=plan,
                run_profiling=False,
                run_segmentation=False,
                include_validation_report=False,
            )

            self.assertTrue(result.ok)
            self.assertEqual(result.stage_count, 1)
            self.assertTrue(result.manifest_path.exists())
            self.assertTrue(result.run_report_path.exists())
            self.assertTrue(result.download_plan_path.exists())
            self.assertIsNone(result.profiling_output_dir)
            self.assertIsNone(result.segmentation_output_dir)
            self.assertIsNone(result.validation_report_path)

            payload = json.loads(result.manifest_path.read_text(encoding='utf-8'))
            self.assertTrue(payload['ok'])
            self.assertEqual(payload['stage_count'], 1)
            self.assertEqual(payload['download_plan_path'], str(result.download_plan_path))
            self.assertIsNone(payload['profiling_output_dir'])
            self.assertIsNone(payload['segmentation_output_dir'])
            self.assertEqual(len(payload['stages']), 1)
            self.assertEqual(payload['stages'][0]['stage'], 'download_plan')
            self.assertEqual(payload['stages'][0]['resolved_prefix'], 'cpg0016-jump/source_4/workspace/')

            report_text = result.run_report_path.read_text(encoding='utf-8')
            self.assertIn('# End-to-End Pipeline Run Report', report_text)
            self.assertIn('- Overall ok: True', report_text)
            self.assertIn('- Download plan: ', report_text)

    @patch('cellpaint_pipeline.cli.run_end_to_end_pipeline')
    def test_cli_run_end_to_end_pipeline_command(self, run_end_to_end_pipeline_mock) -> None:
        output_dir = PROJECT_ROOT / 'outputs' / 'tests_cli_end_to_end'
        run_end_to_end_pipeline_mock.return_value = EndToEndPipelineResult(
            output_dir=output_dir,
            manifest_path=output_dir / 'end_to_end_pipeline_manifest.json',
            run_report_path=output_dir / 'run_report.md',
            data_access_summary_path=output_dir / 'data_access_summary.json',
            download_plan_path=None,
            download_execution_path=None,
            profiling_output_dir=output_dir / 'profiling',
            profiling_manifest_path=output_dir / 'profiling' / 'workflow_manifest.json',
            segmentation_output_dir=output_dir / 'segmentation',
            segmentation_manifest_path=output_dir / 'segmentation' / 'workflow_manifest.json',
            validation_report_path=output_dir / 'validation_report.json',
            profiling_suite='native',
            segmentation_suite='deepprofiler-export',
            deepprofiler_mode='export',
            stage_count=4,
            ok=True,
        )

        returncode = main([
            'run-end-to-end-pipeline',
            '--config',
            str(CONFIG_PATH),
            '--output-dir',
            str(output_dir),
            '--include-data-access-summary',
            '--deepprofiler-mode',
            'export',
        ])

        self.assertEqual(returncode, 0)
        run_end_to_end_pipeline_mock.assert_called_once()
        _, kwargs = run_end_to_end_pipeline_mock.call_args
        self.assertTrue(kwargs['include_data_access_summary'])
        self.assertEqual(kwargs['deepprofiler_mode'], 'export')
        self.assertIsNone(kwargs['download_plan'])

    @patch('cellpaint_pipeline.cli.run_end_to_end_pipeline')
    def test_cli_run_end_to_end_pipeline_with_plan_path(self, run_end_to_end_pipeline_mock) -> None:
        config = ProjectConfig.from_json(CONFIG_PATH)
        with tempfile.TemporaryDirectory() as tmpdir:
            tmpdir_path = Path(tmpdir)
            request = build_data_request(mode='gallery-prefix', prefix='cpg0016-jump/source_4/workspace/', dry_run=True)
            plan = build_download_plan(config, request, validate_with_summary=False)
            plan_path = write_download_plan(plan, tmpdir_path / 'download_plan.json')
            output_dir = tmpdir_path / 'cli_plan_end_to_end'
            run_end_to_end_pipeline_mock.return_value = EndToEndPipelineResult(
                output_dir=output_dir,
                manifest_path=output_dir / 'end_to_end_pipeline_manifest.json',
                run_report_path=output_dir / 'run_report.md',
                data_access_summary_path=None,
                download_plan_path=output_dir / 'download_plan.json',
                download_execution_path=None,
                profiling_output_dir=None,
                profiling_manifest_path=None,
                segmentation_output_dir=None,
                segmentation_manifest_path=None,
                validation_report_path=None,
                profiling_suite=None,
                segmentation_suite=None,
                deepprofiler_mode='off',
                stage_count=1,
                ok=True,
            )

            returncode = main([
                'run-end-to-end-pipeline',
                '--config',
                str(CONFIG_PATH),
                '--plan-path',
                str(plan_path),
                '--skip-profiling',
                '--skip-segmentation',
                '--skip-validation-report',
            ])

            self.assertEqual(returncode, 0)
            _, kwargs = run_end_to_end_pipeline_mock.call_args
            self.assertIsNotNone(kwargs['download_plan'])
            self.assertIsNone(kwargs['data_request'])
            self.assertEqual(kwargs['download_plan'].resolved_prefix, 'cpg0016-jump/source_4/workspace/')


if __name__ == '__main__':
    unittest.main()
