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
from cellpaint_pipeline.cli import main
from cellpaint_pipeline.config import ProjectConfig
from cellpaint_pipeline.orchestration import EndToEndPipelineResult
from cellpaint_pipeline.public_api import PublicApiContractError, run_public_api_entrypoint, run_public_api_entrypoint_to_dict

CONFIG_PATH = PROJECT_ROOT / 'configs' / 'project_config.example.json'


class PublicApiDispatchSmokeTests(unittest.TestCase):
    def test_package_api_exports_dispatch_helpers(self) -> None:
        self.assertTrue(callable(package_api.run_public_api_entrypoint))
        self.assertTrue(callable(package_api.run_public_api_entrypoint_to_dict))

    def test_run_public_api_entrypoint_build_data_request_without_config(self) -> None:
        result = run_public_api_entrypoint(
            'build_data_request',
            mode='gallery-prefix',
            prefix='cpg0016-jump/source_4/images/',
            dry_run=True,
        )
        self.assertEqual(result.mode, 'gallery-prefix')
        self.assertEqual(result.prefix, 'cpg0016-jump/source_4/images/')
        self.assertTrue(result.dry_run)

    @patch('cellpaint_pipeline.public_api.run_end_to_end_pipeline')
    def test_run_public_api_entrypoint_to_dict_for_orchestration(self, run_end_to_end_pipeline_mock) -> None:
        config = ProjectConfig.from_json(CONFIG_PATH)
        with tempfile.TemporaryDirectory() as tmpdir:
            output_dir = Path(tmpdir) / 'end_to_end'
            run_end_to_end_pipeline_mock.return_value = EndToEndPipelineResult(
                output_dir=output_dir,
                manifest_path=output_dir / 'end_to_end_pipeline_manifest.json',
                run_report_path=output_dir / 'run_report.md',
                data_access_summary_path=None,
                download_plan_path=None,
                download_execution_path=None,
                profiling_output_dir=None,
                profiling_manifest_path=None,
                segmentation_output_dir=None,
                segmentation_manifest_path=None,
                validation_report_path=None,
                profiling_suite='native',
                segmentation_suite='mask-export',
                deepprofiler_mode='off',
                stage_count=2,
                ok=True,
            )
            payload = run_public_api_entrypoint_to_dict(
                'run_end_to_end_pipeline',
                config=config,
                output_dir=output_dir,
            )
            self.assertEqual(payload['entrypoint'], 'run_end_to_end_pipeline')
            self.assertEqual(payload['contract']['layer'], 'orchestration')
            self.assertEqual(payload['output_contract']['name'], 'run_end_to_end_pipeline')
            self.assertEqual(payload['output_dir'], str(output_dir))


    def test_run_public_api_entrypoint_requires_config_error_is_clear(self) -> None:
        with self.assertRaises(PublicApiContractError) as ctx:
            run_public_api_entrypoint('run_end_to_end_pipeline', output_dir=Path('/tmp/out'))
        self.assertIn('requires a ProjectConfig', str(ctx.exception))

    def test_run_public_api_entrypoint_rejects_wrong_config_type(self) -> None:
        with self.assertRaises(PublicApiContractError) as ctx:
            run_public_api_entrypoint('run_end_to_end_pipeline', config='not-a-config')
        self.assertIn('expected config to be a ProjectConfig', str(ctx.exception))

    def test_run_public_api_entrypoint_unknown_name_is_clear(self) -> None:
        with self.assertRaises(PublicApiContractError) as ctx:
            run_public_api_entrypoint('not_real_entrypoint')
        self.assertIn('Unknown public API entrypoint', str(ctx.exception))

    def test_cli_list_public_api_entrypoints(self) -> None:
        with patch('sys.stdout.write') as write_mock:
            returncode = main(['list-public-api-entrypoints'])
        self.assertEqual(returncode, 0)
        rendered = ''.join(call.args[0] for call in write_mock.call_args_list)
        payload = json.loads(rendered)
        names = {item['name'] for item in payload}
        self.assertIn('run_deepprofiler_pipeline', names)

    def test_cli_run_public_api_entrypoint_for_build_data_request(self) -> None:
        params = json.dumps({
            'mode': 'gallery-prefix',
            'prefix': 'cpg0016-jump/source_4/workspace/',
            'dry_run': True,
        })
        with patch('sys.stdout.write') as write_mock:
            returncode = main([
                'run-public-api-entrypoint',
                '--entrypoint',
                'build_data_request',
                '--params-json',
                params,
            ])
        self.assertEqual(returncode, 0)
        rendered = ''.join(call.args[0] for call in write_mock.call_args_list)
        payload = json.loads(rendered)
        self.assertEqual(payload['entrypoint'], 'build_data_request')
        self.assertEqual(payload['mode'], 'gallery-prefix')
        self.assertEqual(payload['prefix'], 'cpg0016-jump/source_4/workspace/')


if __name__ == '__main__':
    unittest.main()
