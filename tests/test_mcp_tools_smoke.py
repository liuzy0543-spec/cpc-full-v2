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
from cellpaint_pipeline.mcp_tools import (
    available_mcp_tools,
    get_mcp_tool_definition,
    mcp_tool_definition_to_dict,
    run_mcp_tool_to_dict,
)
from cellpaint_pipeline.skills import PipelineSkillResult

CONFIG_PATH = PROJECT_ROOT / 'configs' / 'project_config.example.json'


class McpToolsSmokeTests(unittest.TestCase):
    def test_package_api_exports_mcp_helpers(self) -> None:
        self.assertTrue(callable(package_api.available_mcp_tools))
        self.assertTrue(callable(package_api.get_mcp_tool_definition))
        self.assertTrue(callable(package_api.run_mcp_tool))
        self.assertTrue(callable(package_api.run_mcp_tool_to_dict))

    def test_available_mcp_tools_contains_core_tools(self) -> None:
        tools = available_mcp_tools()
        self.assertIn('run_public_api_entrypoint', tools)
        self.assertIn('run_pipeline_skill', tools)
        self.assertIn('run_pipeline_preset', tools)

    def test_mcp_tool_definition_roundtrip(self) -> None:
        definition = get_mcp_tool_definition('run_pipeline_skill')
        payload = mcp_tool_definition_to_dict(definition)
        self.assertEqual(payload['target_type'], 'skill')
        self.assertTrue(payload['requires_config'])
        self.assertEqual(payload['cli_command'], 'run-mcp-tool')

    @patch('cellpaint_pipeline.mcp_tools.run_pipeline_skill')
    def test_run_mcp_tool_to_dict_dispatches_skill(self, run_pipeline_skill_mock) -> None:
        config = ProjectConfig.from_json(CONFIG_PATH)
        with tempfile.TemporaryDirectory() as tmpdir:
            output_dir = Path(tmpdir) / 'mcp_skill_run'
            run_pipeline_skill_mock.return_value = PipelineSkillResult(
                skill_key='run-segmentation-masks',
                category='segmentation',
                implementation='cellpaint_pipeline.workflows.segmentation',
                output_dir=output_dir,
                manifest_path=output_dir / 'pipeline_skill_manifest.json',
                primary_outputs={'cellprofiler_output_dir': output_dir / 'cellprofiler_masks'},
                details={},
                ok=True,
            )
            payload = run_mcp_tool_to_dict(
                'run_pipeline_skill',
                config=config,
                skill_key='run-segmentation-masks',
                output_dir=str(output_dir),
            )
            self.assertEqual(payload['tool'], 'run_pipeline_skill')
            self.assertEqual(payload['result']['output_dir'], str(output_dir))
            _, kwargs = run_pipeline_skill_mock.call_args
            self.assertEqual(kwargs['output_dir'], output_dir.resolve())

    def test_cli_list_mcp_tools(self) -> None:
        with patch('sys.stdout.write') as write_mock:
            returncode = main(['list-mcp-tools'])
        self.assertEqual(returncode, 0)
        rendered = ''.join(call.args[0] for call in write_mock.call_args_list)
        payload = json.loads(rendered)
        names = {item['name'] for item in payload}
        self.assertIn('run_pipeline_skill', names)
        self.assertIn('run_public_api_entrypoint', names)

    def test_cli_run_mcp_tool_for_public_api_dispatch(self) -> None:
        params = json.dumps({
            'entrypoint': 'build_data_request',
            'params': {
                'mode': 'gallery-prefix',
                'prefix': 'cpg0016-jump/source_4/workspace/',
                'dry_run': True,
            },
        })
        with patch('sys.stdout.write') as write_mock:
            returncode = main([
                'run-mcp-tool',
                '--tool',
                'run_public_api_entrypoint',
                '--params-json',
                params,
            ])
        self.assertEqual(returncode, 0)
        rendered = ''.join(call.args[0] for call in write_mock.call_args_list)
        payload = json.loads(rendered)
        self.assertEqual(payload['tool'], 'run_public_api_entrypoint')
        self.assertEqual(payload['result']['entrypoint'], 'build_data_request')
        self.assertEqual(payload['result']['prefix'], 'cpg0016-jump/source_4/workspace/')


if __name__ == '__main__':
    unittest.main()
