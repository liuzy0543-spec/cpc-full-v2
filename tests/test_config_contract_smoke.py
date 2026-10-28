from __future__ import annotations

import json
import sys
import tempfile
import unittest
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / 'src'
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

import cellpaint_pipeline as package_api
from cellpaint_pipeline.config import (
    ConfigContractError,
    data_access_config_field_guide,
    project_config_contract_summary,
    project_config_field_guide,
    ProjectConfig,
)


class ConfigContractSmokeTests(unittest.TestCase):
    def test_package_api_exports_config_guides(self) -> None:
        self.assertTrue(callable(package_api.project_config_field_guide))
        self.assertTrue(callable(package_api.data_access_config_field_guide))
        self.assertTrue(callable(package_api.project_config_contract_summary))

    def test_project_config_field_guide_contains_required_roots(self) -> None:
        fields = {item['name']: item for item in project_config_field_guide()}
        self.assertEqual(fields['project_name']['level'], 'required')
        self.assertEqual(fields['workspace_root']['level'], 'required')
        self.assertEqual(fields['python_executable']['level'], 'recommended')

    def test_data_access_guide_contains_gallery_defaults(self) -> None:
        fields = {item['name']: item for item in data_access_config_field_guide()}
        self.assertEqual(fields['gallery_bucket']['default'], 'cellpainting-gallery')
        self.assertEqual(fields['use_unsigned']['default'], True)

    def test_project_config_contract_summary_lists_required_fields(self) -> None:
        summary = project_config_contract_summary()
        self.assertIn('project_name', summary['required_project_fields'])
        self.assertIn('deepprofiler_export_root', summary['required_project_fields'])
        self.assertIn('data_access', summary['recommended_project_fields'])

    def test_project_config_from_json_missing_required_field(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            config_path = Path(tmpdir) / 'bad_config.json'
            config_path.write_text(json.dumps({
                'project_name': 'bad',
                'profiling_backend_root': '/tmp/a',
                'segmentation_backend_root': '/tmp/b',
                'default_output_root': '/tmp/out',
                'deepprofiler_export_root': '/tmp/export',
            }), encoding='utf-8')
            with self.assertRaises(ConfigContractError) as ctx:
                ProjectConfig.from_json(config_path)
        self.assertIn('workspace_root', str(ctx.exception))

    def test_project_config_from_json_reports_missing_backend_config(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            tmp = Path(tmpdir)
            profiling_root = tmp / 'profiling'
            segmentation_root = tmp / 'segmentation'
            workspace_root = tmp / 'workspace'
            profiling_root.mkdir()
            segmentation_root.mkdir()
            workspace_root.mkdir()
            config_path = tmp / 'bad_paths.json'
            config_path.write_text(json.dumps({
                'project_name': 'bad-paths',
                'profiling_backend_root': str(profiling_root),
                'segmentation_backend_root': str(segmentation_root),
                'workspace_root': str(workspace_root),
                'default_output_root': str(tmp / 'outputs'),
                'deepprofiler_export_root': str(tmp / 'exports'),
            }), encoding='utf-8')
            with self.assertRaises(ConfigContractError) as ctx:
                ProjectConfig.from_json(config_path)
        self.assertIn('profiling_backend_config', str(ctx.exception))


if __name__ == '__main__':
    unittest.main()
