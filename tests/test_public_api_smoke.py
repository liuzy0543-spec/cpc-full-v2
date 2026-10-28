from __future__ import annotations

import sys
import unittest
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / 'src'
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

import cellpaint_pipeline as package_api
from cellpaint_pipeline.public_api import (
    available_public_api_entrypoints,
    get_public_api_entrypoint,
    get_public_api_output_contract,
    public_api_contract_summary,
    public_api_entrypoint_to_dict,
    public_api_output_contract_summary,
    recommended_public_api_pathways,
)


class PublicApiSmokeTests(unittest.TestCase):
    def test_package_api_exports_public_api_helpers(self) -> None:
        self.assertTrue(callable(package_api.available_public_api_entrypoints))
        self.assertTrue(callable(package_api.get_public_api_entrypoint))
        self.assertTrue(callable(package_api.public_api_contract_summary))
        self.assertTrue(callable(package_api.public_api_output_contract_summary))
        self.assertTrue(callable(package_api.recommended_public_api_pathways))

    def test_available_public_api_contains_core_entrypoints(self) -> None:
        names = available_public_api_entrypoints()
        self.assertIn('run_end_to_end_pipeline', names)
        self.assertIn('run_pipeline_skill', names)
        self.assertIn('run_deepprofiler_pipeline', names)

    def test_public_api_entrypoint_roundtrip(self) -> None:
        entry = get_public_api_entrypoint('run_end_to_end_pipeline')
        payload = public_api_entrypoint_to_dict(entry)
        self.assertEqual(payload['layer'], 'orchestration')
        self.assertEqual(payload['category'], 'top-level')
        self.assertEqual(payload['cli_command'], 'run-end-to-end-pipeline')

    def test_public_api_contract_summary_groups_by_layer(self) -> None:
        summary = public_api_contract_summary()
        self.assertIn('orchestration', summary)
        self.assertIn('deepprofiler', summary)
        deepprofiler_names = {item['name'] for item in summary['deepprofiler']}
        self.assertIn('run_deepprofiler_pipeline', deepprofiler_names)

    def test_recommended_public_api_pathways_order(self) -> None:
        recommendations = recommended_public_api_pathways()
        self.assertEqual(recommendations[0]['name'], 'run_end_to_end_pipeline')
        self.assertEqual(recommendations[0]['priority'], 1)
        self.assertEqual(recommendations[1]['name'], 'run_pipeline_skill')
        self.assertEqual(recommendations[-1]['name'], 'run_deepprofiler_pipeline')

    def test_public_api_output_contract_for_end_to_end(self) -> None:
        contract = get_public_api_output_contract('run_end_to_end_pipeline')
        self.assertEqual(contract['output_kind'], 'run-artifacts')
        self.assertEqual(contract['primary_fields'][:3], ['ok', 'output_dir', 'manifest_path'])
        artifact_fields = {item['path_field'] for item in contract['primary_artifacts']}
        self.assertIn('run_report_path', artifact_fields)

    def test_public_api_output_contract_summary_contains_deepprofiler(self) -> None:
        summary = public_api_output_contract_summary()
        names = {item['name'] for item in summary}
        self.assertIn('run_deepprofiler_pipeline', names)


if __name__ == '__main__':
    unittest.main()
