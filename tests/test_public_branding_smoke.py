from __future__ import annotations

import json
import sys
import unittest
from pathlib import Path
from unittest.mock import patch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / 'src'
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

import cellpainting_claw as claw
import cellpainting_skills as skills
from cellpainting_claw.cli import main as claw_main
from cellpainting_skills.cli import main as skills_main


class PublicBrandingSmokeTests(unittest.TestCase):
    def test_cellpainting_claw_exports_core_api(self) -> None:
        self.assertTrue(callable(claw.ProjectConfig))
        self.assertTrue(callable(claw.run_end_to_end_pipeline))
        self.assertTrue(callable(claw.run_pipeline_skill))

    def test_cellpainting_skills_exports_skill_api(self) -> None:
        self.assertTrue(callable(skills.available_pipeline_skills))
        self.assertTrue(callable(skills.get_pipeline_skill_definition))
        self.assertTrue(callable(skills.run_pipeline_skill))

    def test_cellpainting_claw_cli_help_uses_public_prog(self) -> None:
        with self.assertRaises(SystemExit) as raised:
            claw_main(['--help'])
        self.assertEqual(raised.exception.code, 0)

    def test_cellpainting_skills_list_outputs_skill_catalog(self) -> None:
        with patch('sys.stdout.write') as write_mock:
            returncode = skills_main(['list'])
        self.assertEqual(returncode, 0)
        rendered = ''.join(call.args[0] for call in write_mock.call_args_list)
        payload = json.loads(rendered)
        keys = {item['key'] for item in payload}
        self.assertIn('run-segmentation-masks', keys)


if __name__ == '__main__':
    unittest.main()
