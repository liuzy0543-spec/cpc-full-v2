from __future__ import annotations

import io
import sys
import unittest
from contextlib import redirect_stdout
from pathlib import Path
from unittest.mock import patch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / 'src'
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from cellpaint_pipeline.test_suites import (
    available_test_suites,
    get_test_suite_modules,
    main,
    test_suite_summary,
)


class TestSuitesSmokeTests(unittest.TestCase):
    def test_available_test_suites_contains_fast_and_extended(self) -> None:
        self.assertEqual(available_test_suites(), ['fast', 'extended'])

    def test_get_test_suite_modules_for_extended_includes_delivery(self) -> None:
        modules = get_test_suite_modules('extended')
        self.assertIn('tests.test_delivery_smoke', modules)
        self.assertIn('tests.test_orchestration_smoke', modules)

    def test_test_suite_summary_reports_module_counts(self) -> None:
        payload = test_suite_summary()
        names = {item['name'] for item in payload['suites']}
        self.assertEqual(names, {'fast', 'extended'})

    def test_main_list_renders_json(self) -> None:
        buffer = io.StringIO()
        with redirect_stdout(buffer):
            returncode = main(['--list'])
        self.assertEqual(returncode, 0)
        rendered = buffer.getvalue()
        self.assertIn('"fast"', rendered)
        self.assertIn('"extended"', rendered)

    @patch('cellpaint_pipeline.test_suites.run_test_suite')
    def test_main_dispatches_selected_suite(self, run_test_suite_mock) -> None:
        run_test_suite_mock.return_value = unittest.TestResult()
        returncode = main(['fast'])
        self.assertEqual(returncode, 0)
        run_test_suite_mock.assert_called_once_with('fast', verbosity=1)


if __name__ == '__main__':
    unittest.main()
