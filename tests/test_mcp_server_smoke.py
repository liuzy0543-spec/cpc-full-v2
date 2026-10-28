from __future__ import annotations

import json
import sys
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / 'src'
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from cellpaint_pipeline.cli import main
from cellpaint_pipeline.mcp_server import _apply_server_settings, _parse_params_json, run_mcp_server


class McpServerSmokeTests(unittest.TestCase):
    def test_parse_params_json_requires_object(self) -> None:
        payload = _parse_params_json(json.dumps({'a': 1}))
        self.assertEqual(payload['a'], 1)
        with self.assertRaises(ValueError):
            _parse_params_json('[]')

    def test_apply_server_settings_updates_proxy(self) -> None:
        fake_server = SimpleNamespace(settings=SimpleNamespace())
        _apply_server_settings(fake_server, host='127.0.0.1', port=9000, path='/mcp')
        self.assertEqual(fake_server.settings.host, '127.0.0.1')
        self.assertEqual(fake_server.settings.port, 9000)
        self.assertEqual(fake_server.settings.streamable_http_path, '/mcp')

    @patch('cellpaint_pipeline.mcp_server.create_mcp_server')
    def test_run_mcp_server_stdio_uses_default_run(self, create_mcp_server_mock) -> None:
        fake_server = SimpleNamespace(run=lambda *args, **kwargs: None, settings=SimpleNamespace())
        create_mcp_server_mock.return_value = fake_server
        run_mcp_server(transport='stdio')
        create_mcp_server_mock.assert_called_once()

    @patch('cellpaint_pipeline.cli.run_mcp_server')
    def test_cli_serve_mcp_dispatches(self, run_mcp_server_mock) -> None:
        returncode = main(['serve-mcp', '--transport', 'stdio'])
        self.assertEqual(returncode, 0)
        _, kwargs = run_mcp_server_mock.call_args
        self.assertEqual(kwargs['transport'], 'stdio')

    @patch('cellpaint_pipeline.cli.run_mcp_server', side_effect=RuntimeError('missing mcp'))
    @patch('sys.stdout.write')
    def test_cli_serve_mcp_reports_runtime_error(self, write_mock, _run_mcp_server_mock) -> None:
        returncode = main(['serve-mcp', '--transport', 'stdio'])
        self.assertEqual(returncode, 1)
        rendered = ''.join(call.args[0] for call in write_mock.call_args_list)
        self.assertIn('missing mcp', rendered)


if __name__ == '__main__':
    unittest.main()
