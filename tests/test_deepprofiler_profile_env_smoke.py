from __future__ import annotations

import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / 'src'
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from cellpaint_pipeline.adapters.deepprofiler_project import run_deepprofiler_profile
from cellpaint_pipeline.config import ProjectConfig
from cellpaint_pipeline.runner import ExecutionResult

CONFIG_PATH = PROJECT_ROOT / 'configs' / 'project_config.example.json'


class DeepProfilerProfileEnvSmokeTests(unittest.TestCase):
    @patch('cellpaint_pipeline.adapters.deepprofiler_project.run_command')
    @patch('cellpaint_pipeline.adapters.deepprofiler_project._stage_deepprofiler_checkpoint')
    @patch('cellpaint_pipeline.adapters.deepprofiler_project.importlib.util.find_spec')
    def test_run_deepprofiler_profile_injects_pythonpath_for_module_mode(
        self,
        find_spec_mock,
        stage_checkpoint_mock,
        run_command_mock,
    ) -> None:
        config = ProjectConfig.from_json(CONFIG_PATH)
        find_spec_mock.return_value = SimpleNamespace(origin='/workspace/DeepProfiler/deepprofiler/__init__.py')
        with tempfile.TemporaryDirectory() as tmpdir:
            project_root = Path(tmpdir) / 'dp_project'
            (project_root / 'inputs' / 'config').mkdir(parents=True)
            (project_root / 'inputs' / 'metadata').mkdir(parents=True)
            stage_checkpoint_mock.return_value = project_root / 'outputs' / 'cell_painting_cnn' / 'checkpoint' / 'Cell_Painting_CNN_v1.hdf5'
            run_command_mock.return_value = ExecutionResult(
                label='deepprofiler_profile',
                command=[str(Path(config.python_executable).resolve()), '-m', 'deepprofiler', 'profile'],
                cwd=project_root,
                log_path=project_root / 'log.txt',
                returncode=0,
            )

            result = run_deepprofiler_profile(config, project_root=project_root)

            self.assertEqual(result.returncode, 0)
            _, kwargs = run_command_mock.call_args
            self.assertIn('env', kwargs)
            self.assertEqual(kwargs['env']['PYTHONPATH'], '/workspace/DeepProfiler')
