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

from cellpaint_pipeline.config import ProjectConfig
from cellpaint_pipeline.profile_summaries import (
    summarize_classical_profiles,
    summarize_deepprofiler_profiles,
)

CONFIG_PATH = PROJECT_ROOT / 'configs' / 'project_config.example.json'

try:
    import pandas as pd
except ImportError:  # pragma: no cover - environment dependent
    pd = None


@unittest.skipUnless(pd is not None, 'pandas is required for profile summary smoke tests')
class ProfileSummarySmokeTests(unittest.TestCase):
    def test_summarize_classical_profiles_writes_summary_artifacts(self) -> None:
        config = ProjectConfig.from_json(CONFIG_PATH)
        with tempfile.TemporaryDirectory() as tmpdir:
            tmpdir_path = Path(tmpdir)
            feature_selected_path = tmpdir_path / 'feature_selected.csv'
            pd.DataFrame([
                {
                    'Metadata_Plate': 'BR00000001',
                    'Metadata_Well': 'A01',
                    'Metadata_Treatment': 'dmso',
                    'Metadata_ControlType': 'negative_control',
                    'Cells_Feature_A': 0.1,
                    'Cells_Feature_B': 1.0,
                },
                {
                    'Metadata_Plate': 'BR00000001',
                    'Metadata_Well': 'A02',
                    'Metadata_Treatment': 'drug_x',
                    'Metadata_ControlType': 'treatment',
                    'Cells_Feature_A': 0.9,
                    'Cells_Feature_B': 2.0,
                },
                {
                    'Metadata_Plate': 'BR00000001',
                    'Metadata_Well': 'A03',
                    'Metadata_Treatment': 'drug_y',
                    'Metadata_ControlType': 'treatment',
                    'Cells_Feature_A': 1.2,
                    'Cells_Feature_B': 3.5,
                },
            ]).to_csv(feature_selected_path, index=False)

            result = summarize_classical_profiles(
                config,
                output_dir=tmpdir_path / 'classical_summary',
                feature_selected_path=feature_selected_path,
            )

            self.assertEqual(result.row_count, 3)
            self.assertEqual(result.feature_count, 2)
            self.assertTrue(result.summary_path.exists())
            self.assertTrue(result.top_variable_features_path.exists())
            summary_payload = json.loads(result.summary_path.read_text(encoding='utf-8'))
            self.assertEqual(summary_payload['row_count'], 3)

    def test_summarize_deepprofiler_profiles_writes_summary_artifacts(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            tmpdir_path = Path(tmpdir)
            single_cell_path = tmpdir_path / 'deepprofiler_single_cell.csv'
            well_aggregated_path = tmpdir_path / 'deepprofiler_well_aggregated.csv'

            pd.DataFrame([
                {
                    'Metadata_Plate': 'BR00000001',
                    'Metadata_Well': 'A01',
                    'Metadata_Site': '1',
                    'Cells_DeepProfiler_0001': 0.2,
                    'Cells_DeepProfiler_0002': 0.4,
                },
                {
                    'Metadata_Plate': 'BR00000001',
                    'Metadata_Well': 'A01',
                    'Metadata_Site': '1',
                    'Cells_DeepProfiler_0001': 0.3,
                    'Cells_DeepProfiler_0002': 0.5,
                },
                {
                    'Metadata_Plate': 'BR00000001',
                    'Metadata_Well': 'A02',
                    'Metadata_Site': '1',
                    'Cells_DeepProfiler_0001': 1.1,
                    'Cells_DeepProfiler_0002': 1.4,
                },
            ]).to_csv(single_cell_path, index=False)

            pd.DataFrame([
                {
                    'Metadata_Plate': 'BR00000001',
                    'Metadata_Well': 'A01',
                    'Metadata_DeepProfiler_Experiment': 'demo',
                    'Cells_DeepProfiler_0001': 0.25,
                    'Cells_DeepProfiler_0002': 0.45,
                },
                {
                    'Metadata_Plate': 'BR00000001',
                    'Metadata_Well': 'A02',
                    'Metadata_DeepProfiler_Experiment': 'demo',
                    'Cells_DeepProfiler_0001': 1.1,
                    'Cells_DeepProfiler_0002': 1.4,
                },
            ]).to_csv(well_aggregated_path, index=False)

            result = summarize_deepprofiler_profiles(
                output_dir=tmpdir_path / 'deepprofiler_summary',
                single_cell_parquet_path=single_cell_path,
                well_aggregated_parquet_path=well_aggregated_path,
            )

            self.assertEqual(result.cell_count, 3)
            self.assertEqual(result.well_count, 2)
            self.assertEqual(result.feature_count, 2)
            self.assertTrue(result.summary_path.exists())
            summary_payload = json.loads(result.summary_path.read_text(encoding='utf-8'))
            self.assertEqual(summary_payload['well_count'], 2)


if __name__ == '__main__':
    unittest.main()
