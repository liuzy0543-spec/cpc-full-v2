"""Print the effective configuration of one profile as a Markdown table.

The point is that the v2/v3 summary is *generated*, not maintained by hand: a
hand-written table drifts the moment a manifest changes, and a drifted table is
worse than none because it looks authoritative.

Usage:
    python dump_profile.py --profile v2
    python dump_profile.py --profile v3 --json
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / 'src'))

from cellpaint_pipeline import models  # noqa: E402
from cellpaint_pipeline.models import registry  # noqa: E402


def _row(manifest) -> dict:
    spec = manifest.preprocess
    return {
        'name': manifest.name,
        'version': manifest.version,
        'input_unit': manifest.input_unit.value,
        'encoding': spec.input_encoding,
        'layout': spec.batch_layout,
        'spatial': manifest.spatial,
        'dtype': spec.to_dtype,
        'scale': spec.scale,
        'source_map': spec.source_map,
        'mask_mode': spec.mask_mode,
        'resize_policy': spec.resize_policy,
        'value_range': spec.value_range,
        'max_channels': manifest.channels.max_channels,
        'output_dim': manifest.output_dim,
        'output_dim_per_channel': manifest.output_dim_per_channel,
        'channels': list(manifest.channels.names),
        'channel_count_fixed': manifest.channels.channel_count_fixed,
        'determinism': manifest.determinism,
        'provenance': manifest.weights_provenance,
        'requires': list(manifest.requires),
        'license': manifest.license,
        'sha256': (manifest.weights.sha256 or '')[:16],
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument('--profile', default='v2', choices=['v2', 'v3'])
    parser.add_argument('--json', action='store_true')
    args = parser.parse_args()

    registry.set_profile(args.profile)
    names = models.declared()
    rows = [_row(registry.manifest(n)) for n in names]

    if args.json:
        print(json.dumps({'profile': args.profile, 'methods': rows},
                         indent=2, ensure_ascii=False))
        return 0

    print(f'# profile `{args.profile}` — {len(rows)} method(s)\n')
    print('| method | provenance | width | input unit | spatial | dtype | scale '
          '| source_map | layout | margin |')
    print('|---|---|---|---|---|---|---|---|---|---|')
    for r in rows:
        # Only spell out the arithmetic when the declared width really is
        # per-channel × channel-count.  ca_mae also declares a per-channel width
        # but pools across channels, so its total stays 384 at any input channel
        # count -- printing "384×5=1920" there would be a plain falsehood.
        width = str(r['output_dim'])
        per_channel = r['output_dim_per_channel']
        if per_channel and r['output_dim'] == per_channel * len(r['channels']):
            width = f"{per_channel}×{len(r['channels'])}={r['output_dim']}"
        elif per_channel and r['output_dim'] != per_channel:
            width = f"{r['output_dim']}（逐通道 {per_channel}，跨通道池化）"
        prov = r['provenance']
        if prov == 'surrogate':
            prov = '**surrogate**'
        print(f"| `{r['name']}` | {prov} | {width} | {r['input_unit']} "
              f"| {r['spatial']} | {r['dtype']} | {r['scale']} "
              f"| {r['source_map']} | {r['layout']} | {r['mask_mode']} |")

    print('\n## 权重与依赖\n')
    print('| method | sha256(16) | license | determinism | requires |')
    print('|---|---|---|---|---|')
    for r in rows:
        print(f"| `{r['name']}` | `{r['sha256']}` | {r['license']} "
              f"| {r['determinism']} | {', '.join(r['requires'])} |")

    surrogates = [r['name'] for r in rows if r['provenance'] == 'surrogate']
    if surrogates:
        print(f"\n> ⚠️ 替身权重：{', '.join(surrogates)}。"
              f"这些数字不是论文性能，只能证明接口能驱动该方法。")
    return 0


if __name__ == '__main__':
    raise SystemExit(main())