# profile `v2` — 4 method(s)

| method | provenance | width | input unit | spatial | dtype | scale | source_map | layout | margin |
|---|---|---|---|---|---|---|---|---|---|
| `deepprofiler` | official | 672 | single_cell_crop | (32, 32) | float32 | per_crop_minmax | per_crop_minmax_bhwc | BHWC | none |
| `ca_mae` | official | 384 | single_cell_crop | (256, 256) | uint8 | div255 | identity | BCHW | none |
| `cellclip` | official | 512 | channel_embeddings | None | float32 | none | None | BCHW | none |
| `o2vae` | official | 512 | single_cell_crop | (64, 64) | float32 | none | mask_to_silhouette | BCHW | none |

## 权重与依赖

| method | sha256(16) | license | determinism | requires |
|---|---|---|---|---|
| `deepprofiler` | `87b756f579336d60` | BSD-3-Clause (code) / CC-BY-4.0 (weights) | deterministic | tensorflow, efficientnet |
| `ca_mae` | `f6e5f1c97101331b` | Recursion Non-Commercial EULA | deterministic | torch, transformers, einops |
| `cellclip` | `66c2d32a44a3e7d6` | CC BY-NC-ND 4.0 | deterministic | torch |
| `o2vae` | `aa70cfa8b080c9d9` | MIT | deterministic | torch, e2cnn |
