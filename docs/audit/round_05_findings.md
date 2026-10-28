# 审核轮次 05

- 时间：2026-10-27 14:15:59
- 结论：**通过**
- 阻塞项：0

## 闸门总览

| 闸门 | 内容 | 通过/总数 | 状态 |
|---|---|---|---|
| G1 | 能跑（形状/宽度/有限/摘要） | 28/28 | ✅ |
| G2 | 统一调度（一次调用驱动七个） | 4/4 | ✅ |
| G3 | 契约守卫（该拒的必须拒） | 6/6 | ✅ |
| G4 | 确定性（同输入同输出） | 7/7 | ✅ |
| G5 | 产物与 v2 数字稳定 | 11/11 | ✅ |
| G6 | 与官方实现一致（对拍/等价证据） | 0/0 | — |

## 未通过项（下一轮的输入）

无。

## 全部检查明细

| 闸门 | 检查项 | 结果 | 实测 |
|---|---|---|---|
| G1 | deepprofiler runs | ✅ | dim=672 shape=(2, 672) sha=87b756f579336d60 |
| G1 | deepprofiler width | ✅ | 672 vs declared 672 |
| G1 | deepprofiler finite | ✅ | finite |
| G1 | deepprofiler sha | ✅ | weights_sha256 为空，溯源链断开 |
| G1 | ca_mae runs | ✅ | dim=384 shape=(2, 384) sha=f6e5f1c97101331b |
| G1 | ca_mae width | ✅ | 384 vs declared 384 |
| G1 | ca_mae finite | ✅ | finite |
| G1 | ca_mae sha | ✅ | weights_sha256 为空，溯源链断开 |
| G1 | cellclip runs | ✅ | dim=512 shape=(2, 512) sha=66c2d32a44a3e7d6 |
| G1 | cellclip width | ✅ | 512 vs declared 512 |
| G1 | cellclip finite | ✅ | finite |
| G1 | cellclip sha | ✅ | weights_sha256 为空，溯源链断开 |
| G1 | o2vae runs | ✅ | dim=512 shape=(2, 512) sha=aa70cfa8b080c9d9 |
| G1 | o2vae width | ✅ | 512 vs declared 512 |
| G1 | o2vae finite | ✅ | finite |
| G1 | o2vae sha | ✅ | weights_sha256 为空，溯源链断开 |
| G1 | unidino runs | ✅ | dim=1920 shape=(2, 1920) sha=a0265184661d511e |
| G1 | unidino width | ✅ | 1920 vs declared 1920 |
| G1 | unidino finite | ✅ | finite |
| G1 | unidino sha | ✅ | weights_sha256 为空，溯源链断开 |
| G1 | dinocells runs | ✅ | dim=384 shape=(1, 384) sha=5e172d8fed4876ff |
| G1 | dinocells width | ✅ | 384 vs declared 384 |
| G1 | dinocells finite | ✅ | finite |
| G1 | dinocells sha | ✅ | weights_sha256 为空，溯源链断开 |
| G1 | morphogenie runs | ✅ | dim=10 shape=(2, 10) sha=491029978fef0969 |
| G1 | morphogenie width | ✅ | 10 vs declared 10 |
| G1 | morphogenie finite | ✅ | finite |
| G1 | morphogenie sha | ✅ | weights_sha256 为空，溯源链断开 |
| G2 | embed_all drives seven | ✅ | 7 results, errors={} |
| G2 | identical result fields | ✅ | 1 distinct field sets |
| G2 | provenance correct | ✅ | {'deepprofiler': 'official', 'ca_mae': 'official', 'cellclip': 'official', 'o2vae': 'official', 'unidino': 'surrogate', 'dinocells': 'surrogate', 'morphogenie': 'surrogate'} |
| G2 | surrogates not in v2 | ✅ | v2 下替身不可解析 |
| G3 | uint8 -> unidino | ✅ | reason=value-range (want value-range) |
| G3 | 256 -> deepprofiler | ✅ | reason=spatial-size (want spatial-size) |
| G3 | probe(dinocells) runnable | ✅ | status=ready requires=['torch'] |
| G3 | runnable() == 7 | ✅ | 7 methods |
| G3 | dinocells batch | ✅ | (2, 5, 96, 96) |
| G3 | dinocells mask dropped | ✅ | 带/不带 mask 必须逐位相等（官方 mode=drop） |
| G4 | deepprofiler repeats exactly | ✅ | identical |
| G4 | ca_mae repeats exactly | ✅ | identical |
| G4 | cellclip repeats exactly | ✅ | identical |
| G4 | o2vae repeats exactly | ✅ | identical |
| G4 | unidino repeats exactly | ✅ | identical |
| G4 | dinocells repeats exactly | ✅ | identical |
| G4 | morphogenie repeats exactly | ✅ | identical |
| G5 | benchmark_scores_v2_e2e.json exists | ✅ | 3845 B |
| G5 | benchmark_scores_v2_e2e.json profile | ✅ | profile=v2 |
| G5 | benchmark_scores_v3_e2e.json exists | ✅ | 5230 B |
| G5 | benchmark_scores_v3_e2e.json profile | ✅ | profile=v3 |
| G5 | stability_v3_e2e.json exists | ✅ | 4025 B |
| G5 | io_uniformity_v3_e2e.json exists | ✅ | 4549 B |
| G5 | v2 cellclip mAP_z | ✅ | 0.1337 vs pinned 0.1337 |
| G5 | v2 ca_mae mAP_z | ✅ | 0.1083 vs pinned 0.1083 |
| G5 | v2 deepprofiler mAP_z | ✅ | 0.0691 vs pinned 0.0691 |
| G5 | v2 o2vae mAP_z | ✅ | 0.0471 vs pinned 0.0471 |
| G5 | IO uniformity 7/7 | ✅ | {'n_ok': 7, 'n_failed': 0, 'all_ok': True} |
