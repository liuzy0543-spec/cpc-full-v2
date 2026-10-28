# v2 / v3 最终代码梳理

> 生成日期：2026-10-27
> 对应实现：`src/cellpaint_pipeline/models/`
> 状态：两版本均通过 66 项审核（第 8 轮 PASS）、全流程 e2e 七步全 0 退出码

---

## 1. 两版本是什么关系

**同一套代码、同一套契约、两套注册表视图。** v3 不是 v2 的分支，也没有第二份适配器实现：

```
                    ┌──────────────────────────────┐
                    │  contract.py（契约，不含框架） │
                    │  preprocessing.py（预处理执行）│
                    │  embedding.py（调度/校验/缓存）│
                    └──────────────┬───────────────┘
                                   │
                    ┌──────────────▼───────────────┐
                    │ registry.py                  │
                    │  _ALL_REGISTRATIONS = 7 条    │
                    │  ADAPTERS  = 4（官方权重）     │ ← v2 见到这些
                    │  SURROGATES= 3（本地自训）     │ ← v3 额外见到这些
                    └──────────────────────────────┘
```

`registry.set_profile('v2'|'v3')` 只是换一个视图，并清空
`embedding.clear_adapter_cache()`。**默认是 `v2`**：不加任何参数时，`declared()`
只有四个方法，`embed('unidino', …)` 会报 `UnknownModel`。

两版本永不合并——把它们并进一张表就等于把"论文性能"和"接口能跑通"
混成一个数字。

---

## 2. 代码地图

| 文件 | 行数 | 职责 | 谁依赖它 |
|---|---|---|---|
| `models/__init__.py` | 267 | 公开门面，PEP 562 惰性导入（`import models` 只拉入 1 个模块） | 调用方 |
| `models/contract.py` | 856 | `ModelManifest` / `PreprocessSpec` / `ChannelSpec` / `WeightsSpec` / 纯函数 `check_compatible()` | 纯数据，无框架依赖 |
| `models/registry.py` | 1031 | 七条注册、剖面切换、`describe()` | contract |
| `models/embedding.py` | 1252 | `embed` / `embed_many` / `embed_all` / `prepare` / `plan` / `probe` / `doctor` / 输出宽度与确定性校验 | registry |
| `models/preprocessing.py` | 577 | `SourceMap` 八种映射、dtype/双缩放守卫、resize 策略 | contract |
| `models/weights.py` | 238 | 权重解析、sha256 校验、ASCII 路径检查 | — |
| `models/errors.py` | 103 | `IncompatibleInput` / `UnknownModel` / `WeightsUnavailable` / `MissingDependency` | — |
| `models/adapters/base.py` | 167 | `BaseAdapter`：依赖检查、权重解析、设备选择、加载 | 七个适配器 |
| `models/adapters/{ca_mae,cellclip,o2vae,deepprofiler}.py` | 202/310/298/197 | v2 四个官方方法 | base |
| `models/adapters/{unidino,dinocells,morphogenie}.py` | 100/137/115 | v3 三个替身方法 | base + `_surrogate.py` |
| `models/adapters/_surrogate.py` | 205 | 替身公用骨干 + 权重 sha256 校验 | 三个替身 |
| `models/adapters/unavailable.py` | 94 | 权重缺失时的占位适配器（给出"缺什么、去哪拿"） | registry |

---

## 3. v2（默认剖面）— 四个官方权重方法

| method | provenance | width | input unit | spatial | dtype | scale | source_map | layout | mask |
|---|---|---|---|---|---|---|---|---|---|
| `deepprofiler` | official | 672 | single_cell_crop | (32, 32) | float32 | per_crop_minmax | per_crop_minmax_bhwc | BHWC | none |
| `ca_mae` | official | 384 | single_cell_crop | (256, 256) | uint8 | div255 | identity | BCHW | none |
| `cellclip` | official | 512 | channel_embeddings | None | float32 | none | None | BCHW | none |
| `o2vae` | official | 512 | single_cell_crop | (64, 64) | float32 | none | mask_to_silhouette | BCHW | none |

| method | sha256(16) | license | determinism | requires |
|---|---|---|---|---|
| `deepprofiler` | `87b756f579336d60` | BSD-3-Clause (code) / CC-BY-4.0 (weights) | deterministic | tensorflow, efficientnet |
| `ca_mae` | `f6e5f1c97101331b` | Recursion Non-Commercial EULA | deterministic | torch, transformers, einops |
| `cellclip` | `66c2d32a44a3e7d6` | CC BY-NC-ND 4.0 | deterministic | torch |
| `o2vae` | `aa70cfa8b080c9d9` | MIT | deterministic | torch, e2cnn |

```python
from cellpaint_pipeline import models          # 默认 v2

r = models.embed('ca_mae', uint16_bhwc, assume='raw_crop')
r.as_numpy()          # (N, 384) float32
r.metadata['weights_provenance']   # 'official'
r.license                          # 随结果带出，报告合规用
```

---

## 4. v3 — 四个官方 + 三个替身

| method | provenance | width | input unit | spatial | dtype | scale | source_map | layout | mask |
|---|---|---|---|---|---|---|---|---|---|
| `deepprofiler` | official | 672 | single_cell_crop | (32, 32) | float32 | per_crop_minmax | per_crop_minmax_bhwc | BHWC | none |
| `ca_mae` | official | 384 | single_cell_crop | (256, 256) | uint8 | div255 | identity | BCHW | none |
| `cellclip` | official | 512 | channel_embeddings | None | float32 | none | None | BCHW | none |
| `o2vae` | official | 512 | single_cell_crop | (64, 64) | float32 | none | mask_to_silhouette | BCHW | none |
| `unidino` | **surrogate** | 384×5=1920 | single_cell_crop | (224, 224) | float32 | div65535 | div65535 | BCHW | none |
| `dinocells` | **surrogate** | 384 | single_cell_crop | (96, 96) | float32 | per_image_selfnorm | per_sample_selfnorm | BCHW | **drop** |
| `morphogenie` | **surrogate** | 10 | single_cell_crop | (256, 256) | float32 | none | matlab_centring | BCHW | none |

| method | sha256(16) | determinism | 说明 |
|---|---|---|---|
| `unidino` | `a0265184…` | deterministic | 逐通道 384 向量拼接，**顺序敏感** |
| `dinocells` | `5e172d8f…` | deterministic | mask 被丢弃（官方训练未用 mask） |
| `morphogenie` | `49102997…` | deterministic | 只吃 3 通道，宽度是 `c_dim=10` 而非 `z_dim=256` |

```python
from cellpaint_pipeline import models
from cellpaint_pipeline.models import registry

registry.set_profile('v3')
batch = models.embed_all(
    ['deepprofiler', 'ca_mae', 'cellclip', 'o2vae', 'unidino', 'dinocells', 'morphogenie'],
    inputs,                                    # {name: array}
    assume={n: 'raw_crop' for n in
            ('deepprofiler', 'ca_mae', 'unidino', 'dinocells')},
    channels={'morphogenie': ('C0', 'C1', 'C2')},
    weights_roots=('E:/cpc_weights',),
)
```

`unit` / `assume` / `masks` / `channels` 四者都接受 `{方法名: 值}` 映射，
未列出的方法取 `None`。这是"一次调用驱动七个"的前提。

---

## 5. 结果（BBBC022 plate 20585，3,857 细胞 / 54 化合物 / 64 孔）

### v2 官方权重

| 排名 | method | mAP_z | kNN_z | silhouette（全部为负） | mock-AUC_z |
|---|---|---|---|---|---|
| 1 | `cellclip` | **0.1337** | 0.6451 | −0.0454 | 0.4763 |
| 2 | `ca_mae` | 0.1083 | 0.4454 | −0.1365 | 0.5861 |
| 3 | `deepprofiler` | 0.0691 | 0.1934 | −0.0341 | 0.5935 |
| 4 | `o2vae` | 0.0471 | 0.0869 | −0.1424 | 0.5054 |

### v3 追加的三个替身

| method | mAP_z | kNN_z | silhouette | mock-AUC_z |
|---|---|---|---|---|
| `unidino` | 0.0684 | 0.2320 | −0.1739 | 0.5842 |
| `morphogenie` | 0.0574 | 0.1281 | −0.7161 | 0.5402 |
| `dinocells` | 0.0460 | 0.0814 | −0.1891 | 0.4772 |

### 必须一起读的基线

| 基线 | mAP_z | kNN_z |
|---|---|---|
| 原始像素（同批裁剪） | 0.0508 | 0.1058 |
| 随机投影 | 0.0450 | 0.0848 |
| 随机 4 折区间下界–上界 | 0.0683–0.1109 | — |

**o2vae（0.0747–0.1179）与随机区间重叠**，dinocells 与原始像素几乎相同
（0.0460 vs 0.0508）。这两个名次**不被本数据支持**，不能写进"某方法更好"的结论。

---

## 6. 怎么验证

```bash
export PYTHONPATH=src CPC_WEIGHTS_ROOT=E:/cpc_weights

# 1) 单元与契约测试
python -m pytest tests -q                       # .venv: 324 passed, 14 failed(既有)
D:/ep_bench/venv/Scripts/python.exe -m pytest \
    tests/test_models_{v3_dispatch,contract,profiles,preprocessing}.py -q   # 116 passed

# 2) 六道闸门审核（每次改代码后必跑）
python scripts/embedding_benchmark/audit_round.py --round N

# 3) 全流程（重算嵌入 → IO 校验 → 打分 → 稳定性）
bash scripts/embedding_benchmark/run_e2e.sh

# 4) 配置表（本文 §3/§4 由它生成，改声明后重新生成）
python scripts/embedding_benchmark/dump_profile.py --profile v2
python scripts/embedding_benchmark/dump_profile.py --profile v3
```

六道闸门：G1 能跑(28) / G2 统一调度(5) / G3 契约守卫(6) / G4 确定性(7) /
G5 产物与 v2 数字稳定(11) / G6 与官方一致(9) = **66 项**。

---

## 7. 已知边界（写进结论时必须一起出现）

1. **v3 三个方法是替身**：权重本地自训（VICReg，只用 mock 细胞、不用标签），
   它们证明"接口能驱动该方法"，**不证明论文性能**。官方权重不可得
   （unidino 在 Zenodo 约 10 KB/s ≈ 180 小时；dinocells/morphogenie 未发布），
   这一豁免已记录在案，只能声明接口一致性。
2. **o2vae 是形状模型**：它吃 MEF 的**细胞**剪影，而 BBBC022 只有核染色可靠，
   所以喂的是**核**剪影。任何把它与强度模型并列的结论都要带这句。
3. **不报 p 值**：一个化合物恰好一个孔，化合物 ≡ 孔。排除自身孔后 64 孔里 53 孔
   无相关项；保留则测的是"认孔"。
4. **silhouette 全为负**：不支持"嵌入形成可分离簇"。
5. **mock-AUC 0.47–0.59，随机 0.4996**：区分处理与对照的能力接近随机。
6. **数据规模**：本地图像只有 72 个视野同时凑齐 5 通道（5 个 zip 曾被截断），
   3,857 细胞是**该批数据的上限**。更大规模的对比需要重新下载完整 zip
   （`fetch_bbbc022_plate.sh`）。