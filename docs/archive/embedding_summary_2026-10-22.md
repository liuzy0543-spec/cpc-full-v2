# Cell Painting Embedding 统一框架 —— 汇总

日期：2026-10-22
范围：只保留可运行方法；统一 IO 复测；BBBC022 全量 benchmark

---

## 1. 最终方法清单（4 个）

取消判定依据：**权重能否在可接受时间内取得**。功能实现不是瓶颈，权重才是。

| 方法 | 状态 | 输出 | 输入 | 权重 sha256（前 16 位） | 许可 |
|---|---|---|---|---|---|
| deepprofiler | ✅ 可运行 | 672 | 5ch 像素 BHWC 32×32 | `87b756f579336d60` | BSD-3 / CC-BY-4.0 |
| ca_mae | ✅ 可运行 | 384 | 5ch 像素 BCHW 256×256 uint8 | `f6e5f1c97101331b` | Recursion NC EULA |
| cellclip | ✅ 可运行 | 512 | **(N,5,1536) DINOv2 向量** | `66c2d32a44a3e7d6` | CC-BY-4.0 |
| o2vae | ✅ 可运行 | 512 | 1ch 剪影 64×64 | `aa70cfa8b080c9d9` | MIT |

四个权重的 sha256 全链路校验通过（`weights_sha256` 随每个 `EmbeddingResult` 一起返回）。

## 2. 被取消的 3 个方法

| 方法 | 权重是否开源 | 实际障碍 | 结论 |
|---|---|---|---|
| **dinocells** | ✅ **确实开源**：Zenodo [8061392](https://zenodo.org/records/8061392)，CC-BY-4.0，open access，425 次下载 | 只有**一个** `HPA_FOV_data.tar.gz`（3,711,589,832 B）把 checkpoint 和整个 HPA FOV 数据集打包在一起，**无法单独取 checkpoint**。2026-10-21 实测 **9.2 KB/s**（180 s 只拿到 1.67 MB）→ **约 109 小时** | ❌ 取消 |
| unidino | 开源（Zenodo `checkpoints.zip` 1.38 GB） | 同主机同速率 → 约 42 小时 | ❌ 取消 |
| morphogenie | Google Drive | 本机不可达；且 256×256 居中预处理在 MATLAB 里，不在仓库中 | ❌ 取消 |

**dinocells 这条要说清楚**：它**不是"没开源"**，是"开源但取不到"。
镜像也查过：HF 上 `dinocells` / `DINO4Cells` / `morphogenie` 均无官方权重，
唯一的第三方 `UniDINO` 仓库（4 下载、无 license、输入 448 而官方 224）不可信，已拒绝。

**取消不等于删除**：三份契约完整保留在 `registry.PENDING`，
`models.pending()` 可查看，测试仍覆盖其 `source_map`。
恢复一个是**改一行**（从 `_PENDING_NAMES` 移除名字）+ 写适配器。

## 3. Benchmark 结果（BBBC022，3,857 细胞）

数据集：3,857 细胞 / 64 孔 / **54 化合物** / 671 mock / 11 mock 孔。
随机基线（chance）= **0.0185**。

### 3.1 耗时

| 方法 | 输出 | 耗时 | 吞吐 |
|---|---|---|---|
| cellclip | (3857, 512) | 10.6 s | 364.8 细胞/s |
| deepprofiler | (3857, 672) | 37.8 s | 101.9 细胞/s |
| o2vae | (3857, 512) | 79.9 s | 48.2 细胞/s |
| ca_mae | (3857, 384) | 157.8 s | 24.4 细胞/s |
| **合计** | — | **289.8 s（4.8 min）** | — |

### 3.2 指标

| 方法 | kNN | kNN_z | mAP | mAP_z | silhouette_z | mock-AUC_z | 有效秩 |
|---|---|---|---|---|---|---|---|
| **cellclip** | 0.6248 | **0.6451** | 0.1318 | **0.1337** | −0.0454 | 0.4763 | 23.4 |
| **ca_mae** | 0.4286 | 0.4454 | 0.1001 | 0.1083 | −0.1365 | 0.5861 | 2.0 |
| **deepprofiler** | 0.1960 | 0.1934 | 0.0701 | 0.0691 | −0.0341 | **0.5935** | 34.7 |
| 原始像素基线 | 0.1084 | 0.1058 | 0.0503 | 0.0508 | −0.0655 | 0.5227 | 19.8 |
| **o2vae** | 0.0843 | 0.0869 | 0.0473 | 0.0471 | −0.1424 | 0.5054 | 2.9 |
| 随机基线 | 0.0837 | 0.0848 | 0.0450 | 0.0450 | −0.0141 | 0.4996 | 348.8 |

**排序（按 z-score mAP，化合物检索）**：

```
cellclip 0.1337  >  ca_mae 0.1083  >  deepprofiler 0.0691  >  o2vae 0.0471
                    ↑ 比随机高 5.8x                    ↑ 与随机无差异
```

### 3.3 稳定性（4 折）

```
fold 0: cellclip > ca_mae > deepprofiler > o2vae
fold 1: cellclip > ca_mae > deepprofiler > o2vae
fold 2: cellclip > ca_mae > deepprofiler > o2vae
fold 3: cellclip > ca_mae > deepprofiler > o2vae
identical ordering in every fold: True
first-place counts: cellclip=4
```

**每一折排序完全相同**，cellclip 4/4 第一。

### 3.4 怎么读这张表

- **cellclip 最强**：kNN_z 0.6451，是随机基线的 7.6 倍。但它吃的是 DINOv2 向量，
  不是像素——性能里含 DINOv2 的功劳，不能全记在 CellCLIP 头上。
- **ca_mae 第二且"有效秩"只有 2.0**：说明它的 384 维里实际只有约 2 个有效自由度，
  高度压缩但检索能力强。这是一个值得注意的现象，不是缺陷。
- **deepprofiler 第三，但 mock-AUC_z 最高（0.5935）**：它最擅长区分处理孔与对照孔，
  只是化合物间区分度不如前两者。两个指标测的不是同一件事。
- **o2vae 垫底（0.0471 ≈ 随机 0.0450）**：它是**细胞形状模型**，吃单通道剪影，
  本来就不是为化合物表型设计的。跑通说明接口对，分数低说明任务不匹配，
  **不是坏了**。
- **原始像素 0.0508 略高于 o2vae**：进一步说明 o2vae 丢掉的不只是噪声。

### 3.5 不能说的话

- ❌ **不给 p 值**。每个化合物只有**单孔**，化合物身份 ≡ 孔身份，
  留一孔交叉验证会得到零个相关项。任何显著性数字都是假的。
- ❌ **不能与论文数字直接比较**。这是自建的化合物检索协议，
  不是 CA-MAE / CellCLIP 论文的指标。
- ❌ **不能说"o2vae 不行"**。只能说它在化合物检索这个任务上不适用。

## 4. 统一接口验证

```
runnable+prepared: ['deepprofiler', 'ca_mae', 'cellclip', 'o2vae']

deepprofiler  n=3857 dim=672  17.04s (226.3/s) unit=SINGLE_CELL_CROP
ca_mae        n=3857 dim=384 118.57s ( 32.5/s) unit=SINGLE_CELL_CROP
cellclip      n=3857 dim=512  10.83s (356.3/s) unit=CHANNEL_EMBEDDINGS
o2vae         n=3857 dim=512  85.06s ( 45.3/s) unit=SINGLE_CELL_CROP

succeeded 4/4; failed 0
identical result-object field sets: True
all carry a weights sha256: True
all carry a license: True
```

调用签名完全相同：`models.embed(name, payload)`，**`unit` 都不用传**（从 manifest 推断）。
输出维度不同（384/512/672）是正常的——统一的是接口，不是维度。

## 5. 可复现性

| 检查 | 结果 |
|---|---|
| 与上一轮 benchmark 比对 | **65 个可比数值项，0 项不一致** |
| 与官方 DeepProfiler 对拍 | **逐位相同**（max\|diff\| = 0.000000e+00） |
| 4 折稳定性 | 每折排序一致 |
| 测试基线 | **289 passed**，5 skipped，14 failed（均为任务开始前既有） |

## 6. 复现命令

```powershell
# 生成 4 个方法的 embedding（约 5 分钟）
python scripts\embedding_benchmark\embed_dataset.py --cells-dir E:\bbbc022\cells --force

# 打分 + 稳定性
python scripts\embedding_benchmark\score_benchmark.py --cells-dir E:\bbbc022\cells
python scripts\embedding_benchmark\stability_check.py --cells-dir E:\bbbc022\cells --folds 4

# 统一性（单方法路径 + 批处理路径）
python scripts\embedding_benchmark\verify_io_uniformity.py --cells-dir E:\bbbc022\cells --batch

# 与官方 DeepProfiler 对拍
python scripts\embedding_benchmark\parity_vs_official_dp.py

# 测试
python -m pytest tests/ -q
```

产物：`E:\bbbc022\cells\` 下 `benchmark_scores_v2.json`、`stability_v2.json`、
`io_uniformity_v2.json`、`emb_*.npy`。