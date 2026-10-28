# CellPainting Embedding 层审查报告

> 日期：2026-10-18
> 审查对象：`src/cellpaint_pipeline/models/`、`src/cellpaint_pipeline/embedding_benchmark.py`、`tests/test_models_contract.py`、`tests/test_embedding_benchmark.py`、`docs/PROJECT_STATUS.md`、真实数据流水线 `D:\项目\cellpainting claw复现\_probe\`
> 审查方式：静态代码审计 + 契约测试 + 真实权重加载 + BBBC022 全量复算 + 4 折分组实验 + 4 细胞端到端 benchmark slice
> 本轮未修改任何已跟踪文件；只新增了实验产物（见 §9）

---

## 0. 审查任务与结论总览

| # | 审查任务 | 结论 |
|---|---|---|
| 1 | 六/七种 embedding 方法能否统一 input-output 接口 | **能**；“契约层 + 注册表 + adapter”方向正确 |
| 2 | 接口是否解耦、能否正确 load 模型 | **基本能**；4 个 ready 方法全部加载成功，输出维度正确 |
| 3 | 现有方法是否符合要求 | **2 个通过、2 个有条件通过、3 个仅声明**；详见 §2 |
| 4 | 代码可读性、易扩展性、软工思想 | **总体良好**；但存在文档漂移、未跟踪代码、若干静默失败缺口 |
| 5 | embedding benchmark 能否出期望结果 | **能复现项目自己的结果**；全量指标逐位一致，4 折排名稳定 |
| 6 | 权重是否与官方一致 | **4/4 本地哈希匹配**；O2-VAE 与上游仓库文件逐字节相同 |
| 7 | 测试集是否合理 | **只能作为该板子集的 smoke/relative benchmark**；每化合物仅 1 孔/1 浓度 |
| 8 | 拿一组做测试 | 已跑 4 细胞 × 4 方法端到端 slice，全部成功，耗时 22.9 s |

### 最重要的单条发现

> **DeepProfiler 通道顺序与上游官方 README/集成测试冲突。**
>
> - 上游官方：`(DNA, ER, RNA, AGP, Mito)`
> - 当前 registry / benchmark：`(DNA, RNA, ER, AGP, Mito)`（RNA/ER 互换）
>
> 现有“逐位 parity”匹配的是**本项目自己的 demo 配置**，而该 demo 配置也用了同一顺序。因此：
> **parity 证明的是“接口层 == 本项目官方树”，不是“接口层 == 上游模型训练语义”。**

---

## 1. 审查背景：为什么要统一接口

官方 CellPainting-Claw 只支持 1 个 embedding 方法（DeepProfiler），且自己不加载模型，用子进程外包给 `deepprofiler` 包。其余方法在官方代码里没有出现，输入契约互不兼容：

| 方法 | 输入单元 | 布局 | dtype / 值域 | 备注 |
|---|---|---|---|---|
| DeepProfiler | 单细胞裁剪 | `(N,H,W,C)` | float `[0,1]` | 官方 crop graph 内做 per-crop min-max |
| CA-MAE / OpenPhenom | 单细胞裁剪 | `(N,C,H,W)` | **uint8 `[0,255]`** | 通道数可变；模型内部 /255 + InstanceNorm |
| CellCLIP | **通道向量** | `(N,5,1536)` | float | 不是像素 |
| O2-VAE | 单细胞裁剪 | `(N,1,64,64)` | float `[0,1]` | 单通道形状剪影 |
| uniDINO | 单细胞裁剪 | 多通道 | float | 输出 `384*C`，非固定维度 |
| DINO4Cells | tape 编码 | `(H,W*(C+1))` | uint8 | 需要显式 tape 语义 |
| MorphoGenie | 单细胞裁剪 | `(N,3,256,256)` | float | 官方抽取路径随机 |

强行统一成一种 tensor 形状会丢信息。当前设计把统一放在**声明式契约**，而不是张量层：

- `ModelManifest` 声明输入/输出/权重/许可/确定性
- `check_compatible` 是纯函数，加载前校验
- `registry.ADAPTERS` 一张表
- adapter 懒加载、缓存、类型化错误
- `EmbeddingResult` 携带 provenance

这个架构符合依赖倒置、开闭原则和 fail-fast，是正确的。

---

## 2. 现有方法逐项审查

| 方法 | 官方权重 | 加载严格性 | 契约正确性 | 可运行 | 结论 |
|---|---|---|---|---|---|
| **deepprofiler** | ✅ Zenodo CP CNN v1，哈希匹配 | ⚠️ `load_weights(by_name=True)` 静默跳过不匹配层 | ❌ 通道顺序 RNA/ER 错误；32×32 与官方测试 128×128 不同 | ✅ | **有条件通过** |
| **ca_mae** | ✅ OpenPhenom-S/16，哈希匹配 | ✅ `strict=True` | ✅ 通道无关声明正确；uint8 契约正确 | ✅ | **通过** |
| **cellclip** | ✅ CellCLIP checkpoint，哈希匹配 | ✅ 逐 key 检查 missing/unexpected | ⚠️ 通道顺序未与官方训练数据对齐；输入是向量 | ✅ | **有条件通过** |
| **o2vae** | ✅ 官方 MEF 权重，与上游仓库文件 sha256 相同 | ⚠️ `strict=False`；实测 missing 全是 buffer，0 个参数 | ✅ 单通道 64×64 `[0,1]` 正确 | ✅ | **有条件通过** |
| **unidino** | ❌ Zenodo 未获取 | — | ⚠️ `output_dim=384` 但实际 `384*C`；`scale` 未声明 | ❌ | **仅契约声明** |
| **dinocells** | ❌ Zenodo 未获取 | — | ✅ tape 契约正确；官方缺 CP 配置 | ❌ | **仅契约声明** |
| **morphogenie** | ❌ Google Drive 不可达 | — | ✅ 已标 `stochastic`；`output_dim=10` 正确 | ❌ | **仅契约声明** |

### 2.1 关于 “deepprofiler cellpainting ssl”

当前注册表里**没有**一个独立的 `cellpainting_ssl` 方法。

- `deepprofiler` = **Cell Painting CNN v1**（EfficientNet-B0），是官方 DeepProfiler 的监督式 checkpoint，不是 SSL 模型。
- 若你指的是 Recursion 的 Cell Painting SSL 系列模型，它没有 manifest、adapter 或本地权重，需要新增。

### 2.2 DeepProfiler 通道顺序问题（P0）

上游官方证据：

- `D:\official_src\1-deepprofiler\README.md:74`：
  `... 5-channel Cell Painting images (DNA, ER, RNA, AGP, Mito)`
- `tests/deepprofiler/profiling/test_cell_painting_cnn_v1.py:44-48,74`：
  `ch5=DNA, ch4=ER, ch3=RNA, ch1=AGP, ch2=Mito`
  `CHANNELS = ["DNA", "ER", "RNA", "AGP", "Mito"]`

本项目：

- `src/cellpaint_pipeline/models/registry.py:96`：
  `DEEPPROFILER_CHANNELS = ('DNA', 'RNA', 'ER', 'AGP', 'Mito')`
- `src/cellpaint_pipeline/models/contract.py:45-46`：重复了同样的错误顺序
- `D:\项目\cellpainting claw复现\_probe\build_bbbc022_cells.py:53-60`：同样按 `DNA, RNA, ER, AGP, Mito` 构建
- `D:\cpc_c_official\demo\backend\profiling_backend\configs\pipeline_config.json:32-41`：
  `1=Mito, 2=AGP, 3=RNA, 4=ER, 5=DNA`，与官方测试的 `1=AGP, 2=Mito` 也冲突

实测影响（BBBC022，3857 细胞）：

| DP 输入 | mAP_z | kNN_z |
|---|---:|---:|
| 当前（DNA,RNA,ER,AGP,Mito；joint min-max） | 0.0684 | 0.1926 |
| 官方顺序 + per-channel min-max | **0.0691** | **0.1934** |

结论：

- 在这份数据上只差约 1%，**“DP ≈ 原始像素”的定性结论不变**；
- 但契约声明本身是错的，正确调用方会被 `channel-order` 拒绝；
- 换数据后影响可能变大，必须修。

### 2.3 Phase 的参照对象需要说清楚

`_probe/parity_vs_known_result.py:40-42` 使用：

```python
CHANNELS = ('DNA', 'RNA', 'ER', 'AGP', 'Mito')
SUFFIX = {'DNA': 'ch5', 'RNA': 'ch3', 'ER': 'ch4', 'AGP': 'ch2', 'Mito': 'ch1'}
```

这与上游官方测试的 `AGP→ch1, Mito→ch2` 也不一致。

因此：

> `bitwise_equal=True` 证明的是“接口层 == 本项目官方树（CellPainting-Claw demo）”，
> 不是“接口层 == 上游 DeepProfiler 模型训练语义”。

这不是说 parity 结果无效，而是说它的**参照系**需要明确标注。

### 2.4 CellCLIP 通道顺序

上游 `preprocessing/preprocess_images_jumpcp.py:79-83` 的注释给出 JUMP-CP 的 w1..w5 语义：

```
w1 = DNA
w2 = ER
w3 = RNA
w4 = AGP
w5 = Mito
```

当前 CellCLIP manifest 直接复用了 DeepProfiler 的通道常量，因此顺序同样可疑。

实测把 CellCLIP 输入从 `DNA,RNA,ER,...` 换成 `DNA,ER,RNA,...`：

| CellCLIP 输入顺序 | mAP_z | kNN_z |
|---|---:|---:|
| 当前 | 0.1338 | 0.6427 |
| 交换 ER/RNA | 0.1337 | 0.6451 |

影响几乎为零，但契约仍然写错了。应单独为 CellCLIP 声明通道顺序，并加回归测试。

### 2.5 O2-VAE 加载情况

`strict=False` 实测结果：

```
checkpoint keys: 194
model state keys: 446
missing: 252（其中 learned parameters = 0，buffers = 188，其余为 e2cnn derived entries）
unexpected: 0
```

所有 learned parameter 都已加载，权重是完整的。但代码没有断言“missing 只能是非参数 buffer”。建议改成：

```python
result = model.load_state_dict(state, strict=False)
missing_params = [k for k in result.missing_keys if k in param_names]
assert not missing_params and not result.unexpected_keys
```

### 2.6 DeepProfiler 裁剪尺度

上游集成测试 `test_cell_painting_cnn_v1.py:75` 使用：

```python
BOX_SIZE = 128
```

当前 demo 和 BBBC022 benchmark 使用 32×32。文档自己也把“给 DeepProfiler 更大裁剪（64×64）”列为下一步假设。由于模型是全卷积的，32 和 128 都能跑，但 32 是否为其原生训练尺度**未被验证**。这可能低估 DeepProfiler。

---

## 3. 统一接口能力评估

### 3.1 已经具备的能力

| 能力 | 实现 |
|---|---|
| 按名字取方法 | `models.embed(name, x)` |
| 列出方法 | `models.available()` |
| 查看契约 | `models.describe(name)` / `models.manifest(name)` |
| 查询本机可运行性 | `models.probe(name)` / `models.doctor()` |
| 类型化错误 | `UnknownModel` / `IncompatibleInput` / `MissingDependency` / `WeightsUnavailable` |
| 加载前校验 | `check_compatible` 先于 adapter 构建 |
| adapter 缓存 | 同一方法第二次调用从 ~7 s 降到 ~0.03 s |
| 输出导出 | `as_numpy` / `as_torch` / `to_frame` / `save_npz` / `to_dict` |
| 随机方法聚合 | `embed_all` 对 `determinism='stochastic'` 重复并记录 spread |
| 许可闸门 | `classify_licence` + `EmbeddingResult.license` |

### 3.2 还不是真正“一个入口调度所有方法”的地方

| 缺口 | 现状 | 影响 |
|---|---|---|
| 同一批原始数据喂给所有方法 | `PreprocessSpec` 只声明、不执行 | 调用方必须自己实现 scale/dtype/通道重排；benchmark 的耦合就在 `build_bbbc022_cells.py` |
| 逐方法指定 `unit` / `channels` | `BenchmarkPlan` 只有一个 `unit`/`channels` | CellCLIP 需要 `channel_embeddings`，其余需要 `single_cell_crop`，一个 plan 无法同时表达 |
| 统一调度入口 | `models.embed_all` **未导出** | `PROJECT_STATUS.md` 文档写的是 `models.embed_all(...)`，实测 `hasattr(models, 'embed_all') == False` |
| 指定 device / weights root | `_build_adapter` 只传 manifest | 无法通过公开 API 控制 CPU/GPU 或权重根 |
| 选择 feature layer / channelwise | 只能构造 adapter 参数 | `models.embed(..., feature_layer='pool5')` 会被转发但被忽略 |
| 输出形状 | 假设 `(N,D)` | uniDINO 的 `(N,C,384)` 无法用 `output_dim=384` 正确表达 |
| 落盘行身份 | `save_npz` 不写 `sample_metadata` | benchmark 产出的 `.npz` 事后无法按 `cell_id` 对齐 |

### 3.3 一个具体的契约漏洞

CellCLIP 的 `input_unit=CHANNEL_EMBEDDINGS`，但 `check_compatible` 只检查 ndim ∈ {3,4}，不检查最后一维。实测：

```python
check_compatible(cellclip_manifest, array=np.zeros((2,5,256,256), dtype='float32'))
# -> ok=True, codes=()
```

也就是说，**不传 `unit` 时，一个 5 通道像素堆会被 CellCLIP 契约放行**。`embed_all` 的 shared-payload 路径如果 `unit=None`，会走到 model load 后才在 forward 里报错。建议：

- `CHANNEL_EMBEDDINGS` 强制 `ndim == 3`；
- manifest 增加 `input_feature_dim=1536` 并校验 `shape[-1]`。

---

## 4. 权重核验

本轮实际计算了四个 ready 方法的 sha256：

| 方法 | 文件 | 实测 sha256 | 与 manifest 一致 |
|---|---|---|---|
| deepprofiler | `E:\cpc_weights\deepprofiler\Cell_Painting_CNN_v1.hdf5` | `87b756f5…833e73` | ✅ |
| ca_mae | `E:\cpc_weights\openphenom_pkg\openphenom\model.safetensors` | `f6e5f1c9…4c4d47` | ✅ |
| cellclip | `E:\cpc_weights\cellclip\model.safetensors` | `66c2d32a…71b550` | ✅ |
| o2vae | `E:\cpc_weights\o2vae\o2vae_mefs_official.pt` | `aa70cfa8…5dfc0f` | ✅ |

额外验证：

- O2-VAE 文件与上游仓库 `o2vae_repo/wandb/pretrained_models/model_mefs.pt` **sha256 完全相同**。
- CA-MAE 的 `config.json` 与 `README.md` 确认是 `OpenPhenom-S/16`（`channel_agnostic: true`, `max_in_chans: 11`）。
- CellCLIP 的 `config.json` 与 adapter 中的 `CROSS_CHANNEL_FORMER_CONFIG` 一致（1536/12/8/512/5）。

### 4.1 DINOv2-Giant 溯源缺口

CellCLIP 链路真正的大模型是 DINOv2-Giant：

```
E:\cpc_weights\dinov2_giant\model.safetensors   4,546,005,432 B
sha256 = 917d3c470db999d32a312f8542149be91c7cbac61ee8fb4b67ae3d82b79ce21f
```

但：

- manifest 只记录了 CellCLIP 的 1.8 GB checkpoint；
- `cellclip_vectors.json` 只有 `model_dir`，没有哈希；
- benchmark 的 provenance 只覆盖 CellCLIP checkpoint，不覆盖产生输入的 DINOv2-Giant。

建议增加 `chain_weights` 或 `upstream_models` 溯源字段。

### 4.2 “官方性”证据强度

| 方法 | 证据 |
|---|---|
| DeepProfiler | Zenodo 7114558 的官方 checkpoint 名称/大小/哈希一致 |
| CA-MAE | `recursionpharma/OpenPhenom` 的包布局、config、README 一致 |
| CellCLIP | `suinleelab/CellCLIP` 的 `config.json` + `model.safetensors` 布局一致 |
| O2-VAE | 与上游仓库内文件逐字节相同 |

本轮没有重新从外网下载并比对官方 checksum；上述结论基于文件名、目录布局、config、上游仓库内文件以及项目记录的下载来源。

---

## 5. Benchmark 实测结果

### 5.1 全量复算：与文档完全一致

用 `_probe/score_benchmark.py` 在现有 3,857 细胞上重新打分：

```
cells 3857  compounds 54  mock 671  mock_wells 11
deepprofiler        knn_z=0.1926  map_z=0.0684  auc_z=0.6135
ca_mae              knn_z=0.4454  map_z=0.1083  auc_z=0.5861
cellclip            knn_z=0.6427  map_z=0.1338  auc_z=0.4773
o2vae               knn_z=0.0869  map_z=0.0471  auc_z=0.5054
baseline_raw_pixels knn_z=0.1592  map_z=0.0664  auc_z=0.5904
baseline_random     knn_z=0.0848  map_z=0.0450  auc_z=0.4996
```

排序：`cellclip > ca_mae > deepprofiler > o2vae`，与 `benchmark_bbbc022_result.md` 完全一致。

随机基线 AUC = 0.4996，指标校准可信。

### 5.2 4 折按孔分组：排名稳定

把 64 个孔按排序后 `i::4` 分成 4 组，各自 z-score 后打分：

| Fold | n | cellclip | ca_mae | deepprofiler | raw_pixels | o2vae |
|---|---:|---:|---:|---:|---:|---:|
| 1 | 927 | 0.2816 | 0.2190 | 0.1764 | 0.1770 | 0.1030 |
| 2 | 984 | 0.2490 | 0.1784 | 0.1229 | 0.1119 | 0.0737 |
| 3 | 981 | 0.2472 | 0.2023 | 0.1693 | 0.1536 | 0.1158 |
| 4 | 965 | 0.2687 | 0.1966 | 0.1606 | 0.1391 | 0.1017 |

每一折的排名都是：

```
cellclip > ca_mae > deepprofiler ≈ raw_pixels > o2vae
```

说明**相对排序在这份数据内部是稳定的**。

### 5.3 统一接口的端到端 slice

通过 `embedding_benchmark.run_benchmark` 跑 4 个细胞 × 4 个方法：

| 方法 | 输出 | 与已有 `emb_*.npy` 对比 | 权重 sha256 |
|---|---|---|---|
| deepprofiler | `(4,672)` | **exact=True**, maxdiff 0 | ✅ |
| ca_mae | `(4,384)` | exact=False, maxdiff `4.77e-07` | ✅ |
| cellclip | `(4,512)` | exact=False, maxdiff `3.76e-06` | ✅ |
| o2vae | `(4,512)` | **exact=True**, maxdiff 0 | ✅ |

总耗时 22.9 秒（含 4 个模型加载）。CA-MAE/CellCLIP 的 1e-6 级差异来自 batch/GPU 数值非确定性，不是加载错误。

### 5.4 测试集结构

| 项 | 值 | 评价 |
|---|---:|---|
| 视野 | 64 | 偏少 |
| 细胞 | 3,857 | 够做检索，不够做稳健结论 |
| 化合物 | 53 + mock | 每化合物**恰好 1 个孔** |
| 浓度 | 每化合物 **1 个** | 无法做剂量-反应 |
| mock 孔 | 11 | 可做 mock-vs-compound AUC |
| 每孔细胞 | 30–62 | 基本均衡 |

**核心混杂**：化合物身份与孔/视野完全绑定。一个嵌入只要识别“这是哪个孔”，就能获得很高的化合物检索分数。原始像素基线（0.0664）明显高于随机（0.0450），证明“孔外观”本身就有信号。

文档已经诚实标注了这一点，但最终表述仍应限制为：

> “在 BBBC022 plate 20585 抢救出的 64 个视野上，按化合物检索能力排序”，不是“通用 Cell Painting 方法排名”。

---

## 6. Benchmark 合理性与风险

### 6.1 预处理偏差（实测）

| 偏差 | 实测影响 | 评价 |
|---|---|---|
| DP 用 joint min-max，官方是 per-channel min-max | 8 个裁剪 maxdiff 7.36，cosine 0.94–0.997；官方顺序+per-channel 的 mAP 0.0691 vs 当前 0.0684 | 结论稳健，但输入不是官方预处理 |
| CA-MAE 每裁剪按 max 缩放到 255 | 0.8× 全局缩放后 cosine 0.9998 | 模型含 InstanceNorm，基本尺度不变 |
| CellCLIP/DINOv2 每裁剪按 max 缩放到 255 | 0.8× 缩放后下游 CellCLIP cosine 0.9866 | 有一定敏感性 |
| CellCLIP 只取 1 个 crop，上游默认 5 个随机 crop 平均 | 文档已注明 | 可复现性提高，但偏离官方抽取 |
| CellCLIP 通道顺序与官方未对齐 | mAP 0.1337 vs 0.1338 | 本次影响很小，但契约错误 |
| DP 用 32×32，官方集成测试用 128×128 | 未重跑 128 裁剪 | **可能低估 DeepProfiler** |
| DP 没有照明校正 | 官方 pipeline 有 illum correction 阶段 | 可能影响 DP 结果 |
| O2-VAE 吃核剪影，官方是 MEF 全细胞剪影 | 文档已注明 | 模态不匹配，低分不能说明实现坏了 |
| CellCLIP 用了 DINOv2-Giant，其他方法吃像素 | 文档已注明 | 赢的是整条链路，不是 CellCLIP 架构单独获胜 |

### 6.2 指标与统计

做得好的部分：

- 随机投影基线 AUC = 0.4996，校准可信。
- 逐维 z-score，避免单维方差主导（CA-MAE 71%、O2-VAE 57%）。
- 按 mock 孔交叉验证 AUC，修掉了质心泄漏。
- 原始像素基线回答“嵌入是否优于直接看像素”。

有问题的部分：

#### ① Bootstrap CI 不成立（P0）

`_probe/bootstrap_ranking.py:60-66` 是有放回地重采样孔，然后直接在包含重复行的样本上算 mAP。重复行会彼此成为最近邻，抬高 AP。实测：

```
cellclip point     = 0.1338
bootstrap mean     = 0.1864
bootstrap CI95     = [0.1615, 0.2176]
```

**点估计 0.1338 根本不在 CI 内。** 这不是“区间偏宽”，而是 bootstrap 分布整体偏移。

后果：

- 表里的 CI 不能解释为“点估计的 95% 置信区间”；
- `P(cellclip > ca_mae)=1.000` 的精确数值被夸大（4 折排名稳定，定性结论仍可信，但不要引用这个 P 值）。

建议改为：留一孔 / out-of-bag 评估、孔级成对置换检验，或“参考集 + 留出 query 集”的检索评估。

#### ② mock 被当成一个化合物类

`score_benchmark.py` 里：

```python
labels = meta['compound'].astype(str)
```

mock 的 NaN 会变成字符串 `'nan'`，成为第 54 个类，贡献约 17% 的 query。建议在 kNN/mAP 中排除 mock，或单独报告。

#### ③ 没有 batch correction

CA-MAE manifest 自己写了 “embeddings need batch correction (PCA-CenterScale or TVN) before comparison”，但 benchmark 只做逐维 z-score，没有 TVN。每化合物一个孔的情况下，batch correction 会与化合物效应完全混淆，但至少应报告“不校正”和“按孔校正”两种结果。

#### ④ 没有与论文指标对齐

这是项目自建的化合物检索协议，不是 CA-MAE/CellCLIP 论文协议。它能回答“在这份数据上谁排序更好”，不能回答“是否复现论文指标”。

### 6.3 “期望结果”的准确表述

| 层面 | 结论 |
|---|---|
| 项目内部期望 | ✅ 完全复现：全量指标一致，4 折排名稳定 |
| 统一接口可用性 | ✅ 4 方法端到端跑通 |
| 官方方法学复现 | ❌ 不能声称：测试集、预处理、评估协议都不足 |
| 通用方法排名 | ❌ 不能声称：每化合物仅 1 孔/1 浓度 |

---

## 7. 代码质量与软工评估

### 7.1 做得好的地方

- 契约层零重依赖，`check_compatible` 纯函数，可在无 torch/TF 环境跑。
- `errors.py` 把“输入错 / 环境缺依赖 / 权重缺失”分开，可操作性强。
- `weights.py` 禁止隐式下载，sha256 记忆化校验。
- `BaseAdapter` 统一了加载、设备选择、幂等 `load()`。
- `registry` 一张表，新增方法只需两处改动。
- `embedding_benchmark.py` 的 RV coefficient、许可闸门、随机方法重复聚合都是合理设计。
- 测试设计区分“契约测试（无框架）”和“集成测试（真实权重）”，并把已发现缺陷钉成回归测试。

### 7.2 关键缺陷清单

#### P0：会直接导致错误结论或文档承诺失效

| # | 缺陷 | 位置 | 影响 |
|---|---|---|---|
| 1 | **DeepProfiler 通道顺序 RNA/ER 错误** | `models/registry.py:96`、`models/contract.py:45-46`、`_probe/build_bbbc022_cells.py:53-60` | 契约拒绝正确输入；benchmark DP 输入不是上游官方语义 |
| 2 | **`models.embed_all` 未导出** | `models/__init__.py:51-80,101-102` | `PROJECT_STATUS.md` 的核心调度 API 实际不可用 |
| 3 | **落盘丢失 `sample_metadata`** | `models/embedding.py:156-187` | benchmark `.npz` 只有 `features` + 摘要，没有 `cell_id`，事后无法对齐 |
| 4 | **`summary.json` 的 `finished_at` / `written` 永远为空** | `embedding_benchmark.py:220-222` | 先写文件再设置字段，summary 记录不完整 |
| 5 | **CellCLIP 在 `unit` 省略时接受 4D 像素堆** | `models/contract.py:583-639` | 违反正层的“加载前拦住静默错误”目标 |
| 6 | **Bootstrap CI 点估计落在区间外** | `_probe/bootstrap_ranking.py:60-66` | 统计推断不可直接引用 |

#### P1：影响调度/复现/可维护性

| # | 缺陷 | 位置 |
|---|---|---|
| 7 | `BenchmarkPlan` 只有一个 `unit`/`channels`，无法表达逐方法输入契约 | `embedding_benchmark.py:78-91` |
| 8 | `PreprocessSpec` 只声明不执行，预处理散落在外部脚本 | `contract.py`、`_probe/build_bbbc022_cells.py` |
| 9 | O2-VAE `strict=False` 未断言 missing key 只能是 buffer | `adapters/o2vae.py:161` |
| 10 | DeepProfiler `by_name=True` 没有加载层数断言 | `adapters/deepprofiler.py:116` |
| 11 | DINOv2-Giant 链路没有哈希/版本溯源 | `registry.py:223-230`、`cellclip_vectors.json` |
| 12 | 整个 `models/`、`embedding_benchmark.py`、两个测试文件**都未被 git 跟踪** | `git ls-files` 返回 0 |
| 13 | benchmark/scoring 脚本在仓库外，且硬编码绝对路径 | `bootstrap_ranking.py:19`、`embed_dataset.py:22` |
| 14 | `models.embed` 不能传 `device`/`weights_roots`；`feature_layer`、`return_channelwise` 等公开参数不可达 | `embedding.py:353-366`、`deepprofiler.py:136-148` |
| 15 | `embed_all` 用魔法键 `__errors__` 混进 `dict[str, EmbeddingResult]` | `embedding.py:687` |
| 16 | 打包缺失 torch extras；`deepprofiler` extra 仍钉 0.3.1（文档明确说该版本 setup.py 有问题） | `pyproject.toml` |

#### P2：文档/卫生问题

- `adapters/o2vae.py:1-28` 仍写“自训 32×32、dim 64”，与 registry 的官方 MEF、512 维、64×64 冲突。
- `adapters/deepprofiler.py:18-21` 仍写 `scale='raw'`、`value_range=None`，与 manifest 的 `per_crop_minmax` 冲突。
- `adapters/unavailable.py:5-7` 仍写“O2-VAE 从未发布权重、四个方法不可用”，与现状冲突。
- `docs/embedding_interface.md` 仍写 O2-VAE dim 64/自训、3 个方法可跑。
- `PROJECT_STATUS.md` 的测试数量 33+34 与实测 66 不完全一致；`models.embed_all` 文档与 facade 不一致。
- 死代码：`embedding.py:_import_guard`、`weights.py:_dedupe` 未被使用。
- `CellCLIP` 把上游架构复制进 adapter；在 CC BY-NC-ND 许可下可能构成衍生作品风险，需法务确认。
- `o2vae` 适配器把上游仓库根加入 `sys.path`，裸 `import models` 会被遮蔽。

### 7.3 测试覆盖

本轮实测：

```
tests/test_models_contract.py + tests/test_embedding_benchmark.py
66 passed in 0.21s

全量套件
14 failed, 192 passed
```

失败集中在 delivery/orchestration/public-API 等既有模块，与 embedding 层无关。

新增层缺少：

- `models.embed_all` 的 facade 存在性测试；
- CellCLIP 4D 像素堆的拒绝测试；
- `save_npz` 保留 `sample_metadata` 的测试；
- `summary.json` 时间戳/写出路径的测试；
- 真实权重集成测试（目前只在 `_probe` 里）；
- benchmark 指标/统计的回归测试（指标脚本不在仓库内）。

---

## 8. 建议的修复顺序

### 8.1 最小 P0 修复

1. **修正通道契约**
   - `DEEPPROFILER_CHANNELS` 改为 `('DNA','ER','RNA','AGP','Mito')`。
   - 为 CellCLIP 单独声明通道顺序，不复用 DeepProfiler 常量。
   - 修正 `build_bbbc022_cells.py` 的 DP 输入顺序，重跑 DP 结果。
   - 在 parity 脚本中增加“官方 README/test 顺序”的对照，明确参照对象。

2. **导出 `embed_all`**
   ```python
   # models/__init__.py
   _declare(_SELF + 'embedding', '...', 'embed_all', ...)
   # __all__ 增加 'embed_all'
   ```
   并加 `hasattr(models, 'embed_all')` 测试。

3. **落盘保留行身份**
   ```python
   for key, values in self.sample_metadata.items():
       payload[f'sample_{key}'] = np.asarray(values)
   ```

4. **修正 summary 写入顺序**
   先设置 `finished_at` 和 `written`，再 `_write_artefacts`。

5. **收紧 CellCLIP 契约**
   ```python
   if manifest.input_unit is InputUnit.CHANNEL_EMBEDDINGS:
       require ndim == 3 and shape[-1] == manifest.input_feature_dim
   ```

6. **修正 bootstrap**
   - 留一孔 / out-of-bag；
   - 或孔级成对置换检验；
   - 或“参考集 + 留出 query 集”的检索评估。

### 8.2 让接口真正适合调度的改造

建议把“声明”和“执行”分开：

```python
@dataclass(frozen=True)
class Payload:
    array: Any
    unit: InputUnit
    channels: tuple[str, ...] | None = None

inputs: dict[str, Payload] = {
    'deepprofiler': Payload(dp_crops, SINGLE_CELL_CROP, CHANNELS),
    'ca_mae':       Payload(ca_crops, SINGLE_CELL_CROP, ...),
    'cellclip':     Payload(clip_vectors, CHANNEL_EMBEDDINGS, CHANNELS),
    'o2vae':        Payload(masks, SINGLE_CELL_CROP, ('shape',)),
}
results = models.embed_all(list(inputs), inputs, on_error='collect')
```

同时：

- `BenchmarkPlan` 的 `unit`/`channels` 改为可选的 per-method `Payload`；
- `embed_all` 返回 `BatchResult(results, errors)`，不再用 `__errors__` 魔法键；
- adapter 构造参数通过 `embed(..., device=..., weights_roots=...)` 传入；
- `PreprocessSpec` 增加可执行 hook：`models.prepare(name, raw)` 或 `ManifestPreprocessor`；
- 输出侧增加 `output_ndim` / `dim_rule`，正确处理 uniDINO 的 `384*C`。

### 8.3 Benchmark 协议建议

- 测试集：至少多孔/多浓度；当前 64 视野只能作为 smoke benchmark。
- 指标：保留 mAP/kNN，但排除 mock 类或单独报告；增加按孔留出的检索评估。
- 预处理：对每个方法使用官方预处理；至少对 DP 使用 per-channel min-max + 官方通道顺序，并测试 64/128 裁剪。
- 统计：按孔成对检验；不要报告无法解释的 bootstrap CI。
- 溯源：记录 CellCLIP 链路中的 DINOv2-Giant 哈希、crop 数、processor 设置。
- 外部验证：如果要声称“方法排名”，需要与论文协议或公开 benchmark 对齐。

---

## 9. 本轮实测产物与命令

### 9.1 已运行的命令

```powershell
# 契约 + benchmark 测试
.\.venv\Scripts\python.exe -m pytest tests/test_models_contract.py tests/test_embedding_benchmark.py -q
# -> 66 passed

# 全量套件
.\.venv\Scripts\python.exe -m pytest tests/ -q
# -> 14 failed, 192 passed

# doctor
PYTHONPATH=src D:\ep_bench\venv\Scripts\python.exe -c "from cellpaint_pipeline import models; ..."
# -> deepprofiler / ca_mae / cellclip / o2vae = ready

# 全量 benchmark 复算
.\.venv\Scripts\python.exe "D:\项目\cellpainting claw复现\_probe\score_benchmark.py" `
  --cells-dir E:\bbbc022\cells `
  --output E:\bbbc022\cells\benchmark_scores_recheck.json

# 4 细胞 × 4 方法统一接口 slice
# -> E:\bbbc022\cells\bench_slice\{summary.json, embeddings\*.npz}
```

### 9.2 新增实验产物

```
E:\bbbc022\cells\bench_slice\
E:\bbbc022\cells\benchmark_scores_recheck.json
E:\bbbc022\cells\emb_deepprofiler_perchan.npy
E:\bbbc022\cells\emb_deepprofiler_official_order.npy
E:\bbbc022\cells\emb_cellclip_er_rna.npy
```

### 9.3 关键文件索引

| 主题 | 文件 |
|---|---|
| 契约原语 | `src/cellpaint_pipeline/models/contract.py` |
| 注册表 | `src/cellpaint_pipeline/models/registry.py` |
| 调度入口 | `src/cellpaint_pipeline/models/embedding.py` |
| 权重解析 | `src/cellpaint_pipeline/models/weights.py` |
| 适配器 | `src/cellpaint_pipeline/models/adapters/` |
| benchmark 执行器 | `src/cellpaint_pipeline/embedding_benchmark.py` |
| 契约测试 | `tests/test_models_contract.py` |
| benchmark 测试 | `tests/test_embedding_benchmark.py` |
| 全量评分 | `D:\项目\cellpainting claw复现\_probe\score_benchmark.py` |
| bootstrap | `D:\项目\cellpainting claw复现\_probe\bootstrap_ranking.py` |
| 数据构建 | `D:\项目\cellpainting claw复现\_probe\build_bbbc022_cells.py` |
| CellCLIP 向量 | `D:\项目\cellpainting claw复现\_probe\make_cellclip_vectors.py` |

---

## 10. 最终判断

**接口层：方向正确，工程质量中上，但还没有完成“统一 benchmark/调度”的最后一步。**

- 作为“带契约校验的模型分发器”，它是合格的。
- 作为“一个数据集喂所有方法”的调度器，它还不合格：预处理在层外、逐方法 unit/channels 不可表达、输出行身份落盘丢失、facade 缺 `embed_all`。
- 作为“官方复现层”，它有一个必须修的通道顺序缺陷；现有 parity 只证明了与项目自己的 demo 一致。

**Benchmark：能复现项目自己的期望结果，但不能支撑“通用方法排名”。**

- 全量数字、4 折排序、统一接口 slice 都通过；
- 但数据集每化合物仅 1 孔/1 浓度，孔混杂严重；
- bootstrap CI 统计上不成立；
- DP 的预处理/裁剪尺度不是上游官方设置；
- CellCLIP 的 DINOv2-Giant 链路未完整溯源。

最准确的表述：

> **“在 BBBC022 plate 20585 的 3,857 个细胞上，按化合物检索能力排序”——不是“Cell Painting embedding 方法排名”。**

建议优先完成：

1. 通道契约修正；
2. `embed_all` 导出；
3. `sample_metadata` 落盘；
4. CellCLIP 契约收紧；
5. bootstrap 修正。

这五项完成后再补 per-method payload 和库内预处理，接口才真正具备“benchmark + 调度”的资格。
