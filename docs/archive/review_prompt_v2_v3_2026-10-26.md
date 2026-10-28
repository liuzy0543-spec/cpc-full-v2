# 审查提示词：Embedding 统一框架 v2 / v3

**用法**：把本文档整段交给审查者（人或 agent）。它按 **黑盒 → 白盒 → benchmark**
三段执行，每段有独立的通过标准。**不要跳段**：黑盒没全过就做白盒，会把
"跑不起来"误诊成"实现错了"。

**一句话任务**：
> 先证明七个 embedding 方法**都能被同一个接口驱动并返回格式正确的输出**；
> 再逐个对照论文与官方 GitHub 代码检查实现逻辑是否一致；
> 最后用已有数据集跑 benchmark 并如实报告。

---

## 0. 审查对象

仓库：`D:\项目\CellPainting-Claw-v2`（对拍树 `D:\cpc_c_v2`，应与前者 byte-identical）

| 版本 | 方法数 | 权重 | 用途 |
|---|---|---|---|
| **v2**（默认） | 4 | 全部官方 | 可发表 |
| **v3** | 7 | 4 官方 + 3 替身 | 接口完整性 |

```python
from cellpaint_pipeline import models
models.declared()            # v2: 4 个
models.set_profile('v3')
models.declared()            # v3: 7 个
models.set_profile('v2')     # 复位
```

**审查者必须记住的前提**：v3 的 unidino / dinocells / morphogenie 用的是
**本地训练的替身权重**（`weights_provenance='surrogate'`），不是官方权重。
它们的接口是真的，分数不是论文的。任何把它们当官方方法报告的审查结论都是错的。

---

## 1. 模块地图（各模块作用）

只列与 embedding 相关的。行号供定位，不必逐行读。

### 1.1 契约层（不含框架依赖，可在无 torch/TF 的机器上 import）

| 文件 | 行数 | 作用 |
|---|---|---|
| `models/contract.py` | 785 | 契约定义：`ModelManifest`（方法全部元数据）、`ChannelSpec`、`PreprocessSpec`、`WeightsSpec`、`check_compatible()`（纯函数，不 import 框架）。**任何"这个方法要什么输入"的答案都在这里** |
| `models/errors.py` | 103 | 错误类型：`UnknownModel` / `IncompatibleInput` / `MissingDependency` / `WeightsUnavailable` / `EmbeddingError`。**拒绝时给 reason code**，是审查时定位问题的入口 |
| `models/weights.py` | 238 | 权重定位与校验：`resolve_weights()` 按 显式 roots → `CPC_WEIGHTS_ROOT` → 默认根 查找，`verify_sha256()` 校验 |
| `models/registry.py` | 824 | 注册表：`ADAPTERS`（v2 的 4 个）、`SURROGATES`（v3 的 3 个）、`PENDING`（已取消但保留契约的 3 个）、剖面切换 `set_profile()`、`describe()`/`probe()` |

### 1.2 执行层

| 文件 | 行数 | 作用 |
|---|---|---|
| `models/preprocessing.py` | 417 | **执行** `PreprocessSpec` 声明的变换：通道重排、resize、`source_map` 数值映射、布局、dtype、dinocells 的磁带折叠。6 种 `source_map` 全实现，`matlab_centring` 明确拒绝 |
| `models/embedding.py` | 1157 | 公共入口：`embed()` / `embed_all()`，契约检查 → 预处理 → 适配器 → 输出宽度校验 → `EmbeddingResult`。`assume='raw_crop'` 是"输入是原始裁片，请帮我预处理"的开关 |
| `models/__init__.py` | 267 | 门面：`embed` / `embed_all` / `declared` / `runnable` / `pending` / `describe` / `plan` / `doctor` / `set_profile`。**用户只应该 import 这个** |
| `models/adapters/base.py` | 167 | 适配器基类：`resolve_weights()`、`select_device()`、`load()`、`embed()` |
| `models/adapters/{deepprofiler,ca_mae,cellclip,o2vae}.py` | 197/153/310/266 | 四个**官方权重**适配器 |
| `models/adapters/{unidino,dinocells,morphogenie}.py` | 100/104/103 | 三个**替身**适配器 |
| `models/adapters/_surrogate.py` | 207 | 替身共享件：`SPECS`（架构规格）、`tiny_vit()`、`tape_spatial()`、`sha256_file()` |
| `models/adapters/unavailable.py` | 94 | 保留的"声明但不可运行"工厂（供 `PENDING` 场景） |

### 1.3 调度与评估

| 文件 | 作用 |
|---|---|
| `models/embedding_benchmark.py` | 基准计划构建（`build_benchmark_plan` / `run_benchmark`），含 chunk 分段 |
| `scripts/embedding_benchmark/embed_dataset.py` | 为 4 个官方方法生成嵌入（读预构建数组） |
| `scripts/embedding_benchmark/build_v3_surrogates.py` | 为 3 个替身生成嵌入（**全程走 `models.embed`**） |
| `scripts/embedding_benchmark/train_surrogates.py` | 训练替身（VICReg 自监督，仅 mock 细胞） |
| `scripts/embedding_benchmark/score_benchmark.py` | 打分：kNN / mAP / silhouette / mock-AUC，`--profile` 选 v2/v3 |
| `scripts/embedding_benchmark/stability_check.py` | 4 折稳定性：稳定前缀 + 互换对 |
| `scripts/embedding_benchmark/verify_io_uniformity.py` | 四方法 IO 一致性验证 |
| `scripts/embedding_benchmark/parity_vs_official_dp.py` | 与官方 DeepProfiler **逐位对拍** |

### 1.4 测试

| 文件 | 覆盖 |
|---|---|
| `tests/test_models_contract.py` | 契约与注册表 |
| `tests/test_models_preprocessing.py` | 6 种 `source_map` 的数值正确性 |
| `tests/test_models_profiles.py` | v2/v3 剖面与 provenance 标注 |
| `tests/test_models_v3_dispatch.py` | 七方法单次 `embed_all` 调度 |
| `tests/test_parity_deepprofiler.py` | 与官方逐位一致（需真实 TF + 权重） |

---

## 2. 阶段一：黑盒测试（**先做，且必须全做**）

### 2.1 目的与通过标准

**目的**：证明接口能驱动每一个方法，输出格式符合 embedding 约定。

**通过标准只有四条** —— 不看分数好坏：

1. 调用**返回**而非抛异常（`MissingDependency` / `WeightsUnavailable` 记为"环境未就绪"而非"失败"，但必须写清楚是哪种）
2. 返回 `EmbeddingResult`，且 `as_numpy()` 是**二维 `(N, dim)`** 浮点数组
3. `values` 全部有限（无 NaN/Inf）
4. `dim` 与 manifest 声明**一致**（见 2.3 的特殊规则）

> 用户明确要求："结果暂时不强求，务必全部测试这几个 embedding"。
> 所以这一步**不需要**判断嵌入有没有生物学意义，只需要"真的跑通了"。

### 2.2 环境

```powershell
# 测试用（有 torch/TF）
$env:PATH = 'C:\Users\DELL\miniconda3\envs\torch_gpu\Library\bin;' + $env:PATH
$env:PYTHONPATH = 'D:\项目\CellPainting-Claw-v2\src'
D:\ep_bench\venv\Scripts\python.exe ...
```

权重根：`E:\cpc_weights`。数据：`E:\bbbc022\cells`。

### 2.3 七个方法的输入构造（照抄即可）

数据文件都是现成的：

| 文件 | 形状 | 用途 |
|---|---|---|
| `cells_ca.npy` | (3857, 5, 256, 256) uint8 | 通用裁片源 |
| `cells_dp.npy` | (3857, 32, 32, 5) float32 | deepprofiler 输入 |
| `cellclip_vectors.npy` | (3857, 5, 1536) float32 | cellclip 输入（DINOv2 向量） |
| `cells_o2.npy` | (3857, 1, 64, 64) float32 | o2vae 输入（剪影） |

```python
import numpy as np
from cellpaint_pipeline import models
from cellpaint_pipeline.models import registry, preprocessing

models.set_profile('v3')
C = r'E:\bbbc022\cells'
block = np.asarray(np.load(C + r'\cells_ca.npy', mmap_mode='r')[:2])   # (2,5,256,256)
bhwc  = np.moveaxis(block, 1, -1).astype('uint16') * 257                # (2,256,256,5)
dna   = block[:, 0]
masks = (dna > dna.mean(axis=(1, 2), keepdims=True)).astype('uint8')

# 1) deepprofiler：32x32 裁片
dp = np.asarray(np.load(C + r'\cells_dp.npy', mmap_mode='r')[:2])
r1 = models.embed('deepprofiler', dp, assume='raw_crop')

# 2) ca_mae：256x256 5 通道
r2 = models.embed('ca_mae', bhwc, assume='raw_crop')

# 3) cellclip：每通道 DINOv2-Giant 向量，不是像素
vec = np.asarray(np.load(C + r'\cellclip_vectors.npy', mmap_mode='r')[:2])
r3 = models.embed('cellclip', vec)

# 4) o2vae：单通道核剪影
sil = np.asarray(np.load(C + r'\cells_o2.npy', mmap_mode='r')[:2])
r4 = models.embed('o2vae', sil)

# 5) unidino（替身）：224x224 5 通道，逐通道过 ViT
r5 = models.embed('unidino', bhwc, assume='raw_crop')

# 6) dinocells（替身）：磁带 (H, W*(C+1))，必须带 mask
r6 = models.embed('dinocells', bhwc[0], masks=masks[0], assume='raw_crop')

# 7) morphogenie（替身）：3 通道；prepare 拒绝 matlab_centring，故直接给张量
r7 = models.embed('morphogenie', block[:, :3].astype('float32') / 255.0,
                  channels=('C0', 'C1', 'C2'))
```

**期望输出宽度**：

| 方法 | dim | 说明 |
|---|---|---|
| deepprofiler | 672 | EfficientNet-B0 池化层 |
| ca_mae | 384 | ViT-S |
| cellclip | 512 | |
| o2vae | 512 | z_mu |
| **unidino** | **1920** | 逐通道 384 × 5 通道，**不是 384** |
| dinocells | 384 | CLS token |
| morphogenie | 10 | c_dim，**不是 256** |

> **dinocells 的 N 是 1，不是 2。** 上面第 6 行传的是 `bhwc[0]`（单条磁带），
> 所以返回 `(1, 384)`。一条磁带只能放一个细胞（多个细胞必须共享 mask），
> 这是契约的正确表现而非缺陷 —— 判它 `(2, 384)` 的审查者会误报。
> 要跑 N>1 请像 `build_v3_surrogates.py` 那样逐细胞循环。

**上面这段代码在本仓库已原样执行过**（2026-10-26，`ep_bench` 环境），输出：

```
deepprofiler  dim=672   shape=(2, 672)   finite=True  weights_sha256[:12]=87b756f57933 prov=official
ca_mae        dim=384   shape=(2, 384)   finite=True  weights_sha256[:12]=f6e5f1c97101 prov=official
cellclip      dim=512   shape=(2, 512)   finite=True  weights_sha256[:12]=66c2d32a44a3 prov=official
o2vae         dim=512   shape=(2, 512)   finite=True  weights_sha256[:12]=aa70cfa8b080 prov=official
unidino       dim=1920  shape=(2, 1920)  finite=True  weights_sha256[:12]=e53838fe8073 prov=surrogate
dinocells     dim=384   shape=(1, 384)   finite=True  weights_sha256[:12]=5475ec885ae6 prov=surrogate
morphogenie   dim=10    shape=(2, 10)    finite=True  weights_sha256[:12]=491029978fef prov=surrogate
```

> **`sha` 列的口径（P1）**：上面这一列是**权重文件 sha256 的前 12 位**
> （`EmbeddingResult.to_dict()['weights_sha256']`），**不是嵌入数组的哈希**。
> 二者不是一回事：同一个嵌入数组的 sha 是 `a05475038143`（deepprofiler）这类值，
> 与权重摘要无关。若把它理解成"结果哈希"，会误报 7/7 不一致。
> 该权重摘要也是 §7 表格里那组数，核对时应与 §7 对照而不是与嵌入数组对照。

### 2.4 必须额外验证的接口性质

| # | 检查 | 判定 |
|---|---|---|
| B1 | 七个结果对象字段集合是否完全相同（`set(r.to_dict())`） | 必须相同 |
| B2 | 每个结果是否带 `weights_sha256` | 必须非空（**在 `to_dict()` 顶层，不在 `metadata` 里** —— 按 `metadata['weights_sha256']` 取会误判为缺失） |
| B3 | 每个结果 `metadata['weights_provenance']` | 4 个 `official`，3 个 `surrogate` |
| B4 | `models.embed_all(names, payload)` **一次**跑七个 | 七个都返回 |
| B5 | v2 下 `models.embed('unidino', ...)` | 必须抛 `UnknownModel` |
| B6 | 同一输入连跑两次，两次结果是否一致 | 记录差异（morphogenie 官方是 stochastic，替身是 deterministic） |
| B7 | `models.pending()` 返回三个已取消契约 | 名字为 dinocells/unidino/morphogenie |
| B8 | `models.probe('dinocells')['runnable']` | **True** —— 诊断面不得对替身报假阴性 |
| B9 | `models.runnable()` | 返回 **7** 个（不是 6 个） |
| B10 | uint8 喂 unidino | **抛 `IncompatibleInput(reason='value-range')`** |
| B11 | 256×256 喂 deepprofiler | **抛 `IncompatibleInput(reason='spatial-size')`**，不是 Keras 的 `ValueError` |
| B12 | `models.prepare('dinocells', …)` 的输出形状 | `(N, 5, 96, 96)`，且 **N>1 不报错**（官方可成批） |
| B13 | dinocells 的 prepared 结果 vs 手工按官方 `fold_channels` 展开 | **逐位相等** |
| B14 | 替身的 `weights_sha256` | 与 manifest 里钉住的值一致 |

> **B4 的正确用法（P8）**：`embed_all` **不提供** `assume=` / `masks=`，
> 且 `channels=` 是**所有方法共享的单一值**（传 morphogenie 的
> `('C0','C1','C2')` 会让 5/7 方法被 `channel-set` 拒绝）。
> 正确姿势是**先逐方法 `models.prepare()` 预制备**，再以
> `{方法: 载荷}` 映射 + `unit` 映射调用 `embed_all`。
> §2.3 的代码块**不能原样**用于 B4。

### 2.5 一行式自检

```powershell
python -m pytest tests\test_models_v3_dispatch.py -q     # 4 项，七方法调度
python -m pytest tests\test_models_profiles.py -q        # 11 项，剖面与标注
```

### 2.6 黑盒记录表（审查者填写）

| 方法 | 能跑 | dim | dtype | 有限 | sha256 | provenance | 异常/原因 |
|---|---|---|---|---|---|---|---|
| deepprofiler | | | | | | | |
| ca_mae | | | | | | | |
| cellclip | | | | | | | |
| o2vae | | | | | | | |
| unidino | | | | | | | |
| dinocells | | | | | | | |
| morphogenie | | | | | | | |

**任一方法"不能跑"就停在这里**，把原因（`MissingDependency` 缺什么模块 /
`WeightsUnavailable` 缺哪个文件 / 其他异常栈）写清楚，**不要进入白盒阶段**。

---

## 3. 阶段二：白盒检查（实现逻辑是否与论文/官方代码一致）

### 3.1 论文与官方代码对照表

官方源码在 `D:\official_src\`，权重在 `E:\cpc_weights\`。

| 方法 | 论文 | 官方 GitHub | 本地源码 | 本地权重 |
|---|---|---|---|---|
| **deepprofiler** | Moshkov et al., *Learning representations for image-based profiling of perturbations*, **Nature Communications 15 (2024)**, doi:[10.1038/s41467-024-45999-1](https://doi.org/10.1038/s41467-024-45999-1) | [cytomining/DeepProfiler](https://github.com/cytomining/DeepProfiler) | `1-deepprofiler/` | `E:\cpc_weights\deepprofiler`（Zenodo [7114558](https://doi.org/10.5281/zenodo.7114558)，Cell Painting CNN v1，EfficientNet-B0，5 通道 DNA/ER/RNA/AGP/Mito） |
| **ca_mae / OpenPhenom** | *Masked Autoencoders are Scalable Learners of **Cellular Morphology***, [arXiv:2309.16064](https://arxiv.org/abs/2309.16064)（NeurIPS 2023）<br>*Masked Autoencoders **for Microscopy** are Scalable Learners of **Cellular Biology***, [arXiv:2404.10242](https://arxiv.org/abs/2404.10242)（CVPR 2024）<br>**两篇标题不可互换**；作者单位为 Recursion | [recursionpharma/maes_microscopy](https://github.com/recursionpharma/maes_microscopy) | `2-camae-openphenom/` | `E:\cpc_weights\openphenom_pkg` |
| **cellclip** | Lu, Weinberger, Kim, Lee, *CellCLIP: Learning Perturbation Effects in Cell Painting via Text-Guided Contrastive Learning*, [arXiv:2506.06290](https://arxiv.org/abs/2506.06290)（NeurIPS 2025） | [suinleelab/CellCLIP](https://github.com/suinleelab/CellCLIP) | `3-cellclip/` | `E:\cpc_weights\cellclip` |
| **o2vae** | Burgess, Nirschl, Zanellati, Lozano, Cohen, Yeung-Levy, *Orientation-invariant autoencoders learn robust representations for shape profiling of cells and organelles*, **Nature Communications 15:1022 (2024)**, doi:[10.1038/s41467-024-45362-4](https://doi.org/10.1038/s41467-024-45362-4)<br>（预印本 bioRxiv [2022.12.08.519671](https://www.biorxiv.org/content/10.1101/2022.12.08.519671v1)）<br>**"mefs" = Mouse Embryonic Fibroblasts**，不是任何缩写展开 | [github.com/jmhb0/o2vae](https://github.com/jmhb0/o2vae)（Zenodo 10.5281/zenodo.10206848）<br>**⚠️ 论文正文写 `d = 128`，而发布权重实测 `zdim=512`（以代码/权重为准）** | **无**（运行期 checkout） | `E:\cpc_weights\o2vae\o2vae_mefs_official.pt`（上游路径 `wandb/pretrained_models/model_mefs.pt`，与上游字节级同一） |
| **unidino** | Morelli, Kim, Hecker, Geibel, Marin Zapata, *uniDINO: assay-independent feature extraction for fluorescence microscopy images*, **Comput Struct Biotechnol J 27:928–936 (2025)**, doi:[10.1016/j.csbj.2025.02.020](https://doi.org/10.1016/j.csbj.2025.02.020) | [Bayer-Group/uniDINO](https://github.com/Bayer-Group/uniDINO) | `5-unidino/` | 官方：Zenodo [14988837](https://doi.org/10.5281/zenodo.14988837)（1.38 GB，**未取得**）<br>**实际用替身** |
| **dinocells** | *Unbiased single-cell morphology with self-supervised vision transformers*, [bioRxiv 2023.06.16.545359](https://www.biorxiv.org/content/10.1101/2023.06.16.545359v1)；复现分析在 [broadinstitute/Dino4Cells_analysis](https://github.com/broadinstitute/Dino4Cells_analysis) | [broadinstitute/DINO4Cells_code](https://github.com/broadinstitute/DINO4Cells_code) | `6-dinocells/` | 官方：Zenodo [8061392](https://zenodo.org/record/8061392)（3.71 GB 打包，**未取得**）<br>**实际用替身** |
| **morphogenie** | bioRxiv [2024.09.24.614684](https://www.biorxiv.org/content/10.1101/2024.09.24.614684v1) | [rashmisrm/MorphoGenie](https://github.com/rashmisrm/MorphoGenie) | `7-morphogenie/` | 官方：Google Drive `Models.7z`（**不可达**）<br>**实际用替身** |

### 3.2 逐方法的对拍锚点（行号已核对）

审查者应逐个确认"本项目实现"与"官方行"行为一致。

#### deepprofiler

| 检查点 | 官方位置 | 本项目位置 | 期望 |
|---|---|---|---|
| 通道顺序 | README：DNA, ER, RNA, AGP, Mito | `registry.py` deepprofiler manifest | **必须一致**。历史缺陷：曾用 DNA/RNA/ER/AGP/Mito（RNA 与 ER 互换）。✅ **该行已核实无误，保留** |
| 逐裁片 min-max | `deepprofiler/imaging/cropping.py:65-67`（`reduce_min`/`reduce_max` → `(x-min)/(max-min+eps)`） | `preprocessing.py` `per_crop_minmax_bhwc` | 逐样本逐通道独立归一化，**不是**整批 |
| **输入尺寸** | 官方尺寸**不是常量**：来自 `config["dataset"]["locations"]["box_size"]`（`cropping.py:125`；官方 test 配置是 **16**）。本仓库发货的 checkpoint 期望 **32×32**（读自其 `input_1` 形状） | manifest `spatial=(32,32)` + **`resize_policy='refuse'`** | 尺寸不符必须**抛 `IncompatibleInput(reason='spatial-size')`**；**不得静默 resize**（官方对裁片做的是 crop 而非 resize） |
| 输出层 | 672（`block6a_activation`）；`pool5` 为 1280 | manifest `output_dim=672`, `output_dim_alternatives=(1280,)` | |
| 逐位一致性 | — | `scripts/embedding_benchmark/parity_vs_official_dp.py` | 应报 `bitwise_equal=True`（样本仅 4 细胞） |

#### ca_mae

| 检查点 | 期望 |
|---|---|
| 输入 | 256×256、C 通道、**uint8**（逐裁片拉伸到 0–255） |
| 输出 | 384 |
| **通道数上限** | 官方 `max_in_chans=11`（`huggingface_mae.py:75`），pos_embed 按 11 模态生成 → 契约必须拒 C>11（`reason='channel-count'`） |
| **归一化定性** | 官方只有 `pixels/255`（`normalizer.py:4-7`）+ checkpoint 内 `InstanceNorm2d`（`huggingface_mae.py:88-91`），**没有任何逐图拉伸，也没有 16→8 转换代码**。本项目的 `per_crop_max255` 是**源域选择，不是官方行为**；它被官方 InstanceNorm 的仿射不变性抵消，实测残差 ≤0.7% std。**可写"残差 ≤0.7% std"，不可写"与官方一致"** |
| 官方映射 | `2-camae-openphenom/` 中 crop/归一化逻辑需逐行核对 |

#### cellclip

| 检查点 | 期望 |
|---|---|
| 输入 | **(N, 5, 1536)** DINOv2-Giant 每通道向量，**不是像素** |
| 链式依赖 | `chain_weights` 必须记录 DINOv2-Giant 的 sha256（4.5 GB） |
| 输出 | 512 |
| 关键风险 | 若把像素直接喂进去，`check_compatible` 必须拒绝。注意 **`check_compatible` 不产生 `chain-required`**：像素栈得到的是 `ndim` / `feature-width` / `channel-count`；`chain-required` 只由 `prepare()` 抛出 |

#### o2vae

| 检查点 | 官方位置 | 期望 |
|---|---|---|
| 输入 | `configs/config_mefs.py:28,33` | 128×128 MEF 裁片 resize 到 **64** |
| 输出 | `configs/config_mefs.py:46` `zdim=512` | 512；可用 `q_net.fully_net.3.bias` 形状 (1024,) = 2×zdim 交叉验证 |
| 特征层 | `models/vae.py:148-149` `z_mu, _ = torch.split(self.q_net(x), self.zdim, dim=1); return z_mu` | `z_mu` |
| 输入语义 | `configs/config_mefs.py:19` `data_dir=.../data/mefs/scaled/cell` | 官方权重训练于**细胞**剪影。本项目喂**核**剪影 → **推理期域偏移**。缺 mask 应拒 `mask-required` |
| 论文一致性 | 论文正文写 "embedding dimensions **d = 128**" | 发布权重实测 `zdim=512`（`p_net.fc.0.weight` 形状 (48,512)）。**以代码/权重为准**，但这条差异要写进补充材料 |
| 仓库依赖 | — | `requires_upstream_repo=True`，架构从上游 checkout import |

#### unidino（替身）

| 检查点 | 官方位置 | 期望 |
|---|---|---|
| 架构 | `inference.py:95-97`：`patch_size=16`，`arch="vit_small"` | 替身用 ViT-S/16 |
| **逐通道** | `source/inference_utils.py` `forward_inference`：`(batch,ch,crops,1,h,w)` → `(batch*ch*crops,1,h,w)`，每通道**单独**过网络 | 输出 `384×C`；本项目 `dim=1920` |
| 通道顺序敏感 | 拼接不是对称运算 | manifest `order_matters=True` |
| 预处理 | `source/image_ops.py` `img/(2**16-1)` | `div65535`，**输入须为 uint16**。喂 uint8 必须被拒（`reason='value-range'`）——它会变成全黑（prepared max ≈ 0.0039 而非 0.98），是静默错误 |
| 权重来源 | Zenodo doi:10.5281/zenodo.14988837；官方脚本 `run_dino_inference.sh:2` `--ckpt ./checkpoints/uniDINO.pth`；权重取自 `state_dict["teacher"]`（`source/dino/utils.py:98-109`） | manifest `weights.source` 应指向官方脚本用的文件名，而不是一个在官方仓库里不存在的名字 |
| 替身差异 | — | 深度 4 而非 12；见 3.4 |

#### dinocells（替身）

| 检查点 | 官方位置 | 期望 |
|---|---|---|
| **磁带折叠** | `data_utils/cellpainting_dataset.py:23-26`：`np.reshape(image, (h, channel_width, -1), order="F")`。**注意方向**：这是把磁盘上的磁带**展开**成 `(h, w, c)`；`mode="drop"`（默认）**丢掉 mask 通道**，模型输入是 `(C, h, w)` 的多通道堆叠（`in_chans=num_channels`） | **必须用 Fortran order**（用 C order 会静默打乱通道）。但**契约必须描述官方几何**：堆叠而非磁带 | 用 C order 会静默打乱通道 |
| 磁带形状 | 磁盘存储格式 `(H, W*(C+1))`，最后一路是 mask | 磁盘格式，不是模型输入格式 |
| 加载器归一化 | `data_utils/file_dataset.py` `norm_loader`：`img -= np.min(img, axis=(0,1)); img *= 255/np.max(img, axis=(0,1))` → uint8。**⚠️ `np.min(2D, axis=(0,1))` 是 0 维标量（已实测），所以这是「全图全局」min-max，不是"逐图逐通道"** | 官方随后由 `utils/augmentations.py:460-468` 的 `self_normalize` 做**逐通道 z-score**（`mean((-2,-1))`/`std(unbiased=False)`）；全局 min-max 对随后的 z-score 是仿射不变、被代数抵消（1e-15 量级） |
| 模型输入宽度 | `config.yaml`：`arch: vit_tiny`、`patch_size: 16`、`num_channels: 4`（HPA 变体）；CP 权重为 `DINO_cell_painting_base/small_checkpoint.pth`（ViT-B 768 / ViT-S 384） | 官方 CP 配置与 HPA 配置不同；官方代码未给出 CP 的裁片尺寸与通道数 |
| batch 语义 | 官方模型输入是 `(C,H,W)` 堆叠 → **天然可批**（`batch_size_per_gpu: 24`） | 本项目对 batch **不得**再以 `tape-batch` 拒绝 |
| 替身差异 | — | 架构与官方 CP 配置不同；见 3.4 |

#### morphogenie（替身）

| 检查点 | 官方位置 | 期望 |
|---|---|---|
| **输出宽度** | `MorphoGenie_Test.py:67` `c_dim=10`、`:68` `z_dim=256` | 嵌入宽 = **c_dim = 10**，不是 256 |
| 潜伏变量语义 | `:227` `get_zdist('gauss', args.c_dim, ...)`；`gan_training/config.py:43` `z_dim = z_dist.dim + c_dim` = 266 | 官方把 c_dim 当 z 分布维度，进一步印证上面的混淆 |
| 提取的是哪个张量 | `gan_training/models/dvae.py:393-396` 解包顺序 `(c, mu, logvar)` vs `MorphoGenie_Test.py:297` `mu_real, logvar, c_real = fvae(...)` → `mu_real` 实际绑定到 **`c`（重参数化采样）**；`:409-413` `c = reparametrize(mu, logvar)` | 官方持久化的是**重新参数化采样** `c` → 非确定性 |
| **引用文件消歧** | 推理路径是 **`gan_training/models/dvae.py`**（`c_dim`、三元组）。`dVAE/model.py` 是 **legacy 训练脚本**（用 `z_dim`、返回**四元组** `(x_recon, mu, logvar, z.squeeze())`），不在推理路径上 | 引用 `dVAE/model.py` 会误导审查者；结论（10 维、stochastic）不受影响 |
| 预处理 | MATLAB 居中。**源码就在官方仓库内**：`ImagePreProcessing/MultipleCells/CPA/GetCentImages.m`；缺的是 **MATLAB 运行时 + Cellpose + 原始数据**。居中裁剪为 **200×200**（`:26` `Crop=[Centerx-100, Centery-100, 200, 200]`），并在此 min-max 拉伸到 uint8（`:38`）；**256×256 缩放在 Python 侧**（`gan_training/utils.py:352-357` `Resize` + `Normalize(0.5)`） | 本项目 `matlab_centring` 以 `unsupported-preprocess` **拒绝**，这是正确行为；**理由**要按左列写，不要说"不在仓库" |
| 替身差异 | — | 返回 mu（确定性）；见 3.4 |

### 3.3 横切检查项

| # | 检查 | 判定 |
|---|---|---|
| W1 | `check_compatible()` 是否**不** import torch/TF | 在无框架解释器里能 import |
| W2 | 每个 manifest 的 `output_dim` 是否与适配器实际输出一致 | 不一致应被 `_check_output_width` 拒绝（`reason='output-width'`） |
| W3 | 通道顺序错时是否**真的**报错而非静默 | 用置换通道的输入测。**边界**：只有调用方**声明了**通道名才会报错；不声明 names 的置换契约无从检测（嵌入会变但不会报） |
| W4 | 16-bit 数据喂给 8-bit 方法 | 应拒 |
| W5 | `assume='raw_crop'` 是否等价于"手动 prepare + embed" | 数值应一致 |
| W6 | 权重 sha256 是否与实际文件相符 | `verify_sha256`；**替身也必须把摘要钉进 manifest** |
| W7 | 切换 profile 是否清空适配器缓存 | 否则同一名字复用错误权重 |
| W8 | `describe()` 是否带 `weights_provenance` | 否则渲染方法表时无法区分替身 |
| W9 | `probe()` / `runnable()` 的依赖来源 | 必须问**实际要跑的适配器**（`requires_frameworks`），不是 manifest 的 `requires`——后者描述的是官方实现 |
| W10 | `div65535` 这类"除以 2**16-1"的 scale 的 **dtype** 守卫 | uint8 必须被拒（`reason='value-range'`）：它数值上落在声明区间内，但暗 257 倍 |
| W11 | 声明了 `spatial` 的方法是否守尺寸 | 尺寸不符必须是 `IncompatibleInput(reason='spatial-size')`，**不是**框架内部异常；且**不得静默 resize** |
| W12 | 声明 `max_channels` 的方法是否守上限 | 超出必须 `reason='channel-count'`，因为官方 checkpoint 的 pos_embed 只支持有限模态 |

### 3.4 审查者必须核实的四个"已知差异"（不是 bug，但必须确认是差异）

| # | 差异 | 性质 |
|---|---|---|
| D1 | unidino 替身：ViT-S **depth 4**（官方 12），训练 300 步 | 架构同类、规模缩小。**必须确认 manifest 标注了 surrogate** |
| D2 | dinocells 替身：架构与官方 Cell Painting 配置不同（官方 CP 发布的是 ViT-S 384 与 ViT-B 768 两个 checkpoint） | 同 D1 |
| D3 | morphogenie 替身返回 **mu**（确定性），官方返回采样（随机） | manifest 已声明 `determinism='deterministic'`；**两者不可混比** |
| D4 | dinocells 的 mask 由 DNA 通道 Otsu 阈值生成，**非** CellProfiler 分割 | 替身模型的替身输入；已在脚本与文档标注 |

### 3.5 审查缺口（需要审查者补，本项目没做到）

| # | 缺口 | 影响 |
|---|---|---|
| **G1** | ~~o2vae 的论文/官方仓库未记录~~ **已补齐（2026-10-26）**：Burgess et al., *Nat Commun* 15:1022 (2024), doi:10.1038/s41467-024-45362-4；代码 `github.com/jmhb0/o2vae` | 已可对照论文。**新增两条必须写进论文的风险**：① 论文正文 `d=128` vs 发布权重 `zdim=512`；② 官方权重训练于**细胞**剪影而本项目喂**核**剪影 → 域偏移，必须声明为域外推理或改喂细胞剪影 |
| **G2** | 只有 deepprofiler 做过**真实数据逐位对拍** | 其余 6 个方法的预处理是"对照官方公式"的推理验证，**不是**实测对拍。报告中必须区分这两类 |
| **G3** | morphogenie 的 `source_map='matlab_centring'` 永远拒绝 | 官方路径无法端到端复现，只能确认"拒绝是对的"。**注意理由**：MATLAB 源码在官方仓库内，缺的是运行时与数据 |
| **G4** | ca_mae / o2vae 未用官方 demo 输入做过实测对比 | 同 G2 |

---

## 4. 阶段三：benchmark（既有数据集）

**前提**：阶段一 7/7 通过。若某方法在阶段一就不可运行，benchmark 里如实标"未参与"。

### 4.1 数据与协议

- 数据：BBBC022 子集，**3,857 细胞 / 64 孔 / 54 化合物 / 671 mock / 11 mock 孔**
- 任务：化合物检索（同化合物细胞应邻近）
- 指标：kNN、mAP、silhouette、mock-AUC；均给 z-scored 版
- 基线：原始像素、随机投影
- chance = 0.0185

### 4.2 命令

```powershell
# v2（4 个官方方法）
python scripts\embedding_benchmark\embed_dataset.py --cells-dir E:\bbbc022\cells --force
python scripts\embedding_benchmark\score_benchmark.py --cells-dir E:\bbbc022\cells --profile v2 --output E:\bbbc022\cells\benchmark_scores_v2_review.json

# v3（含 3 个替身）
python scripts\embedding_benchmark\build_v3_surrogates.py --force
python scripts\embedding_benchmark\score_benchmark.py --cells-dir E:\bbbc022\cells --profile v3 --output E:\bbbc022\cells\benchmark_scores_v3_review.json
python scripts\embedding_benchmark\stability_check.py --cells-dir E:\bbbc022\cells --profile v3 --folds 4
```

### 4.3 必须报告的 caveat（**不得省略**）

1. **不给 p 值**：每化合物只有单孔，化合物身份 ≡ 孔身份，留一孔 CV 无相关项
2. **替身行必须标注**：v3 表里 unidino/dinocells/morphogenie 不是论文性能
3. **不可与论文数字比较**：协议是自建的
4. **o2vae 是形状模型**：吃剪影，与化合物表型任务不匹配
5. **稳定性要报"稳定前缀 + 互换对"**，不能只报一个布尔值

### 4.4 本项目既有结果（供审查者对照，**不一致要标红**）

v2（mAP_z）：

```
cellclip 0.1337 > ca_mae 0.1083 > deepprofiler 0.0691 > o2vae 0.0471
原始像素 0.0508   随机 0.0450
```

v3 追加（mAP_z）：

```
unidino 0.0684 (surrogate)   morphogenie 0.0574 (surrogate)   dinocells 0.0460 (surrogate)
```

v3 稳定性：**稳定前缀 5/7**（cellclip > ca_mae > deepprofiler > unidino > morphogenie），
唯一跨折互换对：`dinocells / o2vae`。

---

## 5. 交付格式

审查者输出一份 `docs/review_report_<日期>.md`，必须含：

1. **黑盒表**（2.6 的表格，填满七行）
2. **白盒表**：逐方法 × 逐检查点，三态标注
   - ✅ 一致
   - ⚠️ 有意差异（引用 D1–D4 编号）
   - ❌ 不一致（**必须给官方行号 + 本项目行号 + 差异**）
3. **缺口确认**：G1–G4 是否仍然存在
4. **benchmark 表**：七方法 + 两基线，替身行加标记
5. **结论**：分三档
   - v2 能否发表
   - v3 的替身是否只用于接口验证
   - 哪些结论**数据不支持**

---

## 6. 禁止事项

| 禁止 | 原因 |
|---|---|
| 只跑部分方法就下结论 | 用户明确要求"务必全部测试" |
| 把替身分数当方法性能 | 权重不是官方的 |
| 给 p 值或"显著优于" | 单孔数据结构上不成立 |
| 用"看起来对"代替运行 | 本项目历史上多个缺陷是静默的（通道序、磁带 order、uint8/uint16） |
| 跳过黑盒直接白盒 | 会把"跑不起来"误诊成"实现错" |
| 声称某方法已与官方逐位一致（除 deepprofiler） | 见缺口 G2 |

---

## 7. 已知事实基线

以下数字在本项目内已验证，审查者若得到不同结果，**先怀疑自己的环境**，
再作为问题上报：

| 项 | 值 |
|---|---|
| 七方法输出宽度 | 672 / 384 / 512 / 512 / 1920 / 384 / 10 |
| 官方权重 sha256（前 16） | deepprofiler `87b756f579336d60`、ca_mae `f6e5f1c97101331b`、cellclip `66c2d32a44a3e7d6`、o2vae `aa70cfa8b080c9d9` |
| 替身权重 sha256（前 16） | unidino `e53838fe807311bf`、dinocells `5475ec885ae6b7d9`、morphogenie `491029978fef0969` |
| DeepProfiler 逐位对拍 | `max|diff| = 0.000000e+00` |
| 测试基线（`.venv`，无 torch） | 300 passed / 9 skipped / 14 failed（14 项为既有失败） |
| 测试基线（`ep_bench`，有 torch，4 个模型测试文件） | 90 passed |

> **注**：上面的 sha256 是**权重文件**的摘要，**不是嵌入数组**的哈希。
> §2.3 输出块的 `sha=` 列与之同源（见 §2.3 的口径说明）。
> `ep_bench` 的 `90 passed` 是**修复前**的基线；修复后新增了守卫的回归测试，
> 数量会上升，但**既有测试一项都不能失败**。
