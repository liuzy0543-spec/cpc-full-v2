# Cell Painting Embedding 统一接口框架：现状、验证与问题汇总

> 日期：2026-10-19
> 范围：`src/cellpaint_pipeline/models/`、`src/cellpaint_pipeline/embedding_benchmark.py`
> 方式：读源码 + 在 torch 2.5.1 / TF 2.15.1 环境实跑 4 个可运行方法 + 上游权重独立核验
> 关联文档：`embedding_interface.md`、`embedding_audit_report.md`、`audit_response.md`、`benchmark_bbbc022_result.md`
> 修订：2026-10-19 逐条结论核查后修订。修复了 §6-P0 的 list 输入绕过，并按核查结果改正 dtype 格、失败归类、溯源项与两处自我误读；修订明细见 §11。

---

## 0. 结论摘要

**可以统一，而且当前实现走的是正确路线：统一“契约”而不是强行统一“张量形状”。**

- 已实现一个入口：`models.embed('方法名', x)`，所有方法返回同一种 `EmbeddingResult`。
- 7 个方法已在同一张注册表声明；**4 个可运行**（deepprofiler、ca_mae、cellclip、o2vae），**3 个如实标注为不可运行**（dinocells、unidino、morphogenie）。
- 新增方法只需两处改动：`registry.py` 一条注册 + 一个 adapter 模块；核心调度与契约不需要改。
- 4 个 ready 方法的权重 sha256 与上游原始文件一致；CellCLIP 的 DINOv2-Giant 链式权重也被记录并透出。
- 契约能在加载模型前拦住形状/dtype/值域/通道顺序错误；4 个方法的实际输出维度均等于声明值；连续两次调用结果完全一致。
- **P0 缺陷为 0（仅在修复后成立）**：核查发现一条漏网通道——以 **Python list** 传入的原始 16-bit 数据会绕过形状/dtype/值域**全部**检查（`check_compatible` 无 numpy、读不到 list 的 `shape`/`min`/`max`），随后被 adapter 的 `np.asarray` 静默转换，产出"看似合理但错误"的 embedding。已在 `embed()`/`embed_many()` 入口归一化为 ndarray（`contract.py` 仍不依赖 numpy），并加回归测试；现 ndarray 与 list 的判定完全一致。主要 P1 里的 `device`/`weights_roots` 不可用已在本轮修复（见 §11.2）；仍开放的 P1 是 CellCLIP 输入向量的溯源文件缺 `model_sha256`、以及 parity 脚本在仓库外且已过时。
- 本轮把“预处理执行”和“输出契约”从声明变成可执行：新增 `models.prepare()`（manifest 驱动，numpy/skimage 惰性导入），`models.embed(..., assume='raw_crop')` 可以直接吃原始 16-bit crop；`ModelManifest` 新增 `output_dim_alternatives`/`output_dim_per_channel`，`embed` 返回前校验实际宽度；tape 契约接受 `(N,H,W*(C+1))` 批量；benchmark 在加载模型前校验行数。实测 `prepare()` 在规范数据上与既有数组**逐位相同**，raw uint16 一次调用得到的 embedding 与预处理后路径的差为 0（CA-MAE）或 6.7e-4（DeepProfiler，仅 16-bit 量化）。
- 仍未完成的不是架构，而是：**3 个方法的权重不可得**、以及 **CellCLIP 的输入（DINOv2 向量）需要另一个模型先行生成**（链式 stage 未内建）。

---

## 1. 任务目标与判断标准

目标可以拆成五条：

1. **一个入口**：`models.embed('方法名', x)` 直接得到 embedding。
2. **统一格式**：所有方法返回同一种结果对象，带形状、维度、权重哈希、许可、确定性。
3. **接口解耦**：调用方不需要知道每个方法的布局、dtype、通道规则。
4. **正确加载模型**：权重来自官方、加载可验证、不静默降级。
5. **可调度 / 可 benchmark**：一批数据能按方法分派，失败可收集，结果可按行对齐。

判断“统一”是否成功，不能只看函数签名，还要看：

- 错误输入是否在**加载前**被拒绝；
- 权重是否真的进了模型，而不是随机初始化；
- 不同方法的结果能否放在同一张表里比较；
- 新方法接入是否不触发核心代码修改。

---

## 2. 七个方法的真实差异

| 方法（代码名） | 输入单元 | 布局 | dtype / 值域 | 通道 | 空间 | 输出 | 状态 |
|---|---|---|---|---|---|---|---|
| DeepProfiler（`deepprofiler`） | 单细胞 crop | BHWC | float32 `[0,1]`，per-crop per-channel min-max | `DNA,ER,RNA,AGP,Mito`，顺序敏感 | 可变 | 672 | ✅ 可运行 |
| Cell Painting SSL / DINO4Cells（`dinocells`） | 单细胞 crop | **tape** `(H,W*(C+1))` | float32，`scale='per_image_selfnorm'`；**`value_range` 未声明**，故契约对它没有 dtype 也没有值域检查 | 5；manifest 断言 `names_are_labels/order_matters`，强于上游证据 | (96,96) | 384 | ⛔ 权重 6.90GB，Zenodo ~180h；缺 `oyaml` |
| CA-MAE / OpenPhenom（`ca_mae`） | 单细胞 crop | BCHW | **uint8 `[0,255]`**，模型内 `/255` + InstanceNorm | **通道无关，数量可变** | (256,256) | 384 | ✅ 可运行 |
| uniDINO（`unidino`） | 单细胞 crop | BCHW | float32 `[0,1]`，`scale='div65535'`（官方 `source/image_ops.py`） | 5，顺序敏感 | (224,224) | **384×C**（`output_dim_per_channel=384`） | ⛔ 权重 1.38GB，Zenodo ~43h |
| O2-VAE（`o2vae`） | 单细胞 crop | BCHW | float32 `[0,1]` | **1（DNA 掩膜剪影）** | (64,64) | 512 | ✅ 可运行 |
| MorphoGenie（`morphogenie`） | 单细胞 crop | BCHW | float32 | 3（RGB） | (256,256) | **10** | ⛔ Google Drive 不可达；随机；预处理在 MATLAB |
| CellCLIP（`cellclip`） | **通道向量，不是像素** | `(N,5,1536)` | float32，`scale='none'` | `DNA,ER,RNA,AGP,Mito` | — | 512 | ✅ 可运行（但需先跑 DINOv2） |

这组差异说明：**强行统一成一个 tensor 形状会丢信息或静默喂错数据**。例如：

- CA-MAE 要 uint8 原始值，DeepProfiler 要已归一化到 `[0,1]` 的 float；只统一形状会让 16-bit 输入静默进入 DP 并产生错误 embedding（实测误差可达 `4.48e6`）。
- CellCLIP 根本不吃像素，吃的是 5×1536 的 DINOv2 向量；把它当图像喂进去，形状检查可能通过，但结果是错的。
- dinocells 的 tape 编码 `(H, W*(C+1))` 与普通 `(H, W*C)` 图像形状相同，只能靠显式声明区分。

因此当前设计是：**统一 API + 统一结果对象 + 每个方法声明自己的输入契约**，而不是统一数据布局。

---

## 3. 当前实现结构

| 文件 | 职责 |
|---|---|
| `models/contract.py` | `InputUnit`、`ChannelSpec`、`PreprocessSpec`、`WeightsSpec`、`ModelManifest`；纯函数 `check_compatible`，不 import numpy/torch |
| `models/registry.py` | 7 个方法的唯一注册表；manifest 与 adapter factory 分离，factory 懒加载 |
| `models/weights.py` | 本地权重解析，**禁止隐式下载**；sha256 校验与记忆化 |
| `models/adapters/base.py` | 共享的权重解析/依赖检查/设备选择；`load()` 幂等 |
| `models/adapters/*.py` | `deepprofiler`、`ca_mae`、`cellclip`、`o2vae` 四个真实 adapter + `unavailable` 占位 |
| `models/embedding.py` | `embed` / `embed_many` / `embed_all` / `EmbeddingResult` / `BatchResult` / adapter 缓存；返回前校验输出宽度 |
| `models/preprocessing.py` | `plan()` / `prepare()`：manifest 驱动的预处理执行层；numpy/skimage 惰性导入 |
| `models/__init__.py` | PEP 562 懒加载 facade；导入不触发 torch/TF/权重 |
| `embedding_benchmark.py` | 对齐行身份、许可闸门、随机方法重复、RV coefficient、产物落盘 |
| `tests/test_models_contract.py` | 契约层测试，不需要 torch/TF/权重 |
| `tests/test_embedding_benchmark.py` | 调度、落盘、汇总测试 |

核心调用链：

```python
from cellpaint_pipeline import models

result = models.embed(
    'deepprofiler',
    crops,                                   # (N, H, W, 5) float32, [0,1]
    unit='single_cell_crop',
    channels=('DNA', 'ER', 'RNA', 'AGP', 'Mito'),
)
# result.values  (N, 672)
# result.shape, result.dim, result.weights_sha256, result.license, result.determinism
```

也可以直接把**原始采集像素**交给同一个入口，由框架按 manifest 预处理（本轮新增；默认关闭以保持旧语义）：

```python
raw = np.load('crop_16bit.npy')              # (32, 32, 5) uint16, 原始计数
result = models.embed(
    'deepprofiler',
    raw,
    unit='single_cell_crop',
    channels=('DNA', 'ER', 'RNA', 'AGP', 'Mito'),
    assume='raw_crop',                       # 先跑 models.prepare()
    device='cpu',
)
# 等价于 models.prepare(...) 之后走默认路径；实测差 6.7e-4（16-bit 量化）
```

批量调度：

```python
out = models.embed_all(
    ['deepprofiler', 'ca_mae'],
    {'deepprofiler': dp_crops, 'ca_mae': ca_crops},
    unit='single_cell_crop',
    on_error='collect',
)
out.results   # {method: EmbeddingResult}
out.errors    # {method: reason}
```

落盘与评测：

```python
result.save_npz('embeddings/deepprofiler.npz')   # 保留 sample_cell_id 等逐行身份
result.to_frame()                                # sample_metadata + 特征列
```

---

## 4. 已经达到的能力（实测）

### 4.1 权重与溯源

| 方法 | 上游核验 | 结果 |
|---|---|---|
| deepprofiler | Zenodo 7114558：size `19,236,600`，md5 `b3667957…` | 本地 md5 相同；本地 sha256 = manifest |
| ca_mae | HF `recursionpharma/OpenPhenom`：LFS sha256 `f6e5f1c9…` | MATCH |
| cellclip | HF `suinleelab/CellCLIP`：LFS sha256 `66c2d32a…` | MATCH |
| o2vae | 上游 git `4c70d58` 的 `model_mefs.pt` | 与本地文件逐字节相同 |
| dinov2-giant（CellCLIP 链） | HF `facebook/dinov2-giant`：sha256 `917d3c47…` | MATCH |

`BaseAdapter.resolve_weights()` 在加载时默认校验声明的 sha256；`EmbeddingResult` 携带 `weights_path`、`weights_sha256`、`license`、`determinism`，CellCLIP 还携带 `metadata['chain_weights']`。

### 4.2 输出形状与确定性

| 方法 | 声明维度 | 实测形状 | 连续两次 |
|---|---|---|---|
| deepprofiler | 672 | `(2,672)` | `identical=True` |
| ca_mae | 384 | `(2,384)` | `identical=True` |
| cellclip | 512 | `(2,512)` | `identical=True` |
| o2vae | 512 | `(2,512)` | `identical=True` |

缓存效果（首次 → 第二次）：

- deepprofiler：`10.49s → 0.023s`
- ca_mae：`10.94s → 0.166s`
- cellclip：`4.57s → 0.039s`
- o2vae：`2.71s → 0.046s`

> 口径提醒：首次耗时对"框架/上游包是否已预导入"极敏感——复核实测 CA-MAE 冷 `29.5s` / 暖 `17.2s`，O2-VAE 冷 `19.6s`（含 import）/ 仅 build `2.38s`。上表是某一预热口径下的值，跨方法比较首次耗时前需先统一口径。

### 4.3 契约拦截

以 CellCLIP 为例：

| 输入 | 结果 |
|---|---|
| `(2,5,256,256)` 4D 像素堆 | `ok=False, ['ndim']` |
| `(2,5,256)` 3D 像素 | `ok=False, ['feature-width']` |
| `(2,5,1536)` 正确向量 | `ok=True` |

通道顺序、dtype、值域、spatial、tape 宽度均有对应 issue code，`raise_for_status()` 会把 code 带进异常的 `reason`。

**容器无关性（修复后）**：`check_compatible` 本身是 duck-typed 纯函数，直接调用时无法识别 list（它没有 `shape`/`dtype`/`min`/`max`）。因此归一化放在入口层：`embed()`/`embed_many()` 在调用契约之前把非 ndarray 载荷转成 ndarray，`contract.py` 依旧不 import numpy。同一组数值无论以 ndarray 还是 list 传入，判定一致；回归测试见 `tests/test_embedding_benchmark.py::TestArrayLikeNormalisation`。

### 4.4 加载正确性

- DeepProfiler：`load_weights(by_name=True)` 后断言第一个**有权重**的层（`stem_conv`）确实变化；反向验证时把加载改成 no-op，adapter 抛出 `weight-load-noop`。
- CellCLIP：`strict=False` 用于跳过非视觉部分；对 visual keys 做 missing/unexpected 检查。实测加载 `visual_keys=153`、`missing=0`、`unexpected=0`；被跳过的非视觉 key 共 208 个 = 201 个 `text.*` + 6 个 `image_pool.*` + 1 个 `logit_scale`（并非只有文本塔）。删掉一个 visual key 会抛 `missing-keys ['channel_embed']`。
- O2-VAE：`strict=False` 由 e2cnn buffer 决定；断言 missing 中没有任何可学参数、unexpected 为空（实测 missing=252，其中可学参数 0，unexpected=0）。
- CA-MAE：`strict=True`。
- 三个不可运行方法统一抛 `MissingDependency` / `WeightsUnavailable`，不会假装成功。

### 4.5 测试与调度

- 契约/调度测试：`131 passed`（`test_models_contract.py` 45 + `test_embedding_benchmark.py` 86；无 torch/TF/权重）。
- 全量套件：`257 passed, 14 failed`（基线 215+14，本轮新增 42 条；失败集合与基线逐条相同）。14 个失败与 embedding 层无关（失败文件对 `models`/`embedding` 的引用数为 0），但**归类不是"全是路径分隔符与 CLI 冒烟"**：4 条路径分隔符、2 条相对路径解析（测试按 cwd 而非配置目录解析）、5 条 CLI 冒烟/缺产物、**3 条真实代码漂移**——`public_api.py` 运行时 `NameError: ProjectConfig`（该名只在 `TYPE_CHECKING` 下导入）、`public_api` 缺 `run_end_to_end_pipeline` 属性（仅 dict key，`mock.patch` 失败）、skills catalog 缺 `run-segmentation-masks`。
- `doctor()` 实测：4 个 `ready`，3 个 `runnable=False`。
- benchmark：`bench_slice2` 的 4 个 npz 均含 `sample_cell_id`/`sample_well`，summary 的 `finished_at`/`written` 已正确写入；行对齐只取各方法共有细胞。
- benchmark 调度实测（本轮，8 cells × 4 方法，`device='cpu'`）：4/4 运行成功、`errors={}`、5 个产物、`aligned_row_count=8`；总耗时 34–96s（同一脚本四次运行：34.58 / 35.07 / 91.35 / 96.00，随磁盘缓存与机器负载波动），warm 计算稳定在 2.44s（DeepProfiler 0.04 / CA-MAE 2.08 / O2-VAE 0.19 / CellCLIP 0.13），加载占 94%。行数不匹配现在 **0.000s** 拒绝且不创建输出目录。

### 4.6 扩展成本

新增方法只需：

1. 在 `registry.py` 的 `ADAPTERS` 增加一条 `AdapterRegistration`；
2. 写一个实现 `ModelAdapter` 协议的 adapter 模块。

`embedding.py` 与 `contract.py` 中没有任何方法名分支；factory 是字符串，`load_factory()` 时才 import。

### 4.7 预处理执行层与输出契约（本轮新增）

前几轮的核心缺口是“`PreprocessSpec` 只被声明、从未被执行”：真正把 16-bit crop 变成模型输入的代码住在 `scripts/embedding_benchmark/build_bbbc022_cells.py` 里。本轮补上执行层，但仍**不改变既有语义**——`embed()` 默认行为不变，执行是**显式选择**的。

- **`models.prepare(name_or_manifest, array, channels=None, masks=None)`**（`models/preprocessing.py`）
  - 驱动字段是新增的 `PreprocessSpec.source_map`，它回答“原始采集值如何变成声明的 `value_range`”，与 `scale`（模型内部怎么归一化）分开——这两件事以前混在一个字段里。
  - 已实现且可复算的映射：`per_crop_minmax_bhwc`(DP)、`per_crop_max255`(CA-MAE)、`div65535`(uniDINO)、`div257`、`per_sample_selfnorm`(dinocells)、`mask_to_silhouette`(O2-VAE)。
  - **拒绝猜测**：`cellclip` 返回 `reason='chain-required'`（它吃 1536 维向量，不是像素）；`morphogenie` 返回 `reason='unsupported-preprocess'`（256×256 居中在 MATLAB 上游）；O2-VAE 无 mask 时返回 `reason='mask-required'`。这三条都是有意的“宁可报错也不产出看似合理的错 embedding”。
  - 顺序固定并由 `models.plan(name)` 输出：`coerce → reorder-channels → resize → source-map → tape → layout → cast`。顺序（先 resize 后缩放）是实测过的语义，不是随意。
- **`models.embed(..., assume='raw_crop')`**：一次性完成“原始像素 → embedding”，等价于先 `prepare` 再 `embed`。默认 `None`，保持旧行为。
- **单样本升批**：`prepare` 保持秩不变（3-D→3-D），而框架占位符要 batch。`embed` 只在形状**可证明**是“单个样本”（对照 `batch_layout` + `spatial` + 通道数）时才补前导轴，避免把 `(N,C,H)` 批量误判成 `(C,H,W)` 单样本。
- **输出契约**：`ModelManifest.output_dim_alternatives`（DeepProfiler `pool5`=1280、dinocells ViT-B=768）与 `output_dim_per_channel`（uniDINO / CA-MAE `return_channelwise` 的 `384*C`）。`embed` 在返回前比对实际宽度，不符抛 `reason='output-width'`——此前 `EmbeddingResult.dim` 会与 `manifest.output_dim` 静默不一致。
- **tape 批量**：契约接受 `(N,H,W*(C+1))`。`prepare` 本身仍逐 cell 产出 2-D tape（折叠没有 batch 轴可放），把准备好的 tape 堆叠起来现在能通过校验。
- **fail-fast**：`run_benchmark` 在 `embed_all` **之前**校验 `sample_metadata` 各列长度一致、`align_on` 存在、以及每个方法输入行数与之相符。此前这条 `ValueError` 出自 `_align`，在全部模型跑完后才抛，且位于 `_write_artefacts` 之前，代价是白跑一轮且零产物。

实测（真实权重，8 cells）：`prepare()` 在既有规范数组上是**逐位恒等**（DP/CA/O2 的 `max|diff|` 均为 `0.00e+00`）；raw uint16 → 一次调用与预处理后路径的 embedding 差 `0.00e+00`(CA) / `6.69e-04`(DP，16-bit 量化)；行数不匹配 0.000s 拒绝。

---

## 5. 与目标的对照

| 目标 | 达成度 | 说明 |
|---|---|---|
| 一个入口 | ✅ 对 4 个可运行方法成立 | 3 个方法调用会得到类型化错误，不是假装成功 |
| 统一结果格式 | ✅ | `EmbeddingResult` 字段完整；`to_dict` 可 JSON 序列化；`save_npz` 保留逐行身份 |
| 接口解耦 | ✅ 基本成立 | 调用方传数组 + 可选 `unit/channels`；adapter 处理布局/dtype/加载细节 |
| 正确加载模型 | ✅ 对 4 个方法成立 | sha256 校验 + 加载断言；`device/weights_roots` 已能送达 4 个 adapter（只构造不加载的回归测试 12 条） |
| 可调度/可 benchmark | ✅ 成立 | `embed_all` 逐方法输入、失败收集、行对齐；`device` 现可用（8×4 实测 4/4、`errors={}`）；行数不符在加载前拒绝 |
| 预处理 | ✅ 声明可执行 | `models.prepare()` 执行 `source_map`，`assume='raw_crop'` 一步到位；无法复现的方法（CellCLIP 链式、MorphoGenie MATLAB）**显式拒绝**而非猜测 |
| 输出契约 | ✅ 已强制 | `output_dim_alternatives` / `output_dim_per_channel`；宽度不符抛 `output-width` |
| 7 个方法全部可跑 | ⛔ 4/7 | 3 个受权重/依赖阻断，已如实标注 |

---

## 6. 现存问题

### P0（会让结论错）

**1 条，已修复。** 曾存在一条从公开入口 `models.embed()` 可达的静默错误通道：`embed()` 把用户原始载荷直接交给 `check_compatible`，而该函数是 duck-typed（只读 `shape`/`dtype`/`min`/`max`）；Python list 这些属性全无，于是形状、dtype、值域检查**全部静默跳过**，随后 `adapters/deepprofiler.py` 的 `np.asarray` 把它转成数组喂进模型。同一组原始 16-bit 数值：以 ndarray 传入被 `value-range` 拒绝，以 list 传入则放行并产出错误 embedding。

修复：`embedding.py` 新增 `_normalise_input()`，在 `embed()`/`embed_many()` 入口把无 `shape` 的对象转成 ndarray（已带 `shape` 的 ndarray/tensor/HDF5 dataset/测试替身原样返回），`contract.py` 保持不依赖 numpy。回归测试 `tests/test_embedding_benchmark.py::TestArrayLikeNormalisation`（3 条）。修复后本节断言成立。

仍存在的相关局限（不是 bug，但需知情）：对 `to_dtype='float32'` 的方法（DP/O2-VAE/CellCLIP），**错 scale 但落在声明值域内的输入**无法被识别——例如把 uint16 除以 65535 而非做 per-crop min-max，值域仍在 `[0,1]`。本轮提供的 `models.prepare()` 能消除这类歧义，但它是**显式选项**（`assume='raw_crop'`）：默认路径仍是“调用方负责预处理”，因此这条局限对默认用法依旧成立。

### P1（影响调度/复现）

1. ✅（本轮已修复）**`device` / `weights_roots` 对 4 个真实 adapter 全部抛 `TypeError`**
   - 原因：4 个 adapter 都是 `@dataclass`，生成的 `__init__` 覆盖了 `BaseAdapter.__init__`；`__post_init__` 调用父类时没有传这两个参数。
   - 后果（修复前）：`models.embed(..., device='cpu')`、`weights_roots=...` 直接失败；`BenchmarkPlan.device` 端到端跑出空结果；O2-VAE 的 `device-unsupported` 保护不可达。
   - 修复：两者声明为 dataclass 字段并在 `__post_init__` 转发；回归测试 `TestAdapterConstructorPassthrough`（12 条，只构造不加载）覆盖 4 个 adapter 的 `device`、`weights_roots` 与“默认不设置”三种情形。实测 `BenchmarkPlan(device='cpu')` 已 4/4 运行。
2. **已有 `cellclip_vectors.json` 缺 `model_sha256`，且 `cells_oldorder` 的溯源错误比"指错目录"更严重**
   - `E:\bbbc022\cells\cellclip_vectors.json` 只有 8 个字段，没有 `model_sha256`/`model_name`/`channel_count`；`scripts/embedding_benchmark/make_cellclip_vectors.py` 现在确实会写这三个字段，但旧产物无法自证向量来源。
   - `cells_oldorder` 的 JSON 与 `cells` 版本**逐字节相同**（同一 sha256），但两目录的 `cells_clip.npy` **哈希不同**，且实测 **ER/RNA 通道互换**（`old[:,:,1] == new[:,:,2]`）。即旧产物的元数据是从新目录复制的，恰好掩盖了数据顺序差异——这比"`source_crops` 指错"更难发现。
   - 另外该 JSON 与其同目录 `.npy` 也不同步（JSON 17:37 / npy 22:14），说明两者不是同一次生成的。
3. **parity 命令不可执行（脚本不在本仓库）**
   - `_probe/parity_vs_known_result.py` **不在本仓库**，只存在于旧工作区 `D:\项目\cellpainting claw复现\_probe\`；按本文档所写路径在本仓库执行必然 `FileNotFoundError`。
   - 该脚本仍用旧通道顺序 `('DNA','RNA','ER','AGP','Mito')`，即使拿到副本也会在 `models.embed` 处被 `channel-order` 拒绝。
   - 绕过契约后 `bitwise_equal=True, max|diff|=0`，说明实质对拍成立；但"按文档命令直接跑出结论"不成立，且失效原因比原记录更强——路径本身已不存在。

### P2（文档/健壮性）

- ✅（本轮已解决）`PreprocessSpec` 是纯声明；调用方仍必须自己产出方法正确的预处理结果。现已提供 `models.prepare()`/`assume='raw_crop'` 作为显式选项（默认行为不变，见 §4.7）。
- ✅（本轮已修复）~~`embed_many` 先加载模型、后校验数组~~：改为先校验全部 chunk，再构建/加载 adapter；回归测试 `TestEmbedManyFailFast` 用“adapter 一旦被构建就断言失败”钉住顺序。
- O2-VAE 加载上游仓库时修改 `sys.path`；之后裸 `import models` 会解析到上游 `models` 包。CA-MAE 也会插入路径。模块 docstring 有说明，但隐患仍在。
- `docs/layers.py` 没有把 `models`/`embedding_benchmark` 归入任何层，分层测试对 embedding 层形同虚设。
- 多处文档漂移：行数表（8/8 全过期）、`contract.py` 中“16-bit raw”和旧通道序、`o2vae` 的“self-trained”、误差 `8.8e4` vs 实测 `4.481e6`（`8.8e4` 是改用 `tf.image.crop_and_resize` 之前那版裁剪的中间值）、`PROJECT_STATUS` 的“device/weights_roots 可传”。
  - 更正：本清单原有一条“O2-VAE buffer `188` vs 实测 `252`”属**误读**——`188` 是缺失 key 中 buffer 的子集，`252` 是缺失 key 总数，`embedding_audit_report.md` 自己写明"252 中 buffers=188"，没有任何文档把 188 当作缺失总数。该条已删除。
- `models.ADAPTERS` 在 `register()` 后不会刷新（facade 缓存旧元组）。
- `probe/doctor` 只检查权重文件存在，不校验哈希；损坏文件仍显示 `ready`（真正加载时才失败）。
- `illum_corrected` 字段声明但从未检查：把 manifest 改成 `illum_corrected=True` 后，`check_compatible` 仍返回 `ok=True`。
- O2-VAE 用错 checkpoint 时先抛裸 `RuntimeError`（size mismatch），未走类型化 `EmbeddingError`；`_loaded_key_count` **必然**为负（官方权重实测 `-58` = 194 个 checkpoint key − 252 个 missing key），不是"可能"。
- 许可风险：CellCLIP 为 CC BY-NC-ND，而 adapter 内逐行复制上游架构；CA-MAE 为非商用 EULA；MorphoGenie 为 GPL-3.0。功能上可用，但发布/再分发前需法务确认。**CellCLIP 权重包内没有 LICENSE 文件**，许可只在上游仓库，再分发风险高于其余方法。
- 运行前提未记录：`D:\ep_bench\venv` 这类"只装框架、没装本包"的解释器需 `PYTHONPATH=src`；并且**用绝对路径直接调用 conda 的 `python.exe` 时，numpy 的 BLAS 会因 `Library\bin` 不在 PATH 而崩溃**（`0xC06D007F` / `ERROR_MOD_NOT_FOUND`），复核前必须先补 PATH，否则会误判成"测试套件自身崩了"。
- `unidino` 本地只有 `5,701,424 B` 的 `checkpoints.partial.zip`，是**损坏的截断包**（`BadZipFile`），不只是"权重不可得"。
- `registry.py` 段注仍写 "three methods have weights … remaining four"，与实况（4 个有 / 3 个无）不符。
- `clear_adapter_cache` 可访问但不在 `__all__`。
- 旧 `bench_slice` 产物缺 `sample_cell_id`、`finished_at=""`、`written=0`（修复前产物）；当前代码已用 `bench_slice2` 验证修复。

---

## 7. 三个未运行方法的具体阻塞

| 方法 | 缺什么 | 影响 |
|---|---|---|
| `dinocells`（Cell Painting SSL） | 6.90GB 权重（Zenodo ~180h）；`oyaml` 未安装 | tape 输入契约与 adapter 无法端到端验证 |
| `unidino` | 1.38GB 权重（Zenodo ~43h） | 无法验证真实输出 `384*C` 与预处理 semantics |
| `morphogenie` | Google Drive 权重不可达；随机采样；256×256 居中预处理在 MATLAB | 无法端到端复现；`determinism='stochastic'` 只能在机制上验证 |

本地实况（2026-10-19 核查）：`E:\cpc_weights\dino4cells` 与 `morphogenie` 为**空目录**，`unidino` 只有 5.7MB 的损坏 partial zip。权重体积与 registry/Zenodo 记录精确对应（dinocells `6,896,284,400 B`、unidino `1,379,562,715 B`）。`morphogenie` 的"Google Drive 不可达"未记录具体 URL，无法独立复验。

这不是架构缺陷：三者的 manifest/契约已经声明，依赖到位后只需补 adapter / 放权重。当前做法是**如实标注不可运行**，而不是用自训的小权重假装。

---

## 8. 是否“只给方法名就能得到 Embedding”？

分三种情况：

1. **4 个 ready 的像素方法（DP/CA-MAE/O2-VAE）**：可以，且本轮已能做到“给原始像素”。两条路等价：
   - 调用方自己预处理后传数组：`models.embed(name, x, unit=..., channels=...)`；
   - 直接给原始 16-bit crop：`models.embed(name, raw, assume='raw_crop', channels=...)`（O2-VAE 需 `masks=`）。
2. **CellCLIP**：不能完全“只给像素”。它需要先由 DINOv2-Giant 生成 `(N,5,1536)` 向量；目前这一步是 `scripts/embedding_benchmark/make_cellclip_vectors.py`，不在 `models.embed` 内部。因此 CellCLIP 是“两段式 pipeline”，不是单入口。`prepare('cellclip', ...)` 会以 `chain-required` 明确拒绝，而不是把像素当向量。
3. **3 个不可运行方法**：不能，原因在权重/依赖，不在接口。

因此，**“统一接口”已经做到；“原始像素到单入口”已对 3 个像素方法做到**。仍差的只有一层：

- 链式输入层：把 CellCLIP 的 DINOv2 提取作为框架内的一等 stage（例如 `models.embed('cellclip', pixels)` 内部先跑 DINO，或 `models.pipeline(...)` 组合）。

---

## 9. 建议的下一步（按优先级）

### 已完成（2026-10-19）

- **P0：list 输入绕过契约**。`embedding.py` 新增 `_normalise_input()`，在 `embed()`/`embed_many()` 入口把不含 `shape` 的载荷转成 ndarray（已带 `shape` 的 ndarray/tensor/HDF5 dataset/测试替身原样返回），`contract.py` 保持不 import numpy。回归测试 `TestArrayLikeNormalisation`（3 条）。修复后同一组数值无论 ndarray 还是 list 传入，判定与报错 code 完全一致；契约/调度测试由 86 → `89 passed`。

### 已完成（本轮 M1–M4）

- **M4：`device`/`weights_roots` 送到 adapter**。四个 `@dataclass` adapter 生成的 `__init__` 覆盖了 `BaseAdapter.__init__`，参数被拒为 `TypeError`；改为把两者声明为 dataclass 字段并在 `__post_init__` 里转发。实测：`BenchmarkPlan(device='cpu')` 从“4 条 TypeError、`methods_run=[]`、0.01s 静默空跑”变成 4/4 成功。回归测试 `TestAdapterConstructorPassthrough`（12 条，只构造不加载）。
- **M1：预处理执行层**。新增 `models/preprocessing.py` 与 `PreprocessSpec.source_map`；`models.prepare()` + `models.embed(..., assume='raw_crop')`。回归测试 `TestPrepareLayer`（13 条）。
- **M2：输出契约**。`output_dim_alternatives`、`output_dim_per_channel`，`embed` 校验实际宽度并抛 `output-width`。回归测试 `TestOutputContract`（5 条）。
- **M3：tape 批量**。契约接受 `(N,H,W*(C+1))`，不再以 rank 错误拒绝堆叠的 tape。
- **B1：benchmark fail-fast**。行数/对齐列在加载模型前校验，不再白跑一轮后零产物。回归测试 `TestRowCountFailFast`（5 条）。
- 顺带修正：uniDINO `scale` 由 `none` 改为 `div65535`（官方 `image_ops.py` 的 `img / (2**16-1)`，见 §11.2）；`PreprocessSpec` 新增 `source_map` 后，`scale` 与“源数据映射”不再混为一谈。

### P1
1. ✅（本轮完成）**修 `device`/`weights_roots`**：让 4 个 dataclass adapter 显式接收并传给 `BaseAdapter.__init__`（或改成 `@dataclass(init=False)` 自建构造器）；给真实 adapter 加“只构造不加载”的回归测试。
2. **补 `cellclip_vectors.json` 溯源**：重新生成向量文件，写入 `model_sha256`/`model_name`/`channel_count`；修正 `cells_oldorder` 的错误 `source_crops`，并注意其 `cells_clip.npy` 与 `cells` 版本存在 ER/RNA 通道互换，不能只改路径。
3. **修 parity 脚本与命令**：脚本目前在仓库外的旧工作区（`D:\项目\cellpainting claw复现\_probe\`），要么移入本仓库，要么改用上游通道顺序，或在脚本中明确 `skip_contract_check` 并说明这是“接口 vs demo”对拍；不要把旧顺序当作可执行默认值。

### P2
4. 把 `models`、`embedding_benchmark` 加入 `docs/layers.py`，让分层规则真正覆盖 embedding 层。
5. ✅（本轮完成，以 `models.prepare()` 实现，见 §4.7）明确 `PreprocessSpec` 的边界：要么实现 `apply()`，要么在 API 文档中把“预处理是调用方责任”写成硬约束。
6. `embed_many` 先校验全部 chunk，再构建/加载 adapter。
7. 隔离 `sys.path` 修改（上下文管理器 / 子进程 / 显式 importlib）；至少不要让裸 `import models` 被静默遮蔽。
8. 统一文档与代码：行数、docstring、误差数字、buffer 计数、`available()` 与 `doctor()` 的语义。
9. 修复 `models.ADAPTERS` 在 `register()` 后的刷新；把 `clear_adapter_cache` 加入 `__all__`。
10. 处理许可问题：在 README / 结果产物中明确标注 CC BY-NC-ND、非商用 EULA、GPL-3.0 的使用边界。
11. 把 CellCLIP 的 DINOv2 提取纳入框架（或至少纳入同一 manifest/产物溯源）。
12. ✅（本轮完成）若要让 `unidino` 真正可评测，解决 `output_dim=384` 无法表达 `384*C` 的问题（例如动态输出维度或 `output_dim_per_channel`）。已采用 `output_dim_per_channel`。

---

## 10. 最终判断

**“统一 embedding 输入输出框架”这件事本身已经基本实现，且方向正确。**

- 统一的是 API、结果格式、契约、错误类型、调度与溯源，而不是数据形状；
- 4/7 方法可端到端运行，4/4 权重与上游一致，输出维度与确定性已实测；
- 新增方法只需两处改动，说明抽象没有漏；
- 3/7 的不可运行是权重/依赖问题，已如实标注，不是“没做完”。

**但距离“任意方法名 + 原始像素 → 可直接 benchmark 的 embedding”还差两步（另有 1 个 P1 待修）**：

1. 预处理目前只声明不执行；
2. CellCLIP 需要独立的 DINOv2 前置阶段；
3. 另外有一个 P1 的调度缺陷（device/weights_roots）需要先修。

修完这三点，统一框架就可以作为 Cell Painting embedding 评测的稳定底座。

---

## 附录：本次复核的关键实测结果

```text
# 契约/调度测试（无 torch/TF/权重；含 3 条 list 归一化回归测试）
89 passed in 0.48s

# 全量套件
215 passed, 14 failed in 5.28s

# doctor（torch 2.5.1 + TF 2.15.1）
deepprofiler  runnable=True   status=ready
ca_mae        runnable=True   status=ready
cellclip      runnable=True   status=ready
dinocells     runnable=False  status=needs-deps-and-weights   missing=['oyaml']
unidino       runnable=False  status=needs-weights
o2vae         runnable=True   status=ready
morphogenie   runnable=False  status=needs-weights

# 真实输出
deepprofiler  (2, 672)  identical=True
ca_mae        (2, 384)  identical=True
cellclip      (2, 512)  identical=True
o2vae         (2, 512)  identical=True
deepprofiler feature_layer='pool5' -> (2, 1280)

# 契约
CellCLIP 4D 像素堆      ok=False ['ndim']
CellCLIP 3D 像素(256)   ok=False ['feature-width']
CellCLIP (2,5,1536)     ok=True

# 容器无关性（P0 修复验证，经 models.embed / embed_many 公开入口）
ndarray uint16 raw       -> IncompatibleInput reason='value-range'
python list, same values -> IncompatibleInput reason='value-range'   # 修复前：放行
ndarray float32 [0,1]    -> 通过契约（合法输入未被误杀）
python list float [0,1]  -> 通过契约
embed_many(list chunk)   -> IncompatibleInput reason='value-range'

# 加载断言反向验证
DP  no-op load          -> EmbeddingError: weight-load-noop
CellCLIP 少一个 key     -> EmbeddingError: missing-keys ['channel_embed']
O2VAE 缺失 252 个 key   -> 可学参数 0，unexpected 0

# 已知 P1（调度）
models.embed('deepprofiler', x, device='cpu')
  -> TypeError: DeepProfilerAdapter.__init__() got an unexpected keyword argument 'device'
BenchmarkPlan(..., device='cpu') -> results=[], errors={'cellclip': "TypeError ... 'device'"}
```

---

## 11. 修订记录（2026-10-19 核查后）

本节逐条对应核查发现，说明改了什么。

### 代码

| 文件 | 改动 | 原因 |
|---|---|---|
| `src/cellpaint_pipeline/models/embedding.py` | 新增 `_normalise_input()`；`embed()`、`embed_many()` 在调用契约前归一化载荷 | 关闭 §6-P0 的 list 绕过，同时让 `contract.py` 保持不依赖 numpy |
| `tests/test_embedding_benchmark.py` | 新增 `TestArrayLikeNormalisation`（3 条） | 钉住"同一数值、不同容器、同一判定" |

### 文档

| 位置 | 原表述 | 改为 |
|---|---|---|
| §0、§6-P0 | "P0 缺陷为 0 / 未发现漏网案例" | 记录曾存在的 list 绕过、修复方式与遗留局限 |
| §2 表格 dinocells 行 | dtype "uint8" | float32，`value_range` 未声明（故无 dtype/值域检查） |
| §4.3 | — | 补"容器无关性由入口保证"的说明 |
| §4.4 CellCLIP | "只跳过文本塔" | 实际跳过 208 个非视觉 key（text + image_pool + logit_scale） |
| §4.5 | 86 passed / 212+14；"失败全是路径分隔符与 CLI" | 89 passed / 215+14，并把 3 条真实代码漂移单列 |
| §4.2 缓存 | — | 补首次耗时对 import 预热的口径敏感性提醒 |
| §6-P1-2 | `cells_oldorder` 仅"`source_crops` 指错" | 补 JSON 逐字节相同、ER/RNA 通道互换、JSON 与 npy 不同步 |
| §6-P1-3 | `_probe/...` 在仓库内 | 脚本不在本仓库，所写路径已失效 |
| §6-P2 | "O2-VAE buffer `188` vs `252`" | 该条属误读，已删除并说明 188 与 252 的量纲关系 |
| §6-P2 | `_loaded_key_count` "可能"为负 | 必然为负（实测 `-58`） |
| §6-P2 | list 绕过记为 P2 | 提升为 P0 并修复 |
| §6-P2 | — | 新增：运行前提（PATH/BLAS）、unidino 损坏包、`registry.py` 段注过期、CellCLIP 包内无 LICENSE |
| §7 | — | 补本地权重实况（空目录 / 损坏 partial zip）与体积对照 |
| §9、§10 | — | 增"已完成"小节；修正 parity 脚本与 device 两处表述 |
| 附录 | 旧的测试数字 | 更新为修复后数字，并新增容器无关性验证输出 |

### 仍开放

- ✅（本轮完成）**P1**：`device`/`weights_roots` 已能送达 4 个真实 adapter（见 §11.2）。
- **P1**：`cellclip_vectors.json` 缺溯源字段；`cells_oldorder` 存在 ER/RNA 通道互换（需重新生成产物，未改）。
- **P2**：`docs/layers.py` 仍未覆盖 `models`；`models.ADAPTERS` 注册后缓存不刷新等。完整清单见 §6、§9。（`embed_many` 的“先加载后校验”已在本轮修复。）

### 11.2 本轮（2026-10-19 继续：M1–M4）

第二轮结论是“入口层统一、输入准备层未统一、输出语义半统一，且 `device` 通路失效导致 benchmark 静默空跑”。本轮按该结论逐条落地，目标是“调用一个方法名就得到 embedding，并且能被调度/benchmark”。

| 文件 | 改动 | 为什么 |
|---|---|---|
| `models/adapters/{deepprofiler,ca_mae,cellclip,o2vae}.py` | 新增 `device`/`weights_roots` dataclass 字段；`__post_init__` 转发给 `BaseAdapter.__init__` | `@dataclass` 生成的 `__init__` 覆盖父类，两参数被拒为 `TypeError`，`device` 形同虚设 |
| `models/contract.py` | 新增 `SourceMap` 与 `PreprocessSpec.source_map`；`ModelManifest.output_dim_alternatives`/`output_dim_per_channel`；tape 接受 3-D | 声明无法表达“源数据→值域”；输出宽度无契约；批量 tape 被 rank 拒绝 |
| `models/preprocessing.py`（新） | `plan()`/`prepare()`，7 种 source_map，无法复现者显式拒绝 | 预处理此前只活在数据集脚本里 |
| `models/embedding.py` | `embed`/`embed_many` 新增 `assume='raw_crop'`、`masks`；`_promote_single_sample`；`_check_output_width`；`embed_many` 改为先校验后加载 | 单入口、输出校验、批量 fail-fast |
| `embedding_benchmark.py` | 新增 `_validate_row_counts`，`run_benchmark` 加载前调用 | 行数不符此前在 `_align` 才炸，白跑一轮且零产物 |
| `models/registry.py` | 7 个 manifest 补 `source_map`；DP/dinocells 补 `output_dim_alternatives`；uniDINO/CA-MAE 补 `output_dim_per_channel`；uniDINO `scale` 改 `div65535` | 让声明可执行、可校验；修正一处有据可查的错误声明（官方 `source/image_ops.py` 的 `img = img / (2**16-1)`） |

实测（真实权重 `E:\cpc_weights`，8 cells）：

- `prepare()` 在既有规范数组上**逐位恒等**（DP/CA/O2 的 `max|diff|` 均为 `0.00e+00`）；
- raw uint16 一次调用 vs 预处理后路径：CA `0.00e+00`、DP `6.69e-04`（仅 16-bit 量化）；
- `BenchmarkPlan(device='cpu')`：4/4 方法运行、`errors={}`、5 个产物、`aligned_row_count=8`，总 35.07s；
- warm 计算 2.44s（DP 0.04 / CA 2.08 / O2 0.19 / CLIP 0.13），加载占 94%——与第二轮“加载主导”一致；
- 行数不匹配 **0.000s** 拒绝且不创建输出目录。

仍开放：CellCLIP 的 DINOv2 链式 stage（唯一挡住“纯像素单入口”的项）、3 个方法权重不可得、`cellclip_vectors.json` 溯源与 `cells_oldorder` 通道互换、`docs/layers.py` 覆盖。

### 核查来源

逐条判定、证据与复现脚本见仓库外核查目录 `D:\项目\_embed_audit_verify\`（`_FINAL_核查报告.md`、`A_static.md`、`B_runtime_suite.md`、`C_loads.md`、`D_weights_docs.md`、`lead_findings_P0.md`、`verify_fix_p0.py`）。本次核查与修订未改动仓库其他文件。

第二轮核查与本轮实现验证在 `D:\项目\_embed_review2\`（`_FINAL_审查报告2.md`、`_FINAL_实现报告.md`、`verify_e2e.py`、`probe_timing.py`、`bench_out/e2e/summary.json`）。
