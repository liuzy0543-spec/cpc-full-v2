# Embedding 层审查路线图（可直接作为审查提示词）

> 用途：把本文档整体交给审查者（人或 AI），按 §4 的四条路线逐项核查。
> 每条都给了**可执行命令**与**判定标准**，避免"看起来对"就通过。
> 项目位置：`D:\项目\CellPainting-Claw-v2`（主开发树）

---

## 1. 这个任务要解决什么

### 1.1 起点

官方 CellPainting-Claw **只支持 1 个 embedding 方法**（DeepProfiler），而且它自己
**不加载模型**——用子进程把特征提取外包给 `deepprofiler` 包。其余 6 个方法
（CA-MAE / CellCLIP / uniDINO / DINO4Cells / MorphoGenie / O2-VAE）在官方代码里
**一个都没出现**。

这些方法的输入契约**互不兼容**：

| 方法 | 布局 | dtype | 通道 | 输入是像素吗 |
|---|---|---|---|---|
| DeepProfiler | BHWC | float32 `[0,1]` | 5 固定 | 是 |
| CA-MAE | BCHW | **uint8 `[0,255]`** | **可变** | 是 |
| CellCLIP | BCHW | float32 | 5 | ❌ **是 DINOv2 向量** |
| O2-VAE | BCHW | float32 `[0,1]` | **1** | 是（剪影） |
| DINO4Cells | **tape 编码** `(H, W*(C+1))` | uint8 | 5 | 是 |
| uniDINO | BCHW | float32 | 5 | 是 |
| MorphoGenie | BCHW | float32 | 3 | 是 |

强行统一成一个 tensor 形状会丢信息，或者把错的数据喂给模型而不报错。

### 1.2 目标（验收口径）

1. **一个入口**：`models.embed('方法名', x)` 直接拿到 Embedding
2. **统一格式**：所有方法返回同一种结果对象，带形状、维度、权重哈希、许可、确定性
3. **接口解耦**：调用方不需要知道每个方法的布局/dtype/通道规则
4. **正确加载模型**：权重来自官方、加载可验证、不静默降级
5. **可调度 / 可 benchmark**：一批数据能按方法分派，失败可收集，结果可对齐

### 1.3 明确的非目标

- 不重做 v1/v2 的架构重构（已有成果保持）
- 不追求 7 个方法全部可跑（权重不可得的必须**如实标注**，不能假装）

---

## 2. 已有功能（截至本版）

### 2.1 代码规模

| 文件 | 行数 | 职责 |
|---|---:|---|
| `models/contract.py` | 622 | 契约原语 + `check_compatible`（纯函数，不依赖框架） |
| `models/embedding.py` | 740 | `embed` / `embed_many` / `embed_all` / `BatchResult` / adapter 缓存 |
| `models/registry.py` | 518 | 7 个方法的 `ModelManifest` 注册表 |
| `models/weights.py` | 199 | 权重解析（**禁止隐式下载**）+ sha256 记忆化校验 |
| `models/errors.py` | 81 | 类型化错误 |
| `models/__init__.py` | 152 | PEP 562 懒加载 facade |
| `models/adapters/` | 962 | `base` + 4 个实现 + `unavailable` |
| `embedding_benchmark.py` | 381 | benchmark 执行器（对齐、许可闸门、溯源落盘） |

### 2.2 公开 API

```
embed  embed_many  embed_all  BatchResult
available  describe  doctor  probe  manifests  register
check_compatible  resolve_weights
ModelManifest  ChannelSpec  PreprocessSpec  WeightsSpec  InputUnit
EmbeddingResult  ModelAdapter  ResolvedWeights  AdapterRegistration  ADAPTERS
EmbeddingError  IncompatibleInput  MissingDependency  WeightsUnavailable  UnknownModel
CompatibilityReport  CompatibilityIssue  ParityMode
```

### 2.3 契约原语（审查重点，见 §4.4）

- `InputUnit`：`single_cell_crop` / `field` / `channel_embeddings`
- `ChannelSpec(names, order_matters, names_are_labels, channel_count_fixed)`
- `PreprocessSpec(scale, value_range, to_dtype, batch_layout, illum_corrected, input_encoding)`
  - `scale` ∈ `raw` / `div255` / `div65535` / `per_image_selfnorm` / `per_crop_minmax` / `none`
- `WeightsSpec(source, format, sha256, host, downloadable, approx_bytes, weight_file_layer_count, …)`
- `ModelManifest(..., output_dim, input_feature_dim, spatial, determinism, chain_weights, license, requires, ascii_path_required)`

### 2.4 测试

```
tests/test_models_contract.py        契约层，无需 torch/TF/权重
tests/test_embedding_benchmark.py    调度/落盘/汇总
合计 86 用例；全量套件 212 passed
```

全量套件另有 **14 个失败，是本任务开始前就存在的**（`ProjectConfig` NameError、
Windows 路径分隔符、需要 demo 数据的端到端冒烟），与本层无关。

---

## 3. 七个 embedding 方法（审查主要对象）

### 3.1 四个可运行（本地有官方权重，sha256 已校验 MATCH）

| 方法 | 版本 / 论文 | 输入单元 | 布局 | dtype / 值域 | 通道 | 空间 | 输出维度 | 权重 sha256 |
|---|---|---|---|---|---|---|---|---|
| **deepprofiler** | Cell Painting CNN v1 (Zenodo 7114558) | `single_cell_crop` | BHWC | float32 `[0,1]`，**per-crop per-channel min-max** | `DNA, ER, RNA, AGP, Mito`（顺序敏感） | 可变 | **672** | `87b756f579336d606fd3d387b04bd2d422f9d410e5635a5d6b2cda7615833e73` |
| **ca_mae** | OpenPhenom-S/16 (Recursion) | `single_cell_crop` | BCHW | **uint8 `[0,255]`**，模型内 /255 + InstanceNorm | `C0..C4`（**通道无关**，数量可变） | (256,256) | **384** | `f6e5f1c97101331b1574c9e4b99623260191c55eea5d98e40460849c0e4c4d47` |
| **cellclip** | CellCLIP (NeurIPS 2025) | **`channel_embeddings`** | BCHW | float32，`scale=none` | `DNA, ER, RNA, AGP, Mito`（顺序敏感） | — | **512** | `66c2d32a44a3e7d68deb489047df0df7676940308683b3285b1c8d28c471b550` |
| **o2vae** | O2-VAE (Nat Commun 2024)，**MEF 官方权重** | `single_cell_crop` | BCHW | float32 `[0,1]` | `shape`（**单通道剪影**） | (64,64) | **512** | `aa70cfa8b080c9d9989c11de316a2f7f074e8bef29366c4aaec24c807c5dfc0f` |

**CellCLIP 的链路权重**（它不吃像素）：

```
facebook/dinov2-giant   917d3c470db999d32a312f8542149be91c7cbac61ee8fb4b67ae3d82b79ce21f
                       （4,546,005,432 B，hidden 1536，patch 14，40 层）
```

### 3.2 三个仅声明（本机无权重，注册表如实标注）

| 方法 | 输入单元 | 布局 | 通道 | 空间 | 输出 | 缺什么 |
|---|---|---|---|---|---|---|
| **dinocells** | `single_cell_crop` | BCHW | 5 | (96,96) | 384 | 权重（Zenodo 180 h）+ 依赖 |
| **unidino** | `single_cell_crop` | BCHW | 5 | (224,224) | 384 | 权重（Zenodo 43 h） |
| **morphogenie** | `single_cell_crop` | BCHW | 3 | (256,256) | **10** | 权重（Google Drive 不通）；且 `determinism='stochastic'` |

`doctor()` 应输出：这 3 个 `runnable=False`，其余 4 个 `ready`。

### 3.3 许可（审查时需要留意的合规点）

| 方法 | 许可 | 影响 |
|---|---|---|
| deepprofiler | BSD-3-Clause（代码）/ CC-BY-4.0（权重） | 可商用 |
| o2vae | MIT | 可商用 |
| **ca_mae** | **Recursion Non-Commercial EULA** | 仅研究 |
| **cellclip** | **CC BY-NC-ND 4.0** | **禁止衍生作品** → adapter 内复制上游架构有风险 |
| **morphogenie** | **GPL-3.0** | 传染性 |
| dinocells | 权重 CC-BY-4.0 / 代码**未声明** | 不明 |
| unidino | 代码 BSD-3 / 权重 CC BY-NC（GitHub API 报 CC-BY-NC-ND） | 上游自相矛盾 |

---

## 4. 审查路线图

四条路线，**建议按顺序做**：A → B → C → D。
A/B/C 是"这个方法对不对"，D 是"这个框架好不好"。

---

### 路线 A：权重是否与官方一致

#### A1 哈希与来源

```powershell
cd D:\项目\CellPainting-Claw-v2
$env:PATH = 'C:\Users\DELL\miniconda3\envs\torch_gpu\Library\bin;' + $env:PATH
& 'D:\ep_bench\venv\Scripts\python.exe' -c @"
import sys
sys.path.insert(0, r'D:\项目\CellPainting-Claw-v2\src')
from cellpaint_pipeline.models import registry
from cellpaint_pipeline.models.weights import resolve_weights, verify_sha256
for reg in registry.ADAPTERS:
    m = reg.manifest
    if not m.weights.sha256:
        print('  %-13s (no weights declared)' % m.name); continue
    r = resolve_weights(m.name, m.weights.source)
    h = verify_sha256(r.path)
    print('  %-13s %s  %s' % (m.name, h, 'MATCH' if h == m.weights.sha256 else 'MISMATCH'))
"@
```

**判定**：4 个 ready 方法必须全部 `MATCH`；3 个仅声明方法必须输出 `(no weights declared)`，
不能出现"声明了哈希但文件不存在"。

#### A2 与上游原始文件比对（不要只信 manifest 自称）

必须**独立**核对，不能只跑上面的脚本（那只是自己跟自己对）：

| 方法 | 上游权威来源 | 核对方式 |
|---|---|---|
| deepprofiler | Zenodo record **7114558** `Cell_Painting_CNN_v1.hdf5` | 文件大小 = **19,236,600 B**；哈希 = manifest 值 |
| ca_mae | `recursionpharma/OpenPhenom` 的 `openphenom/model.safetensors` | 大小 **712,434,294 B**；`config.json` 含 `channel_agnostic: true`, `max_in_chans: 11` |
| cellclip | `suinleelab/CellCLIP` 的 `model.safetensors` | 大小 **1,799,564,952 B**；`config.json` 中 cross-channel former 为 1536/12/8/512/5 |
| o2vae | 上游仓库 `wandb/pretrained_models/model_mefs.pt`（commit `4c70d58`） | **与仓库内该文件逐字节相同**（sha256 相等）；大小 **7,428,429 B** |

本地权重根：`E:\cpc_weights\`。上游仓库副本在 `D:\official_src\{1-deepprofiler,…}`。

#### A3 链式权重

CellCLIP 的输入由 DINOv2-Giant 产生。检查：
1. `registry.manifest('cellclip').chain_weights` 是否含 DINOv2 哈希；
2. `EmbeddingResult.metadata['chain_weights']` 是否透出；
3. `cellclip_vectors.json` 是否记录 `model_sha256`（应由 `make_cellclip_vectors.py` 写入）。

**反例（必须判 FAIL）**：只记录 CellCLIP 的 1.8 GB checkpoint，不记录产生其输入的
4.5 GB 模型——那样的溯源到 CellCLIP 就断了。

---

### 路线 B：黑盒输入输出是否一致

"黑盒"= 不看内部实现，只喂官方给的输入，比对官方给的输出。

#### B1 与官方实现逐位对拍（DeepProfiler 必做）

这是唯一有**官方可复现产物**的方法（官方仓库带 demo 数据与完整流水线）。

```powershell
& 'D:\ep_bench\venv\Scripts\python.exe' `
  'D:\项目\CellPainting-Claw-v2\scripts\embedding_benchmark\..\..\..\cellpainting claw复现\_probe\parity_vs_known_result.py'
```

**判定**：
- `bitwise_equal=True` 且 `max|diff| = 0.000000e+00`
- 官方树 `.npz` 与 v2 树 `.npz` 的 **sha256 相同**

**⚠️ 参照系警告（审查时必须确认这一点）**：
该对拍证明的是"**接口层 == 本项目官方树**"，**不是**"== 上游 DeepProfiler 训练语义"。
因为本项目 demo 的通道顺序也曾与上游不同（详见 §5 陷阱 1）。
审查者必须明确写出这条限定，不能笼统称"与官方一致"。

#### B2 输出形状与维度

```powershell
& 'D:\ep_bench\venv\Scripts\python.exe' -c @"
import sys, numpy as np
sys.path.insert(0, r'D:\项目\CellPainting-Claw-v2\src')
from cellpaint_pipeline import models
from cellpaint_pipeline.models import registry
for m in registry.manifests():
    print(m.name, 'declared dim =', m.output_dim, '| ready =', m.name in models.available())
"@
```

再对 4 个 ready 方法各喂一个合成输入，检查返回 `result.dim == manifest.output_dim`
且 `result.shape == (N, output_dim)`。

**注意**：`unidino` 的真实输出是 `384*C`（随通道数变），声明 `output_dim=384` 只是
单通道时的值。这属于**已知的表达能力不足**，审查者应指出而不是放过。

#### B3 契约是否在加载前拦住错输入

```powershell
& 'D:\ep_bench\venv\Scripts\python.exe' -c @"
import sys, numpy as np
sys.path.insert(0, r'D:\项目\CellPainting-Claw-v2\src')
from cellpaint_pipeline.models import registry
from cellpaint_pipeline.models.contract import check_compatible
mc = registry.manifest('cellclip')
for label, arr in (
    ('4D 像素堆', np.zeros((2,5,256,256),'float32')),
    ('3D 像素(宽256)', np.zeros((2,5,256),'float32')),
    ('正确向量(1536)', np.zeros((2,5,1536),'float32')),
):
    r = check_compatible(mc, array=arr)
    print('  %-18s ok=%-6s %s' % (label, r.ok, [i.code for i in r.issues]))
"@
```

**判定**：前两个必须 `ok=False`（分别给 `ndim` / `feature-width`），第三个 `ok=True`。
若像素堆被放行，说明契约没起到"加载前拦住静默错误"的作用。

#### B4 确定性

对 `determinism='deterministic'` 的方法，连续两次嵌入同一输入必须**数组完全相等**。
`morphogenie` 标记为 `stochastic`，`embed_all` 会重复采样并记录 spread——
检查这个分支是否真的生效（而不是把一次采样当结果）。

---

### 路线 C：白盒逻辑是否一致（**最容易出"传参失误"**）

"白盒"= 打开实现，逐条核对"官方怎么算" vs "我们怎么算"。

#### C1 通道顺序与文件映射

这是本任务**实际踩过的最严重错误**。上游证据：

```
D:\official_src\1-deepprofiler\README.md:74
  "trained on 5-channel Cell Painting images (DNA, ER, RNA, AGP, Mito)"

tests\deepprofiler\profiling\test_cell_painting_cnn_v1.py
  L44-48: ch5=DNA, ch4=ER, ch3=RNA, ch1=AGP, ch2=Mito
  L74:    CHANNELS = ["DNA", "ER", "RNA", "AGP", "Mito"]
```

**审查动作**：
1. 确认 `registry.DEEPPROFILER_CHANNELS == ('DNA','ER','RNA','AGP','Mito')`
2. 确认 `CELLCLIP_CHANNELS` 是**独立常量**（不 alias DP 的）
3. 确认旧顺序 `('DNA','RNA','ER','AGP','Mito')` **会被契约拒绝**（`channel-order`）
4. 确认数据构建脚本 `scripts/embedding_benchmark/build_bbbc022_cells.py` 的
   `CHANNELS` 与契约一致

**注意**：这里有**两处独立的错位**，只检查一处不够：
- 张量顺序：RNA/ER 互换
- 文件映射：`AGP→ch1, Mito→ch2`（上游）vs `AGP→ch2, Mito→ch1`（本项目 demo）

#### C2 预处理是否与官方算子等价

| 方法 | 官方行为 | 检查点 |
|---|---|---|
| deepprofiler | `cropping.py:65-67`：`reduce_min/max(axis=[1,2], keepdims=True)` ⇒ **逐通道** per-crop min-max | `PreprocessSpec.scale == 'per_crop_minmax'`、`value_range == (0.0, 1.0)`；数据构建用 `axis=(0,1)` 的 per-channel 归一化，**不是** joint |
| ca_mae | 模型内部 `/255` + InstanceNorm | `to_dtype='uint8'`、`scale='div255'`、`value_range=(0,255)` |
| cellclip | `convert_npz_to_avg_emb.py`：`RandomCrop(224)` → 逐通道复制成 RGB → processor（`do_resize=False`）→ `pooler_output`（CLS, 1536） | `scale='none'`；向量生成脚本的 crop_size=224、do_resize=False |
| o2vae | `config_mefs.py`：128 → resize 64、`do_sigmoid=True` | `spatial=(64,64)`、`value_range=(0,1)` |

**判定**：`PreprocessSpec` 的声明必须与**官方算子**一致，而不只是"能跑通"。
历史上 `deepprofiler` 曾声明 `scale='raw'`，导致 16-bit 输入被静默接受，
最大绝对误差达 **4.5e6**——这就是"能跑通但错"的典型。

#### C3 参数字段 vs 构造函数参数（"传参失误"专项）

本任务出过这类错：`feature_layer` 被放进了 `**kwargs`，一路转发到**前向函数**，
而前向函数忽略它——调用者以为换了特征层，实际没有。

**审查动作**：
1. `models.embed` 的签名必须显式包含 `device`、`weights_roots`、`adapter_kwargs`
2. `adapter_kwargs` 必须进**构造函数**，不能进前向
3. adapter 缓存键必须包含 `device` 与 `adapter_kwargs`（否则换设备会拿到 CPU 实例）
4. 构造一个"错误参数"用例：传 `adapter_kwargs={'feature_layer':'pool5'}`，
   确认结果维度的**确变化**（DeepProfiler `pool5` = 1280，`block6a_activation` = 672）

**反例（判 FAIL）**：
```python
# 转移后仍返回 672，说明参数被吞了
r = models.embed('deepprofiler', x, adapter_kwargs={'feature_layer': 'pool5'})
assert r.dim == 1280   # 若不成立，就是传参没到达构造函数
```

#### C4 权重加载是否可验证（不能 `strict=False` 蒙混）

| 方法 | 检查点 |
|---|---|
| **deepprofiler** | `load_weights(by_name=True)` 会**静默跳过**名字不匹配的层。必须有断言证明权重真的进去了：加载前后对比第一个**有权重**的层，若逐字节相同则抛 `weight-load-noop` |
| **o2vae** | `strict=False` 不可避免（188 个 buffer 由 e2cnn 重算）。但必须断言 **missing 里没有任何可学参数**、`unexpected` 为空，否则抛 `weight-mismatch` |
| **ca_mae** | `strict=True`；记录 missing/unexpected |
| **cellclip** | 逐 key 检查 missing/unexpected |

**判定**：加载断言必须能**真的失败**。建议构造一次反向验证（例如临时改一个层名），
确认断言会触发——只写不断言等于没写。

#### C5 输出语义

| 方法 | 官方取哪一层 | 检查点 |
|---|---|---|
| deepprofiler | 本层取 `block6a_activation`（672）；上游 README 默认 `pool5`（1280） | `feature_layer` 默认值是否**有据可依**；两个都可达 |
| ca_mae | 官方 `predict()` 的返回 | adapter 是否与官方 `huggingface_mae.py` 一致 |
| cellclip | `pooler_output`（CLS，1536）→ CrossChannelFormer | 是否取 CLS 而非最后一层 hidden |
| o2vae | 隐变量均值 `z_mu`，**不是**重建、不是 logvar | 是否 `torch.split(latent, zdim)[0]` 的 mean |

---

### 路线 D：软件设计规范（侧重 embedding 这一块）

#### D1 分层与依赖方向

项目有分层规则（`docs/layers.py`，测试 `tests/test_layering.py`）。

```powershell
& 'D:\项目\CellPainting-Claw-v2\.venv\Scripts\python.exe' -m pytest tests/test_layering.py -q
```

**检查**：
1. `models/` 归在哪一层？是否被更高层依赖而**不反向依赖**？
2. `embedding_benchmark.py` 是否在中层（它依赖 models，不被 models 依赖）？
3. `contract.py` 是否**零重依赖**（能在无 torch/TF 的环境里跑）？

**判定**：`check_compatible` 必须是纯函数；若它 import 了 numpy/torch，判 FAIL。

#### D2 开闭原则 / 扩展成本

**检查**：新增一个方法需要改几处？
正确答：**两处**——`registry.py` 加一条 `AdapterRegistration`，`adapters/` 加一个类。
若需要改 `embedding.py` 或 `contract.py`，说明抽象漏了，判 FAIL。

#### D3 依赖倒置

**检查**：`embedding.py` 是否 `import` 任何具体 adapter？
正确答：**否**，通过 `registry` 的 factory 字符串懒加载。
`registry.ADAPTERS` 里的 `factory` 应是**模块级字符串**，在 `load_factory()` 时才 import。

#### D4 类型化错误

**检查**：错误是否区分了
「输入不合法」`IncompatibleInput` /「环境缺依赖」`MissingDependency` /
「权重不可得」`WeightsUnavailable` /「方法不存在」`UnknownModel`。
四者不可混用；不得抛裸 `ImportError` / `KeyError` / `AssertionError`。

#### D5 契约是否"可执行"

**检查**：`check_compatible` 的每一条 issue 是否都有 `code`，且
`CompatibilityReport.raise_for_status()` 能把 code 带进异常。
断言"声明了但从不检查"的字段——这是本任务修过的一类缺陷
（`input_feature_dim` 未校验、`CHANNEL_EMBEDDINGS` 未限制 `ndim`）。

#### D6 结果对象的信息完整性

`EmbeddingResult` 的**实际**字段（扁平结构，**没有**嵌套的 `provenance` 子对象——
按 dataclass 字段核对，不要凭直觉找 `result.provenance`）：

```
model  values  dim  input_unit  backend  feature_layer  n_samples
weights_path  weights_sha256  license  determinism
metadata  sample_metadata
```

属性方法：`shape`、`as_numpy()`、`as_torch()`、`to_dict()`、`to_frame()`、`save_npz()`。

`metadata` 里应含 `channels`、`preprocess_scale`、`contract_codes`、`chain_weights`。

**检查** `to_dict()` / `save_npz()`：
- `save_npz` 是否保留**逐行身份**（`sample_cell_id` 等）？不保留则 benchmark 产物无法事后对齐。
- `to_dict` 是否可 JSON 序列化（无 ndarray / Path 泄漏）？

#### D7 缓存与状态

**检查**：
1. adapter 缓存是**可清理**的（`clear_adapter_cache()`）
2. 缓存键包含权重摘要 + device + 构造参数
3. 缓存复用是否安全（adapter 在 `load()` 之后只读？）

#### D8 副作用与可测试性

**检查**：
1. 导入 `cellpaint_pipeline.models` 是否**不触发**权重下载、不建模型、不 import torch/TF？
2. 契约测试能否在**没有框架、没有权重**的环境跑通？
3. `sys.path` 是否被 adapter 修改（`o2vae` 曾把上游仓库根加入 `sys.path`，
   裸 `import models` 会被遮蔽——这是隐患）

#### D9 文档与代码一致性

**检查**：`adapters/*.py` 顶部的 docstring 是否与 `registry.py` 的 manifest 一致。
本任务曾发现 3 处漂移（`o2vae` 仍写"自训 32×32 dim64"、`deepprofiler` 仍写
`scale='raw'`、`unavailable.py` 仍写"O2-VAE 从未发布"）。这类漂移会让审查者误判。

---

## 5. 已知陷阱清单（审查时务必逐条验证，不要重蹈覆辙）

这些是**真实发生过**的错误，每一条都已修并加了回归测试。
审查者的任务是确认"确实修了"，而不是相信文档。

| # | 陷阱 | 后果 | 现在应表现为 |
|---|---|---|---|
| 1 | DP 通道顺序 `DNA,RNA,ER,…` | RNA/ER 静默互换，不报错 | 常量 = `DNA,ER,RNA,AGP,Mito`；旧顺序被 `channel-order` 拒 |
| 2 | DP 归一化用 joint min-max | 原始像素基线被抬高，**结论反转** | `per_crop_minmax` + 数据侧 per-channel |
| 3 | DP 声明 `scale='raw'` | 16-bit 输入被接受，误差 **4.5e6** | `value_range=(0.0,1.0)` |
| 4 | `unidino` 声明 `order_matters=False` | 置换输入静默通过（实测差 2.109） | `order_matters=True` |
| 5 | `morphogenie` 声明 `output_dim=256` | 与实测 `(4,10)` 不符 | `output_dim=10` |
| 6 | `o2vae` 用自训手写数字权重 | 不是细胞模型，结论无意义 | 官方 MEF 权重；`zdim=512`；解码器 6 层；空间 64 |
| 7 | `embed()` 每次重载权重 | 7 s/次，benchmark 不可用 | 缓存后 0.031 s，两次结果 `identical=True` |
| 8 | `feature_layer` 进了 `**kwargs` | 参数被前向忽略，调用者以为生效 | 走 `adapter_kwargs` 进构造函数；`pool5` 应得 1280 |
| 9 | `models.embed_all` 文档写了但未导出 | 文档承诺的调度入口不可用 | `hasattr(models,'embed_all') == True` |
| 10 | `save_npz` 丢 `sample_metadata` | benchmark 产物无法按 `cell_id` 对齐 | npz 含 `sample_cell_id` |
| 11 | `summary.json` 的 `finished_at` 恒空 | 先写文件后赋值 | 赋值在 `_write_artefacts` 之前 |
| 12 | CellCLIP 放行 4D 像素堆 | 错输入到 forward 才炸 | `ndim` + `feature-width` 双拒 |
| 13 | 加载断言取 `base.layers[0]` | 那是 InputLayer，**没有权重** | 取"第一个有权重的层" |
| 14 | `embed_all` 用 `__errors__` 魔法键 | 字符串被当成 `EmbeddingResult` | `BatchResult(results, errors)` |

---

## 6. 环境与命令

```powershell
# 契约与调度测试（最快，无需 torch/TF/权重）——0.4 s
cd D:\项目\CellPainting-Claw-v2
& '.\.venv\Scripts\python.exe' -m pytest tests/test_models_contract.py tests/test_embedding_benchmark.py -q

# 全量套件（14 个失败为本任务之前既有，不属于本层）
& '.\.venv\Scripts\python.exe' -m pytest tests/ -q

# 真实权重环境（torch 2.5.1 + tf 2.15.1）
$env:PATH = 'C:\Users\DELL\miniconda3\envs\torch_gpu\Library\bin;' + $env:PATH
$env:TF_CPP_MIN_LOG_LEVEL = '3'
& 'D:\ep_bench\venv\Scripts\python.exe' -c "import sys; sys.path.insert(0,'D:\项目\CellPainting-Claw-v2\src'); from cellpaint_pipeline import models; print(models.doctor())"

# benchmark 全流程
& 'D:\ep_bench\venv\Scripts\python.exe' D:\项目\CellPainting-Claw-v2\scripts\embedding_benchmark\score_benchmark.py --cells-dir E:\bbbc022\cells
& 'D:\ep_bench\venv\Scripts\python.exe' D:\项目\CellPainting-Claw-v2\scripts\embedding_benchmark\stability_check.py --cells-dir E:\bbbc022\cells --folds 4
```

**目录**

| 内容 | 路径 |
|---|---|
| 主开发树 | `D:\项目\CellPainting-Claw-v2` |
| 官方对照树 | `D:\项目\CellPainting-Claw-official` |
| 上游 7 仓库原码 | `D:\official_src\{1-deepprofiler,…,7-morphogenie}` |
| 权重根 | `E:\cpc_weights\` |
| 数据集与产物 | `E:\bbbc022\` |
| benchmark 脚本 | `scripts\embedding_benchmark\` |

---

## 7. 判定标准

### 每条检查给出三档，不要只写"通过"

| 档 | 含义 |
|---|---|
| **PASS** | 有实测证据（命令输出/数值）支撑 |
| **CONDITIONAL** | 行为正确但有已知偏差，必须写明偏差是什么、影响多大 |
| **FAIL** | 断言不成立，或"声明了但没检查" |

### 必答的五个结论

1. **权重**：4/4 哈希是否与上游原始文件一致？链式权重是否完整？
2. **黑盒**：输出形状/维度是否与官方一致？契约是否在**加载前**拦住错输入？
3. **白盒**：通道顺序、预处理算子、参数落点、加载断言，四项是否都经过了**独立核对**？
4. **设计**：分层、开闭、依赖倒置、类型化错误、契约可执行性，五项是否成立？
5. **信任边界**：哪些结论是**实测**的，哪些只是**文档声称**的？必须分开列。

### 已经知道的、不构成缺陷的事实（避免误报）

- 全量套件 **14 个失败**是本任务之前既有的，与本层无关
- `unidino` 的 `output_dim=384` 表达不了真实的 `384*C`——已知不足，不是回归
- `PreprocessSpec` 目前是**纯声明**，真正做预处理的是 `scripts/embedding_benchmark/`
  ——这是已知的职责边界问题，不建议在审查中当作"新发现"
- 3 个方法 `runnable=False` 是**如实标注**，不是"没做完"
- `parity` 只证明了"接口层 == 本项目官方树"，`docs/parity_check_c.md` 已声明该限定

---

## 8. 交付物

审查者应产出：

1. **逐项结论表**：路线 A/B/C/D 下每条检查的 PASS / CONDITIONAL / FAIL
2. **证据**：每条结论附命令与输出（或关键数值），不要只写判断
3. **缺陷清单**：按 P0（会让结论错）/ P1（影响调度/复现）/ P2（文档卫生）分级
4. **信任边界**：实测项 vs 声称项分列
5. **不应该被外推的结论**：例如"这份 benchmark 的排序只在该数据集子集上成立"
