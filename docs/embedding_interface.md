# Embedding 统一接口 — 使用与扩展说明

> 对应实现：`src/cellpaint_pipeline/models/`
> **覆盖方法（2026-10-27 现状）：v2 四个官方方法 + v3 三个替身方法，七个全部可运行。**
>
> 本文档的 §2 是 2026-10-09 的快照，保留作历史；当前状态以 §2.1 为准。

---

## 1. 一句话

```python
from cellpaint_pipeline import models

result = models.embed('deepprofiler', crops)   # ← 一个方法名拿到 Embedding
result.values      # (N, 672) float32
result.dim         # 672
result.license     # 随结果带出，便于评测报告合规
```

---

## 2. 实测状态（2026-10-09）

| 方法 | `embed()` 实测 | 输出 | 确定性 | 权重 |
|---|---|---|---|---|
| `deepprofiler` | ✅ 跑通 | `(4, 672)` | `array_equal=True` | 19,236,600 B |
| `ca_mae` | ✅ 跑通 | `(2, 384)` | `array_equal=True` | 712,434,294 B |
| `cellclip` | ✅ 跑通 | `(3, 512)` | `array_equal=True` | 1,799,564,952 B |
| `o2vae` | ⏳ 适配器已就绪、`probe` 报 `ready`，**端到端未实测** | `(N, 64)` | — | 9,297,256 B |
| `dinocells` / `unidino` / `morphogenie` | ❌ 权重不可得 | — | — | — |

契约层自检 **26/26 通过**；`import models` 在干净解释器下**只拉入 1 个模块**。

`models.doctor()` 会区分「本解释器跑不了」与「真的缺东西」：

```
torch 解释器:  ca_mae ready · cellclip ready · o2vae ready
               deepprofiler wrong-interpreter
               unidino needs-deps-and-weights · dinocells/morphogenie needs-weights
conda 解释器:  deepprofiler ready · 其余 wrong-interpreter / needs-*
```

---

## 2.1 当前状态（2026-10-27）

| 剖面 | 方法 | 权重 | 宽度 | 原生输入 |
|---|---|---|---|---|
| `v2`（默认） | `deepprofiler` | official | 672 | `(N,32,32,5)` float [0,1] |
| `v2` | `ca_mae` | official | 384 | uint8 `(N,C,256,256)`，`C≤11` |
| `v2` | `cellclip` | official | 512 | `(N,5,1536)` DINOv2 向量 |
| `v2` | `o2vae` | official | 512 | 二值核剪影 `(N,1,64,64)` |
| `v3` | `unidino` | **surrogate** | **1920** | uint16 `(N,H,W,5)`（逐通道 384×5） |
| `v3` | `dinocells` | **surrogate** | 384 | uint16 5 通道堆栈，mask 被**丢弃** |
| `v3` | `morphogenie` | **surrogate** | **10** | `(N,3,256,256)` float（`c_dim`，非 `z_dim`） |

三个替身的权重是**本地自训**的，`result.metadata['weights_provenance'] == 'surrogate'`。
它们的分数只能证明"接口能驱动该方法"，**不能当作论文性能**。

---

## 2.2 统一调度：一次调用驱动全部七个

```python
from cellpaint_pipeline import models
from cellpaint_pipeline.models import registry

registry.set_profile('v3')          # 默认是 'v2'，只含四个官方方法
batch = models.embed_all(
    names,                          # 七个方法名
    inputs,                         # {name: array} —— 每个方法自己的载荷
    assume={                        # 按方法声明"这是原始采集值"
        'deepprofiler': 'raw_crop', 'ca_mae': 'raw_crop',
        'unidino': 'raw_crop', 'dinocells': 'raw_crop',
    },                              # 未列出的方法 → None = 载荷已就绪
    channels={'morphogenie': ('C0', 'C1', 'C2')},
    on_error='raise',
    weights_roots=('E:/cpc_weights',),
)
batch.results['unidino'].as_numpy()     # (N, 1920)
batch.errors                            # {} 表示七个全成功
```

**`unit` / `assume` / `masks` / `channels` 都接受 `{方法名: 值}` 映射**，未列出的方法取
`None`。这一点是接口成立的关键：七个方法的输入单位、通道数、是否需要先跑
`prepare` 各不相同，若这些参数只能全局广播，调用方就得自己按方法分支——
那正是 `embed_all` 要消灭的东西。

（此前 `assume`/`masks`/`channels` 只能广播，2026-10-27 改为可按方法传；
回归测试见 `tests/test_models_v3_dispatch.py::TestOneCallDrivesEveryMethod`。）

---

## 3. 为什么统一在「契约层」而不是「张量层」

七个方法的输入契约在四个轴上互不相容，硬塞成同一张量形状会丢信息：

| 轴 | 冲突 |
|---|---|
| 维度序 | `(B,H,W,C)`（DeepProfiler）vs **`(B,C,H,W)`**（CA-MAE / CellCLIP） |
| 通道数 | 5 vs **可变 C**（CA-MAE 实测 1/5/11 均可） |
| dtype / 值域 | 16-bit 原值 vs **uint8 0–255** |
| **输入单元** | 单细胞裁剪 / 整视野 / **预计算向量** |

★ **CellCLIP 吃的不是图像**：它接收 5 个 DINOv2-Giant 逐通道向量（宽度 1536）。
所以 `InputUnit` 必须能表达 `CHANNEL_EMBEDDINGS`，`embed(x: 图像)` 不能是唯一入口。

统一落在三处：

1. **`ModelManifest`** —— 声明式契约，**零重依赖**即可导入；
2. **`check_compatible`** —— 纯函数，在加载权重**之前**判定兼容性；
3. **`models.embed()`** —— 校验 → 分发 → 用带溯源的结果包装。

---

## 4. 契约字段（`models/contract.py`）

```python
class InputUnit(str, Enum):        # 3.10 兼容写法，不用 enum.StrEnum
    SINGLE_CELL_CROP / FIELD / CHANNEL_EMBEDDINGS

@dataclass(frozen=True)
class ChannelSpec:
    names: tuple[str, ...]
    order_matters: bool = True
    names_are_labels: bool = True   # ★ 见下

@dataclass(frozen=True)
class PreprocessSpec:
    scale: 'raw'|'div255'|'div65535'|'per_image_selfnorm'|'none'
    value_range: tuple[float, float] | None
    to_dtype: str
    batch_layout: 'BHWC' | 'BCHW'
    illum_corrected: bool

@dataclass(frozen=True)
class WeightsSpec:
    source / format / sha256 / host / downloadable / approx_bytes
    weight_file_layer_count / rebuilt_layer_count   # DeepProfiler: 233 / 232

@dataclass(frozen=True)
class ModelManifest:
    name / version / input_unit / channels / preprocess / weights
    output_dim / feature_layer / spatial
    license / requires / ascii_path_required
    parity_mode / parity_prerequisites / notes
```

### 4.1 两个字段值得单独解释

**`names_are_labels`** —— 名字是否**有语义**。
DeepProfiler 里 `DNA` 就是 DNA 染色，把 Mito 的数据贴这个标签会**静默**污染嵌入，
所以名字集合必须校验；而 CA-MAE 是**通道无关架构**（channelwise cross-attention），
名字只是标注，传什么都行，**只校验数量**。这两件事此前被混为一谈，
是本轮实测（CA-MAE 传入 Cell Painting 通道名被误拒）暴露出来的真实缺陷。

**`to_dtype`** —— 值是 dtype，作用是抓**尺度写错**。
`0..1` 是 `0..255` 的**子集**，所以纯数值域检查**抓不到**「已经除过 255」这个最常见错误。
真正的信号是 dtype：声明 `to_dtype='uint8'` 的方法会直接拒绝 float 数组。
（曾考虑「观测跨度 ≥ k×声明跨度」的启发式，**已否决**——视野本身很暗时是假阳性，
而假阳性会训练操作者忽略这条检查。）

### 4.2 `check_compatible` 的判据（每条可失败）

| code | 触发 |
|---|---|
| `input-unit` | 喂错输入单元（如把整视野喂给 CellCLIP） |
| `channel-set` | 通道名集合不符（当 `names_are_labels=True`） |
| `channel-count` | 通道数不符（含 channel-agnostic 方法的数量校验） |
| `channel-order` | 顺序不符且 `order_matters=True` |
| `ndim` | 不是 3D/4D |
| `spatial-size` | 空间尺寸与声明不符 |
| **`dtype`** | 已除过 255 却传 float（声明 uint8 时） |
| **`value-range`** | 数值超出声明值域 |
| `ascii-path` | `ascii_path_required=True` 但运行路径含非 ASCII |

---

## 5. 三个环境的分工

方法运行在不同的解释器里，**适配器本身与解释器无关**：

| 用例 | 解释器 | 关键依赖 |
|---|---|---|
| 契约层 / `doctor()` / `describe()` | 任意 | **无** |
| DeepProfiler | `cellpainting-claw` conda 环境 | tensorflow 2.10.1 + efficientnet 1.1.1 |
| CA-MAE / CellCLIP / O2-VAE | `D:\ep_bench\venv` | torch 2.5.1+cu124, timm, transformers 4.46.1 |

权重根：`E:\cpc_weights`（**纯 ASCII**）。可用 `CPC_WEIGHTS_ROOT` 覆盖。
**永不隐式下载**——缺失时抛 `WeightsUnavailable` 并列出所有探测过的路径。

---

## 6. 新增一个方法（扩展点）

两步，不需要动其它任何文件。适配器继承
:class:`~cellpaint_pipeline.models.adapters.base.BaseAdapter`，
后者已经处理好**权重解析（禁止隐式下载）、框架导入守卫、设备选择、幂等加载**——
这四件事每个适配器都要做，写四遍必然有一遍写错，而其中最贵的一遍是
"禁止隐式下载"（上游默认会静默从 Zenodo 拉 19 MB，本机约 40 分钟）。

```python
# 1) 在 models/registry.py 的 ADAPTERS 里追加一条
AdapterRegistration(
    manifest=ModelManifest(
        name='my_method',
        input_unit=InputUnit.SINGLE_CELL_CROP,
        channels=ChannelSpec(names=('DNA','RNA','ER','AGP','Mito')),
        preprocess=PreprocessSpec(scale='div255', value_range=(0.0,255.0),
                                  to_dtype='uint8', batch_layout='BCHW'),
        weights=WeightsSpec(source='my_method/model.pt', format='torch-pth',
                            host='local', downloadable=False),
        output_dim=768,
        license='Apache-2.0',
        requires=('torch',),
    ),
    module='cellpaint_pipeline.models.adapters.my_method',
    factory='MyMethodAdapter',
)

# 2) 写适配器：只需 3 个方法
from cellpaint_pipeline.models.adapters.base import BaseAdapter

@dataclass
class MyMethodAdapter(BaseAdapter):
    manifest: ModelManifest = None            # 注册表注入
    requires_frameworks = ('torch',)          # 供 require_frameworks() 用
    dependency_hint = '需要 torch 环境'

    def check_dependencies(self) -> None:
        self.require_frameworks()             # 缺依赖 -> MissingDependency

    def _build(self) -> Any:
        checkpoint = self.resolve_weights()   # 本地优先，缺失即报错
        model = ...                            # 构造网络并 load_state_dict
        self._device = self.select_device(torch)
        return model.to(self._device)

    def _forward(self, array, **kwargs):
        with torch.no_grad():
            return self._model(array).cpu().numpy()
```

`load()` / `embed()` / `embed_many()` 由基类提供且**幂等**，
所以作者不会忘记"已加载就别再加载"的守卫，也不会每次调用重载 700 MB。

运行时注册也支持：`models.register(registration)`。

**纪律**：`check_compatible` 必须先于任何权重加载与框架导入；
缺依赖与输入不符必须是**可区分**的错误（`MissingDependency` / `IncompatibleInput`）。

---

## 6.1 输出侧的导出（基准评测需要）

`EmbeddingResult` 自带导出，调用方不必自己拼表：

```python
res = models.embed('ca_mae', crops, unit='single_cell_crop',
                   sample_metadata={'cell_id': [...], 'plate': [...]})

res.as_numpy()             # (N, D) —— 最常用，numpy 已是传递依赖
res.as_torch(device='cuda')# 惰性导入 torch，TF-only 环境仍可用 numpy 路径
res.to_frame()             # DataFrame：元数据列 + Cells_<model>_0001..D
res.save_npz(path)         # 与上游 DeepProfiler .npz 同构（features + metadata）
res.to_dict()              # JSON 摘要（不含数组本体）
```

**`sample_metadata` 为什么进契约**：只有 `(N, D)` 数组无法说明每行对应哪个细胞，
而任何下游比较都需要这个对应。适配器把它知道的东西放进来，本层不臆造。

`to_frame()` 的列名沿用项目既有的 `Cells_DeepProfiler_0001` 约定，
所以本层产出的表可以**直接与流水线自己的输出拼接**而不必改名。

---

## 6.2 已用统一模式落地的方法

| 方法 | 输入单元 | 布局 | 输出 | 依赖 |
|---|---|---|---|---|
| `deepprofiler` | 单细胞裁剪 | `BHWC` | 672 | tensorflow + efficientnet |
| `ca_mae` | 单细胞裁剪 | `BCHW` | 384 | torch（+ 权重自带编码器代码） |
| `cellclip` | **通道嵌入** | `BCHW` | 512 | torch + safetensors |
| `o2vae` | 单细胞裁剪 | `BCHW` | 64 | torch + e2cnn + 上游仓库 |

四者共用同一个 `embed()` 调用；差异全部落在 manifest 与各自的 `_build`/`_forward` 里。


---

## 7. 实测暴露的坑（写进适配器，不再重犯）

1. **CA-MAE 的包布局**：checkpoint 目录**本身就是包**（`openphenom/` 内含
   `huggingface_mae.py`、`vit.py`… 与 `model.safetensors`），编码器用**相对导入**。
   必须把**包的父目录**加入 `sys.path` 并 `import openphenom.huggingface_mae`；
   把 checkpoint 目录本身加进去会报 `attempted relative import with no known parent package`。
2. **CellCLIP 的 MLP 子层必须具名**：checkpoint 存的是
   `transformer.resblocks.N.mlp.c_fc.weight`，用匿名 `nn.Sequential` 会生成
   `mlp.0.weight`，导致**全部 MLP 权重被判为 unexpected 而静默保留随机初始化**。
   这是本层最有价值的 `strict` 断言换来的——没有它，模型会「能跑但结果错」。
3. **`np.typing` 陷阱**：`pycytominer==1.4.0` 用 `np.typing.ArrayLike` 却不导入该子模块，
   NumPy ≥1.25 下 `import pycytominer` 直接失败。已用 `sitecustomize.py` 修（见环境 yml 注释）。

### 7.1 ★ O2-VAE 的 `models` 包名遮蔽（接线前必读）

O2-VAE 上游仓库有一个**顶层 `models/` 包**（`models/vae.py`、`models/encoders_o2/e2scnn.py`…），
而本层也叫 `models`。实测结论：

```
sys.path 加入 o2vae_repo 后：
  cellpaint_pipeline.models  → 不受影响（全限定名，仍是本层）
  bare `import models`       → 变成 E:\cpc_weights\o2vae_repo\models\__init__.py
```

即**本层不会被劫持，但裸 `models` 这个名字会被遮蔽**。因此接线 O2-VAE 时必须：

* 只在**需要时**把仓库根加入 `sys.path`，并在其之前先 `import cellpaint_pipeline.models`；
* 或改为按**文件路径**加载仓库模块（`importlib.util.spec_from_file_location`），
  完全不动 `sys.path`。

**当前状态**：O2-VAE 适配器已写好（`adapters/o2vae.py`），权重与上游仓库的就位关系
已通过 `requires_upstream_repo` + `repo_source` 显式声明，`probe('o2vae')` 报 `ready`。
**但 `models.embed('o2vae', ...)` 尚未实测** —— 按用户要求「先优化已有权重的几个方法」，
本轮的**验证范围是 DeepProfiler / CA-MAE / CellCLIP 三个**，O2-VAE 属于下一步。

`E:\cpc_weights\o2vae\` 下用目录联接（junction）指向 `o2vae_repo\` 的
`models/`、`utils/`、`configs/`、`data/`、`registration/`、`prealignment/`，
使「权重目录」与「仓库目录」在同一个根下可解析，而无需复制 471 MB 的仓库。

---

## 8. 已知边界（不要误读）

- **CellCLIP 的完整链路未跑通**。它消费 DINOv2-Giant 的逐通道向量，
  而 DINOv2-Giant **不在这份 1.8 GB 权重里**。本适配器验证的是
  「**权重正确加载 + 前向正确 + 接口正确**」，不是「像素 → 嵌入」的端到端。
  生成那些向量是独立的上游阶段。
- **`o2vae` 的权重是本项目自训的**（50 epoch，o2-mnist 合成数据，zdim=64），
  **不是**论文级 O2-VAE。它验证 VAE 族的接口形状，**不可**用于方法间公平比较。
- **`dinocells` / `unidino` / `morphogenie` 声明了契约但不可运行**：
  三者权重分别在 Zenodo（实测 ~10 KB/s）和 Google Drive（不可达）。
  `models.probe(name)` 会给出具体原因。
- **许可差异会影响 benchmark 的可分发性**：只有 DeepProfiler 代码与权重双双宽松。
  许可已记入 manifest 并随 `EmbeddingResult` 输出。
