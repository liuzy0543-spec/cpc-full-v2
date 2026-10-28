# 剩余 3 个 Embedding 方法：能否嵌入 + 能否统一 IO

> 日期：2026-10-10
> 验证依据：`docs/embedding_verification_7methods.md`
> 被测代码：`D:\cpc_c_v2\src\cellpaint_pipeline\models\`

---

## 第一部分：能否嵌入现有优化代码

### 1.1 依赖缺口 —— 几乎为零

实测 `D:\ep_bench\venv`：

| 方法 | 需要的包 | 已具备 | 缺 |
|---|---|---|---|
| `unidino` | torch, einops, timm, pycytominer, pytorch_lightning, sklearn | **全部具备** | **无** |
| `dinocells` | torch, kornia, oyaml, skimage, PIL, pandas | kornia 等已具备 | **`oyaml`**（1 个纯 python 小包） |
| `morphogenie` | torch, yaml, sklearn, seaborn, torchmetrics | **全部具备** | **无** |

**结论：依赖不是障碍。** 三个方法里只有 DINO4Cells 缺 1 个小包。

> 注：`pytorch_lightning` / `pycytominer` / `kornia` 是我本轮为了跑官方示例**已经装上的**。
> 也就是说，验证过程本身把它们准备好了。

### 1.2 权重缺口 —— 这才是真障碍

| 方法 | 权重 | 大小 | 现状 | 障碍 |
|---|---|---|---|---|
| `unidino` | `checkpoints.zip` | 1,379,562,715 B | `E:\cpc_weights\unidino\checkpoints.partial.zip` **5,701,424 B (0.41%)** | Zenodo ~9 KB/s → **≈43 h** |
| `dinocells` | `Cell_Painting_data.tar.gz` | 6,896,284,400 B | `E:\cpc_weights\dino4cells\` **空** | Zenodo ~10 KB/s → **≈180 h** |
| `morphogenie` | `Models.7z` | 未知 | `E:\cpc_weights\morphogenie\` **空** | **Google Drive 本机不可达** |

**结论：依赖能解决，权重不能。**

### 1.3 适配器复杂度 —— 三者差异很大

| 方法 | 适配器要做的事 | 复用 `BaseAdapter` 后估计行数 | 评价 |
|---|---|---|---|
| `unidino` | 逐通道前向 + 拼接；**模块级 import 会拖入 `scellseg` 等可选依赖**，需绕过 | ~180 | 中等，有一个 import 陷阱 |
| `dinocells` | tape→stack 折叠（`order="F"`）、丢掩膜通道、权重键重映射（`teacher`/剥 `module.`/剥 `backbone.`/`strict=False`）、**还需复刻那套很长的 transform 管线** | ~250 | **最难**，transform 管线是主要工作量 |
| `morphogenie` | 两阶段构建 + 权重键重命名（`encode.`→`encoder.layer.`）+ **修掉官方的元组错位** | ~200 | 中等，但**必须偏离官方代码** |

### 1.4 逐个判定

#### ✅ `unidino` —— 可以接，等权重

- 依赖齐备
- 架构已验证（`(2,3,1920)` = 5×384，官方拼接逻辑确认）
- 需要修 manifest 的 `order_matters`（见 §2.2）
- **唯一阻塞：43 小时下载**

#### ⚠️ `dinocells` —— 可以接，但有两个额外缺口

- 缺 `oyaml`（小）
- **官方仓库没有 Cell Painting 的配置**：`config.yaml` 是 HPA（`num_channels: 4`），
  Cell Painting 的 checkpoint 只在 `README.md:211-213` 列出，
  **`num_channels` 该填几在代码里查不到**
- **必须复刻整套 transform 管线**（`config.yaml` 有 420 行，含 `Warp_cell`、
  `rescale_protein`、`remove_channel`、`self_normalize` 等自定义算子）
- **阻塞：180 小时下载** + 论文场景配置缺失

#### ❌ `morphogenie` —— 现在不该接

三个理由，任何一个单独都足够：

1. **权重不可达**：Google Drive 本机不通，且大小未知
2. **官方代码有元组错位**：持久化的"嵌入"是**随机**的重参数化采样
   （实测同一输入两次调用 max 差 2.979）
3. **官方说需要 MATLAB + Cellpose 做 256×256 居中预处理**，
   而仓库里**没有这段代码**（它是 MATLAB 侧）

**即：接它必须"故意偏离官方实现"**——这需要一个明确的决策，不能默认进行。

---

## 第二部分：能否统一 IO 接口为 benchmarking 铺垫

### 2.1 结论：**能，而且现有契约原语已经够用**

我实测了当前 `contract.py` 的取值域，逐个方法对照：

| # | 方法 | `InputUnit` | `Scale` | 布局 | 能否表达 |
|---|---|---|---|---|---|
| 1 | DeepProfiler | `single_cell_crop` | `raw` | `BHWC` | ✅ |
| 2 | CA-MAE | `single_cell_crop` | `div255`(内置) | `BCHW` | ✅ |
| 3 | CellCLIP | `channel_embeddings` | `none` | `BCHW` | ✅ |
| 4 | O2-VAE | `single_cell_crop` | `none`(0-1) | `BCHW` | ✅ |
| 5 | uniDINO | `single_cell_crop` | `div65535`**?** | `BCHW` | ✅（值待考） |
| 6 | DINO4Cells | `single_cell_crop` | `per_image_selfnorm` | `BCHW` | ⚠️ 缺「输入是 tape 不是 stack」这个概念 |
| 7 | MorphoGenie | `single_cell_crop` | `none` | `BCHW` | ✅ |

**实测证据**（`check_compatible` 真的在按契约判定，不是摆设）：
```
3ch vs declared C0,C1,C2 -> PASS []
5ch (Cell Painting)     -> FAIL ['channel-set', 'channel-count']
```

**7 个方法里 6 个能被现有原语完整表达，只有 1 个需要加字段。**

### 2.2 必须先修的 3 个 manifest 错误（`docs/embedding_verification_7methods.md` §4）

| # | 方法 | 字段 | 现在 | 应为 | 危害 |
|---|---|---|---|---|---|
| 1 | `unidino` | `order_matters` | `False` | **`True`** | **静默错误**：换通道顺序不报错，嵌入错 |
| 2 | `morphogenie` | `output_dim` | `256` | **`10`** | 契约不符 |
| 3 | `unidino` | `scale` | `div65535` | 无据可查，标注来源或降级 | 可能误判值域 |

**这 3 个错误都是在"没有真实权重、也没读官方代码"时写下的。**
其中 #1 恰恰是这一层设计出来要拦的那类静默错误——**结果它自身的申报就是错的**。

**这是对"统一 IO 接口"最强的支持性证据**，也是最强的警示：
申报错一次，整层守卫就失效。

### 2.3 需要扩展的唯一原语

```python
@dataclass(frozen=True)
class PreprocessSpec:
    scale: Scale
    ...
    #: 新增：输入张量在应用 scale 之前的物理排布
    #:   'stack' = (N,C,H,W) 已是通道堆叠（6/7 个方法）
    #:   'tape'  = (H, W*C) 通道横向拼接，需先 reshape(order='F') 再丢掩膜
    input_encoding: Literal['stack', 'tape'] = 'stack'
```

**为什么需要**：DINO4Cells 的输入不是 `(N,C,H,W)`，而是"胶带"格式 `(h, w*c)`，
官方 `data_utils/cellpainting_dataset.py:23-26` 用 `np.reshape(..., order="F")` 还原，
`:31-33` 再丢掉最后一通道（掩膜）。**只看 shape 抓不到这个差异**——
`(32, 192)` 和 `(32, 192)` 长得一样，但一个是 6 通道 tape、一个是别的东西。

**为什么这是"加一个字段"而不是"改设计"**：它落在 `PreprocessSpec`
（本来就负责"shape 检查抓不到的东西"）里，一行默认值，其余 6 个方法不受影响。

### 2.4 统一后能否真的做 benchmarking

**能，但前提是先把 3 件事定死**，否则 benchmark 的结论不可信：

| 前提 | 为什么 |
|---|---|
| **① 确定性** | MorphoGenie 官方嵌入是**随机**的（实测差 2.979）。benchmark 必须要么固定种子、要么改用后验均值。**否则每次跑出不同数字** |
| **② 通道契约正确** | uniDINO 的 `order_matters` 错误会让置换输入静默通过。**benchmark 的对比会建立在错误嵌入上** |
| **③ 输入来源一致** | 7 个方法的值域要求互不兼容（raw 16-bit / uint8 / 0-1 / 向量）。**同一批细胞必须按各自契约转换，转换错误不报错** |

**这三条正好是统一契约层负责的东西**——所以它确实是为 benchmarking 铺垫的**正确位置**。

### 2.5 但有一个 benchmark 层面的真实缺口

`EmbeddingResult` 现在**没有任何关于"这是第几次运行/种子"的字段**，
也没有「同一输入重复 N 次的方差」概念。

对**随机方法**（MorphoGenie）和**受 GPU 非确定性影响的算子**，
benchmark 需要一个显式的确定性标记。建议在 manifest 加：

```python
#: 'deterministic' | 'stochastic'
determinism: Literal['deterministic', 'stochastic'] = 'deterministic'
```

并在 `EmbeddingResult` 里回传，这样下游看到 `stochastic` 就知道要跑多次取统计。

---

## 第三部分：建议的推进顺序

| 优先级 | 事项 | 理由 | 成本 |
|---|---|---|---|
| **P0** | 修 3 个 manifest 错误 | 现在的申报是**错的**，不修则整层守卫失效 | 10 分钟 |
| **P0** | 把 26 条契约检查搬进 pytest | 目前 `models/` **被 0 个测试断言**，改了没人拦 | ~1 h |
| **P1** | 加 `PreprocessSpec.input_encoding` + `ModelManifest.determinism` | 为 DINO4Cells 和随机方法预留 | ~30 min |
| **P1** | 补"新接口层产出 == 官方路径产出"的等价性验证 | 目前**未验证**，是最大缺口 | ~2 h |
| **P2** | 接 `unidino`（等权重） | 依赖齐、架构已验证，只差 43 h 下载 | 权重到手后 ~2 h |
| **P3** | 接 `dinocells` | 还缺论文场景配置 + transform 管线复刻 | 权重到手后 ~1 天 |
| **P4** | `morphogenie` | 权重不可达 + 官方代码有缺陷 + 需 MATLAB 预处理 | **暂缓** |

**关于 P4 的一个判断**：MorphoGenie 目前**不适合作为 benchmark 基准**，
因为它的官方实现本身不确定、且复现需要的预处理步骤不在仓库里。
如果一定要包含它，应当**明确标注"非官方实现复现"**，像 O2-VAE 那样。
