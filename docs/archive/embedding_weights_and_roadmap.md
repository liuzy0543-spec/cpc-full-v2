# Embedding 方法：权重可得性汇总 + 接入技术路线图

> 日期：2026-10-10
> 用途：为「统一调度 + benchmark」提供决策依据
> 前置文档：`embedding_verification_7methods.md`、`embedding_integration_feasibility.md`、`embedding_layer_hardening.md`

---

# 第一部分：权重可得性汇总

## 1.1 结论表

| # | 方法 | 权重状态 | 大小 | 下载链接 | 本机实测 |
|---|---|---|---|---|---|
| 1 | **DeepProfiler** | ✅ **已就位** | 19,236,600 B | [zenodo 7114558](https://zenodo.org/records/7114558/files/Cell_Painting_CNN_v1.hdf5?download=1) | HTTP 200 · sha256 已记录 |
| 2 | **CA-MAE / OpenPhenom** | ✅ **已就位** | 712,434,294 B | [hf-mirror recursionpharma/OpenPhenom](https://hf-mirror.com/recursionpharma/OpenPhenom/resolve/main/model.safetensors) | HTTP 200 · 302 重定向可用 |
| 3 | **CellCLIP** | ✅ **已就位** | 1,799,564,952 B | [hf-mirror suinleelab/CellCLIP](https://hf-mirror.com/suinleelab/CellCLIP/resolve/main/model.safetensors) | HTTP 200 · 302 重定向可用 |
| 4 | **O2-VAE** | ⚠️ **自训**（官方未发布） | 9,297,256 B | 无官方链接 | 本地 50 epoch |
| 5 | **uniDINO** | ❌ **下载受阻** | 1,379,562,715 B | [zenodo 14988837](https://zenodo.org/records/14988837/files/checkpoints.zip) | URL 有效，**~9 KB/s → ≈43 h** |
| 6 | **DINO4Cells** | ❌ **下载受阻** | 6,896,284,400 B | [zenodo 8061428](https://zenodo.org/records/8061428/files/Cell_Painting_data.tar.gz?download=1) | URL 有效，**~10 KB/s → ≈180 h** |
| 7 | **MorphoGenie** | ❌ **不可达** | 未知 | Google Drive 文件夹 | **本机不通** |

## 1.2 一句分割线

> **4 个方法（1-3 官方权重 + 4 自训）已可运行；3 个方法（5-7）的官方权重在本机网络条件下不可获得。**

## 1.3 权重目录现状（`E:\cpc_weights`）

```
cellclip/         1,799,565,137 B   2 files   config.json + model.safetensors
deepprofiler/        19,236,600 B   1 file    Cell_Painting_CNN_v1.hdf5
openphenom_pkg/     712,537,960 B  22 files   openphenom/ 包（代码+权重）
o2vae/                9,297,256 B   1 file    o2vae_50ep.pt         ← 自训
unidino/              5,701,424 B   1 file    checkpoints.partial.zip  ← 0.41%，停滞
dino4cells/                   0 B   空
morphogenie/                  0 B   空
o2vae_repo/           471 MB       上游仓库（用目录联接挂载，未复制）
```

## 1.4 本机网络实测（2026-10-10）

| 主机 | 状态 | 速率 |
|---|---|---|
| `hf-mirror.com` | ✅ 可用 | ~460 KB/s 单流 / ~1.33 MB/s 4 并发 |
| `pypi.tuna.tsinghua.edu.cn` | ✅ 可用 | 1.25 MB/s |
| `gh-proxy.com` | ✅ 可用（git clone / tarball） | — |
| `zenodo.org` **文件下载** | ⚠️ 可用但极慢 | **8.8–10.6 KB/s** |
| `zenodo.org` **REST API** | ❌ **TLS 阻断**（curl exit 35） | — |
| `zenodo.org`（2026-10-10 复测） | ❌ **完全无响应** | — |
| `huggingface.co` / `github.com` 直连 | ❌ 阻断 | — |
| Google Drive | ❌ 不可达 | — |

**两个必须知道的限制**：
1. **Zenodo REST API 被 TLS 阻断**（exit 35），所以**无法列出大包内的文件清单**——
   因此不能只下 DINO4Cells 里那两个 `*.pth`（约 330 MB + 84 MB），**只能整包 6.9 GB**。
2. Zenodo 在 2026-10-10 复测时**连文件下载也无响应**，说明它比记录的更不稳定。

---

# 第二部分：技术路线图（面向调度 + Benchmark）

## 2.0 设计前提：三条不变量

benchmark 的可信度取决于三件事，**它们必须在阶段 0 就锁死**：

| 不变量 | 为什么 | 现状 |
|---|---|---|
| **确定性可比** | 用随机方法跑一次对比确定方法，是拿噪声比信号 | ✅ 已加 `ModelManifest.determinism`，MorphoGenie 标为 `stochastic` |
| **输入契约正确** | 7 个方法值域/布局/通道语义互不兼容，错了不报错 | ✅ 已有 `check_compatible` + 30 个测试；已修 4 个申报错误 |
| **权重可追溯** | "这个数字是哪份权重跑出来的"必须可回答 | ⚠️ 已记录 sha256，但 `EmbeddingResult` 未强制校验 |

## 2.1 阶段总览

```
阶段 0  契约固化          ← 已完成大半
阶段 1  等价性验证        ← ★ 最大缺口，必须先做
阶段 2  统一调度入口
阶段 3  Benchmark 执行器
阶段 4  评测指标
阶段 5  受阻方法处置
```

---

## 阶段 0：契约固化 ✅ 基本完成

| 事项 | 状态 |
|---|---|
| `ModelManifest` / `PreprocessSpec` / `ChannelSpec` / `WeightsSpec` | ✅ |
| `check_compatible` 9 类判据 | ✅ |
| `input_encoding`（tape vs stack） | ✅ |
| `channel_count_fixed`（与 `names_are_labels` 解耦） | ✅ |
| `determinism` 上到结果对象 | ✅ |
| 30 个契约测试（不需要 torch/TF/权重，0.15 s） | ✅ |
| 4 个实测申报错误修正 | ✅ |

**遗留**：
- `EmbeddingResult.weights_sha256` 只是**照抄 manifest**，**没有真正校验文件**
  → benchmark 要求"这个数字来自这份权重"，必须真校验

---

## 阶段 1：等价性验证 ★ 最高优先级

**问题**：`models/` 这 2516 行与官方路径的**等价性从未验证**。
之前做的一致性对拍走的是**官方 CLI 路径**，而流水线**根本不引用 `models/`**——
所以那 2516 行只有"能跑"的证据，没有"与官方一致"的证据。

**要做的**：

```
对每个已就位的方法（1-3）：
  同一批输入
    ├─ 路径 A：官方 CLI / 官方原代码   → 嵌入 X
    └─ 路径 B：models.embed(name, ...) → 嵌入 Y
  断言 np.array_equal(X, Y)  （要求逐位，不是 allclose）
```

**验收标准**：`max|diff| == 0.000e+00`，与 DeepProfiler 双树对拍同等级。

**为什么排最前**：如果这一层与官方不等价，后面所有 benchmark 数字都不可信；
而它同时也决定了这 2516 行**值不值得留**。

**参考**：`_probe/compare_c.py` 的分层比对脚本可直接改造复用。

---

## 阶段 2：统一调度入口

### 2.1 调度需要的能力

| 能力 | 现状 | 需要 |
|---|---|---|
| 按名字取方法 | ✅ `models.embed(name, x)` | — |
| 列出可用方法 | ✅ `models.available()` | — |
| 查询就绪状态 | ✅ `models.doctor()` | — |
| **批量跨方法跑同一批输入** | ❌ | 新增 `models.embed_all(names, x)` 或调度器 |
| **从配置驱动** | ❌ | 新增加 `configs/embedding_benchmark.json` 支持 |
| **结果落盘可追溯** | ⚠️ 有 `save_npz` | 需固定命名 + 权重校验 |

### 2.2 建议的调度接口

```python
# 面向 benchmark 的入口（新增）
from cellpaint_pipeline import models

# 同一批输入，跑多个方法，返回统一契约的结果
results = models.embed_all(
    ['deepprofiler', 'ca_mae', 'cellclip'],
    crops,                                  # 或按方法各自转换
    unit='single_cell_crop',
    channels=('DNA','RNA','ER','AGP','Mito'),
    sample_metadata={'cell_id': ids},
)

# 每个结果自带：值、维度、权重 sha256、许可、确定性、契约报告
for r in results:
    print(r.model, r.shape, r.determinism, r.weights_sha256)
```

### 2.3 调度必须处理的 3 个真问题

| 问题 | 说明 |
|---|---|
| **输入形态不统一** | CellCLIP 吃 `(N,5,1536)` 向量而非图像。调度器必须按 `input_unit` **分发到不同输入源**，不能假设"同一批 crops 喂所有方法" |
| **值域不兼容** | DeepProfiler 要 raw 16-bit，CA-MAE 要 uint8。调度器需按 `preprocess.scale`/`to_dtype` 做**各自转换**，且转换失败要报错而非静默 |
| **随机方法** | `determinism='stochastic'` 的方法需**重复 N 次**并记录均值/方差，不能只取一次 |

---

## 阶段 3：Benchmark 执行器

### 3.1 输入侧

```
输入：一份 Cell Painting 数据集（单细胞裁剪）
      + 一份方法清单（含各自权重路径）
输出：每个方法一份嵌入 + 一份溯源记录
```

**关键设计**：**同一批细胞 ID**。`sample_metadata` 已支持逐行标注，
所以不同方法的输出必须能按 `cell_id` **对齐**——这是能做对比的前提。

### 3.2 输出侧契约

每个方法产出：

| 字段 | 用途 |
|---|---|
| `values` `(N, D)` | 嵌入本体 |
| `cell_id` / `well` / `plate` | **对齐键**，缺了就无法跨方法比较 |
| `weights_sha256` | 溯源 |
| `determinism` | 决定是否要重复跑 |
| `license` | 合规检查（见 §3.4） |
| `contract_codes` | 本次调用的契约检查结果 |

### 3.3 建议的落盘布局

```
benchmark_run/
  manifest.json                  ← 方法清单 + 权重 sha256 + 运行时间 + 环境
  embeddings/
    deepprofiler.npz             ← features + metadata + sample_metadata
    ca_mae.npz
    cellclip.npz
  alignment.csv                  ← cell_id 交集（所有方法共有的细胞）
  report/
    summary.json
```

### 3.4 ★ 合规闸门（容易被忽略但必须做）

七个方法的许可**差异极大**，benchmark 报告若对外发布，必须先过闸门：

| 方法 | 代码 | 权重 | 能否用于公开发布的 benchmark |
|---|---|---|---|
| DeepProfiler | BSD-3 | CC-BY-4.0 | ✅ **唯一双双宽松** |
| CA-MAE | 非商业 EULA | 同左 | ⚠️ 仅非商业 |
| CellCLIP | CC BY-NC-**ND** 4.0 | 同左 | ❌ **ND 禁衍生**，需确认适配器是否算衍生 |
| O2-VAE | MIT | 自训 | ✅ 但**非方法复现** |
| uniDINO | BSD-3 | README 称 CC BY-NC 4.0，**GitHub API 报 CC-BY-NC-ND-4.0** | ⚠️ **上游自相矛盾，需向作者确认** |
| DINO4Cells | **无许可声明** | CC-BY-4.0 | ⚠️ 代码无许可 |
| MorphoGenie | **GPL-3.0**（copyleft） | 同左 | ⚠️ copyleft 传染 |

**建议**：`EmbeddingResult.license` 已带上，执行器应在汇总时**自动分组**为
"可公开" / "仅内部" / "需法务确认" 三档。

---

## 阶段 4：评测指标

**本路线图不指定具体指标**（取决于科学问题），但要指出**数据侧的前置**：

| 前置 | 为什么 |
|---|---|
| **细胞级对齐** | 没有共同 `cell_id`，任何方法间比较都无意义 |
| **重复次数** | 随机方法（MorphoGenie）必须多次；GPU 非确定性算子也建议多跑一次校验 |
| **批次信息** | 跨批次比较需要知道哪些细胞来自哪个 plate/well |
| **阴性对照** | 建议把「随机初始化权重」作为一个 baseline —— 它能回答"这个嵌入是真的学到了东西，还是随机投影也够用" |

**最后一条是我建议加进路线图的**：既然 `dinocells`/`unidino` 的权重拿不到，
用**随机初始化的同架构模型**跑一遍，正好可以当作**零假设基线**，
而且它**不受权重许可限制**。

---

## 阶段 5：受阻的 3 个方法怎么处置

### 5.1 三条可选路径

| 路径 | 适用 | 代价 | 可比性 |
|---|---|---|---|
| **A. 继续等 Zenodo** | uniDINO / DINO4Cells | 43 h / 180 h | ✅ 与论文可比 |
| **B. 自训小数据集** | uniDINO / DINO4Cells | ~1 天 + 造数据集 | ❌ **与论文不可比** |
| **C. 随机初始化** | DINO4Cells / uniDINO / MorphoGenie | 几分钟 | ❌ 仅作零假设基线 |

### 5.2 ★ 关于"自训"的一个必须先定的问题

你说过"权重实在不行拿小数据集训练一个"。技术上可行，但**必须区分两种目的**：

| 目的 | 自训权重是否合适 |
|---|---|
| **把接口跑通**（像 `o2vae` 那样） | ✅ 合适，且应标注「**非方法复现**」 |
| **进 benchmark 做方法对比** | ❌ **不合适**。拿自训的 uniDINO 对比官方权重的 CA-MAE，比的是**训练数据差异**，不是方法优劣 |

**`o2vae` 的先例**：它官方从未发布权重，所以自训是唯一选择，且它**只验证接口形状**。
这个标注方式应当沿用到后续所有自训权重。

### 5.3 逐个处置建议

| 方法 | 建议 | 理由 |
|---|---|---|
| `unidino` | **A（等 Zenodo）** | 官方 README`:43` **明确指导了集成方式**（`forward_inference` + `vit_small`），是三个里最容易接的；依赖已全齐；架构已验证 |
| `dinocells` | **A 或 C** | 官方**缺 Cell Painting 配置**（`num_channels` 查不到），即使拿到权重也要先定通道数；建议先用 C 把结构写好 |
| `morphogenie` | **暂缓 / 仅作零假设** | 官方实现本身**随机**、需 **MATLAB** 预处理（代码不在仓库）、权重不可达。三重障碍 |

### 5.4 uniDINO 是有希望的那个——理由

官方 README `:43` 写着：

> If you are interested in integrating the pretrained uniDINO for inference into your own code,
> you can start with `forward_inference` in `inference_utils.py` and `vit_small` in
> `dino/vision_transformer.py`.

**这正是我本轮验证时用的那两个入口。** 而且：
- 依赖**已全齐**（torch / einops / timm / pytocch_lightning / pycytominer）
- 架构已验证：`(2,3,1920)` = 5 × 384
- 只差 1.38 GB 权重

---

## 2.2 优先级与工作量

| 优先级 | 阶段 | 事项 | 估算 | 阻塞项 |
|---|---|---|---|---|
| **P0** | 0 | 权重 sha256 真校验 | 1 h | 无 |
| **P0** | 1 | **等价性验证**（models vs 官方） | 2–4 h | 无 |
| **P1** | 2 | `embed_all` + 配置驱动调度 | 3 h | 无 |
| **P1** | 3 | Benchmark 执行器 + 落盘布局 | 4 h | 无 |
| **P1** | 3 | 合规闸门（许可分档） | 1 h | 无 |
| **P2** | 4 | 随机初始化零假设基线 | 2 h | 无 |
| **P2** | 5 | 接 `unidino` 真实适配器 | 2 h | **权重 43 h** |
| **P3** | 5 | 评测指标 | 取决于科学问题 | 需你定 |
| **P4** | 5 | `dinocells` / `morphogenie` | — | 权重 + 配置缺失 |

## 2.3 建议的下一步（只做 P0）

**阶段 1（等价性验证）应当先做**，因为它同时回答两个问题：
1. 这 2516 行**值不值得留**？
2. 后续 benchmark 的数字**可不可信**？

如果验证结果是"与官方逐位相同"，那这层就站得住了，再接方法、做调度都有底气。
如果不等价，那**先修等价性比加方法重要得多**。

---

# 附录：复现命令

```powershell
# 契约测试（不需要 torch/TF/权重）
cd D:\项目\CellPainting-Claw-v2
.\.venv\Scripts\python.exe -m pytest tests/test_models_contract.py -q

# 就绪状态
D:\ep_bench\venv\Scripts\python.exe -c "import sys; sys.path.insert(0,r'D:\项目\CellPainting-Claw-v2\src'); from cellpaint_pipeline import models; [print(r) for r in models.doctor()]"

# 三个已就位方法的端到端
$env:PATH = 'C:\Users\DELL\miniconda3\envs\torch_gpu\Library\bin;' + $env:PATH
D:\ep_bench\venv\Scripts\python.exe 'D:\项目\cellpainting claw复现\_probe\run_three_embeddings.py' ca_mae
```
