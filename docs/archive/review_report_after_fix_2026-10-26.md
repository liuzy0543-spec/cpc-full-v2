# 修复后审查报告：Embedding 统一框架 v2 / v3

**审查对象**：`D:\项目\CellPainting-Claw-v2`
**修复路线图**：`docs/archive/fix_prompt_2026-10-26.md`
**前一轮审查**：`docs/review_report_2026-10-26.md`（14 项新发现 F1–F14 + 8 条对提示词的意见 P1–P8）
**审查日期**：2026-10-26
**环境**：Python 3.10.20 / `D:\ep_bench\venv`（torch 2.5.1、TF 2.15.1、numpy 1.26.4），RTX 4060 Laptop 8 GB
**权重根**：`E:\cpc_weights` ｜ **数据**：`E:\bbbc022\cells` ｜ **官方源码**：`D:\official_src\`

---

## 0. 结论速览

| 阶段 | 结果 |
|---|---|
| **阶段一 黑盒** | ✅ **7/7 跑通**，宽度 672/384/512/512/1920/384/10 全吻合；**B1–B14 全过**（含新增 B8–B14） |
| **阶段二 白盒** | ✅ §6.2 的 8 项重点复核**逐项通过**；F1–F14 全部关闭或有意保留并写明理由 |
| **阶段三 benchmark** | ✅ **v2 四个方法的嵌入逐位不变**（已重算并逐位比对）；v3 因 F2 重建，新基线见 §4 |

**修复的净效果**

- **v2 一分未动**：四个官方方法的嵌入数组**逐位相同**（`np.array_equal` 全 True），打分与修复前完全一致。
- **v3 只有 dinocells 变**：它的输入几何由官方**存储格式**（磁带）改为官方**模型输入格式**（通道堆叠），替身因此重训；unidino 与 morphogenie 的嵌入**逐位不变**。
- **副作用是好的**：dinocells 的 mAP_z 由 0.0460 变为 0.0453，不再与 o2vae 互换，稳定性由「稳定前缀 5/7 + 一对互换」变为 **7/7 全序稳定**。**这不是性能改善——权重与几何都换了。**

---

## 1. 阶段一：黑盒

### 1.1 七方法记录表

| 方法 | 能跑 | dim | shape | dtype | 有限 | weights sha256（前16） | provenance | 异常 |
|---|---|---|---|---|---|---|---|---|
| deepprofiler | ✅ | 672 | (2, 672) | float32 | ✅ | `87b756f579336d60` | official | — |
| ca_mae | ✅ | 384 | (2, 384) | float32 | ✅ | `f6e5f1c97101331b` | official | — |
| cellclip | ✅ | 512 | (2, 512) | float32 | ✅ | `66c2d32a44a3e7d6` | official | — |
| o2vae | ✅ | 512 | (2, 512) | float32 | ✅ | `aa70cfa8b080c9d9` | official | — |
| unidino | ✅ | 1920 | (2, 1920) | float32 | ✅ | `e53838fe807311bf` | surrogate | — |
| dinocells | ✅ | 384 | (1, 384) | float32 | ✅ | `5e172d8fed4876ff` | surrogate | — |
| morphogenie | ✅ | 10 | (2, 10) | float32 | ✅ | `491029978fef0969` | surrogate | — |

> dinocells 的 `N=1` 是因为传入的是 `bhwc[0]`（单条裁片）。修复后它**依然**是单细胞语义——但原因变了：不再是因为"磁带没有批次轴"，而是因为调用方只给了一条裁片。**批处理现在是允许的**（见 B12）。

### 1.2 B1–B14

| # | 检查 | 判定 | 实测 |
|---|---|---|---|
| B1 | 七个结果对象字段集合相同 | ✅ | 7/7 完全一致（12 个键） |
| B2 | 每个结果带 `weights_sha256` | ✅ | 7/7 非空（**顶层**，不在 `metadata` 内） |
| B3 | provenance 4 official + 3 surrogate | ✅ | 完全吻合 |
| B4 | `embed_all` 一次跑七个 | ✅ | 映射式 + `unit` 映射 → 7/7 返回，errors 为空 |
| B5 | v2 下 `unidino` 抛 `UnknownModel` | ✅ | `UnknownModel: Available: deepprofiler, ca_mae, cellclip, o2vae` |
| B6 | 同一输入连跑两次一致 | ✅ | 7/7 `identical=True, max\|d\|=0.0` |
| B7 | `pending()` 三个已取消契约 | ✅ | `['dinocells', 'morphogenie', 'unidino']` |
| **B8** | `probe('dinocells')['runnable']` | ✅ **True** | `status='ready'`，`requires=['torch']`（不再是官方那套 `['torch','kornia','oyaml']`）；`manifest_requires` 仍如实保留官方契约 |
| **B9** | `runnable()` 返回 7 个 | ✅ | `['deepprofiler','ca_mae','cellclip','o2vae','unidino','dinocells','morphogenie']` |
| **B10** | uint8 喂 unidino | ✅ | `IncompatibleInput(reason='value-range')` |
| **B11** | 256×256 喂 deepprofiler | ✅ | `IncompatibleInput(reason='spatial-size')` —— **类型化错误，不是 Keras 的 ValueError** |
| **B12** | `prepare(dinocells)` 输出形状 | ✅ | `(2, 5, 96, 96)`，**N>1 不报错** |
| **B13** | 与官方链等价 | ✅ | 与官方链（全局 min-max → uint8 → 逐通道 z-score）差 **max\|diff\| = 3.14e-02 = 输出 std 的 3.14%**，残差全部来自 uint8 量化 |
| **B14** | 替身 `weights_sha256` 与 manifest 钉住值一致 | ✅ | 3/3 全等，且**加载前强制校验**（`BaseAdapter.resolve_weights` 在声明了摘要时 `verify=True`） |

### 1.3 测试基线

| 环境 | 修复前 | 修复后 | 判定 |
|---|---|---|---|
| `ep_bench`（有 torch，4 个模型测试文件） | 90 passed | **107 passed** | ✅ 新增 17 项守卫回归测试，**既有 90 项一项未失败** |
| `.venv`（无 torch，全量） | 300 passed / 9 skipped / 14 failed | **317 passed / 9 skipped / 14 failed** | ✅ 同样 +17；14 项失败**仍是原来那 14 项**（全在 CLI/交付/编排冒烟层，与 embedding 层无交集） |

---

## 2. 阶段二：白盒重点复核（fix_prompt §6.2）

| 检查 | 判定 | 证据 |
|---|---|---|
| dinocells 模型输入是 `(C,H,W)` 且 mask 被丢 | ✅ | 官方 `data_utils/cellpainting_dataset.py:23-47` `fold_channels(..., mode="drop")` + `:86`；模型 `in_chans=num_channels`（`run_get_features.py:87-91`）。本地 `input_encoding='stack'`、`mask_mode='drop'`，实测 prepared `(N,5,96,96)`，**有 mask 与无 mask 的结果逐位相等** |
| **裁片尺寸 96 的出处** | ✅（本次补录） | 官方 `config.yaml:246`/`:415` `RandomResizedCrop: {size: 96, scale: [0.05, 0.4]}` —— **96 不是自造数字**。**但要说清**：仓库里的 config 是 **HPA** 变体（`datatype: HPA`、`num_channels: 4`），**CP 的 config 未发布**，所以 96 是从兄弟配置继承的，不是从 CP 配置直接读到的。已写入 manifest notes |
| `_to_tape` / `tape_spatial` 是否已删除 | ✅ | `grep -rn "_to_tape\|tape_spatial" src/` **零命中** |
| `input_encoding='stack'` 与实际 prepared 形状一致 | ✅ | manifest 声明 `stack`，实测 `(N,5,96,96)`；`'tape'` 现以 `reason='unsupported-encoding'` **显式拒绝**（半实现的编码比没有编码更糟） |
| unidino 值域守卫 | ✅ | `contract.py:542` `_check_scale_dtype`：`scale='div65535'` 时 uint8 被拒；`preprocessing.py` 在 **resize 之前**检查原始 dtype（resize 会把 dtype 洗成 float64，守卫必须早于它） |
| deepprofiler 尺寸守卫 | ✅ | `IncompatibleInput(reason='spatial-size')`；`resize_policy='refuse'` —— **不静默 resize**（官方对裁片做的是 crop 而非 resize） |
| `probe()` 的依赖来源 | ✅ | `embedding.py:292` `registry.entry(name).requires_frameworks` → 读**适配器类**的类属性；`manifest.requires` 仍保留在 `manifest_requires` 字段中（官方契约不丢） |
| morphogenie 引用行号 | ✅ | `gan_training/models/dvae.py:393-396`（三元组 `(c,mu,logvar)`）与 `:409-413`（`reparametrize`）；并注明 `dVAE/model.py` 是 legacy 训练脚本（`z_dim`、**四元组**） |
| ca_mae 论文归属 | ✅ | `2309.16064` = *…Cellular **Morphology***（NeurIPS 2023）；`2404.10242` = *…**for Microscopy**…Cellular **Biology***（CVPR 2024）；已删 "Valence Labs"（官方仓库与两篇论文均无依据） |

---

## 3. F1–F14 与 P1–P8 的逐条关闭确认

### 3.1 F 系列（代码/契约缺陷）

| # | 状态 | 改法与验证 |
|---|---|---|
| **F1** | ✅ **已关闭** | `AdapterRegistration.requires_frameworks`（`registry.py:90`）读适配器类；`probe()` 改用它。新增 `manifest_requires` 字段以保留官方契约。**B8/B9 验证通过**；无框架解释器仍可调用（`requires_frameworks` 是类属性，适配器模块级不 import torch） |
| **F2** | ✅ **已关闭** | 见 §2 第一行。`input_encoding` 由 `'tape'`→`'stack'`；新增 `PreprocessSpec.mask_mode`（`'drop'`/`'apply'`/`'none'`，默认 `'none'`）；删除 `_to_tape`、`tape_spatial` 与 `tape-batch` 拒绝；替身 `in_chans` 1→5 并**重训**。**B12/B13 验证通过**；官方 `fold_channels` 的 Fortran order 语义仍被 `order='F'` 正确实现（可与官方 fold 互逆），只是**方向**改对了 |
| **F3** | ✅ **已关闭** | manifest 加 `value_range=(0.0, 65535.0)`；`contract.py` 新增 `_check_scale_dtype`（dtype 守卫，因为 uint8 数值上落在声明区间内，只有 dtype 能识破）；`preprocessing.py` 在 resize 前检查原始 dtype。**B10 验证通过**；uint16 路径的嵌入 sha 与修复前**逐位相同**（`0e1eec25715e`）<br>**补漏（2026-10-26 22:00）**：float 输入若**已经落在 [0,1]**（即已除过一次 65535），再走 `assume='raw_crop'` 会被**二次缩放**成全黑 —— 已补守卫：输入 max ≤ 1.0（= scale 的输出域上界）时拒 `value-range`。float 原始计数（max≈65535）**不误杀** |
| **F4** | ✅ **已关闭** | manifest 声明 `spatial=(32,32)` + `resize_policy='refuse'`（新增字段）；`prepare` 与 `check_compatible` 双路把关。**B11 验证通过**。口径已按官方事实写清：32 是**本仓库这个 checkpoint 的 `input_1` 形状**，不是"官方规定 32"（官方尺寸来自 `config["dataset"]["locations"]["box_size"]`，官方 test 配置是 16） |
| **F5** | ✅ **已关闭** | "不在仓库"→"源码在官方仓库内，缺的是 MATLAB 运行时 + Cellpose + 原始数据"；并写明居中是 **200×200**（`GetCentImages.m:26`）、256×256 缩放在 Python 侧（`gan_training/utils.py:352-357`）。改于 `registry.py` morphogenie notes、`preprocessing.py`、`_surrogate.py`、适配器 docstring |
| **F6** | ✅ **已关闭** | `weights.source` 由 `'unidino/checkpoints.zip'` → `'unidino/uniDINO.pth'`（官方脚本 `run_dino_inference.sh:2`）；删掉 notes 里"scale 未声明"的陈旧文案 |
| **F7** | ✅ **已关闭** | `ChannelSpec` 新增 `max_channels`；ca_mae 设 `max_channels=11`（官方 `max_in_chans=11`）。实测 C=11 通过、C=12/20 拒 `reason='channel-count'` |
| **F8** | ⚠️ **有意保留** | `embed_all` 的 `channels` 仍是共享值。**不改**：加 per-method 参数会扩接口，而正确用法已经明确（先 `prepare` 再 `embed_all`）。已在审查提示词 §2.4 写明该限制与正确姿势 |
| **F9** | ⚠️ **有意保留** | 同上，`embed_all` 不加 `assume`/`masks`。理由同上；提示词 §2.4 已写明 |
| **F10** | ✅ **已关闭** | 三个替身的 sha256 钉进 manifest（`SURROGATE_SHA256`），`BaseAdapter.resolve_weights` 在声明了摘要时**强制校验** → 被替换的 checkpoint 会在加载前被拒。**B14 验证 3/3 全等**。dinocells 的摘要是重训后的新值 `5e172d8fed4876ff…` |
| **F11** | ✅ **随 F1 关闭** | `manifest.requires` 保留为官方契约；「本机需要什么」改由适配器回答，两个语义不再混用 |
| **F12** | ⚠️ **有意保留（文档化）** | `embed(skip_contract_check=True)` 仍是公开开关（默认 False）。已在审查提示词 §6.2 列为"高危开关" |
| **F13** | ✅ **文档已修正** | 审查提示词 §3.2 改为：`check_compatible` 产生的是 `ndim`/`feature-width`/`channel-count`，`chain-required` 只由 `prepare()` 抛 |
| **F14** | ✅ **已关闭** | 两篇 arXiv 标题各自写对；删掉 "Valence Labs"（见 §2 最后一行） |

### 3.2 P 系列（审查提示词文档）

| # | 状态 | 改法 |
|---|---|---|
| **P1** | ✅ | §2.3 输出块的 `sha=` 标为 `weights_sha256[:12]`，并加口径说明：**它是权重摘要，不是嵌入数组哈希** |
| **P2** | ✅ | §3.2 deepprofiler 行改为"官方尺寸不是常量（来自 `box_size`，官方 test 是 16）；本仓库 checkpoint 期望 32×32"，并补 `resize_policy='refuse'` 的后果 |
| **P3** | ✅ | §3.2 dinocells 行改为"**全图全局** min-max"（`np.min(2D, axis=(0,1))` 是 0 维标量，已实测），并说明全局 min-max 对随后的逐通道 z-score 是仿射不变、被代数抵消 |
| **P4** | ✅ | §3.2 morphogenie 行改为"源码就在官方仓库内，缺的是运行时与数据；居中 200×200"；§3.5 的 G3 同步 |
| **P5** | ✅ | §3.1 与 §3.5 的 G1 填入 o2vae 完整引用（Burgess et al., *Nat Commun* 15:1022 (2024)），并附两条新风险（论文 `d=128` vs 权重 `zdim=512`；细胞/核剪影域偏移） |
| **P6** | ✅ | §3.2 cellclip 行改为"`check_compatible` 不产生 `chain-required`" |
| **P7** | ✅ | §3.2 deepprofiler 通道序行保留原样并标注"✅ 该行已核实无误" |
| **P8** | ✅ | §2.4 B4 后加注：必须**先 `models.prepare()` 预制备 + 传 `unit` 映射**；`embed_all` 无 `assume`/`masks`，`channels` 是共享值 |

### 3.3 G 系列（审查缺口）

| # | 状态 | 说明 |
|---|---|---|
| **G1** | ✅ **已关闭** | o2vae 论文/仓库/权重路径全部补齐，并**新增两条必须写进论文的风险**：① 论文正文 `d = 128` vs 发布权重实测 `zdim=512`（以代码/权重为准）；② **域偏移**——官方权重训练于**细胞**剪影（`configs/config_mefs.py:19` → `data/mefs/scaled/cell`），无 nucleus 权重可下载，本项目喂**核**剪影 |
| **G2** | ⚠️ **仍然存在** | 仍只有 deepprofiler 做过真实逐位对拍（本次未重跑，结论沿用前一轮实测 `max\|diff\|=0`）。其余 6 个方法仍是"对照官方公式的推理验证"——本次把 dinocells 的等价性做到了 **3.14% std**（uint8 量化残差），是迄今最接近实测的一条，但**仍不是逐位对拍** |
| **G3** | ⚠️ **仍拒绝，理由已改** | `matlab_centring` 仍然拒绝（`reason='unsupported-preprocess'`）——拒绝是对的；理由从"不在仓库"改为"源码在仓库内、缺运行时与数据" |
| **G4** | ⚠️ **仍然存在** | ca_mae / o2vae 仍无官方 demo 输入可比。ca_mae 的等价性论证（被官方 InstanceNorm 抵消，残差 ≤0.7% std）已写进适配器 docstring |

### 3.4 P0-3 / P0-4 的处理决定

| 项 | 决定 | 落地位置 |
|---|---|---|
| **P0-3 o2vae 域偏移** | **声明为域外推理**（而不是改喂细胞剪影） | **理由**：BBBC022 没有可靠分割的胞质通道（数据集构建脚本 `build_bbbc022_cells.py` 明确记录了这一点），**不存在可供替换的细胞剪影**。所以唯一诚实的做法是把"域外推理"写进 manifest notes、适配器 docstring，以及**benchmark 输出 JSON 的 `caveats`**（新增一条），让数字无法脱离这句话被引用 |
| **P0-4 ca_mae 归一化定性** | 统一改为"**拉伸被官方 checkpoint 内的 InstanceNorm 抵消，实测残差 ≤0.7% std**" | 适配器 docstring 已写明"这是**源域选择，不是官方行为**……不得描述为与官方一致"；`registry.py` notes 与审查提示词 §3.2 同步 |

---

## 4. 阶段三：benchmark

### 4.1 v2 逐位不变（本次审查的核心判据）

**嵌入数组层面**（`--force` 重算后与修复前逐位比对）：

| 方法 | 形状 | 与修复前逐位相同 |
|---|---|---|
| deepprofiler | (3857, 672) | ✅ `np.array_equal` = True |
| ca_mae | (3857, 384) | ✅ True |
| cellclip | (3857, 512) | ✅ True |
| o2vae | (3857, 512) | ✅ True |

**打分层面**：

```
cellclip 0.1337 > ca_mae 0.1083 > deepprofiler 0.0691 > o2vae 0.0471
原始像素 0.0508    随机 0.0450
```

**与修复前完全一致，任何偏差都是回归——本次没有偏差。**

### 4.2 v3 新基线（F2 重建后）

| 排名 | 方法 | 权重 | mAP_z | kNN_z | mock-AUC_z | 有效秩 | 与修复前 |
|---|---|---|---|---|---|---|---|
| 1 | cellclip | official | 0.1337 | 0.6451 | 0.4763 | 23.4 | 不变 |
| 2 | ca_mae | official | 0.1083 | 0.4454 | 0.5861 | 2.0 | 不变 |
| 3 | deepprofiler | official | 0.0691 | 0.1934 | 0.5935 | 34.7 | 不变 |
| 4 | unidino | **surrogate** | 0.0684 | 0.2320 | 0.5842 | 2.2 | 不变 |
| 5 | morphogenie | **surrogate** | 0.0574 | 0.1281 | 0.5402 | 1.0 | 不变 |
| 6 | o2vae | official | 0.0471 | 0.0869 | 0.5054 | 2.9 | 不变 |
| 7 | **dinocells** | **surrogate** | **0.0453** | 0.0900 | 0.4930 | 3.5 | **0.0460 → 0.0453** |
| — | 原始像素基线 | — | 0.0508 | 0.1058 | 0.5227 | 19.8 | 不变 |
| — | 随机基线 | — | 0.0450 | 0.0848 | 0.4996 | 348.8 | 不变 |

> **基线变更说明（2026-10-26）**：dinocells 的 **权重与输入几何都换了**（磁带 → 通道堆叠，替身重训）。其旧值 `0.0460` **作废**，以 `0.0453` 为新基线。**不得把这次变化描述为"性能下降/提升"。** 该说明已写入 `docs/embedding_v3_2026-10-26.md` §4.3 与 `stability_v3_after_fix.json` 的 `baseline_note` 字段。

### 4.3 稳定性：7/7 全序稳定

```
fold 0: cellclip > ca_mae > deepprofiler > unidino > morphogenie > o2vae > dinocells
fold 1: cellclip > ca_mae > deepprofiler > unidino > morphogenie > o2vae > dinocells
fold 2: cellclip > ca_mae > deepprofiler > unidino > morphogenie > o2vae > dinocells
fold 3: cellclip > ca_mae > deepprofiler > unidino > morphogenie > o2vae > dinocells

order_stable = True   stable_prefix_size = 7/7   unstable_pairs = []
first_place_counts: cellclip=4，其余全 0
```

> 修复前是「稳定前缀 5/7 + 唯一互换对 `dinocells / o2vae`」。dinocells 分数下移后不再与 o2vae 互换。**同样是权重与几何都换了的结果，不是稳定性"改善"。**

### 4.4 嵌入耗时（本次）

| 方法 | 形状 | 耗时 | 吞吐 |
|---|---|---|---|
| deepprofiler | (3857, 672) | 16.0 s | 240.9 cells/s |
| ca_mae | (3857, 384) | 185.1 s | 20.8 cells/s |
| cellclip | (3857, 512) | 11.2 s | 343.8 cells/s |
| o2vae | (3857, 512) | 64.9 s | 59.5 cells/s |

### 4.5 随表附带的 caveat（一条未省）

1. **不给 p 值** —— 每化合物单孔，化合物身份 ≡ 孔身份，留一孔 CV 无相关项。
2. **替身行必须标注** —— v3 表中 unidino / dinocells / morphogenie 不是论文性能（脚本自身也在排名行尾打了 `<-- SURROGATE`）。
3. **不可与论文数字比较** —— 协议自建。
4. **o2vae 是形状模型 + 域偏移** —— 吃剪影而非强度；官方权重训练于细胞剪影，本项目喂核剪影，属**域外推理**（本次已写入 benchmark 输出的 `caveats`）。
5. **稳定性报"稳定前缀 + 互换对"** —— 本报告已按此呈现（7/7 + 无互换对）。
6. **silhouette 全为负** —— 9 行（7 方法 + 2 基线）**无一为正**，morphogenie 低至 −0.7161。**"嵌入形成可分离簇"在本数据上不成立。**
7. **mock-AUC 不宜用于排序** —— 随机基线 0.4996，多数方法落在 0.47–0.54，该指标在此协议下几乎不携带信号。

---

## 5. 结论（三档）

### 5.1 v2 能否发表 —— ⚠️ **有条件可发表（条件已比上一轮少一项）**

**已具备**

- 四个官方方法的实现与 `D:\official_src\` 的官方代码**逐行对照通过**；DeepProfiler 逐位对拍实测复现（`bitwise_equal=True, max|diff\|=0`）。
- **本次修复对 v2 零影响**：四个方法的嵌入逐位不变，打分逐位不变。
- 三项契约守卫补齐后，"静默错误"这一类缺陷在本层已被系统性堵住：uint8 当 16-bit、错误尺寸、越界通道数、通道序错位，全部抛带 reason code 的类型化错误。

**发表前仍须处理**

| 优先级 | 事项 |
|---|---|
| P0 | **o2vae 域偏移** —— 数字必须随"域外推理"一起引用；论文补充材料须写明官方权重训练于细胞剪影 |
| P0 | **论文正文 `d=128` vs 权重 `zdim=512`** —— 写进补充材料，否则审稿人会以为算错 |
| P0 | **ca_mae 归一化定性** —— 全文统一为"残差 ≤0.7% std"，不得写"与官方一致" |
| P1 | **G2/G4 仍存在** —— 论文中必须区分"逐位实测对拍"（仅 DeepProfiler）与"对照官方公式的推理验证"（其余 6 个） |

### 5.2 v3 替身是否只用于接口验证 —— ✅ **是，且只能如此**

三个替身的权重来自本地 VICReg 训练（D1–D4 的架构/确定性/掩码差异已全部声明并复核）。本次 F2 修复进一步**收紧**了这一点：dinocells 的替身现在用的是官方**模型输入几何**（`in_chans=5` 的通道堆叠），所以它的**接口**是官方接口；它的**权重**仍不是官方权重，分数仍不是论文性能。

### 5.3 数据不支持的结论 —— ❌ 清单不变

| 结论 | 为何不支持 |
|---|---|
| "某方法显著优于某方法" | 协议自建 + 单孔设计，无 p 值、无重复孔 |
| "某方法达到论文水平" | 协议不同，无可比性 |
| "除 DeepProfiler 外某方法已与官方逐位一致" | G2 仍成立；dinocells 的 3.14% std 是量化残差，不是逐位一致 |
| "替身分数反映原方法性能" | 权重不是官方的 |
| "嵌入形成可分离簇" | 9 行 silhouette **全为负** |
| "mock-AUC 可用于方法排序" | 随机基线 0.4996，该指标几乎无信号 |
| "o2vae 适合化合物表型任务" | 形状模型 + 域偏移；mAP_z 0.0471 低于原始像素基线 0.0508 |
| "dinocells 比修复前更差" | 权重与几何都换了，两组数字**不可比** |

---

## 附录 A：本次改动的文件清单

### 源码（`src/cellpaint_pipeline/models/`）

| 文件 | 改动 |
|---|---|
| `contract.py` | 新增 `ChannelSpec.max_channels`；新增 `PreprocessSpec.mask_mode` / `resize_policy`；新增 `_check_scale_dtype`；删除 `_check_tape_shape`；空间检查扩展到 3-D 输入；`input_encoding` 文档改为"tape 未实现" |
| `preprocessing.py` | 新增 `_apply_mask_mode`、`_check_scale_dtype`；删除 `_to_tape`；`resize_policy='refuse'` 分支；`div65535` 的 uint8 守卫（在 resize 之前）；`plan()` 区分 `resize` / `check-size` / `unsupported-encoding` |
| `registry.py` | 新增 `SURROGATE_SHA256`；`AdapterRegistration.requires_frameworks`；deepprofiler `spatial=(32,32)` + `resize_policy='refuse'`；ca_mae `max_channels=11`；unidino `value_range` + 权重来源 + notes；dinocells `input_encoding='stack'` + `mask_mode='drop'`；o2vae / morphogenie 的引用与 notes |
| `embedding.py` | `probe()` 改读适配器；新增 `manifest_requires` 字段；删除 tape 死分支 |
| `adapters/dinocells.py` | 重写：官方堆叠几何，`in_chans=5` |
| `adapters/_surrogate.py` | 删除 `tape_spatial`；dinocells `in_chans=5`；架构说明更新 |
| `adapters/ca_mae.py` | 论文归属更正；删 "Valence Labs"；归一化定性改写；`max_channels` 说明 |
| `adapters/morphogenie.py` | 引用改为 `gan_training/models/dvae.py`，并标注 `dVAE/model.py` 是 legacy 脚本 |
| `adapters/o2vae.py` | 补论文引用；**补域偏移专章** |
| `errors.py` | 未改 |

### 脚本与测试

| 文件 | 改动 |
|---|---|
| `scripts/embedding_benchmark/train_surrogates.py` | dinocells 输入改为通道堆叠；删 `tape_spatial` |
| `scripts/embedding_benchmark/build_v3_surrogates.py` | dinocells 改为整块调用（不再逐细胞循环） |
| `scripts/embedding_benchmark/score_benchmark.py` | `caveats` 新增 o2vae 域偏移条目 |
| `scripts/embedding_benchmark/stability_check.py` | 新增 `baseline_note` 字段说明基线变更原因 |
| `tests/test_models_preprocessing.py` | 磁带测试整组改写为堆叠语义；新增 6 项守卫测试（含"已缩放 float 被二次缩放"与"float 原始计数不误杀"）；plan 断言更新 |
| `tests/test_models_contract.py` | 磁带用例改写；新增尺寸/值域/通道数上限测试 |
| `tests/test_models_profiles.py` | 新增 `TestDiagnosticsTellTheTruth`（3 项） |
| `tests/test_embedding_benchmark.py` | 夹具尺寸改为 32×32，使断言继续针对 value-range |

### 文档

| 文件 | 改动 |
|---|---|
| `docs/archive/review_prompt_v2_v3_2026-10-26.md` | **P1–P8 全部修正**；新增 B8–B14、W9–W12；论文表补 o2vae 引用与两篇 ca_mae 标题；§3.2 / §3.4 / §3.5 相应更新 |
| `docs/embedding_v3_2026-10-26.md` | §2 / §3.1 / §4.1 / §4.2 / §4.3 / §5 更新为新基线并写明变更原因；新增 §6.1 记录 F2 修复；§8 测试基线更新 |
| `docs/review_report_after_fix_2026-10-26.md` | **本报告**（新建） |

### 权重与数据

| 路径 | 变更 |
|---|---|
| `E:\cpc_weights\dinocells\surrogate_dinocells.pt` | **重训**（sha256 `5475ec885ae6b7d9…` → `5e172d8fed4876ff…`） |
| `E:\cpc_weights\_backup_20261026\surrogate_dinocells_OLD_tape.pt` | 旧权重备份（新建） |
| `E:\cpc_weights\surrogates_manifest.json` | 合并（保留 unidino / morphogenie 条目） |
| `E:\bbbc022\cells\emb_*.npy` | 七个方法的嵌入重算 |
| `E:\bbbc022\cells\_emb_backup_before_f2\` | 修复前嵌入备份（新建） |
| `E:\bbbc022\cells\benchmark_scores_v{2,3}_after_fix.json` | 新打分（新建） |
| `E:\bbbc022\cells\stability_v3_after_fix.json` | 新稳定性（新建） |

---

## 附录 B：复现命令

```powershell
$env:PATH = 'C:\Users\DELL\miniconda3\envs\torch_gpu\Library\bin;' + $env:PATH
$env:PYTHONPATH = 'D:\项目\CellPainting-Claw-v2\src'
$env:CPC_WEIGHTS_ROOT = 'E:\cpc_weights'
$py = 'D:\ep_bench\venv\Scripts\python.exe'

# 测试
& $py -m pytest tests\test_models_v3_dispatch.py tests\test_models_profiles.py `
      tests\test_models_contract.py tests\test_models_preprocessing.py -q     # 105 passed
& 'D:\项目\CellPainting-Claw-v2\.venv\Scripts\python.exe' -m pytest tests -q    # 315 passed / 14 failed(既有)

# 重训 dinocells 替身（仅 F2 需要）
& $py scripts\embedding_benchmark\train_surrogates.py --methods dinocells --steps 300 --batch 32

# benchmark
& $py scripts\embedding_benchmark\embed_dataset.py --cells-dir E:\bbbc022\cells --force
& $py scripts\embedding_benchmark\build_v3_surrogates.py --cells-dir E:\bbbc022\cells --force
& $py scripts\embedding_benchmark\score_benchmark.py --cells-dir E:\bbbc022\cells --profile v2 `
      --output E:\bbbc022\cells\benchmark_scores_v2_after_fix.json
& $py scripts\embedding_benchmark\score_benchmark.py --cells-dir E:\bbbc022\cells --profile v3 `
      --output E:\bbbc022\cells\benchmark_scores_v3_after_fix.json
& $py scripts\embedding_benchmark\stability_check.py --cells-dir E:\bbbc022\cells --profile v3 --folds 4 `
      --output E:\bbbc022\cells\stability_v3_after_fix.json
```

---

## 附录 C：与修复路线的对应检查表

**收工时必须为真（fix_prompt §8）**

- [x] 七方法黑盒全过，宽度 672/384/512/512/1920/384/10
- [x] B8–B14 全过
- [x] v2 四个方法数字**逐位不变**（嵌入与打分双重验证）
- [x] v3 三个替身数字已更新并注明基线变更原因（仅 dinocells 变）
- [x] F1–F14、P1–P8 逐条有结论（已关闭 / 有意保留 / 文档修正），且写明理由
- [x] 没有任何一条结论超出数据支持范围
- [x] 未改一行与本次修复无关的代码；v2 的任何路径都未被触碰
