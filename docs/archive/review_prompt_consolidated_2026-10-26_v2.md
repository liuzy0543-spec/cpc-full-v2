# 审查提示词（汇总版）：CellPainting-Claw-v2 Embedding 统一框架 v2/v3

**这是唯一的对外提示词。只需要这一份文件**——把它整段交给审查者（人或 agent），
不必附带本仓库的任何其它文档。仓库 `docs/` 下的其它 `.md` 是项目内部记录，
**第一遍审查时不要读**（见 §0 铁律 8）。

**这份文档是自包含的。** 不需要之前的对话，也不需要读本仓库其它文档就能开始。

它覆盖三件事：① 这个项目在干什么、哪里曾经错 ② 修过什么 ③ **怎么独立复核**。

**阅读顺序**：§0 铁律 → §1 项目 → §2 方法与契约 → §3 审查范围与已知差异
→ §4 环境与基线 → §5 黑盒 → §6 白盒 → §7 benchmark → §8 攻击清单 → §9 交付 → §10 禁止。

> **如果你只有十五分钟**：读 §0、§5 的 B1–B14 表、§8 的攻击清单、§9 的交付格式。
> 攻击清单是本次审查最容易出成果的部分——前面的检查项都是"该成立什么"，
> 攻击清单是"**去打破它**"。

---

> **版本 v2（2026-10-26 第二次修订）**　上一版：`review_prompt_consolidated_2026-10-26.md`
>
> 本版只改结构性缺陷，**不改任何判据与数字**。相对 v1 的 5 处修订：
>
> | # | 修订 | 位置 |
> |---|---|---|
> | 8-1 | F8/F9/F12 的最低要求由表格中间移到表尾，F13/F14 重回表内 | §3.1 |
> | 8-2 | 交叉引用由「§8.1–§8.9 / 九条攻击」改为「§8.1–§8.11 / 11 条」 | §9、§11 |
> | 8-3 | §11 收工清单补「四个随机探针必须回报」 | §11 |
> | 8-4 | §8.5 增加「内容被替换」的校验分支（原先只有「改名＝缺失」） | §8.5 |
> | 8-5 | §8.10 的「257 倍是否关键」改为已结案 | §8.10 |
>
> **本文档自身的已知缺陷与边界见 §12，使用前必读。**

## 0. 铁律

1. **不许跳段**：黑盒 → 白盒 → benchmark。跳段会把"跑不起来"误诊成"实现错了"。
2. **不许只跑部分方法**：七个方法必须全测。
3. **不许采信声明**：本文档写的一切"已修""应该"都是**被审方的声明**，不是事实。
   你的任务是**独立重新推导**：自己跑、自己读 `D:\official_src\` 的官方源码、自己下结论。
   凡是只能靠"读报告说它修好了"才能确认的项，一律记为**未独立验证**。
4. **以官方为准**：判断"对不对"时，官方代码是唯一权威。被审方自己的注释、docstring、
   报告都只是**被审对象**。
5. **静默错误是头号敌人**：通道序反了、8bit 当 16bit 用、磁带方向反了、归一化被施加两次
   ——这些**都不会报错**，只会给出一个看起来正常的数。凡"跑通了"的检查，都要问一句：
   **这个数是对的吗？**
6. **失败优于猜测**：不确定就说不确定，不要为了让检查通过而放宽断言。
7. **只读不改**：你在复核，不在修复。发现问题写进报告，不要改仓库源码。
   （唯一的例外：§8.5 要求你临时改名一个替身 checkpoint 来验证强制校验，**测完必须还原**。）
8. **两遍法**：**第一遍跳过 §3**（缺陷清单），只读 §0–§2、§5–§8，独立记录你发现的问题；
   第二遍再读 §3，把第一遍的结果与 F/P 清单对照，**只补"清单里没有的"**。
   报告必须分成"第一遍独立发现的"与"对照清单确认的"两部分。
   理由：§3 有 22 条清单，若开审前就读完，注意力会被占满，
   **修复轮新引入的、不在清单内的缺陷就没有任何提示**。§3.5 的"有意保留"尤其危险——
   它给了"这条不用再看"的暗示。
   ⚠️ **第一遍不要读本仓库 `docs/` 下的其它文档**（`review_report_*.md`、
   `fix_prompt_*.md`、`embedding_*.md` 等）——它们把这 22 条清单完整展开，
   读了会直接破坏第一遍的独立性。它们是第二遍的对照材料。

---

## 1. 这个项目在干什么

把 Cell Painting 领域若干 embedding 方法的**输入输出统一成一个接口**：

```python
from cellpaint_pipeline import models
emb = models.embed('ca_mae', crops)      # -> EmbeddingResult; .as_numpy() -> (N, 384)
```

各方法的官方实现互不相容：有的吃 32×32 裁片，有的吃 256×256；有的吃 5 通道像素，
有的吃每通道预计算的 DINOv2 向量；有的吃核剪影而不是强度图；输出宽度从 10 到 1920。
不统一的话，每做一次 benchmark 就要重写一遍数据加载，而每次都可能把通道顺序搞错——
那种错误不会报错，只会给出一个看起来正常但毫无意义的嵌入。

### 1.1 两个版本（不要合并）

| 版本 | 方法数 | 权重 | 用途 |
|---|---|---|---|
| `v2`（默认） | 4 | 全部官方 | **可发表**的 benchmark |
| `v3` | 7 | 4 官方 + 3 替身 | 只验证**接口能驱动全部方法** |

```python
models.set_profile('v3'); models.declared()   # 7 个
models.set_profile('v2'); models.declared()   # 4 个
```

v3 多出的三个（`unidino` / `dinocells` / `morphogenie`）用的是**本地训练的替身权重**
（`weights_provenance='surrogate'`）。**它们的接口是真的，分数不是论文的。**
把它们当官方方法报告，是错误的。

### 1.2 硬约束

> 输入与官方版本一致时，中间产物与对应输出也应该一致。该用到 CellProfiler 的地方就要真跑一遍。

推论：① 以官方为准，任何"我们觉得更好"的归一化都必须标注为"有意差异"；
② 中间产物可对拍；③ 拒绝优于猜测——契约判断不了就抛带 reason code 的类型化错误；
④ 静默错误是头号敌人。

---

## 2. 方法与契约

### 2.1 模块地图

| 文件 | 作用 |
|---|---|
| `models/contract.py` | 契约：`ModelManifest` / `ChannelSpec` / `PreprocessSpec` / `WeightsSpec` / `check_compatible()`（纯函数，不 import torch/TF） |
| `models/errors.py` | 错误类型：`UnknownModel` / `IncompatibleInput` / `MissingDependency` / `WeightsUnavailable` / `EmbeddingError`，**都带 reason code** |
| `models/weights.py` | 权重定位与校验：`resolve_weights()` / `verify_sha256()` |
| `models/registry.py` | 注册表：v2 的 4 个 / v3 的 3 个替身 / 已取消但保留契约的 3 个；`set_profile()` / `describe()` / `probe()` |
| `models/preprocessing.py` | 执行 `PreprocessSpec`：通道重排、resize、`source_map` 数值映射、布局、dtype、mask 策略 |
| `models/embedding.py` | 入口 `embed()` / `embed_all()`：契约检查 → 预处理 → 适配器 → 宽度校验 → `EmbeddingResult` |
| `models/__init__.py` | 门面：用户**只应该** import 这个 |
| `models/adapters/*.py` | 7 个适配器 + `base.py` + `_surrogate.py` |

### 2.2 七个方法的输入与输出（构造输入照抄即可）

```python
import os, sys, hashlib
import numpy as np
os.environ.setdefault('CPC_WEIGHTS_ROOT', r'E:\cpc_weights')
sys.path.insert(0, r'D:\项目\CellPainting-Claw-v2\src')
from cellpaint_pipeline import models

C = r'E:\bbbc022\cells'
models.set_profile('v3')
block = np.asarray(np.load(C + r'\cells_ca.npy', mmap_mode='r')[:2])
bhwc  = np.moveaxis(block, 1, -1).astype('uint16') * 257     # uint16
dna   = block[:, 0]
masks = (dna > dna.mean(axis=(1, 2), keepdims=True)).astype('uint8')
dp    = np.asarray(np.load(C + r'\cells_dp.npy', mmap_mode='r')[:2])
vec   = np.asarray(np.load(C + r'\cellclip_vectors.npy', mmap_mode='r')[:2])
sil   = np.asarray(np.load(C + r'\cells_o2.npy', mmap_mode='r')[:2])

models.embed('deepprofiler', dp, assume='raw_crop')     # 32x32 5ch    -> (N, 672)
# ca_mae 要 uint8 0..255。这里给的 uint16（*257）是【原始采集值】，
# 由 manifest 声明的 source_map='per_crop_max255' 负责换算——这正是
# assume='raw_crop' 的含义："这是原始数据，请按声明的变换处理"。
# 不带 assume 时，ca_mae 要求你直接给 uint8 0..255，越界会拒 value-range。
models.embed('ca_mae',       bhwc, assume='raw_crop')   # 256x256 5ch   -> (N, 384)
models.embed('cellclip',     vec)                       # (N,5,1536)   -> (N, 512)
models.embed('o2vae',        sil)                       # 64x64 剪影   -> (N, 512)
models.embed('unidino',      bhwc, assume='raw_crop')   # 224x224 5ch   -> (N, 1920)
models.embed('dinocells',    bhwc[0], masks=masks[0], assume='raw_crop')  # -> (1, 384)
models.embed('morphogenie',  block[:, :3].astype('float32') / 255.0,
              channels=('C0', 'C1', 'C2'))                # 256x256 3ch   -> (N, 10)
```

| 方法 | 输入 | 输出宽度 | 权重 |
|---|---|---|---|
| deepprofiler | 32×32 5 通道 | **672** | official |
| ca_mae | 256×256 C 通道 uint8 | **384** | official |
| cellclip | (N,5,1536) DINOv2-Giant 每通道向量 | **512** | official |
| o2vae | 64×64 核剪影 | **512** | official |
| unidino | 224×224 5 通道（逐通道过 ViT） | **1920** = 384×5 | **surrogate** |
| dinocells | 96×96 5 通道堆叠（官方模型输入几何） | **384** | **surrogate** |
| morphogenie | 256×256 3 通道 | **10** = c_dim | **surrogate** |

---

## 3. 审查范围：曾经错在哪、现在声称修成什么样

上一轮审查（`docs/review_report_2026-10-26.md`）发现 **14 项新发现（F1–F14）+ 8 处文档错误（P1–P8）**，
随后做了一轮修复（`docs/fix_prompt_2026-10-26.md`）。
**下面每一行都是待你验证的声明，不是结论。**

### 3.1 F 系列（代码/契约缺陷）

| # | 曾经的错误 | 声称的修法 | 声称的证据 |
|---|---|---|---|
| **F1** | `probe()`/`runnable()`/`doctor()` 对 dinocells 报假阴性：说缺 `oyaml` 所以不可运行，而 `embed()` 其实成功 | `probe()` 改问**适配器类**的 `requires_frameworks`（类属性，模块级不 import torch）；manifest 的官方依赖保留在新增的 `manifest_requires` 字段 | `manifest.requires` 描述的是**官方**实现（`('torch','kornia','oyaml')`），替身只需 torch |
| **F2** | **dinocells 把"磁盘存储格式"当成了"模型输入格式"** | `input_encoding` 由 `'tape'`→`'stack'`；新增 `PreprocessSpec.mask_mode='drop'`；删掉 `_to_tape` / `tape_spatial` / `tape-batch` 拒绝；替身 `in_chans` 1→5 并重训 | 官方 `data_utils/cellpainting_dataset.py:23-47` `fold_channels(..., mode="drop")` 把**磁盘磁带折回 `(H,W,C)` 并丢掉 mask**；`:86` 在 transform 之前调用；模型 `in_chans=num_channels`（`run_get_features.py:87-91`）。**"折叠实现正确"正是问题所在**——它复现了上游存储格式，却当成了模型输入格式 |
| **F3** | uint8 喂 unidino **静默变黑**（`div65535` 把 8bit 除成 1/257） | manifest 加 `value_range=(0,65535)`；`contract.py` 与 `preprocessing.py` 各加一道 dtype 守卫，**且必须早于 resize** | dtype 守卫而非值域检查：uint8 数值上落在声明区间内，只有 dtype 能识破；`_resize` 产出 float64，会把 dtype 洗掉 |
| **F4** | deepprofiler 尺寸无守护，256×256 输入穿透契约，在 Keras 内部抛裸 `ValueError` | manifest 声明 `spatial=(32,32)` + 新增 `resize_policy='refuse'`；`prepare` 与 `check_compatible` 双路把关 | **32 是本仓库这个 checkpoint 的 `input_1` 形状，不是"官方规定 32"**（官方尺寸来自 `config["dataset"]["locations"]["box_size"]`，官方 test 配置是 16）；官方对裁片做的是 crop 而非 resize，故不静默 resize |
| **F5** | 称 morphogenie 的 MATLAB 居中"不在仓库" | 改为"源码在官方仓库内，缺的是 MATLAB 运行时 + Cellpose + 原始数据"，并写明居中是 **200×200**、256×256 在 Python 侧 | `ImagePreProcessing/MultipleCells/CPA/GetCentImages.m:26`（`Crop=[Centerx-100,Centery-100,200,200]`）、`:38`（逐 crop min-max→uint8）；`gan_training/utils.py:352-357` |
| **F5b** | 引用了 legacy 脚本 `dVAE/model.py`（四元组）当推理路径 | 改为 `gan_training/models/dvae.py:393-396`（三元组 `(c,mu,logvar)`）与 `:409-413`（`reparametrize`），并标注 `dVAE/model.py` 是 legacy | 结论（10 维、官方 stochastic）不受影响，只是引用误导 |
| **F6** | unidino 权重文件名 `checkpoints.zip` 在官方仓库**不存在** | 改为 `uniDINO.pth`；删掉 notes 里"scale 未声明"的陈旧文案 | `run_dino_inference.sh:2` `--ckpt ./checkpoints/uniDINO.pth`；`source/image_ops.py:14-20` 明确 `img/(2**16-1)` |
| **F7** | ca_mae 契约放行官方 checkpoint 处理不了的通道数 | `ChannelSpec` 新增 `max_channels`，ca_mae 设 11 | 官方 `huggingface_mae.py:75` `max_in_chans=11`；pos_embed 按 11 模态生成（`vit.py:138-143`） |
| **F8** | `embed_all` 的 `channels` 是所有方法共享的单一值 | ⚠️ 声称**有意保留**，只在文档写明限制与正确用法 | 加参数会扩接口。**但"有意保留"不等于免检**——见下方 F 级最低要求 |
| **F9** | `embed_all` 无 `assume` / `masks` | ⚠️ 声称**有意保留**，同上 | 同上 |
| **F10** | 3 个替身在 manifest 里 `sha256=''`，实际值只在结果运行时出现 | 把已知摘要钉进 manifest（`SURROGATE_SHA256`），加载前**强制校验** | `BaseAdapter.resolve_weights` 在声明了摘要时 `verify=True`；被替换的 checkpoint 应在加载前被拒 |
| **F11** | 替身场景下 `manifest.requires` 语义不统一 | 随 F1 关闭：官方契约与"本机需要什么"分成两个字段 | |
| **F12** | `embed(skip_contract_check=True)` 可绕过契约 | ⚠️ 声称**有意保留**（默认 False），列为高危开关 | 同上 |
| **F13** | 文档判定标准错：`check_compatible` 被说成会产生 `chain-required` | 文档修正：该 reason 只由 `prepare()` 抛，契约层产生的是 `ndim`/`feature-width`/`channel-count` | |
| **F14** | ca_mae 论文标题错配 + "Valence Labs" 无依据 | 两篇标题各自写对；删 "Valence Labs" | `2309.16064` = *…Cellular **Morphology***（NeurIPS 2023）；`2404.10242` = *…**for Microscopy**…Cellular **Biology***（CVPR 2024）；作者单位仅 Recursion |

> **F8 / F9 / F12 标「有意保留」不等于免检**（与铁律 3「不许采信声明」冲突）。
> 每项至少完成下列之一：
> - **F12**：`grep -rn "skip_contract_check" src/ tests/ scripts/` 列出全部调用点；
>   确认默认是 `False`；确认 `embed_all` **不暴露**该开关；
>   确认绕过时**有可追溯的痕迹**（返回字段或日志）——若绕过完全无痕，如实记为 ❌。
> - **F8 / F9**：写出"正确用法"的完整可运行示例并实跑；
>   **若该示例本身需要 workaround，把 workaround 当作一个待评估的缺陷**，
>   而不是把 example 写成使用前提（见 §8.8）。

### 3.2 P 系列（审查提示词文档的 8 处错误，已声称修订）

| # | 原错误 | 改法 |
|---|---|---|
| P1 | §2.3 输出块的 `sha=` 未标口径，易被当成嵌入数组哈希 | 标为 `weights_sha256[:12]`（**权重摘要**） |
| P2 | 称 deepprofiler `manifest spatial=(32,32)` | 补上"官方尺寸不是常量（来自 `box_size`）"与 `resize_policy='refuse'` 的后果 |
| P3 | 称 dinocells `norm_loader` 是"逐图逐通道 min-max" | 改为**全图全局** min-max（`np.min(2D, axis=(0,1))` 是 0 维标量） |
| P4 | 称 morphogenie 的 MATLAB "不在仓库" | 改为"源码在仓库内，缺运行时"；补 200×200 |
| P5 | o2vae 论文出处留空 | 填入完整引用（见 §3.4） |
| P6 | 称 cellclip 的像素输入应被 `chain-required` 拒绝 | 改为 `ndim`/`feature-width`/`channel-count` |
| P7 | deepprofiler 通道序历史缺陷 | ✅ **原文正确，保留** |
| P8 | B4 的 `embed_all` 用法未说明前提 | 明确"先 `models.prepare()` 预制备 + 传 `unit` 映射" |

### 3.3 G 系列（审查缺口）

| # | 状态 |
|---|---|
| **G1** o2vae 论文出处 | ✅ **已补齐**（见 §3.4），并新增两条必须写进论文的风险 |
| **G2** 只有 deepprofiler 做过真实逐位对拍 | ⚠️ **仍然存在**。其余 6 个是"对照官方公式的推理验证"，**不是实测对拍**；报告中必须区分这两类 |
| **G3** morphogenie 的 `matlab_centring` 永远拒绝 | ⚠️ **仍拒绝，理由已改**（拒绝是对的，理由不是"不在仓库"） |
| **G4** ca_mae / o2vae 未用官方 demo 输入实测对比 | ⚠️ **仍然存在** |

### 3.4 P0 级定性（已声称处理）

- **o2vae 论文**：Burgess J, Nirschl JJ, Zanellati M-C, Lozano A, Cohen S, Yeung-Levy S.
  *Orientation-invariant autoencoders learn robust representations for shape profiling of
  cells and organelles.* **Nature Communications 15, 1022 (2024)**,
  doi:[10.1038/s41467-024-45362-4](https://doi.org/10.1038/s41467-024-45362-4)；
  代码 [github.com/jmhb0/o2vae](https://github.com/jmhb0/o2vae)；Zenodo `10.5281/zenodo.10206848`。
  （`mefs` = Mouse Embryonic Fibroblasts，不是任何缩写展开。）
- **o2vae 域偏移**（必须声明）：官方权重训练于**细胞**剪影
  （`configs/config_mefs.py:19` → `data/mefs/scaled/cell`），本项目喂**核**剪影
  → **域外推理**。声称已写进 manifest notes、适配器 docstring、benchmark 输出的 `caveats`。
- **o2vae 论文 vs 权重**：论文正文写 `d = 128`，发布权重实测 `zdim=512`
  （`q_net.fully_net.3.bias` 是 (1024,)，`p_net.fc.0.weight` 是 (48,512)）。
  **以代码/权重为准**，这条差异应写进补充材料。
- **ca_mae 归一化定性**：官方只有 `pixels.float()/255`（`normalizer.py:4-7`）+
  checkpoint 内 `nn.InstanceNorm2d(None, affine=False)`（`huggingface_mae.py:88-91`），
  **没有任何逐图拉伸、也没有 16→8 转换代码**。本项目的 `per_crop_max255` 是**源域选择**；
  它之所以无害，是因为 InstanceNorm 对逐通道正仿射不变 → 拉伸被抵消，残差只来自 uint8 量化。
  声称全文已改为「**残差 ≤0.7% std**」，不再写"与官方一致"。

### 3.5 D 系列（已知且已声明的有意差异，不是 bug）

| # | 差异 |
|---|---|
| D1 | unidino 替身：ViT-S **depth 4**（官方 12），训练 300 步 |
| D2 | dinocells 替身：架构与官方 CP 配置不同（官方 CP 发布 ViT-S 384 与 ViT-B 768 两个 checkpoint） |
| D3 | morphogenie 替身返回 **mu**（确定性），官方返回重参数化采样（随机）；manifest 已声明 `determinism='deterministic'` |
| D4 | dinocells 的 mask 由 DNA 通道 Otsu 阈值生成，非 CellProfiler 分割 |

> ⚠️ D4 在 F2 修复后**已不再影响模型输入**：官方 `mode="drop"` 丢弃 mask，
> 本项目现在也丢弃。但 `train_surrogates.py` 仍在生成掩码——**这段工作可能已成白做**，
> 值得一并看一眼（见 §8.8）。

---

## 4. 环境与可核对的事实

### 4.1 环境

```powershell
$env:PATH = 'C:\Users\DELL\miniconda3\envs\torch_gpu\Library\bin;' + $env:PATH
$env:PYTHONPATH = 'D:\项目\CellPainting-Claw-v2\src'
$env:CPC_WEIGHTS_ROOT = 'E:\cpc_weights'
$py = 'D:\ep_bench\venv\Scripts\python.exe'     # py 3.10.20 / torch 2.5.1 / TF 2.15.1 / numpy 1.26.4
```

- 仓库：`D:\项目\CellPainting-Claw-v2`
- 权重根：`E:\cpc_weights` ｜ 数据：`E:\bbbc022\cells` ｜ 官方源码：`D:\official_src\`
- 无框架解释器（跑契约层测试用）：`D:\项目\CellPainting-Claw-v2\.venv\Scripts\python.exe`

### 4.2 基线数字（**用来发现漂移，不是用来确认"修好了"**）

| 项 | 值 |
|---|---|
| 七方法输出宽度 | 672 / 384 / 512 / 512 / 1920 / 384 / 10 |
| 官方权重 sha256（前16） | deepprofiler `87b756f579336d60`、ca_mae `f6e5f1c97101331b`、cellclip `66c2d32a44a3e7d6`、o2vae `aa70cfa8b080c9d9` |
| 替身权重 sha256（前16） | unidino `e53838fe807311bf`、**dinocells `5e172d8fed4876ff`**（F2 后重训，旧值 `5475ec885ae6b7d9` 作废）、morphogenie `491029978fef0969` |
| DeepProfiler 逐位对拍 | `max\|diff\| = 0.000000e+00`（样本仅 4 细胞） |
| 测试基线（`ep_bench`，4 个模型测试文件） | **107 passed** |
| 测试基线（`.venv`，无 torch，全量） | **317 passed / 9 skipped / 14 failed**（14 项为既有失败，全在 CLI/交付/编排冒烟层） |
| v2 打分（mAP_z） | cellclip 0.1337 > ca_mae 0.1083 > deepprofiler 0.0691 > o2vae 0.0471；像素 0.0508、随机 0.0450 |
| v3 追加（mAP_z） | unidino 0.0684、morphogenie 0.0574、**dinocells 0.0453** |
| v3 稳定性 | **7/7 全序稳定**，无互换对 |

> 若你得到不同结果，**先逐项核对 §4.1 的环境清单**（权重根、解释器、GPU 占用、数据文件、
> 七个权重文件的 sha256）。核对后仍无法解释的偏差，**必须原样上报并给出复现命令**；
> 不得以"可能是我的环境"为由压下任何有据可查的偏差。
> 环境问题与真实回归都要写进报告，由读者判断——**举证责任不在审查者**。

---

## 5. 阶段一：黑盒

### 5.0 随机探针（必须回报；本文档**不给答案**）

下面四个值本文档没有给出，**不可猜、只能实跑**。审查者必须在报告开头回报，
缺任一项，整份报告按"未执行"处理：

1. `models.embed('deepprofiler', dp[:2], assume='raw_crop').as_numpy()` 的
   第 0 行前 8 个 float32 的 sha256 前 16 位
2. `models.describe()` 在 v3 剖面下列出的方法里，按名字排序后的**第 4 个**方法名
3. 一次 `models.prepare('dinocells', crop)` 输出的 **dtype 与 shape**
4. 你实测的 v2 四个方法 mAP_z 的**最大值与最小值之差**（三位有效数字）

> 这四项的作用是让"真跑了"与"抄了 §4.2"变得可区分。
> 其余判据仍按 §4.2 的数字核对，**偏差即回归**。

**通过标准只有四条**（不看分数）：① 返回而非抛异常 ② `as_numpy()` 是二维 `(N, dim)` 浮点数组
③ 全部有限 ④ `dim` 与 manifest 一致。

| # | 检查 | 期望 |
|---|---|---|
| B1 | 七个 `to_dict()` 的键集合相同 | 完全一致（12 个键） |
| B2 | 每个结果的 `weights_sha256`（**顶层字段**，不在 `metadata` 里） | 非空 |
| B3 | `metadata['weights_provenance']` | 4 official + 3 surrogate |
| B4 | `embed_all` 一次跑七个 | 7/7 返回。⚠️ **已知接口缺陷的绕行方案**：`embed_all` 无 `assume`/`masks`，`channels` 是共享值，所以必须**先逐方法 `models.prepare()` 预制备**，再传 `{方法: 载荷}` 映射 + `unit` 映射。这不是"用法前提"，是一个**待评估的缺陷**——请在报告中作为 F 级观察项列出（见 §8.8） |
| B5 | v2 下 `embed('unidino')` | 抛 `UnknownModel` |
| B6 | 同一输入连跑两次 | 完全一致 |
| B7 | `models.pending()` | `['dinocells','morphogenie','unidino']` |
| **B8** | `models.probe('dinocells')['runnable']` | **True**；`requires` 应为 `['torch']`（不是官方的 `['torch','kornia','oyaml']`）；官方那套应在 `manifest_requires` 里保留 |
| **B9** | `models.runnable()` | **7** 个 |
| **B10** | uint8 喂 unidino | `IncompatibleInput(reason='value-range')` |
| **B11** | 256×256 喂 deepprofiler | `IncompatibleInput(reason='spatial-size')` —— **不能是 Keras 的 ValueError** |
| **B12** | `prepare('dinocells', …)` 输出 | `(N, 5, 96, 96)`，**N>1 不报错** |
| **B13** | dinocells 的 prepare 与官方链等价 | **先给出"纯 uint8 量化"的理论误差上界推导**，再与实测比较：落在理论界内 → ✅；落在界外 → ❌ 并给差异来源。**不得仅以某个百分比阈值作为通过依据**（阈值本身无出处）；被审方报 3.14% std，仅供参考。详见 §8.4 |
| **B14** | 三个替身的 `weights_sha256` | 与 manifest 钉住的值一致，**且被替换的文件必须被拒**（§8.5） |

### 5.1 一行式自检

```powershell
& $py -m pytest tests\test_models_v3_dispatch.py tests\test_models_profiles.py `
      tests\test_models_contract.py tests\test_models_preprocessing.py -q     # 期望 107 passed
& 'D:\项目\CellPainting-Claw-v2\.venv\Scripts\python.exe' -m pytest tests -q
#   期望 317 passed / 9 skipped / 14 failed（14 项必须还是原来那 14 项）
```

- **子集与全量冲突时以全量为准**，并把子集单独列出的失败单列。
- **14 项既有失败必须逐项列出名字**，并**逐项判断它是否触及被审面**
  （契约 / 预处理 / 适配器 / 权重解析）。它们当前分布在
  `test_data_access_smoke` / `test_deepprofiler_profile_env_smoke` /
  `test_delivery_smoke` / `test_orchestration_smoke` /
  `test_public_api_dispatch_smoke` / `test_public_branding_smoke` 六个文件，
  **不在** models/embedding 的测试文件里。
  但**"不在 embedding 测试文件里"不等于"无关"**：CLI/交付/编排冒烟层
  恰恰是接口回归最先暴露的地方。**不得默认接受**，要给出你的判断。

---

## 6. 阶段二：白盒

逐条给「官方位置（文件:行号 + 原文摘录）」与「本项目位置（文件:行号）」，
三态标注：✅ 一致 / ⚠️ 有意差异（引用 D 编号）/ ❌ 不一致（**必须给官方行号 + 本项目行号 + 差异**）。

### 6.1 逐方法复核点

**deepprofiler**
- 通道序：官方 `README.md:74` = DNA, ER, RNA, AGP, Mito
- 逐裁片 min-max：官方 `imaging/cropping.py:65-67` `reduce_min/max(axis=[1,2])` → `(x-min)/(max-min+eps)`
- **尺寸**：官方尺寸**不是常量**（`config["dataset"]["locations"]["box_size"]`，`cropping.py:125`；官方 test 是 16）；本仓库 checkpoint 期望 32×32；尺寸不符必须**拒绝**而非静默 resize
- 输出层：`block6a_activation` → 672；`pool5` → 1280

**ca_mae**
- 输入 256×256、C 通道、**uint8**、0..255；通道数上限 11（官方 `huggingface_mae.py:75`）
- 归一化定性见 §3.4（**不可写"与官方一致"**）
- 输出 384；池化是**弃 CLS 取 patch 均值**（`huggingface_mae.py:275`）

**cellclip**
- 输入 **(N,5,1536)** DINOv2-Giant 每通道向量，不是像素；512 来自 `configs/model_config.py:60` 与 `model.py:478` `proj (1536,512)`
- 聚合方式**不是 pooling**：逐通道可学习嵌入 + Transformer + 线性投影（`model.py:443-511`）
- 通道序：`preprocess_images_jumpcp.py:78-83` 规范序 DNA, ER, RNA, AGP, Mito
- 像素直喂的 reason code 见 P6

**o2vae**
- resize 到 64：`configs/config_mefs.py:28,33`；潜变量 `zdim=512`（`:46`）；特征层 `models/vae.py:148-149` `z_mu`
- 论文与域偏移见 §3.4

**unidino（替身）**
- 逐通道：`source/inference_utils.py:19-21`、`:29-34`（按 ch 升序拼成 `384×C`）
- 缩放：`source/image_ops.py:14-20` `img/(2**16-1)`；官方次序是 **scale→crop**，本项目是 **resize→scale**

**dinocells（替身）** —— **本次修复的核心**
- 模型输入 `(C,H,W)`，mask 被丢：`cellpainting_dataset.py:23-47`、`:86`；`run_get_features.py:87-91`
- **Fortran order**：`np.reshape(image, (H, channel_width, -1), order="F")`（C order 会静默打乱通道）
- **裁片尺寸 96 的出处**：官方 `config.yaml:246`/`:415` `RandomResizedCrop: {size: 96}`。
  **但要说准**：仓库里的 config 是 **HPA** 变体（`datatype: HPA`、`num_channels: 4`），
  **CP 的 config 未发布** → 96 是从兄弟配置继承的，不是从 CP 配置直接读到的
- 官方归一化：`file_dataset.py:77-82` `norm_loader`（**全图全局** min-max → uint8）
  + `utils/augmentations.py:460-468` `self_normalize`（逐通道 z-score，`unbiased=False`）
- 官方可批（`batch_size_per_gpu: 24`）→ 不得再以 `tape-batch` 拒绝

**morphogenie（替身）**
- 嵌入宽 = `c_dim = 10`（`MorphoGenie_Test.py:67`），不是 `z_dim=256`（`:68`）
- 官方持久化的是重参数化采样 `c`（`gan_training/models/dvae.py:393-396`、`:409-413`；
  `MorphoGenie_Test.py:297` 的解包顺序导致 `mu_real` 实际绑定到 `c`）→ 官方 stochastic

### 6.2 横切（W1–W12）

| # | 检查 | 期望 |
|---|---|---|
| W1 | `check_compatible()` 不 import torch/TF | 无框架解释器里可调用，且 `sys.modules` 中无框架 |
| W2 | `output_dim` 与实际一致 | 不一致被 `reason='output-width'` 拒 |
| W3 | 通道序错时真的报错 | 声明 names → `channel-order`；**不声明 names 的置换契约无从检测**（嵌入会变但不报）——固有限制，报告要写明 |
| W4 | 16-bit 喂 8-bit 方法 | **逐方法测**：对每个声明了 `value_range` 的方法，喂一个越界数组**且不带** `assume='raw_crop'`，必须拒 `value-range`。<br>⚠️ **带 `assume='raw_crop'` 时不适用**：那条路径的语义是"这是原始采集值，请按 manifest 声明的 `source_map` 换算"，换算本身是契约的一部分（例如 ca_mae 的 `per_crop_max255` 把 16-bit 域归一到 0..255）。<br>判据要写成"**未经声明的换算**必须被拒"，而不是"16-bit 输入必须被拒" |
| W5 | `assume='raw_crop'` ≡ 手动 prepare + embed | 数值一致 |
| W6 | 权重 sha256 与实际文件相符 | 官方 4/4；替身 3/3 |
| W7 | 切 profile 清空适配器缓存 | 是 |
| W8 | `describe()` 带 `weights_provenance` | 是 |
| W9 | `probe()` 的依赖来源 | 读**适配器**不是 manifest；官方依赖仍保留在 `manifest_requires` |
| W10 | `div65535` 的 dtype 守卫 | uint8 拒；**已落在输出域内的 float 也要拒**（§8.2） |
| W11 | 声明了 `spatial` 的方法守尺寸 | `IncompatibleInput(reason='spatial-size')`，非框架异常；**不得静默 resize** |
| W12 | 声明 `max_channels` 的方法守上限 | 超出 → `channel-count` |

---

## 7. 阶段三：benchmark

```powershell
& $py scripts\embedding_benchmark\embed_dataset.py --cells-dir E:\bbbc022\cells --force
& $py scripts\embedding_benchmark\build_v3_surrogates.py --cells-dir E:\bbbc022\cells --force
& $py scripts\embedding_benchmark\score_benchmark.py --cells-dir E:\bbbc022\cells --profile v2 `
      --output E:\bbbc022\cells\benchmark_scores_v2_recheck.json
& $py scripts\embedding_benchmark\score_benchmark.py --cells-dir E:\bbbc022\cells --profile v3 `
      --output E:\bbbc022\cells\benchmark_scores_v3_recheck.json
& $py scripts\embedding_benchmark\stability_check.py --cells-dir E:\bbbc022\cells --profile v3 --folds 4 `
      --output E:\bbbc022\cells\stability_v3_recheck.json
```

**判据（两条是核心）**

1. **v2 四个方法必须与修复前逐位一致**（嵌入数组层面与打分层面双重验证）：
   ```
   cellclip 0.1337 > ca_mae 0.1083 > deepprofiler 0.0691 > o2vae 0.0471
   原始像素 0.0508    随机 0.0450
   ```
   **任何偏差都是回归** —— 本轮修改不动 v2 的任何路径。
2. **v3 的三个替身**：unidino 0.0684、morphogenie 0.0574 应**不变**；dinocells 应为 **0.0453**。
   旧值 0.0460 **作废**（权重与几何都换了）。**不得把这种变化描述为"性能下降/提升"。**

**必须随表附带（一条都不能省）**

1. 不给 p 值（每化合物单孔，化合物身份 ≡ 孔身份）
2. 替身行必须标注
3. 不可与论文数字比较（协议自建）
4. o2vae 是形状模型 + **域偏移**
5. 稳定性报"稳定前缀 + 互换对"，不只报布尔
6. **silhouette 全为负** → "嵌入形成可分离簇"在本数据上不成立
7. mock-AUC 随机基线 0.4996 → 不宜用于方法排序

---

## 8. 攻击清单（**本节最重要**）

前面是"该成立什么"，这一节是"**去打破它**"。每条都要**动手**，不要只读代码。
能打破的写成 ❌ 并给出官方行号 + 本项目行号 + 复现命令；打不动的也要写出攻击代码。

### 8.1 F1 的诊断面：换个环境还准吗

- 在**没有 torch** 的解释器（`.venv`）里跑 `probe('dinocells')`：`runnable=False` 是**因为缺 torch**（正确）还是别的原因？`requires` 里还有没有 `kornia`/`oyaml`？
- `probe()` 在**适配器模块 import 失败**时会怎样？会不会静默退化？
- `doctor()` 对 7 个方法逐个看，有没有第二个假阴性/假阳性？

### 8.2 F3 的守卫：能不能绕过

| 输入 | 期望 |
|---|---|
| uint8 + `assume='raw_crop'` | 拒 `value-range` |
| **已除过 65535 的 float（max≤1）+ `assume='raw_crop'`** | **拒 `value-range`**（二次缩放→全黑） |
| 已除过 65535 的 float（max≤1）**不带** `assume` | **接受**（这已是准备好的张量） |
| float 原始计数（max≈65535）+ `assume` | **接受**（不能误杀） |
| uint16 / int16 + `assume` | 接受 |
| **Python list**（无 dtype 属性） | 契约是 duck-typed 的，list 没有 dtype → **守卫会不会形同虚设**？实测 `embed('unidino', uint8_array.tolist())` 曾返回 OK。**这是不是漏网？** |
| `skip_contract_check=True` | 放行（确认默认是 False） |
| 阈值边界：max = 0.999 / 1.0 / 1.001 | 阈值定在 1.0（scale 的输出域上界），**检查边界是否自洽** |
| dtype guard 的位置 | 必须在 **resize 之前**（`_resize` 产出 float64，dtype 会被洗掉） |

### 8.3 F4/F7 的边界

- deepprofiler：31×31 / 32×32 / 33×33 / 64×64 → 只 32×32 放行？
- deepprofiler：3-D 单裁片 `(32,32,5)` 与 4-D `(1,32,32,5)` → 都放行？都拒？
- deepprofiler：`resize_policy='refuse'` 会不会把**合法的 32×32 路径**也拒掉？（跑一遍 benchmark 用的那条路）
- ca_mae：C = 0 / 1 / 11 / 12 → C=0 放行了吗？0 通道的数组该不该放？
- ca_mae：C=12 的 reason 是不是 `channel-count`（不是别的）？

### 8.4 F2 的几何：和官方链逐位对得上吗

**自己重新实现一遍官方链**，不要复用被审方的脚本：

```
官方磁盘磁带 (H, W*(C+1))  --fold_channels(order='F')-->  (H, W, C+1)  --drop mask-->  (H, W, C)
官方数值链：norm_loader(全图全局 min-max -> uint8)  -->  self_normalize(逐通道 z-score, ddof=0)
本项目链： resize -> per_sample_selfnorm(逐样本逐通道 z-score) -> mask 丢弃
```

- 差应只来自 uint8 量化（声称 ≤5% std，给的是 3.14%）。**数字不同就查原因**——
  两个高频坑：① `np.min(2D, axis=(0,1))` 是**标量**不是逐通道；
  ② 官方缩放是 `* 255 / max`（**别漏掉 ×255**）。
- `prepare('dinocells', crop)` 与 `prepare('dinocells', crop, masks=mask)` 结果**逐位相等**吗？（mask 必须被丢弃）
- 替身 state dict 的 `patch_embed` 第一层是 `in_chans=5` 吗？（打开 `surrogate_dinocells.pt` 看 shape）
- 官方 `self_normalize` 用 `unbiased=False`（ddof=0），本项目 `np.std` 默认 ddof=0 ✅ 一致吗？

### 8.5 F10 的强制校验是真的吗

- 把 `E:\cpc_weights\unidino\surrogate_unidino.pt` **临时改名**，看加载是否被拒。**测完必须还原。**
- **再测「内容被替换」这条分支**（B14 承诺的是「被替换的文件必须被拒」）：
  把一个内容不同的替身（如 `surrogate_morphogenie.pt`）**复制到** `surrogate_unidino.pt` 的路径上——
  文件存在但 sha256 不符，验证是否因**摘要不匹配**被拒，而不是因为文件缺失。
  测完还原原始文件，并重新核对三个替身的 sha256 与 §4.2 一致。
- manifest 钉的摘要与 `E:\cpc_weights\surrogates_manifest.json` 一致吗？
- dinocells 的摘要是 `5e172d8fed4876ff…`（重训后的新值），不是旧的 `5475ec885ae6b7d9…` 吧？

### 8.6 v2 有没有被碰到

- **自己重算** v2 四个方法的嵌入，与 `E:\bbbc022\cells\_emb_backup_before_f2\` 里的旧文件**逐位比对**。
  任何 `np.array_equal=False` 都是回归。
  ⚠️ **该目录由被审方产出，只能证明"没有漂移"，不能证明"正确"。**
  因此**额外要求**：在 v2 的四个方法中自选一个（**不得选 deepprofiler**，
  因为它已有逐位对拍），**直接依据 `D:\official_src\` 重新实现到嵌入层**，
  与该方法的现有输出逐位对比，并报告 `max|diff|` 与差异来源。
  做不完就如实记为"未独立验证"——但**必须尝试**。
- 重跑 v2 打分，与 §4.2 的数字**逐位比对**。
- 特别注意 F4 给 deepprofiler 加了 `spatial=(32,32)`，会不会把 32×32 的正常路径也拒掉？

### 8.7 文档里还有没有过期的话

`grep -rn "磁带\|tape\|96×576\|0.0460\|5 / 7\|5/7" docs/ src/ scripts/`。
历史审计文档（`embedding_audit_2026-10-21.md` 等）可保留历史表述；
**当前状态文档**（`embedding_v3_2026-10-26.md`、`PROJECT_STATUS.md`）里的磁带描述与旧基线应当已更新。
逐条判断并列出**仍需更新的**。

### 8.8 还有没有别的洞

- `mask_mode='apply'` 这条路径**实现了吗、有测试吗**？（声明只强调 `'drop'`）
- `input_encoding='tape'` 现在以 `reason='unsupported-encoding'` 拒绝。有没有代码路径仍声明它？
- `embed_all` 传一个**共享的** `channels` 给 7 个方法会怎样？（应该只有部分被拒）
- `models.describe()` 能否一眼看出 official 还是 surrogate？
- `registry.PENDING` 的 dinocells 契约（`stack` + `drop`）与 v3 替身是否一致？
- `train_surrogates.py` 的 `_masks_from_dna` 还在生成掩码，但 dinocells 的掩码已被丢弃 —— **这段工作现在是白做的吗**？
- `stability_check.py` 的 `rejected_approaches` 里那几个硬编码数字（如 `0.1338 vs [0.1615, 0.2176]`）是旧测量，**现在还成立吗**？

### 8.9 契约层有没有被改坏

- 新增的三个字段默认值（`mask_mode='none'`、`resize_policy='auto'`、`max_channels=None`）
  有没有导致既有方法行为变化？
- 删掉 `_check_tape_shape` 后，`check_compatible` 对一个声明了 `'tape'` 的 manifest 会怎样？
- `PreprocessSpec` 是 `frozen` dataclass —— 新字段是 `Literal` 类型，有没有在运行时被绕过？
### 8.10 攻击 value_range 守卫本身

对**每一个**在 manifest 里声明了 `value_range` 的方法，各喂一次越界输入。
**注意区分两条路径**：`assume='raw_crop'` 会先按声明的 `source_map` 换算，
所以越界原始值在那条路径上是**合法**的；守卫针对的是**未经声明的换算**。

| 方法 | 声明值域 | `source_map` | 越界输入 | 不带 `assume` 的期望 |
|---|---|---|---|---|
| deepprofiler | (0, 1) | `per_crop_minmax_bhwc` | 16-bit 原始值 | 拒 value-range |
| ca_mae | (0, 255) | `per_crop_max255` | uint16 max=65535（**布局正确的 BCHW**） | 拒 value-range |
| ca_mae | (0, 255) | `per_crop_max255` | float32 max=65535 | 拒 value-range |
| o2vae | (0, 1) | `mask_to_silhouette` | 越界灰度 | 拒 value-range |
| unidino | (0, 65535) | `div65535` | uint8 | 拒 value-range |
| morphogenie | **None**（未声明） | `matlab_centring` | — | 不适用（无值域声明） |

**顺带验证一个常被搞错的性质**：ca_mae 的 `per_crop_max255` 是**尺度不变**的
（除以裁片自身的最大值，任何正的标量倍数都被约掉）。
所以把同一批像素乘 1 / 16 / 257 倍再走 `assume='raw_crop'`，
**应当得到逐位相同的嵌入**——这说明 16-bit→8-bit 的换算没有静默截断。
（**已结案，无需重复**：×1 / ×2 / ×16 / ×257 实测 `max|diff|` 全为 0，机制是 `per_crop_max255` 的尺度不变性，与乘数取值无关。）
（另注：乘数大到让 uint16 **溢出**时当然会不同，那是溢出，不是尺度问题。）

### 8.11 攻击 benchmark 本身的有效性

§7 要求逐位复现排序，但**没有要求判断这个排序有没有意义**。请补做：

1. **最低的方法与随机基线的差**：`o2vae 0.0471 − 随机 0.0450 = 0.0021`。
   这个差相对**跨 fold 波动**有多大？把 `stability_check.py` 的 4 折结果拿出来，
   给出每个方法的 fold 区间。
2. **silhouette 全为负**意味着嵌入不可分簇。那么"方法 A 优于方法 B"这一判断，
   在**没有可分离结构**的数据上究竟依赖什么？
3. 若某个方法的 fold 区间与随机基线重叠，**必须在 §9 结论里写明"该方法的排序不成立"**。
4. 明确回答：在 mock-AUC 的随机基线为 0.4996、silhouette 全为负的前提下，
   **"v2 可发表"是否仍然成立**？若不成立，**需要补什么实验**才能成立？


---

## 9. 交付格式

一份 `docs/review_report_recheck_<日期>.md`，含：

1. **黑盒表**（七行 + B1–B14 逐项判定）
2. **白盒表**（逐方法 × 逐检查点，三态；❌ 必须带官方行号 + 本项目行号 + 差异）
3. **攻击清单的逐条结果**（§8.1–§8.11）：每条写「攻击方式 → 实际结果 → 判定」。
   **打不动也要写"攻击无效"和你的攻击代码**，不能只写"通过"
4. **F1–F14、P1–P8 的关闭确认表**：
   `已关闭 / 有意保留（写明理由）/ 未关闭（写明证据）/ 文档未改（指出位置）`
5. **benchmark 表**（v2 逐位对比 + v3 新基线，并写明 dinocells 旧值作废的原因）
6. **结论三档**，且必须逐条给出下列判据的成立情况（**判据先于结论**，不许只写结论）：
   - **v2 可发表** ⇔ ① 4 个官方权重 sha256 全部核对通过；
     ② 4 个方法的输出宽度与 manifest 一致；
     ③ **至少 1 个方法在 v2 打分上高于随机基线，且该差超出跨 fold 波动**
     （见 §8.11 第 1 条；若无法证明，v2 就**不**满足本条）；
     ④ v2 嵌入相对 `_emb_backup_before_f2\` 逐位不变，**并且**至少 1 个非
     deepprofiler 方法已从官方源码独立推导验证（见 §8.6）。
   - **v3 仅接口验证** ⇔ 3 个替身行的 `weights_provenance='surrogate'` 在表内可见，
     且正文未出现任何跨方法性能排序结论。
   - **哪些结论数据不支持** ⇔ 逐条列出被判据否决的声明（尤其 §8.11 得出的排序失效项）。
7. **明确列出"被审方声明但你未能独立验证"的项**
8. **§5.0 的四个随机探针**必须出现在报告开头

---

## 10. 禁止事项

| 禁止 | 原因 |
|---|---|
| 只跑部分方法就下结论 | 用户要求"务必全部测试" |
| 把替身分数当方法性能 | 权重不是官方的 |
| 给 p 值或"显著优于" | 单孔数据结构上不成立 |
| 用"看起来对"代替运行 | 缺陷都是静默型的 |
| 跳过黑盒直接白盒 | 会把"跑不起来"误诊成"实现错" |
| 声称除 deepprofiler 外某方法已与官方逐位一致 | 只有那一个有实测对拍 |
| 为了"通过检查"而放宽断言 | 契约的价值在于拒绝 |
| 把 F2 引起的 v3 数字变化当作回归 | 权重与几何都换了，不可比 |
| 未经核实就照抄本文档的行号 | 行号会漂移，**改前先确认** |
| 修改被审仓库源码来让检查通过 | 你在复核，不在修复（§8.5 的临时改名除外，且必须还原） |

---

## 11. 交接检查清单

开工前确认你能回答：

- [ ] 这个项目为什么需要统一接口？
- [ ] v2 与 v3 的区别？为什么不能合并？
- [ ] F2 到底错在哪？"折叠实现正确"为什么反而是问题？
- [ ] 判据里哪两条是"任何偏差都是回归"？
- [ ] 官方 `fold_channels` 的语义是什么？做完一遍后用一句话描述，**不得从 §6 抄**；mask 最后去哪了？
- [ ] 官方 `norm_loader` 的 min-max 是逐通道还是全图全局？（`np.min(2D, axis=(0,1))` 是标量）
- [ ] 为什么 dtype 守卫必须放在 resize **之前**？
- [ ] 96 这个裁片尺寸的官方出处是哪一行？有什么保留？

收工时，以下必须为真：

- [ ] 报告开头回报了 §5.0 的四个随机探针（本文档不给答案）
- [ ] 七方法黑盒全过，宽度 672/384/512/512/1920/384/10
- [ ] B1–B14 逐条有结论
- [ ] §8 的 11 条攻击（§8.1–§8.11）**逐条动手试过**，并记录了结果与判定
- [ ] v2 四个方法的嵌入与打分**逐位不变**
- [ ] v3 新基线已确认，并写明 dinocells 旧值作废的原因
- [ ] F1–F14、P1–P8 逐条有结论
- [ ] 没有任何一条结论超出数据支持范围

---

## 12. 本文档自身的已知缺陷与边界

> 自评节。前 11 节要求审查者「去打破被审方」；这一节说明**本提示词本身哪里还立不住**。
> 使用前必须读，避免把下列残余风险当成「已经覆盖」。

### 12.1 答案纸问题只是缓解，没有解决

§4.2 仍给出全部期望数字，§7 的核心判据仍是「逐位一致」。§5.0 的四个探针只能证明**跑过一遍**，不能证明**独立判断过**——一个足够仔细的照抄者把探针也跑一遍，依然可以通过。

真正的解法没被采纳：**让至少一个判断不依赖 §4.2 的数字**，例如要求审查者自行判定「v2 四个方法里哪一个最可能实现有误」并给出证据。这是本提示词最大的未解缺口。

### 12.2 §7 与 §8.11 的优先级没有规定

§7 要求逐位复现排序，§8.11 要求质疑排序本身是否有效。两者冲突时（审查者判定排序无效）文档没说报告以哪个为准，也没说「判定排序失效」是否影响「逐位一致」这一项的通过。

**建议补一句**：两项判据**相互独立**，必须同时给出「数字复现了」与「这个排序是否有意义」，任一为否都要写明。

### 12.3 B13 的「理论误差上界」没给推导路径

B13 要求「先给出纯 uint8 量化的理论误差上界」，但没说怎么推。审查者可以写一个凑出来的界照样通过。

**推导骨架应当写进文档**：uint8 量化 → 每像素 ±0.5 LSB → 归一化后的相对误差 → 经嵌入网络按 Lipschitz 常数放大 → 与实测 3.14% std 比较。未给骨架之前，B13 仍可被糊弄。

### 12.4 14 项既有失败只给了文件名，没给测试名

§5.1 列了 6 个测试文件，但没列 14 个具体测试名。若修复轮**新增**了失败，审查者只数到 14 便无法区分「原有」与「新增」，而这恰恰是最该发现的回归。

### 12.5 §8.6 的「独立推导」没有验收标准，也没有时间预算

要求「至少 1 个非 deepprofiler 方法直接依据官方源码重新实现到嵌入层」，但没说推到什么程度算成功（嵌入逐位？同分布？`max|diff|` 多大算通过），也没给预算。实际执行时很可能被记为「未独立验证」，于是 §9 第 ④ 条判据永远不成立，v2 也就永远判不了「可发表」。

### 12.6 v3 的数字没有「不可引用」标记

§9 写了「v3 仅接口验证」，但 §4.2 的 v3 数字（unidino 0.0684、morphogenie 0.0574、dinocells 0.0453）与 v2 数字同表并列，**没有任何标记**说明它们来自 surrogate 权重、不可用于发表。这些数字极易被单独摘出去当结果。

**建议**：在 §4.2 的 v3 行末直接加 ❌ 与一句「surrogate 权重，禁止引用为性能结论」。

### 12.7 没有版本控制约定

本版为 v2，v1 中已有结论被推翻（例如 ca_mae 的 `value_range` 曾被判为未守约，实测那是 `assume='raw_crop'` 的设计意图）。两份文件同时流出会在「按哪个执行」上冲突。

**建议**：只分发 v2，v1 归档到 `docs/archive/`。
