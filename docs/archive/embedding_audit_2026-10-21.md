# Embedding 统一接口 —— 审查报告与路线图

> **已被取代（2026-10-22）**：方法范围从 7 个收敛到 4 个可运行，其余 3 个已取消。
> 现行版本见 `docs/embedding_roadmap_v2_2026-10-22.md`（路线图）与
> `docs/embedding_summary_2026-10-22.md`（benchmark 汇总）。
> 本文保留作为 v1 审查记录：§2 的逐位对拍证据、§4 的问题清单、§8 的 A/B 执行结果仍然有效。

审查对象：`D:\项目\CellPainting-Claw-v2`
审查日期：2026-10-21
审查范围：Cell Painting 的 Embedding 方法能否统一 input/output、解耦、正确加载模型

---

## 0. 结论摘要

| 维度 | 结论 | 证据强度 |
|---|---|---|
| **功能：与 DeepProfiler 原版是否完全相同** | ✅ **逐位相同**（max\|diff\| = 0.000000e+00） | 实测，本次重跑 |
| **功能：统一 input-output** | ✅ **成立**，4 个可运行方法走同一签名、返回同一结构 | 实测，3,857 细胞 |
| **功能：正确加载模型** | ✅ 4 个官方权重 sha256 全部 MATCH；加载有断言可失败 | 实测 |
| **结构：便于调度** | ✅ 分层通过、依赖倒置成立、批处理可分块 | 实测 + 代码审查 |
| **结构：便于维护** | ✅ 命名陷阱与测试失衡**已修复**（§8） | 实测 + 代码审查 |
| **预处理** | ✅ 已实现、逐位保真，**6 种 `source_map` 已有测试**（§8） | 实测 |
| **覆盖度** | ⚠️ **6 个中 3 个可运行**，另 3 个权重拿不到 | 实测网速 |

**一句话**：框架目标已达成——`models.embed('方法名', x)` 直接拿到统一格式的 Embedding，
**连 `unit` 都不用传**；预处理由库自己按契约执行，且与官方逐位相同；
DeepProfiler 与官方流水线**逐位相同**。但用户点名的 6 个方法里只有 3 个真正能跑
（deepprofiler / CA-MAE / CellCLIP），uniDINO / MorphoGenie 卡在权重下载（实测 6 KB/s）。

**审查发现的问题已在阶段 A/B 修复**（详见 §8）：
1. ✅ `available()` 会让人误以为 7 个方法都可用 → 拆成 `declared()` / `runnable()`
2. ✅ `preprocessing.py` 只有 1 条测试 → 补到 28 条，并**顺带修掉两个掩码 resize 缺陷**

---

## 1. 覆盖度核查：点名的 6 个 vs 实际注册

用户点名的 6 个：`deepprofiler`、`CA-MAE`、`uniDino`、`o2-vae`、`morphogenie`、`cellclip`

| 方法 | 注册 | 可运行 | 权重 | 输入类型 | 输出 | 诚实结论 |
|---|---|---|---|---|---|---|
| deepprofiler | ✅ | ✅ | 19,236,600 B，sha256 MATCH | 5ch 像素 BHWC 32×32 | 672 | ✅ 可用，逐位对拍通过 |
| ca_mae | ✅ | ✅ | 712,434,294 B，MATCH | 5ch 像素 BCHW 256×256 uint8 | 384 | ✅ 可用，通道无关 |
| cellclip | ✅ | ✅ | 1,799,564,952 B，MATCH + DINOv2 链 4.5 GB | **(N,5,1536) 向量** | 512 | ✅ 可用，但吃的是 DINOv2 输出 |
| o2vae | ✅ | ✅ | 7,428,429 B，MATCH | 1ch 剪影 64×64 | 512 | ✅ 能跑，但**形状模型**，非化合物表型 |
| unidino | ✅ | ❌ | Zenodo 1.38 GB，**实测 6 KB/s → ≈62 h** | 5ch 像素 224×224 | 384 | ⚠️ 仅声明，无权重 |
| morphogenie | ✅ | ❌ | Google Drive，**本机不可达** | 3ch 像素 256×256 | 10 | ⚠️ 仅声明；且 `stochastic`、需 MATLAB 预处理 |
| *dinocells* | ✅ | ❌ | Zenodo 6.90 GB，≈312 h | 5ch 像素 96×96 | 384 | ⚠️ **用户未点名但已注册**，同样无权重 |

**需要你确认的一点**：`dinocells` 不在你点名的 6 个里，但注册表里有它（共 7 个）。
如果不需要，可以移除；如果保留，它会一直显示为"声明但不可运行"。

**必须说清的局限**：3 个不可运行的方法**没有真实适配器**——`registry.py` 里用的是
`make_unavailable_adapter`。它们有完整的契约（布局/通道/空间/输出维度），
但**没有实现**。硬写一个不验证的实现，等于假装完成。

---

## 2. 功能核查：与 DeepProfiler 原版是否完全相同

这是本次审查最重要的一项，我用**官方流水线自己产出的 `.npz` 作为已知结果**重跑对拍。

### 2.1 实测结果

```
official features : (4, 672)
crop value range  : [850.5, 53131.4]   ← 原始 16-bit

A. demo 顺序 + per-crop min-max    bitwise_equal=True   max|diff| = 0.000000e+00
B. demo 顺序 + 原始像素            bitwise_equal=False  max|diff| = 4.481230e+06
C. 上游顺序 + per-crop min-max     bitwise_equal=False  max|diff| = 1.740073e+01
```

**A = 逐位相同**。我们的 adapter 完整复现了官方 DeepProfiler 的
裁剪 → 归一化 → 前向 → 池化归约 整条链路，误差为 **0**，不是"接近"。

### 2.2 但"完全相同"必须分两层说

| 层次 | 结论 | 说明 |
|---|---|---|
| **实现保真度** | ✅ 完全相同（0 误差） | 同样的代码路径、同样的通道顺序，产出与官方运行逐位一致 |
| **上游训练语义** | ⚠️ **有意与旧 demo 不同** | 旧 demo 把 RNA/ER 装反了；我们按上游 README 改回 `DNA,ER,RNA,AGP,Mito`，所以 C 必然与旧 demo 差 17.4 |

上游权威依据：
```
D:\official_src\1-deepprofiler\README.md:74
  "trained on 5-channel Cell Painting images (DNA, ER, RNA, AGP, Mito)"
tests\deepprofiler\profiling\test_cell_painting_cnn_v1.py:44-48,74
  ch5=DNA, ch4=ER, ch3=RNA, ch1=AGP, ch2=Mito
  CHANNELS = ["DNA", "ER", "RNA", "AGP", "Mito"]
```

**所以准确的说法是**：与**官方流水线在旧 demo 配置下的运行结果**逐位相同；
与**上游训练语义**对齐（且这是修正，不是回退）。**不能说"与官方完全一样"而不加限定**——
那会掩盖"旧 demo 本身是错的"这个事实。

### 2.3 契约守卫实测通过

```
旧通道顺序被拒        PASS (IncompatibleInput)
16-bit 原始输入被拒   PASS (IncompatibleInput)
```

B 这一组说明：如果归一化搞错，误差是 **4.48e6** 量级——不是小偏差，是量级错误。
契约在**加载模型之前**就拦住它，这是"正确加载模型"的真实含义。

### 2.4 复现命令

```powershell
python scripts\embedding_benchmark\parity_vs_official_dp.py
```

---

## 3. 功能核查：统一 input-output 是否成立

### 3.1 同一签名（实测，3,857 细胞）

```python
models.embed(name, payload, unit=unit)     # 只有 name 和 unit 不同
```

最简调用（**连 unit 都不用给**，从 manifest 推断）已实测可用：

```
embed('o2vae',   (4,1,64,64))   -> (4, 512)  OK
embed('cellclip',(4,5,1536))    -> (4, 512)  OK
```

这满足你最初的诉求——"直接调用一个方法名字就得到 Embedding"。

### 3.2 同一返回结构

`EmbeddingResult`（扁平结构）：

```
model  values  dim  input_unit  backend  feature_layer  n_samples
weights_path  weights_sha256  license  determinism  metadata  sample_metadata
+ shape / as_numpy() / as_torch() / to_dict() / to_frame() / save_npz()
```

4 个方法的**字段集合完全一致**（实测 `identical field sets: True`），
全部带权重哈希、许可、确定性标记。**输出维度不同（384/512/672）是正常的**——
统一的是接口，不是维度。

### 3.3 批处理/调度

```python
plan = build_benchmark_plan(methods, output_dir=..., unit={...}, align_on='cell_id')
run_benchmark(plan, {method: array}, sample_metadata=...)
```

- `unit` 支持 `{method: unit}` 映射（必需：CellCLIP 吃向量、其余吃像素）
- 全量 3,857 细胞实测：`errors=none`、`aligned=3857`、5 个产物
- **批处理与逐方法等价**：DeepProfiler 逐位相同；其余差 ≤5.5e-06，
  判别实验证明是 float32 批边界舍入（同 chunk 重跑逐位相同），
  对分数影响 ≤3.11e-08

---

## 4. 结构核查：调度与维护

### 4.1 做得对的地方（有实测支撑）

| 项 | 证据 |
|---|---|
| **依赖倒置** | `embedding.py` 只 import `registry` / `errors` / `contract`，**不 import 任何具体 adapter** |
| **懒加载** | `registry` 的 `factory` 是**字符串**（`'DeepProfilerAdapter'`），用到才 import |
| **契约层纯净** | `contract.py` 只依赖 `dataclasses`/`enum`/`typing`，**零重依赖**，无框架也能测 |
| **分层不破** | `tests/test_layering.py` 通过 |
| **开闭原则** | 新增方法 = 改 2 处（registry 一条 + adapters 一个类），不动 `embedding.py`/`contract.py` |
| **可测性** | 契约/调度/预处理 **165 用例**，在无 torch/TF/权重环境 ~2 s 跑完 |
| **加载可验证** | O2-VAE 断言 missing 只能是 buffer；DP 断言权重确实变了（否则 `weight-load-noop`） |
| **内存可控** | `chunk_size` 分块，全量 3,857 细胞 5.5 min 通过（此前不分块会 OOM） |

### 4.2 发现的问题（按严重度）

#### P0 — 会让使用者判断错误

**1. `available()` 名字骗人** —— ✅ **已修复（A1）**

```
修复前：available() -> 7 个（全部声明，含 3 个不可运行）
修复后：declared()  -> 7 个（名字如实）
        runnable()  -> 4 个（这才是"能用"）
        available() -> declared() 的别名，docstring 里说明它从来不回答"能不能用"
```
字面看 `available()` 会让人以为"可用的方法有 7 个"。这是本次审查最容易踩的坑。

#### P1 — 影响可维护性

**2. 预处理已实现且保真，但几乎没有测试**
上一版审查（本次修正）曾判断"`PreprocessSpec` 只声明不执行"——**这是错的**。
实测 `preprocessing.py`（312 行）确实执行契约：

```
prepare('deepprofiler', raw_16bit)  vs  官方 cropping.py:65-67 per-crop min-max
  bitwise_equal = True    max|diff| = 0.0
embed('deepprofiler', raw, assume='raw_crop')
  与"手动 prepare 后 embed" 逐位相同 = True
```

真正的问题是**覆盖**：`preprocessing.py` 实现了 **6 种 `source_map`**

| source_map | 使用方 | 审查时 | 现在 |
|---|---|---|---|
| `per_crop_minmax_bhwc` | deepprofiler | ❌（人工验证逐位一致） | ✅ 对照 `cropping.py:65-67` 断言逐位 |
| `per_crop_max255` | ca_mae | ❌ | ✅ |
| `div65535` | unidino | ❌ | ✅ |
| `div257` | — | ❌ | ✅ |
| `per_sample_selfnorm` | dinocells | ❌ | ✅ |
| `mask_to_silhouette` | o2vae | ❌ | ✅（含 resize 回归） |
| `matlab_centring` | morphogenie | 明确不支持 | ✅ 写进模块 docstring |

而评审时测试里跟预处理有关的只有 **1 条**：
`test_prepare_is_reachable_from_the_facade`（只验证能 import，不验证算得对）。

312 行、决定模型输入对不对、错了代价是 4.48e6 量级，却只有"能导入"级别的测试。
**已补到 28 条**，并顺带发现两个掩码 resize 缺陷（§8.2）。

**3. 3 个不可运行方法只有契约、没有适配器**
`make_unavailable_adapter` 提供契约但不实现。这是**诚实的**，但意味着一半方法
无法参与 benchmark。现在至少可以通过 `runnable()` 一眼看出（见 P0-1）。

#### P2 — 代码卫生

**4. 重复导入 `Mapping`** —— ✅ **已修复（A2）**
```python
修复前：from collections.abc import Mapping, Sequence
        from typing import Any, Mapping, Protocol, runtime_checkable   ← 遮蔽前者
修复后：from collections.abc import Mapping, Sequence
        from typing import Any, Protocol, runtime_checkable
```
两个 `Mapping` 语义略有差别，运行时靠后者遮蔽前者才没出事。

**5. `o2vae` 目录里有自训权重残留** —— ✅ **已归档（A3）**
`o2vae_50ep.pt`（9.3 MB）从加载路径移到 `E:\cpc_weights\o2vae\_archive\`。
manifest 的 `source` 本来就按文件名 + sha256 钉死官方权重，所以它从未被加载过，
但人容易误认。移动而非删除，原文件仍在。

---

## 5. 路线图

按"先能跑通、再好用、最后补全"排。**每阶段都有可验收的判据。**

> **执行状态（2026-10-21 更新）**：阶段 A、B **已完成**，结果见 §8。
> C、D 按约定不做。

### 阶段 A：把"能跑"变成"跑得对" ✅ 已完成

| # | 事项 | 验收判据 | 结果 |
|---|---|---|---|
| A1 | `available()` → 拆成 `declared()` / `runnable()`（保留旧名做别名） | `runnable()` 返回 4 个；现有测试全绿 | ✅ `runnable()` = 4，`declared()` = 7 |
| A2 | 去掉重复的 `Mapping` 导入 | 无重复导入 | ✅ |
| A3 | 归档 `o2vae_50ep.pt`，或在 manifest 里明确标注"非官方" | 目录里只剩官方权重 | ✅ 移到 `_archive/` |
| A4 | 把 `parity_vs_official_dp.py` 纳入 CI 冒烟（无权重时 skip） | 有权重环境自动跑，输出 `bitwise_equal=True` | ✅ 5 条测试，有权重环境 5 passed |

### 阶段 B：给预处理补上正确性测试 ✅ 已完成

预处理**已经实现且对 DeepProfiler 逐位保真**（本次实测）。缺的不是功能，是**证据**：
6 种 `source_map` 里只有 1 种被验证过，而整个模块只有 1 条"可导入"测试。

| # | 事项 | 验收判据 | 结果 |
|---|---|---|---|
| B1 | 新建 `tests/test_models_preprocessing.py`，每种 `source_map` 一个用例，对照**官方公式** | 6 种全覆盖；`per_crop_minmax_bhwc` 断言与 `cropping.py:65-67` 逐位相同 | ✅ 26 条，全绿 |
| B2 | `assume='raw_crop'` 端到端测试：原始 16-bit → 预处理 → 嵌入 | 与"手动 prepare 后 embed"逐位相同 | ✅ 真实权重下 `array_equal=True` |
| B3 | `tape` 编码测试（dinocells 用） | 形状与掩码列位置符合官方 `(H, W*(C+1))` | ✅ 含 `order='F'` 逐元素对照 |
| B4 | `matlab_centring` 的处置写进文档：不支持、为什么 | 调用方不会误以为能用 | ✅ 写进模块 docstring 的覆盖表 |

> **B 阶段顺带发现并修掉两个真实缺陷**（原计划没有预计到）：
> `o2vae` 的掩码分支和 `dinocells` 的 tape 掩码都**没有按声明的 `spatial` 做 resize**，
> `plan()` 却宣称有 `resize` 这一步。128×128 掩码会产出 128×128 张量，
> 与 manifest 声明的 `(64,64)` 冲突。见 §8.2。

### 阶段 C：补全方法覆盖（依赖外部条件，不可控）

| # | 事项 | 前置条件 | 现实预期 |
|---|---|---|---|
| C1 | uniDINO 真实适配器 | 1.38 GB 权重（实测 62 h） | 权重到手后约 2 h 接入（架构已验证） |
| C2 | dinocells 真实适配器 | 6.90 GB 权重（≈312 h）+ 通道配置 | 权重到手后仍需先定 `num_channels` |
| C3 | MorphoGenie 真实适配器 | Google Drive + MATLAB 预处理 | 三重障碍，短期不可行 |

**建议**：C1/C2 走"下好了就接"的路径，不要现在硬写。
C3 建议**明确放弃**并在文档里写清理由（`preprocessing.plan` 已如实返回
`unsupported-preprocess`，这点做得对）。

### 阶段 D：benchmark 方法论（可选，但有价值）

| # | 事项 | 说明 |
|---|---|---|
| D1 | 补齐 BBBC022 剩余 3.58 GB | 才能做留一孔 CV，进而做显著性检验 |
| D2 | 与论文协议对齐 | 现在是自建化合物检索协议，不是 CA-MAE/CellCLIP 论文的指标 |

---

## 6. 需要你决策的三件事

1. **`dinocells` 留不留？** 它不在你点名的 6 个里，但已注册、无权重。（**仍待定**）
2. ~~**阶段 B 现在做吗？**~~ —— **已做**，见 §8。
3. **3 个不可运行方法**：保持"如实标注"（现状），还是下掉注册表？（**仍待定**）

---

## 7. 附：可复现命令

```powershell
# 功能对拍（第 2 节）
python scripts\embedding_benchmark\parity_vs_official_dp.py

# 统一性 + 批处理（3,857 细胞）
python scripts\embedding_benchmark\verify_io_uniformity.py --cells-dir E:\bbbc022\cells --batch

# 打分与稳定性
python scripts\embedding_benchmark\score_benchmark.py --cells-dir E:\bbbc022\cells
python scripts\embedding_benchmark\stability_check.py --cells-dir E:\bbbc022\cells --folds 4

# 契约与调度测试（无需 torch/TF/权重）
python -m pytest tests/test_models_contract.py tests/test_embedding_benchmark.py -q

# 预处理正确性（无需权重，但需要 numpy + skimage）
python -m pytest tests/test_models_preprocessing.py -q

# 对拍冒烟（无 TF/权重时自动 skip）
python -m pytest tests/test_parity_deepprofiler.py -q
```

**当前测试基线**：`291 passed`、`5 skipped`、14 failed（均为本任务开始前既有，与本层无关）。

---

## 8. 阶段 A / B 执行结果（2026-10-21 补全）

### 8.1 交付清单

| 文件 | 变更 |
|---|---|
| `models/registry.py` | 新增 `declared()`；`available()` 降级为别名并注明它从来不回答"能不能用" |
| `models/__init__.py` | 新增 `declared()` / `runnable()`；`doctor()` 改走 `declared()` |
| `models/embedding.py` | 去掉重复的 `Mapping` 导入 |
| `models/preprocessing.py` | 修两个掩码 resize 缺陷；模块 docstring 补 `source_map` 覆盖表 |
| `tests/test_models_preprocessing.py` | **新增** 28 条 |
| `tests/test_parity_deepprofiler.py` | **新增** 5 条（无权重 skip） |
| `scripts/embedding_benchmark/parity_vs_official_dp.py` | **新增**，区分两种"与官方一致"的基准 |
| `E:\cpc_weights\o2vae\_archive\o2vae_50ep.pt` | 自训权重移出加载路径 |

### 8.2 B 阶段发现的两个真实缺陷

两者是同一个模式：`plan()` 宣称有 `resize`，但掩码分支提前返回，从未 resize。

**缺陷 1 — O2-VAE 掩码**

```
修复前：prepare('o2vae', masks=128x128) -> (3, 1, 128, 128)   ← 与声明 (64,64) 冲突
修复后：prepare('o2vae', masks=128x128) -> (3, 1, 64, 64)     ← 二值保持
```

**缺陷 2 — dinocells tape 掩码**

```
修复前：掩码按"resize 之后"的尺寸校验（96x96），原分辨率掩码直接报 mask-shape
修复后：掩码跟随 crop 一起 resize，再二值化
```

严重程度：**中**。下游 `check_compatible` 会因 spatial 不符而拒绝，所以不会静默产出错结果；
但 `prepare()` 的契约是"给我原始数据，我产出模型可直接吃的张量"，这个承诺之前不成立。

回归测试：`test_mask_is_resized_to_the_declared_spatial`、
`test_tape_resizes_an_acquisition_resolution_mask`、
以及把 `plan()` 与行为绑定的 `test_plan_resize_step_is_backed_by_behaviour`
（后者是**模式级**防护：以后任何方法只要在 plan 里写了 `resize`，
就必须真的 resize，否则测试失败）。

### 8.3 验证记录

```
tests/test_models_preprocessing.py     28 passed          （无框架、无权重，0.8 s）
tests/test_parity_deepprofiler.py       5 passed  (36.6 s) （真实 TF + 官方权重）
tests/                                 291 passed, 5 skipped, 14 failed（14 项为既有）
```

对拍 5 条覆盖：逐位复现官方嵌入、`prepare` 对齐 `cropping.py:65-67`、
拒绝旧通道顺序、拒绝 16-bit 原始输入、`assume='raw_crop'` 与手动两步等价。

`available()` / `runnable()` 实测：

```
无框架环境   runnable() = []                                  ← 正确地反映"这里跑不了"
有框架环境   runnable() = [deepprofiler, ca_mae, cellclip, o2vae]
declared()   = 7（含 3 个仅声明的）  available() == declared()
```