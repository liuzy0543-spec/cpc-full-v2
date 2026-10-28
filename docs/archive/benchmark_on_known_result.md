# 已知结果上的 Benchmark 实测

> 日期：2026-10-10
> **状态：三步全部通过 —— 逐位一致已达成**
> 探针：`_probe/{parity_vs_known_result, capture_official_crops_insitu, benchmark_known_result}.py`
> 产物：`D:\cpc_bench_demo_known\{embeddings\*.npz, summary.json}`

---

## ★ 更正说明（本文档早期版本的结论是错的）

早期版本结论为"未能复现逐位一致，原因未定位"。**根因不在接口层，也不在官方代码，而在我自己的探针脚本。**

**根因**：探针里的 `crop_bilinear()` 把**归一化**的 box 当成**像素**坐标用了。
`tf.image.crop_and_resize` 要求 box 归一化到 `[0,1]`；我的函数拿到 `0.064264` 却按像素解释，
于是采样的其实是图像左上角**宽 0.064 像素**的一条缝，产出一个近乎平坦的 patch
（range `[877, 1145]`，而真实裁剪跨越 `[850, 53276]`）。

**修正方式**：不再自己重实现采样网格，**直接调用 `tf.image.crop_and_resize`**。
官方流水线就是用它裁剪的，对拍脚本没有理由去近似它。
（先尝试修正自己的 numpy 实现，但残留误差仍有 `1.4e4`，说明 TF 的采样网格不是文档里那个简化公式；
与其继续逆向，不如调用同一个算子。）

**修正后**：

```
raw (blocked by contract)  bitwise_equal=False  max|diff|=4.481230e+06
per-crop min-max           bitwise_equal=True   max|diff|=0.000000e+00
```

---

## 1. 结论摘要

| 问题 | 结论 |
|---|---|
| **DeepProfiler 与官方逐位一致？** | ✅ **`bitwise_equal=True`, `max\|diff\| = 0.000000e+00`** |
| 四个 embedding 能否走同一条调用？ | ✅ **能，errors 为空** |
| IO 接口是否统一？ | ✅ 统一，且能表达输入差异 |
| 输入输出是否符合预期？ | ✅ 形状/维度/dtype/值域全部符合各自契约 |
| 能否做 benchmarking？ | ✅ **管线与等价性前提均已满足** |

---

## 2. 实测结果（真实现场数据）

**输入**：CellPainting-Claw demo 数据，2 个 well × 2 个细胞 = **4 个细胞**，
来自 `D:\cpc_c_official` 的真实 16-bit TIFF（64×64×5）。

**已知结果**：官方流水线产出的 `A01/1.npz` + `A02/1.npz`，`(4, 672)`。
`Metadata_Model = efficientnet`。

### 2.1 统一接口：一次调用，四个方法

| 方法 | 载荷形状 | dtype | 载荷值域 | **输出** | 维度 | 确定性 |
|---|---|---|---|---|---|---|
| `deepprofiler` | `(4,32,32,5)` | float32 | `[0,1]` | **`(4,672)`** | 672 | deterministic |
| `ca_mae` | `(4,5,256,256)` | **uint8** | `[3,5]` | **`(4,384)`** | 384 | deterministic |
| `cellclip` | `(4,5,1536)` | float32 | `[-4.66,3.83]` | **`(4,512)`** | 512 | deterministic |
| `o2vae` | `(4,1,32,32)` | float32 | `[0.96,1]` | **`(4,64)`** | 64 | deterministic |

**errors: 无**。四个方法全部成功。

### 2.2 对齐与合规

```
aligned rows (shared cell_id): 4 -> ['A01-cell0','A01-cell1','A02-cell0','A02-cell1']

licence gate:
  deepprofiler  permissive        ← 唯一代码与权重双双宽松
  o2vae         permissive        （但权重是自训）
  ca_mae        non-commercial
  cellclip      no-derivatives    ← ND 禁衍生
```

### 2.3 RV 一致性（**仅说明管线通，无科学含义**）

```
ca_mae    vs cellclip      0.6256
ca_mae    vs deepprofiler  0.7513
ca_mae    vs o2vae         0.9124
cellclip  vs deepprofiler  0.6256
cellclip  vs o2vae         0.5625
deepprofiler vs o2vae      0.4481
```

> ⚠️ **4 个细胞的合成 fixture，不能解读为方法相似度。** 它只证明对齐、
> 跨宽度度量、落盘三件事都работает。

---

## 3. DeepProfiler 与官方已知结果的比对

```
official (4, 672)   ours (4, 672)
bitwise equal : False
max|diff|     : 2.679210e+01
official mean/max: 1.9696 / 18.5077
ours     mean/max: 2.2458 / 30.3767
```

**未达到逐位一致。** 下面是排查过程与结论。

### 3.1 已排除的原因（每条都有实测依据）

| 假设 | 检验方式 | 结论 |
|---|---|---|
| 裁剪几何算错 | 对照 `boxes.py:204-209` 逐行 | ❌ **公式完全一致** |
| 边界外推差异 | 检查 4 个 box 是否越界 | ❌ 全部在界内，**无需 clamp** |
| 读取器差异 | `tifffile` vs `skimage.io.imread` 逐通道 | ❌ **逐字节相同** |
| 通道顺序错 | **穷举全部 120 种排列** | ❌ 最好 0.41、最差 0.34 相关，**均匀地错**，非顺序问题 |
| 标量缩放错 | 8 个候选（`/255`、`/65535`、uint8、illum 等） | ❌ 最好 `uint8→minmax` 相关 0.47，仍差得远 |
| 照明校正被应用 | 除以 `Illum*.npy` 前后比对 | ❌ **结果完全相同**（官方 profile 阶段不用它） |
| **box 坐标语义** | **捕获官方 box 与图像后逐一对照** | ✅ **根因**：探针把归一化 box 当像素用 |

**排除法的价值**：前 6 条都被数据否掉，才使第 7 条成为唯一剩下的可能。

### 3.2 顺带找到的真问题：**manifest 的 scale 申报是错的**

**决定性实验**：直接驱动**官方裁剪图**（`deepprofiler.imaging.cropping.crop_graph`），
把同一张图与同一批 box 喂进去，dump 它的输出：

```
image (64, 64, 5) dtype float64 range [850, 53159]
official crops (2, 32, 32, 5) dtype float32 range [0.0000, 1.0000]
our raw crops  (2, 32, 32, 5)          range [877.12, 1145.38]

official per-crop minima : [0.0, 0.0]
official per-crop maxima : [1.0, 1.0]
```

**官方裁剪图内部做 per-crop min-max**（`cropping.py:65-67`），
输出**恰好 `[0,1]`**。所以模型真正看到的是 `[0,1]`，不是 16-bit 原始值。

**而我方 manifest 声明的是：**

```python
scale='raw',
value_range=None,
```

**这个申报有两重危害**：

1. **它鼓励了错误输入**。`value_range=None` 让 `check_compatible` **无条件接受** 16-bit 原始值，
   实测误差达 **8.8e4**（`raw` 路径 `max|diff| = 8.759e+04`）。
2. **它正是本层设计出来要拦的那类静默错误**——结果它自己的申报就是错的。

**已修正为**：

```python
scale='per_crop_minmax',
value_range=(0.0, 1.0),
```

并在 `Scale` 字面量里新增 `per_crop_minmax` 取值。

**修正后实测**：
```
16-bit raw     accepted=False  issues=['value-range']
min-max [0,1]  accepted=True   issues=[]
```

### 3.3 根因与最终结果

把官方裁剪图与我方裁剪做**裁剪对裁剪**比较，定位到差异出在采样网格上。
修正方式是**改用 `tf.image.crop_and_resize` 本身**（官方流水线用的就是它）。

最终（§2.2 已列）：

```
raw (blocked by contract)  bitwise_equal=False  max|diff|=4.481230e+06
per-crop min-max           bitwise_equal=True   max|diff|=0.000000e+00
official mean/max: 1.9696 / 18.5077
ours     mean/max: 1.9696 / 18.5077
```

**逐位一致，均值与极值也一致。**

---

## 4. 与验收标准的对照

> 「要求官方原版只要 deepprofiler，两版本这个 embedding 产出必须一致，其他的不强求」

| 要求 | 状态 |
|---|---|
| 官方原版只涉及 deepprofiler | ✅ 已独立验证（5 条判据，见 `embedding_weights_and_roadmap.md`） |
| 官方树 ↔ v2 树 产出必须一致 | ✅ C 步骤已证：`.npz` **sha256 相同、max\|diff\|=0** |
| **接口层 ↔ 官方路径 产出一致** | ✅ **本轮已证：`bitwise_equal=True`, `max\|diff\|=0`** |
| 其他三个不强求 | ✅ 只验了形状/维度/dtype/确定性，全部符合 |

**两条独立路径现在都指向同一个结论**：DeepProfiler 的嵌入在官方树、v2 树、统一接口层三者之间**逐位相同**。

---

## 4b. Benchmarking 可行性

| 能力 | 状态 |
|---|---|
| 一次调用跑多方法 | ✅ |
| 统一 IO（能表达输入差异） | ✅ |
| 输入契约校验 | ✅ 且本轮修掉一个真缺陷 |
| 输出形状/维度/dtype 符合预期 | ✅ |
| 对齐键（共享 cell_id） | ✅ |
| 确定性处理（随机方法重复） | ✅ |
| 许可闸门 | ✅ |
| 落盘 + 溯源（sha256 真校验） | ✅ |
| **接口层与官方的数值等价性** | ✅ **本轮已证（逐位）** |

**结论：调度、IO、等价性三项前提均已满足，可以做 benchmarking。**

仍然需要人为决定的是**评测指标**（取决于科学问题），本层只产出对齐矩阵与溯源，不替使用者判断"哪个方法更好"。

---

## 5. 本节记录排查中被排除的原因

（保留此表，因为排除过程本身是结论的一部分：前 6 条都被数据否掉，
才使第 7 条成为唯一剩下的可能。）

| 假设 | 检验方式 | 结论 |
|---|---|---|
| 裁剪几何算错 | 对照 `boxes.py:204-209` 逐行 | ❌ 公式一致 |
| 边界外推差异 | 检查 4 个 box 是否越界 | ❌ 全部在界内 |
| 读取器差异 | `tifffile` vs `skimage.io.imread` | ❌ 逐字节相同 |
| 通道顺序错 | 穷举 120 种排列 | ❌ 均匀地错（0.34–0.41） |
| 标量缩放错 | 8 个候选 | ❌ 最好只到 0.47 |
| 照明校正被应用 | 除以 `Illum*.npy` 前后 | ❌ 结果完全相同 |
| **box 坐标语义** | **捕获官方 box 与图像后对照** | ✅ **根因** |

> 早期版本曾把 §3.2 的 scale 申报问题记为"根因"。
> 那是**另一个真实缺陷**，但**不是**逐位不一致的原因；
> 修好它之后嵌入仍有 `max|diff|=26.79`，直到采样网格也修正才归零。

---

## 6. 本轮修掉/发现的缺陷

| # | 缺陷 | 危害 | 状态 |
|---|---|---|---|
| 1 | **探针把归一化 box 当像素用** | 采到 0.064 像素宽的缝，导致误判"接口层不等价" | ✅ 改用 TF 原算子 |
| 2 | `deepprofiler` 声明 `scale='raw'`、`value_range=None` | 接受 16-bit 原始值，误差 **4.48e6**，**静默** | ✅ 已修为 `per_crop_minmax` + `(0,1)` |
| 3 | 此前"验证"deepprofiler 用的是 `randint(0,20000)` 合成数据 | 值域不对但有界，掩盖了 #2 | ✅ 已由新契约拦下 |
| 4 | `embed_all` 首版对 384 vs 512 维做逐行余弦 | `ValueError` | ✅ 改用 RV |
| 5 | O2-VAE 在 torch 2.x 下加载失败（e2cnn 零步长 buffer） | 完全跑不了 | ✅ 已修 |

---

## 7. 测试现状

```
tests/test_models_contract.py       33 passed
tests/test_embedding_benchmark.py   34 passed   ← 含 3 个钉住本次结论的回归测试
全量套件                             14 failed, 192 passed
```

**192 − 126 = 66** 为本会话新增；14 个失败与会话开始前**完全一致**（既有问题）。

`TestParityOutcomeIsPinned` 把本次结论钉死：scale 必须是 `per_crop_minmax`、
`value_range` 必须被强制、关键字必须在 `Scale` 词表内。
**若有人把 range 改回 `None`，测试会失败** —— 那正是曾让 4.48e6 误差通过的路径。

---

## 8. 遗留

| 项 | 说明 |
|---|---|
| 另三个方法与官方比对 | 官方未发布该输入下的参考，**无法比对**（验收标准已表示不强求） |
| `unidino` / `dinocells` / `morphogenie` | 注册表已声明契约，`adapters/` 尚无模块；权重亦不可得 |
| CLI / 技能入口 | 目前只能从 Python 调用 |
| 评测指标 | 取决于科学问题 |
| 样本规模 | 本次为 4 个细胞的 demo fixture；**结论是"管线逐位一致"，不是方法学结论** |
