# Embedding v3 —— 七个方法全部跑通（含替身权重）

日期：2026-10-26
v2：只有官方权重的 4 个方法（可发表）
v3：v2 + 3 个替身权重方法（跑通接口，**不可发表**）

---

## 0. 一句话结论

七个方法在同一个 `models.embed(name, x)` 接口下**全部跑通**，
输出类型一致、可统一调度、可比较；但其中三个的分数**不是那篇论文的性能**，
因为它们的权重拿不到，用的是本地训练的替身。

这不是妥协，是**把两件事分开**：接口能不能跑 ≠ 方法有多好。

---

## 1. 双剖面（v2 / v3）

| | v2 | v3 |
|---|---|---|
| 方法数 | 4 | 7 |
| 权重来源 | 全部 official | 4 official + 3 surrogate |
| 用途 | 可发表的 benchmark | 接口完整性验证 |

```python
from cellpaint_pipeline import models
models.declared()                  # 4 个（默认 v2）
models.set_profile('v3')
models.declared()                  # 7 个
models.active_profile()            # 'v3'
```

**默认是 v2**，理由很直接：默认答案必须是能发表的那个。
替身只有被点名要时才出现，所以不会有人"不小心"把替身分数写进论文。

切换是**视图**，不是另一条代码路径——适配器、manifest、预处理都是同一批对象。

---

## 2. 替身是什么、不是什么

### 是
- **官方架构**：unidino 和 dinocells 用 ViT-S/16（patch 16），morphogenie 用 dVAE 的
  卷积编码器 + 10 维瓶颈。所以张量形状、通道处理、**官方输入几何**都是真的在跑
  （dinocells 的输入几何在 2026-10-26 由「磁带」修正为「通道堆叠」，见 §6）
- **真训练**：VICReg 自监督，300 步，seed 20261026。产物有完整 hash
- **明确标注**：manifest、每个 `EmbeddingResult.metadata`、benchmark 表都带
  `weights_provenance='surrogate'`

### 不是
- **不是那篇论文的方法**。任何用它得出的分数不得当作该方法性能、
  不得与论文比较、不得与官方权重结果混在同一张表里而不加标注

### 三个替身的实测

| 方法 | 架构 | 参数量 | 嵌入 std | sha256（前 16 位） |
|---|---|---|---|---|
| unidino | ViT-S/16，逐通道 | 29.1 MB | 0.6681 | `a0265184661d511e` |
| dinocells | ViT-S/16，通道堆叠 | 30.4 MB | 0.7908 | `5e172d8fed4876ff` |
| morphogenie | dVAE 编码器 | 2.8 MB | 0.5333 | `491029978fef0969` |

嵌入标准差都在 0.4–0.74，**没有塌缩成常数**——这是自监督训练是否真的学到的
最直接证据（塌缩时 std → 0）。

---

## 3. 训练时做的三个关键决定

### 3.1 训练用 `preprocessing.prepare`，与推理同一条通路

如果训练时自己写一遍变换，就会出现"训练喂的格式 ≠ 推理喂的格式"，
而**失败是静默的**：模型照样输出正确宽度，只是毫无意义。

这条决定直接抓到两个 bug：
- unidino 必须**按通道展平**(`N*C,1,H,W`)训练，否则学到的是 5 通道第一层卷积，
  而推理时喂的是单通道
- dinocells 的 ViT 是**多通道**的（官方 `in_chans=num_channels`），
  吃的是折叠展开后的 `(N, 5, 96, 96)` 堆叠。
  早期实现按官方的**磁盘存储格式**（磁带 `(96, 576)`）建模型，
  于是位置编码尺寸和官方模型对不上——这个缺陷是「把存储格式当成模型输入格式」
  造成的，2026-10-26 已修正（见 §6）

### 3.2 只用 mock 细胞训练，标签一次没用

benchmark 的任务是**化合物检索**。若用 compound 标签做监督，
替身会因为"被告知答案"而检索得很好——那测的是记忆，不是表征。

所以：只用 mock/对照细胞（671 个），零化合物暴露。
副产物是**替身的分数是诚实的泛化结果**，低就是低。

### 3.3 替身输入的两个替代，都记录了

- **dinocells 的掩码**：该数据集没有 CellProfiler 掩码，用 DNA 通道 Otsu 阈值
  生成。这是替身模型的替身输入，不写清楚就会被误当真实分割
- **morphogenie 的预处理**：官方是 MATLAB 居中（`matlab_centring`，`prepare` 拒绝），
  替身用普通缩放。**这正是该方法被取消的原因**，也是它分数不可比的根源

---

## 4. Benchmark（3,857 细胞，54 化合物，7 方法）

### 4.1 结果

| 排名 | 方法 | 权重 | mAP_z | kNN_z | mock-AUC_z | 有效秩 |
|---|---|---|---|---|---|---|
| 1 | **cellclip** | official | **0.1337** | **0.6451** | 0.4763 | 23.4 |
| 2 | **ca_mae** | official | 0.1083 | 0.4454 | 0.5861 | 2.0 |
| 3 | **deepprofiler** | official | 0.0691 | 0.1934 | 0.5935 | 34.7 |
| 4 | unidino | **surrogate** | 0.0646 | 0.2393 | 0.5789 | 2.1 |
| 5 | morphogenie | **surrogate** | 0.0574 | 0.1281 | 0.5402 | 1.0 |
| 6 | **o2vae** | official | 0.0471 | 0.0869 | 0.5054 | 2.9 |
| 7 | dinocells | **surrogate** | 0.0453 | 0.0900 | 0.4930 | 3.5 |
| — | 原始像素基线 | — | 0.0508 | 0.1058 | 0.5227 | 19.8 |
| — | 随机基线 | — | 0.0450 | 0.0848 | 0.4996 | 348.8 |

chance = 0.0185。

### 4.2 怎么读

- **4 个官方方法排序与 v2 完全一致**：cellclip > ca_mae > deepprofiler > o2vae。
  v3 没有改变任何已有结论
- **unidino 替身 (0.0684) 几乎追平 deepprofiler (0.0691)**。
  只训练 300 步、671 个 mock 细胞就能到这个水平，说明该架构对细胞形态确实敏感。
  **但这不能说"uniDINO ≈ DeepProfiler"**——替身不是原方法
- **dinocells 替身 (0.0453) ≈ 随机 (0.0450)**。它是唯一比原始像素还低的。
  5 通道在 96×96 上 patch 16，信息密度本就不高，加上只有 671 个训练样本，
  学不到东西是合理结果
- **morphogenie 只有 10 维**却在第 5 名，nats 上算不错；
  但它的 silhouette_z = −0.7161，是所有方法里最差，说明 10 维里几乎没有簇结构

### 4.3 稳定性：稳定前缀 3/7，互换对 morphogenie / unidino（2026-10-27 最新）

```
fold 0: cellclip > ca_mae > deepprofiler > morphogenie > unidino > o2vae > dinocells
fold 1: cellclip > ca_mae > deepprofiler > unidino > morphogenie > o2vae > dinocells
fold 2: cellclip > ca_mae > deepprofiler > unidino > morphogenie > o2vae > dinocells
fold 3: cellclip > ca_mae > deepprofiler > unidino > morphogenie > o2vae > dinocells

稳定前缀：3 / 7 —— cellclip > ca_mae > deepprofiler
跨折互换的对：morphogenie / unidino
```

> **基线变更说明（两次）**：
> ① 2026-10-26 的 F2 修复把 dinocells 的输入几何从官方**存储格式**（磁带）
> 改成官方**模型输入格式**（通道堆叠），并据此重训了替身；
> dinocells 的 mAP_z 由 0.0460 变为 0.0453。当时的稳定性是 7/7 全序稳定。
> ② 2026-10-27 的 F19 修复过程中，unidino 的替身被**重训**
> （重建**非确定性**：同一 recipe 得到不同权重，sha 由 `e53838fe…` 变为 `a0265184…`），
> mAP_z 由 0.0684 变为 0.0646，于是它与 morphogenie 互换，稳定前缀变为 3/7。
>
> **两次都不是"性能改善/退化"，是权重换了。** 旧值一律作废，以本表为准。
>
> ⚠️ **并且**：morphogenie 的 4 折区间（0.1092..0.1417）与**随机基线区间**
> （0.0683..0.1109）**重叠** → 按"区间重叠即不可区分"的口径，
> **morphogenie / o2vae / dinocells 三个方法的名次不被数据支持**。
> 详见 `docs/review_report_selfaudit_2026-10-27.md` 的 F18。

**顺带修了一个报告缺陷**：原脚本只有"完全相同 / 不稳定"二元判定，
7 个方法时必然输出"不稳定"，而实际上只有末两位在换。
现在改为报告**稳定前缀长度 + 具体哪几对互换**，这才是数据支持的说法。

### 4.4 不能说的话

- ❌ 不能说"unidino 与 deepprofiler 相当"——一个是替身
- ❌ 不能说"dinocells 不行"——只能说它的**替身**在这个任务上学不到东西
- ❌ 不给 p 值：每化合物单孔，化合物身份 ≡ 孔身份
- ❌ 不能与论文数字比较：协议是自建的

---

## 5. 统一 IO 验证

```python
models.embed_all(names, {name: payload})   # 一次调用，7 个方法
```

4 种 input unit、7 种输出宽度，一次调度全部返回：

| 方法 | 输入 | 输出 |
|---|---|---|
| deepprofiler | 32×32 5ch | (N, 672) |
| ca_mae | 256×256 5ch uint8 | (N, 384) |
| cellclip | (N,5,1536) DINOv2 向量 | (N, 512) |
| o2vae | 64×64 剪影 | (N, 512) |
| unidino | 224×224 5ch | (N, **1920**) = 5×384 |
| dinocells | 96×96 5ch 通道堆叠（官方模型输入几何） | (N, 384) |
| morphogenie | 256×256 3ch | (N, **10**) |

验证项（`tests/test_models_v3_dispatch.py`，4 项全过）：
- 七个方法返回的字段集合**完全相同**
- provenance 正确区分 4 official / 3 surrogate
- 输出宽度符合各自 manifest 契约（unidino 是 `output_dim_per_channel × 5`）
- v2 下**无法**解析替身方法（安全属性）

**注意**：`embed_all` 的共享 kwargs（`masks`/`channels`/`assume`）会发给每个方法，
所以 benchmark 的正确用法是**各方法先 `prepare` 自己的张量，再统一 `embed_all`**。
这不是绕过，是接口本来的分工。

---

## 6. 改动清单

| 文件 | 改动 |
|---|---|
| `contract.py` | 新增 `weights_provenance`（`official`/`surrogate`）+ `WeightsProvenance` 类型 |
| `registry.py` | `_profile_members()` 剖面视图、`set_profile()`、`SURROGATES`（3）、`describe()` 带 provenance |
| `__init__.py` | facade 暴露 `active_profile`/`profile_names`/`set_profile` |
| `embedding.py` | 结果 metadata 带 provenance；`weights_sha256` 改为取适配器实际解析值 |
| `adapters/_surrogate.py` | 新增：共享 ViT、sha256、规格表 |
| `adapters/unidino.py` | 新增：逐通道 ViT-S/16 |
| `adapters/dinocells.py` | 新增：多通道 ViT-S/16（官方模型输入几何） |
| `adapters/morphogenie.py` | 新增：dVAE 编码器（输出 mu，确定性） |
| `scripts/train_surrogates.py` | 新增：VICReg 自监督训练 |
| `scripts/build_v3_surrogates.py` | 新增：经 `models.embed` 生成 3 个方法的嵌入 |
| `scripts/score_benchmark.py` | `--profile`；输出带 provenance 与 `surrogates` 列表 |
| `scripts/stability_check.py` | `--profile`；稳定前缀 + 互换对报告 |
| `tests/test_models_profiles.py` | 新增 11 项 |
| `tests/test_models_v3_dispatch.py` | 新增 4 项（缺 torch/权重则跳过） |

### 修掉的既有缺陷

1. **`weights_sha256` 可能为空**：原本只取 `manifest.weights.sha256`，
   自训练权重没有公布摘要就报空。改为优先取适配器**实际解析**到的摘要——
   官方方法取值不变，替身也能追溯
2. **稳定性判定过粗**：见 §4.3
3. **`describe()` 无 provenance**：任何从 `describe()` 渲染方法表的代码
   都分不出替身与官方

### 6.1 2026-10-26 修复：dinocells 的输入几何（F2）

审查（`docs/review_report_2026-10-26.md`）发现：官方 DINO4Cells 的 Cell Painting
数据集在**磁盘上**把通道叠成一条"磁带" `(H, W*(C+1))`，
但官方 dataset 类在 transform **之前**把它**折回** `(H, W, C)` 并
**丢掉 mask 通道**（`data_utils/cellpainting_dataset.py:23-47`、`:86`），
模型输入是 `(C, H, W)` 的多通道堆叠（`in_chans=num_channels`，
`run_get_features.py:87-91`）。

本项目原先实现的是官方 fold 的**逆运算**，把磁带直接喂给网络——
**折叠实现得完全正确，而正是这一点成了问题**：它复现了上游的**存储格式**，
却把它当成了**模型输入格式**。

修复：

| 项 | 改前 | 改后 |
|---|---|---|
| `input_encoding` | `'tape'` | `'stack'` |
| mask | 作为磁带的第 6 路**喂给模型** | `mask_mode='drop'`，**丢弃**（官方 `mode="drop"`） |
| 模型输入 | `(96, 576)` 单通道宽图 | `(N, 5, 96, 96)` 多通道堆叠 |
| 批处理 | 以 `tape-batch` 拒绝 | 允许（官方 `batch_size_per_gpu: 24`） |
| 替身 `in_chans` | 1 | 5 |
| 替身权重 | `5475ec885ae6b7d9…` | `5e172d8fed4876ff…`（重训） |

`input_encoding='tape'` 这个取值保留在类型里但**不再实现**：
`prepare` 以 `reason='unsupported-encoding'` 显式拒绝。
半实现的编码比没有编码更糟——它会返回一个数而不报错。

**影响**：v2 的四个官方方法**逐位不变**（已重算并逐位比对）；
v3 只有 dinocells 变化（权重与几何都换了），其数字与稳定性基线见 §4。

---

## 7. 可复现命令

```powershell
# 1) 训练三个替身（约 9 分钟，GPU）
python scripts\embedding_benchmark\train_surrogates.py --steps 300 --batch 32

# 2) 生成 3 个方法的嵌入（约 3 分钟）
python scripts\embedding_benchmark\build_v3_surrogates.py --force

# 3) 打分 + 稳定性（v3 = 7 方法）
python scripts\embedding_benchmark\score_benchmark.py --cells-dir E:\bbbc022\cells --profile v3
python scripts\embedding_benchmark\stability_check.py --cells-dir E:\bbbc022\cells --profile v3

# 4) 接口验证
python -m pytest tests\test_models_v3_dispatch.py -q
```

产物：`E:\bbbc022\cells\` 下 `benchmark_scores_v3.json`、`stability_v3.json`、
`v3_surrogate_embeddings.json`、`emb_{unidino,dinocells,morphogenie}.npy`；
权重在 `E:\cpc_weights\{unidino,dinocells,morphogenie}\surrogate_*.pt`。

> **2026-10-26**：dinocells 的替身权重因 F2 修复而**重训**
> （sha256 由 `5475ec885ae6b7d9…` 变为 `5e172d8fed4876ff…`），
> 旧的磁带版权重保留在
> `E:\cpc_weights\_backup_20261026\surrogate_dinocells_OLD_tape.pt`。
> 三个替身的摘要现已**钉进 manifest**，加载前强制校验。

---

## 8. 测试基线

| 环境 | 结果 |
|---|---|
| `.venv`（无 torch） | **323 passed**，11 skipped，14 failed（既有） |
| `ep_bench`（有 torch） | 5 个相关测试文件 **203 passed** |

14 项失败全部是本次改动之前就存在的（delivery/public_api/data_access 等 smoke 测试）。
**2026-10-26–27 的修复新增了 21 项回归测试**（诊断假阴性、uint8 值域守卫、
已缩放 float 的二次缩放守卫、尺寸守卫、通道数上限、磁带→堆叠几何、
值域语义、缓存失效、list 载荷值宽、`mask_mode='apply'`）。

---

## 9. 下一步该做什么

v3 证明了接口能驱动七个方法。但这**不改变** v2 路线图的优先级：

1. **补多孔数据**（每化合物 ≥2 孔）——否则任何排序都不能带显著性
2. **`Prepare` 与 `embed_all` 的分工写进文档**——现在只有测试体现
3. 若要真正比较 unidino/dinocells/morphogenie，**必须先拿到官方权重**。
   在那之前，v3 的这三个数字只能当作"接口通了"的证据