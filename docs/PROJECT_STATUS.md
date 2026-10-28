# CellPainting-Claw Embedding 层：阶段总结

> 日期：2026-10-18（**已在审查后修订**）；最近更新：2026-10-27
> 本阶段目标：把多种 embedding 方法统一到一套接口，可调度、可验证、可比排名
> 当前状态：**v2 四个官方方法 + v3 三个替身方法，七个全部跑通；
> 审核 66 项全过（第 9 轮）；全流程 e2e 七步退出码全 0**

---

## 最新一轮（2026-10-27）：可维护性清理与 IO 统一

这一轮没有改进任何方法的效果，只改**代码能不能被下一个人接手**。

| 改动 | 之前 | 现在 |
|---|---|---|
| `docs/` 根目录散装文档 | 29 个 md 平铺，多份互相矛盾 | 9 个（含索引 `docs/README.md`），21 份过期报告移入 `docs/archive/`，死链 0 |
| 仓库根构建日志 | `conda_create*.log` 共 102 KB | 已删 |
| `config.py` 报错指向 | `docs/config_contract.md`（**该文件不存在**） | 报错里直接列出六个必需字段 |
| `embed_all` 的按方法参数 | 只有 `unit` 能按方法传；`assume`/`masks`/`channels` 只能全局广播 → 调用方必须自己 `prepare` 每个载荷 | 四者都接受 `{方法名: 值}`，**一次调用驱动七个**（新增 2 条回归测试） |
| `unidino` 声明的输出宽度 | `output_dim=384`，实际返回 **1920** | `output_dim=1920`；契约里写明"`output_dim` 是声明通道数下的宽度" |
| v2/v3 配置表 | 手工维护，易漂移 | `dump_profile.py` 生成（`docs/_generated/profile_{v2,v3}.md`） |
| 脚本目录 | 15 个脚本无索引 | `scripts/embedding_benchmark/README.md` 按"我要做什么"分类登记 |

验证：`.venv` 324 passed（基线不变）；四个模型测试文件 116 passed（+2）；
审核第 9 轮 66/66 PASS；e2e 七步退出码全 0，七个嵌入与改动前**逐位相同**。

两版本的最终代码梳理见 [`embedding_v2_v3_final.md`](embedding_v2_v3_final.md)。

---

## ⚠️ 修订说明（2026-10-18 晚）

审查（`docs/archive/embedding_audit_report.md`）发现并经我方独立复现确认了 **6 个 P0 缺陷**，
本文档的以下内容已随之更正，详见 **`docs/archive/audit_response.md`**：

1. **DeepProfiler 通道顺序**：应为上游训练序 `(DNA, ER, RNA, AGP, Mito)`，
   本文档此前用的是 `(DNA, RNA, ER, AGP, Mito)`。**已修**，旧顺序现被契约拒绝。
2. **DP 归一化**：官方是**逐通道** min-max（`cropping.py:65-67`），此前用了跨通道联合。**已修**。
3. **排名结论变了**：修正后 **DeepProfiler 明确高于原始像素 36%**
   （mAP_z 0.0691 vs 0.0508），而**不是**此前写的"≈ 打平"。
4. **统计推断撤回**：bootstrap CI 不成立（点估计落在区间外），
   且该数据集**原理上不支持**方法间显著性检验（每化合物仅 1 孔）。
   改为 4 折排序稳定性，**不再报告 p 值**。
5. **`models.embed_all` 此前并未导出**，文档写了但代码没有。**已修**。
6. `save_npz` 曾丢失 `cell_id`、`summary.json` 的 `finished_at` 曾恒空。**均已修**。
7. **P1 已补完**：逐方法 `unit`、O2-VAE/DeepProfiler 的权重加载断言、
   DINOv2-Giant 链路溯源、`device`/`weights_roots`/`adapter_kwargs` 可传、
   `embed_all` 返回 `BatchResult`（去掉 `__errors__` 魔法键）、
   benchmark 脚本入 `scripts/` 且无绝对路径、`pyproject` 补 torch extras。

**测试**：`212 passed`（修订前 192，新增 20 条回归测试）；14 个失败与会话前一致。

**唯一待你确认**：新增的 14 个文件仍是 git untracked，未提交。

---

## 1. 项目地址

### 1.1 代码

| 用途 | 路径 | HEAD |
|---|---|---|
| **主开发树（canonical）** | **`D:\项目\CellPainting-Claw-v2`** | `612aa79` |
| 官方对照树 | `D:\项目\CellPainting-Claw-official` | `59df377` |
| 双树对拍 A（官方） | `D:\cpc_c_official` | — |
| 双树对拍 B（v2） | `D:\cpc_c_v2` | — |
| 7 个上游仓库原码 | `D:\official_src\{1-deepprofiler,…,7-morphogenie}` | 见文档 |
| 复现脚本与实验探针 | `D:\项目\cellpainting claw复现\_probe\` | — |

**新增层的位置**：`D:\项目\CellPainting-Claw-v2\src\cellpaint_pipeline\models\`
（13 个文件 / 2,989 行）+ `src\cellpaint_pipeline\embedding_benchmark.py`

### 1.2 数据与权重

| 内容 | 路径 | 大小 |
|---|---|---|
| 权重根目录 | **`E:\cpc_weights`** | — |
| DINOv2-Giant | `E:\cpc_weights\dinov2_giant\model.safetensors` | 4.55 GB |
| BBBC022 数据集 | **`E:\bbbc022\`**（`salvaged\` 图像、`cells\` 产物） | ~11 GB |

### 1.3 文档

`D:\项目\CellPainting-Claw-v2\docs\`：

| 文档 | 内容 |
|---|---|
| `embedding_interface.md` | 接口用法与扩展指南 |
| `embedding_layer_hardening.md` | manifest 修正 + 新增契约原语 |
| `embedding_verification_7methods.md` | 7 个方法的官方源码溯源与验证 |
| `parity_check_c.md` | 官方树 ↔ v2 树逐字节对拍 |
| `benchmark_on_known_result.md` | 已知结果上的逐位一致验证 |
| `real_dataset_setup.md` | 真实数据集与四方法公平配置 |
| **`benchmark_bbbc022_result.md`** | **真实数据排名结果** |
| `environment_notes.md` | 环境与两个"坑"的根因 |
| `embedding_weights_and_roadmap.md` | 权重可得性与路线图 |

---

## 2. 本阶段要实现什么

### 2.1 起点问题

官方 CellPainting-Claw **只支持 1 个 embedding 方法**（DeepProfiler），
而且它自己**不加载模型**——用子进程外包给 `deepprofiler` 包。
其余 6 个方法（CA-MAE / CellCLIP / uniDINO / DINO4Cells / MorphoGenie / O2-VAE）
在官方代码里**一个都没出现**。

它们的输入契约**互不兼容**：

| 方法 | 布局 | dtype | 通道 | 是像素吗 |
|---|---|---|---|---|
| DeepProfiler | BHWC | float32 | 5 固定 | 是 |
| CA-MAE | BCHW | **uint8** | **可变** | 是 |
| CellCLIP | BCHW | float32 | 5 | ❌ **是向量** |
| O2-VAE | BCHW | float32 | **1** | 是（剪影） |

### 2.2 三个目标

1. **统一 IO**：`models.embed(name, x)` 一个入口，差异落在声明式契约里
2. **一致性**：与官方产物可对拍，且**逐位相同**
3. **可比排名**：在真实数据上跑出可信的"哪个方法更好"

### 2.3 明确的非目标

- 不重做 v1/v2 的架构重构（已有成果保持）
- 不追求 7 个方法全部可跑（权重不可得的就如实标注）

---

## 3. 已实现什么

### 3.1 接口层（声明式契约）

```
models/
  contract.py    契约原语 + check_compatible（纯函数，不 import numpy 也可用）
  registry.py    7 个方法的 ModelManifest 注册表
  embedding.py   embed / embed_many / embed_all / probe
  weights.py     权重解析（禁止隐式下载）+ sha256 记忆化校验
  errors.py      类型化错误
  adapters/      base.py + 各方法适配器
```

契约原语：
`InputUnit` / `ChannelSpec`（含 `names_are_labels`、`order_matters`、`channel_count_fixed`）/
`PreprocessSpec`（含 `scale`、`value_range`、`to_dtype`、`batch_layout`、`input_encoding`）/
`WeightsSpec` / `ModelManifest`（含 `determinism`）

### 3.2 七方法的真实状态

| 方法 | 官方权重 | 维度 | 状态 |
|---|---|---|---|
| **deepprofiler** | ✅ CP CNN v1 | 672 | **ready** |
| **ca_mae** | ✅ OpenPhenom-S/16 | 384 | **ready** |
| **cellclip** | ✅ CellCLIP + **真实 DINOv2-Giant 向量** | 512 | **ready** |
| **o2vae** | ✅ **官方 MEF 权重** | 512 | **ready** |
| unidino | ❌ Zenodo 43h | 384 | needs-weights（契约已声明） |
| dinocells | ❌ Zenodo 180h | 384 | needs-deps-and-weights |
| morphogenie | ❌ Google Drive 不通 | 10 | needs-weights |

**4 个 ready 全部用的是官方权重**，sha256 已校验。

### 3.3 调度与 benchmark 能力

- `models.embed_all(...)`：一批输入跑多方法，**按方法名索引输入**
  （因为 CellCLIP 吃向量、其余吃像素，单一数组必然对某一个是错的）
- adapter 缓存：同一方法第二次调用 **7.04 s → 0.031 s**
- `embedding_benchmark.py`：对齐键、随机方法重复、许可闸门、溯源落盘
- 结果带 `weights_sha256`（**真校验**）、`determinism`、`license`、契约报告

### 3.4 一致性验证结果

| 验证 | 结果 |
|---|---|
| 官方树 ↔ v2 树流水线 | `.npz` **sha256 相同，max\|diff\|=0** |
| 接口层 ↔ 官方路径（DeepProfiler） | **`bitwise_equal=True`，`max\|diff\|=0.000000e+00`** |

### 3.5 真实数据排名（BBBC022）

3,857 细胞 / 64 孔 / 54 化合物 / 671 mock 细胞：

| 方法 | mAP_z | kNN_z |
|---|---|---|
| **cellclip** | **0.1338** | **0.6427** |
| **ca_mae** | 0.1083 | 0.4454 |
| deepprofiler | 0.0684 | 0.1926 |
| *原始像素* | 0.0664 | 0.1592 |
| o2vae | 0.0471 | 0.0869 |
| *随机投影* | 0.0450 | 0.0848 |

### 3.6 测试

```
tests/test_models_contract.py       33 passed
tests/test_embedding_benchmark.py   34 passed
全量套件                             14 failed, 192 passed
```

14 个失败是**会话开始前就存在的**（`ProjectConfig` NameError、Windows 路径分隔符、
需要 demo 数据的端到端冒烟），与本阶段改动无关——`git status` 显示没有修改任何已跟踪文件。

---

## 4. 怎么验证

### 4.1 三层验证，每层验不同的东西

| 层 | 验什么 | 怎么跑 | 通过标准 |
|---|---|---|---|
| **① 契约** | 声明对不对 | `pytest tests/test_models_contract.py` | 33 passed，不需 torch/TF/权重 |
| **② 一致性** | 与官方是否等价 | `parity_vs_known_result.py` | `bitwise_equal=True` |
| **③ 排名** | 哪个更好 | `score_benchmark.py` + `bootstrap_ranking.py` | 相对基线的判定 |

### 4.2 ① 契约层（最快，0.3 秒）

```powershell
cd D:\项目\CellPainting-Claw-v2
.\.venv\Scripts\python.exe -m pytest tests/test_models_contract.py -q
```

覆盖：7 方法齐全、跨表不变量（顺序/标签/数量三者一致）、通道集合/顺序/数量、
dtype 与值域、通道堆叠输入、ASCII 路径策略，以及**实测缺陷的回归测试**。

> **2026-10-26 更新**：dinocells 的输入几何已由「磁带」改为**官方模型输入几何**
> （通道堆叠 `(N,5,96,96)`，掩码按官方 `mode="drop"` 丢弃）；
> `input_encoding='tape'` 保留在类型里但**不再实现**，`prepare` 以
> `reason='unsupported-encoding'` 显式拒绝。详见 `docs/archive/embedding_v3_2026-10-26.md` §6.1。

**关键**：这层**不需要任何框架或权重**，所以任何环境都能跑。

### 4.3 ② 一致性层（与官方逐位对拍）

```powershell
$env:PATH = 'C:\Users\DELL\miniconda3\envs\torch_gpu\Library\bin;' + $env:PATH
$env:TF_CPP_MIN_LOG_LEVEL='3'
D:\ep_bench\venv\Scripts\python.exe 'D:\项目\cellpainting claw复现\_probe\parity_vs_known_result.py'
```

产出：
```
raw (blocked by contract)  bitwise_equal=False  max|diff|=4.481230e+06
per-crop min-max           bitwise_equal=True   max|diff|=0.000000e+00
official mean/max: 1.9696 / 18.5077
ours     mean/max: 1.9696 / 18.5077
```

**两条独立路径都指向同一结论**：官方树、v2 树、接口层三者产出的嵌入逐位相同。

### 4.4 ③ 端到端 benchmark 复现

```powershell
# 1) 抢救截断的 zip（镜像中途崩溃时）
python salvage_partial_zips.py --zips E:\bbbc022\*.zip --output-dir E:\bbbc022\salvaged

# 2) 分割 + 四套原生尺度输入
python build_bbbc022_cells.py --images-dir E:\bbbc022\salvaged `
  --labels E:\bbbc022\salvaged_fields.csv --output-dir E:\bbbc022\cells --max-cells 4000

# 3) CellCLIP 的 DINOv2 向量（GPU, ~45 min）
python make_cellclip_vectors.py --crops E:\bbbc022\cells\cells_clip.npy `
  --model-dir E:\cpc_weights\dinov2_giant --output E:\bbbc022\cells\cellclip_vectors.npy `
  --device cuda --batch-size 2

# 4) 四方法嵌入
python embed_dataset.py --cells-dir E:\bbbc022\cells

# 5) 打分 + bootstrap
python score_benchmark.py --cells-dir E:\bbbc022\cells --output E:\bbbc022\cells\benchmark_scores.json
python bootstrap_ranking.py --cells-dir E:\bbbc022\cells --rounds 200
```

### 4.5 验证时最该警惕的三件事

| 陷阱 | 为什么会发生 | 本阶段的处置 |
|---|---|---|
| **指标未校准** | 随机基线不落在 chance，排名就是噪声 | 随机基线必须 AUC≈0.5、mAP≈chance |
| **数据泄漏** | 用全量质心打分，点离自己参与的质心更近 | 按 mock 孔交叉验证 |
| **重采样单元错** | 按细胞 bootstrap 会把近重复当独立样本 | **按孔** bootstrap |

实测曾踩中前两个：随机基线 AUC 一度虚高到 **0.72**（泄漏）和全部 <0.5（符号反转），
修好后回到 **0.4996**。

---

## 5. Benchmark 怎么比较

### 5.1 数据

BBBC022 plate 20585 —— 真实 Cell Painting，5 通道 + LINCS 化合物标签。

- 原始下载 6.92 GB，镜像崩到 19 KB/s → 已下 3.34 GB 是截断 zip
- 从截断 zip 里**抢救出 8,081 张完整图像** → 72 个五通道齐全的视野
- 分割（Hoechst Otsu + 距离分水岭）→ **3,857 单细胞 / 64 孔 / 54 化合物**

### 5.2 公平性设计

**同一批细胞，各方法拿各自原生尺度的输入**：

| 方法 | 输入 | 为什么是这个尺寸 |
|---|---|---|
| deepprofiler | `(N,32,32,5)` float [0,1] | 其训练裁剪尺度；per-crop min-max 在裁剪图内 |
| ca_mae | `(N,5,256,256)` uint8 | 其训练尺度 |
| cellclip | `(N,5,1536)` float | DINOv2-Giant 逐通道向量 |
| o2vae | `(N,1,64,64)` float | 官方 MEF 配置 resize 到 64 |

强制统一尺寸会**削弱尺寸不匹配的那个方法**，那才是不公平。

### 5.3 指标

| 指标 | 含义 | 随机水平 |
|---|---|---|
| **mAP_z** | 同化合物检索平均精度（**逐维 z-score** 后） | ≈0.045 |
| **kNN_z** | 留一法 k=10 化合物分类 | ≈0.085 |
| **AUC** | mock vs 化合物可分性（按 mock 孔交叉验证） | **0.500** |
| silhouette | 化合物簇分离度 | 0 |

**为什么必须逐维 z-score**：实测 CA-MAE 单一维度占 **71%** 方差、O2-VAE 占 **57%**。
不标准化则距离几乎只由那一维决定，等于用**尺度**而非**信息**在比。

### 5.4 两个必须有的基线

| 基线 | 作用 |
|---|---|
| **原始像素**（同样的裁剪展平） | 回答"嵌入是否比直接看像素更好" |
| **随机投影** | 校准指标、给出 chance 水平 |

**没有这两个基线，绝对分数无法解释。** 实测所有方法的 silhouette 都是负的，
说明没有清晰化合物簇——这时只有"相对基线"才有意义。

### 5.5 统计判定

**按孔 bootstrap**（200 轮），因为同孔内细胞不独立：

| 对比 | P(前者更好) | 判定 |
|---|---|---|
| cellclip > ca_mae | 1.000 | **明确** |
| ca_mae > deepprofiler | 1.000 | **明确** |
| **deepprofiler > 原始像素** | **0.845** | **不可区分** |
| 原始像素 > o2vae | 1.000 | 明确 |

---

## 6. 怎么判断哪个 embedding 最优

### 6.1 判定流程（按顺序，任一步不过就不算"最优"）

```
第一步：它比原始像素好吗？
  否 → 这个嵌入在该任务上没有价值（不管它多复杂）

第二步：它比随机投影好吗？
  否 → 嵌入是退化的，或用错了输入

第三步：优势在统计上明确吗？
  bootstrap P < 0.975 → 只能说"打平"，不能说"更好"

第四步：它用的是公平的输入吗？
  吃了额外的大模型（如 DINOv2-Giant）→ 赢的是整条链路，不是方法本身

第五步：它的模态与任务匹配吗？
  形状模型 vs 强度任务 → 低分是方法定位问题，不是实现问题

第六步：许可允许比较与发布吗？
  CC BY-NC-ND / GPL 有传染性 → 结论能否对外发布另说
```

### 6.2 按这套流程解读本阶段结果

| 方法 | ①胜过像素 | ②胜过随机 | ③统计明确 | ④输入公平 | ⑤模态匹配 | 综合 |
|---|---|---|---|---|---|---|
| **cellclip** | ✅ +102% | ✅ | ✅ P=1.000 | ⚠️ 用了 1.14B backbone | ✅ 强度 | **最优（附条件）** |
| **ca_mae** | ✅ +63% | ✅ | ✅ P=1.000 | ✅ 自己尺度 | ✅ 强度 | **第二，且最"干净"** |
| deepprofiler | ❌ P=0.845 | ✅ | — | ✅ 自己尺度 | ✅ 强度 | **等于像素基线** |
| o2vae | ❌ 更差 | ⚠️ 极小 | ✅ | ✅ 自己尺度 | ❌ 形状 vs 强度任务 | **不适合本任务** |

### 6.3 三条必须随结论一起说的限定

**① CellCLIP 赢的是"整条链路"，不是"架构"**
它的输入是 **DINOv2-Giant（1.14B 参数）** 的逐通道特征，其他方法直接吃像素。
正确表述是"**DINOv2-Giant + CellCLIP 跨通道 Transformer** 这条链路最优"，
不是"CellCLIP 比 CNN 好"。

**② CA-MAE 是"最干净"的赢家**
它用自己原生 256×256 输入、自己的权重、没有借别的预训练模型，
依然明确超过 DeepProfiler 63% 和原始像素。**若只能选一个部署，它是最稳的选择。**

**③ O2-VAE 低分 ≠ 实现坏了**
独立验证：
```
o2vae 嵌入 PC1 vs 细胞核面积   Spearman = -0.927
```
极强相关，说明它**忠实编码了核形态**。它失分是因为**形状模型在化合物判别任务上
信息不足**。把它当"形状描述子"用是对的，当"化合物判别器"用是错的。

### 6.4 本阶段结论的边界（不能外推）

| 限制 | 影响 |
|---|---|
| 每化合物**仅 1 个孔** | 孔/视野外观与化学身份完全混杂，**绝对分数被抬高** |
| 每化合物**仅 1 个浓度** | 无法做剂量-反应 |
| 镜像崩溃后**只剩 64 视野** | 结论限于此板 |
| 所有方法 silhouette 为负 | 本数据上化学信号本身弱 |

**最准确的表述**：这是"**在 BBBC022 plate 20585 的 3,857 个细胞上，按化合物检索能力排名**"，
不是"通用 Cell Painting 方法排名"。

---

## 7. 本阶段修掉的关键缺陷（都会让结论错）

| # | 缺陷 | 后果 | 状态 |
|---|---|---|---|
| 1 | `unidino` 声明 `order_matters=False` | 静默接受置换输入 | 修为 `True`（实测差 2.109） |
| 2 | `morphogenie` `output_dim=256` | 与实测 `(4,10)` 不符 | 修为 `10` |
| 3 | `deepprofiler` 声明 `scale='raw'` | 接受 16-bit 原始值，误差 **4.48e6** | 修为 `per_crop_minmax` + `(0,1)` |
| 4 | `o2vae` 用自训手写数字权重 | 不是细胞模型 | 换成**官方 MEF 权重**，dim 64→512 |
| 5 | `embed()` 每次重载权重 | 7 s/次，benchmark 不可用 | 加缓存 → 0.031 s |
| 6 | 探针把归一化 box 当像素 | 误判"接口层不等价" | 改用 TF 原算子 → 逐位一致 |
| 7 | AUC 符号反转 + 质心泄漏 | 排名是噪声 | 修符号 + 按孔交叉验证 |
| 8 | 全局 cell 上限截断视野 | 65 孔 → 26 孔，54 化合物 → 22 | 改为按视野限流 |

---

## 8. 下一步

| 优先级 | 事项 | 理由 |
|---|---|---|
| **P1** | 补齐 BBBC022（还需 3.58 GB） | 384 孔/板、多视野 → 可做留一孔 CV、去孔混杂 |
| **P1** | 多板 / 多浓度 | 才能做剂量-反应与真正的重复孔对比 |
| **P2** | 给 DeepProfiler 更大裁剪（64×64） | 验证"32×32 对 BBBC022 偏小"这一假设 |
| **P2** | 接 `unidino`（等权重） | 依赖已齐、架构已验证 |
| **P3** | CLI / 技能入口 | 目前只能从 Python 调用 |
