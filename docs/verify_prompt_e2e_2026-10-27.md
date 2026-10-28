# 验证提示词：Embedding 统一框架端到端复现（v2 + v3）

**本文档自包含。** 不需要之前的对话，不需要读仓库其它文档。
按它执行，你会得到一份「能跑 / 接口统一 / 数字可复现」的独立判断。

**这份文档的立场**：下面所有数字都是**被验证方的声明**。你的任务是自己跑出来、
自己比对，然后给出你自己的结论。凡是你没有亲手跑出来的项，一律记为
**未独立验证**——不要因为"文档说是这样"就写"通过"。

---

## 0. 铁律

1. **七个方法必须全测**。v2 的四个 + v3 的三个，一个都不能跳。
2. **不许只看分数**。"跑出数了"不等于"跑对了"。每个结果都要问：形状对吗？有限吗？
   权重摘要对得上吗？
3. **不许跳步**：先确认能跑（黑盒）→ 再确认接口统一（IO）→ 最后才看 benchmark。
   顺序颠倒会把"跑不起来"误诊成"性能差"。
4. **v2 与 v3 不可混比**。v3 多出的三个方法用的是**本地训练的替身权重**
   （`weights_provenance='surrogate'`），它们的分数**不是那篇论文的性能**。
5. **失败优于猜测**。不确定就写不确定。为了让检查通过而放宽断言，是这里最严重的错误。
6. **只读不改**。你在验证，不在修复。发现问题写进报告。

---

## 1. 这是什么项目

把 Cell Painting 领域若干 embedding 方法的**输入输出统一成一个接口**：

```python
from cellpaint_pipeline import models
emb = models.embed('ca_mae', crops)      # -> EmbeddingResult; .as_numpy() -> (N, 384)
```

各方法官方实现互不相容：吃 32×32 还是 256×256、吃像素还是吃预计算向量、
吃强度图还是吃核剪影，输出宽度从 10 到 1920。统一之后，**benchmark 调度只需要一个调用形状**。

| 版本 | 方法数 | 权重 | 用途 |
|---|---|---|---|
| `v2`（默认） | 4 | 全部官方 | 可发表 |
| `v3` | 7 | 4 官方 + 3 替身 | 只证明"接口能驱动全部方法" |

---

## 2. 怎么跑

### 2.1 一键全流程（推荐）

```bash
cd "/d/项目/CellPainting-Claw-v2"
bash scripts/embedding_benchmark/run_e2e.sh
```

它按顺序做六步，日志写到 `E:\bbbc022\cells\_e2e_run.log`，
每步打印 `exit=`，末尾有 SUMMARY。**任一步非 0 就停下**，不要把后续步骤当结果。

**本轮实测总耗时 11 分 16 秒**（2026-10-27 13:26:21 → 13:37:37，GPU：RTX 4060 Laptop 8 GB）。

### 2.2 分步（想看细节时）

```bash
export PATH="/c/Users/DELL/miniconda3/envs/torch_gpu/Library/bin:$PATH"
export PYTHONPATH="/d/项目/CellPainting-Claw-v2/src"
export CPC_WEIGHTS_ROOT="E:/cpc_weights"
cd "/d/项目/CellPainting-Claw-v2"
PY="D:/ep_bench/venv/Scripts/python.exe"
CELLS="E:/bbbc022/cells"

# 1) v2 四个官方方法
$PY scripts/embedding_benchmark/embed_dataset.py --cells-dir $CELLS --force
# 2) v3 三个替身
$PY scripts/embedding_benchmark/build_v3_surrogates.py --cells-dir $CELLS --force
# 3) IO 统一性（v2 / v3）
$PY scripts/embedding_benchmark/verify_io_uniformity.py --cells-dir $CELLS --profile v2 --limit 64
$PY scripts/embedding_benchmark/verify_io_uniformity.py --cells-dir $CELLS --profile v3 --limit 64
# 4) 打分
$PY scripts/embedding_benchmark/score_benchmark.py --cells-dir $CELLS --profile v2 --output $CELLS/b_s2.json
$PY scripts/embedding_benchmark/score_benchmark.py --cells-dir $CELLS --profile v3 --output $CELLS/b_s3.json
# 5) 稳定性
$PY scripts/embedding_benchmark/stability_check.py --cells-dir $CELLS --profile v3 --folds 4
```

---

## 3. 本轮实测基线（你的结果应当与之一致）

### 3.1 七方法黑盒

| 方法 | 权重来源 | dim | 单次调用返回 | 权重 sha256（前 16） |
|---|---|---|---|---|
| deepprofiler | official | 672 | (2, 672) | `87b756f579336d60` |
| ca_mae | official | 384 | (2, 384) | `f6e5f1c97101331b` |
| cellclip | official | 512 | (2, 512) | `66c2d32a44a3e7d6` |
| o2vae | official | 512 | (2, 512) | `aa70cfa8b080c9d9` |
| unidino | **surrogate** | **1920** | (2, 1920) | `a0265184661d511e` |
| dinocells | **surrogate** | 384 | **(1, 384)** | `5e172d8fed4876ff` |
| morphogenie | **surrogate** | **10** | (2, 10) | `491029978fef0969` |

**三个容易误判的地方**：

1. **unidino 是 1920，不是 384**。它逐通道过 ViT 再拼接（384 × 5 通道）。
2. **dinocells 是 (1, 384)，不是 (2, 384)**。单细胞磁带输入，N=1 是正确行为；
   成批它是支持的（见 §4 的 V6），但用单条磁带调用时就是 1 行。
3. **morphogenie 是 10，不是 256**。持久化的是 `c_dim=10`（官方 `MorphoGenie_Test.py:67`），
   `z_dim=256` 是 GAN 生成器的隐变量，不进嵌入。

### 3.2 v2 benchmark（3,857 细胞 / 54 化合物 / chance 0.0185）

| 方法 | mAP_z | kNN_z |
|---|---|---|
| cellclip | **0.1337** | **0.6451** |
| ca_mae | 0.1083 | 0.4454 |
| deepprofiler | 0.0691 | 0.1934 |
| o2vae | 0.0471 | 0.0869 |
| 原始像素基线 | 0.0508 | 0.1058 |
| 随机基线 | 0.0450 | 0.0848 |

### 3.3 v3 追加（**替身，禁止引用为方法性能**）

| 方法 | mAP_z | kNN_z | sil_z |
|---|---|---|---|
| unidino ❌ | 0.0646 | 0.2393 | −0.1891 |
| morphogenie ❌ | 0.0574 | 0.1281 | −0.7161 |
| dinocells ❌ | 0.0453 | 0.0900 | −0.1754 |

### 3.4 稳定性（4 折按孔划分）

```
fold 0: cellclip > ca_mae > deepprofiler > morphogenie > unidino > o2vae > dinocells
fold 1: cellclip > ca_mae > deepprofiler > unidino > morphogenie > o2vae > dinocells
fold 2: cellclip > ca_mae > deepprofiler > unidino > morphogenie > o2vae > dinocells
fold 3: cellclip > ca_mae > deepprofiler > unidino > morphogenie > o2vae > dinocells

稳定前缀 3/7：cellclip > ca_mae > deepprofiler
唯一跨折互换对：morphogenie / unidino
```

### 3.5 测试基线

| 环境 | 结果 |
|---|---|
| `.venv`（无 torch）全量 | **324 passed / 11 skipped / 14 failed** |
| `ep_bench`（有 torch）全量 | **333 passed / 15 failed** |
| `ep_bench` 4 个模型测试文件 | **114 passed** |

14 / 15 项失败全部在 CLI/交付/编排/下载冒烟层，**没有一项**在 embedding 测试文件里。
`ep_bench` 比 `.venv` 多的那一条是
`test_data_access_smoke.py::DataAccessSmokeTests::test_download_gallery_prefix_downloads_files`
（下载/缓存类，与 embedding 无关）。

---

## 4. 必须逐项验证的断言

每条都要你自己跑，给出「实测 → 判定」。

| # | 断言 | 怎么验 | 期望 |
|---|---|---|---|
| **V1** | 七个方法都能返回结果 | §3.1 的七个调用 | 七个都返回，无异常 |
| **V2** | 输出都是二维浮点且有限 | 对每个结果取 `as_numpy()` | `(N, dim)`，`np.isfinite` 全真 |
| **V3** | 宽度与契约一致 | 对比 §3.1 | 672/384/512/512/**1920**/384/**10** |
| **V4** | 每个结果都带权重摘要 | `result.weights_sha256` | 非空，且与 §3.1 一致 |
| **V5** | provenance 标注正确 | `result.metadata['weights_provenance']` | 4 official + 3 surrogate |
| **V6** | dinocells 支持成批 | `prepare(man, bhwc_2cells)` | `(2, 5, 96, 96)`，不报 tape-batch |
| **V7** | dinocells 的 mask 被丢弃 | 带 mask / 不带 mask 各跑一次 | 逐位相等，`max|diff| = 0` |
| **V8** | v2 下够不到替身 | `set_profile('v2')` 后 `embed('unidino',…)` | 抛 `UnknownModel` |
| **V9** | 诊断不再假阴性 | `probe('dinocells')` | `runnable=True`，`requires=['torch']` |
| **V10** | `runnable()` 是 7 个 | 调一次 | 不是 6 个 |
| **V11** | 16-bit 方法拒 8-bit | uint8 喂 unidino | 抛 `IncompatibleInput(reason='value-range')` |
| **V12** | 尺寸错误是类型化拒绝 | 256×256 喂 deepprofiler | 抛 `IncompatibleInput(reason='spatial-size')`，**不是** Keras `ValueError` |
| **V13** | IO 统一性 | `verify_io_uniformity --profile v3` | 7/7，`identical field sets: True` |
| **V14** | 结果对象字段一致 | 七个 `to_dict()` 的键集合 | 完全相同 |
| **V15** | 一切可复现 | 见 §5 | 七个方法逐位一致 |

---

## 5. 可复现性（最硬的一条）

仓库预置了两个对照目录：

| 目录 | 含义 |
|---|---|
| `E:\bbbc022\cells\_emb_pre_e2e\` | 本轮重算**之前**的七个嵌入 |
| `E:\bbbc022\cells\_emb_backup_before_f2\` | dinocells 几何修复（F2）**之前**的七个嵌入 |

跑完 §2 之后，逐个比对：

```python
import numpy as np, pathlib
cells = pathlib.Path(r'E:\bbbc022\cells')
for m in ('deepprofiler','ca_mae','cellclip','o2vae','unidino','dinocells','morphogenie'):
    cur = np.load(cells / f'emb_{m}.npy')
    pre = np.load(cells / '_emb_pre_e2e' / f'emb_{m}.npy')
    old = np.load(cells / '_emb_backup_before_f2' / f'emb_{m}.npy')
    print(m, 'vs pre_e2e:', np.array_equal(cur, pre),
          '| vs before_f2:', np.array_equal(cur, old) if cur.shape == old.shape else 'shape differs')
```

**本轮实测结果**：

| 方法 | vs `_emb_pre_e2e` | vs `_emb_backup_before_f2` |
|---|---|---|
| deepprofiler / ca_mae / cellclip / o2vae | **逐位相等** | **逐位相等** |
| unidino | **逐位相等** | 不同（max\|d\|≈5.42，替身重训过） |
| dinocells | **逐位相等** | 不同（max\|d\|≈6.19，几何修正 + 重训） |
| morphogenie | **逐位相等** | 逐位相等 |

**判据**：

- 四个官方方法若**不**与这两个目录逐位相等 → **回归**，立即上报。
- 三个替身若与 `_emb_pre_e2e` 不等 → 回归（它们被声明为确定性）。
- 替身与 `_emb_backup_before_f2` 不等是**预期的**，不是回归。

---

## 6. 已知边界（不是缺陷，但结论里必须体现）

| # | 边界 | 后果 |
|---|---|---|
| E1 | **不给 p 值** | 每化合物单孔，化合物身份 ≡ 孔身份 |
| E2 | v3 三个数字是替身的 | 只能当"接口驱动成功"的证据 |
| E3 | 不可与论文数字比较 | 协议是自建的 |
| E4 | o2vae 是形状模型且存在域偏移 | 官方权重训练于**细胞**剪影，这里喂的是**核**剪影 |
| E5 | silhouette 全为负 | **"嵌入形成可分离簇"在本数据上不成立** |
| E6 | mock-AUC 多在 0.47–0.54（随机 0.4996） | 不宜用于方法排序 |
| E7 | 只有 deepprofiler 做过真实数据逐位对拍 | 其余是"对照官方公式"的推理验证 |
| E8 | 契约不校验输入数据产物 | `cellclip_vectors.npy` 被换掉不会报错 |
| E9 | 随机基线 4 折区间 0.0683–0.1109 | 见下方逐折表：o2vae、morphogenie、dinocells、原始像素**与它重叠** → 这几个名次不被数据支持 |

**E9 的逐折实测**（`stability_v3_e2e.json`，4 折，按孔划分）：

| 项 | fold 0 | fold 1 | fold 2 | fold 3 | 区间 |
|---|---|---|---|---|---|
| 随机基线 | 0.0953 | 0.0683 | 0.1109 | 0.0931 | **0.0683–0.1109** |
| 原始像素 | 0.1102 | 0.0823 | 0.1297 | 0.1126 | 0.0823–0.1297 ⚠️ 重叠 |
| dinocells ❌ | 0.0963 | 0.0700 | 0.1114 | 0.0918 | 0.0700–0.1114 ⚠️ 重叠 |
| o2vae（官方） | 0.1015 | 0.0747 | 0.1179 | 0.0972 | 0.0747–0.1179 ⚠️ 重叠 |
| morphogenie ❌ | 0.1284 | 0.1092 | 0.1417 | 0.1240 | 0.1092–0.1417 ⚠️ 重叠 |
| unidino ❌ | 0.1256 | 0.1176 | 0.1451 | 0.1335 | 0.1176–0.1451 高于 |
| deepprofiler | 0.1600 | 0.1298 | 0.1687 | 0.1697 | 0.1298–0.1697 高于 |
| ca_mae | 0.1964 | 0.1865 | 0.1943 | 0.1922 | 0.1865–0.1964 高于 |
| cellclip | 0.2556 | 0.2563 | 0.2547 | 0.2565 | 0.2547–0.2565 高于 |

**关于 E9**：o2vae 是 v2 的**官方**方法，它的折区间与随机基线重叠。
所以"o2vae 低于原始像素"这个排序**不被本数据支持**，必须出现在结论里。
（同时注意：原始像素基线自己也与随机重叠——这是同一个数据集的固有噪声水平，
不是某个方法的缺陷。）

---

## 7. 失败怎么分类

看到不一致时，先分三类，不要直接判"回归"：

| 类别 | 特征 | 处理 |
|---|---|---|
| **环境** | 只在某个解释器出现；涉及下载/网络/路径 | 换解释器复现一次，确认是否只在该环境 |
| **声明错误** | 文档数字与实测不符，但行为正确 | 记为"文档不准"，不是代码缺陷 |
| **回归** | 四个官方方法嵌入或分数变了；替身与 `_emb_pre_e2e` 不等 | 立即上报，附实测与期望 |

---

## 8. 交付格式

一份 `docs/verify_report_<日期>.md`，必须含：

1. **V1–V15 逐项表**：每条写「怎么验 → 实测 → 判定」
2. **七个 sha256** 与 §3.1 的对照
3. **v2 四个数字**与 §3.2 的对照（允许 ±0.0001 的四舍五入）
4. **v3 三个数字**与 §3.3 的对照，并标注 surrogate
5. **§5 的逐位比对表**（七个方法 × 两个备份目录）
6. **测试三项**（`.venv` / `ep_bench` 全量 / 4 文件）与 §3.5 的对照，
   并把失败项逐条列出、判断是否触及被审面
7. **稳定性**：稳定前缀长度 + 互换对
8. **结论三档**，且先给判据再给结论：
   - **能跑** ⇔ V1–V5 全过
   - **接口统一** ⇔ V13、V14 过，且七个方法输出宽度符合各自契约
   - **可复现** ⇔ §5 的七个方法全部逐位相等
9. **你未能独立验证的项**（不要留空，没有就写"无"）

---

## 9. 禁止事项

| 禁止 | 原因 |
|---|---|
| 只跑部分方法 | 要求是七个全测 |
| 把替身分数当方法性能 | 权重不是官方的 |
| 给 p 值或"显著优于" | 单孔数据结构上不成立 |
| 用"看起来对"代替运行 | 本项目的缺陷都是静默型的 |
| 跳过黑盒直接看 benchmark | 会把"跑不起来"误诊成"性能差" |
| 修改仓库源码让检查通过 | 你在验证，不在修复 |
| 把 dinocells 的 (1,384) 判成 bug | 单细胞磁带输入，N=1 是正确的 |
| 把 unidino 的 1920 判成 bug | 逐通道拼接，契约如此 |

---

## 10. 环境

| 项 | 值 |
|---|---|
| 解释器 | `D:\ep_bench\venv`（torch 2.5.1 / TF 2.15.1 / numpy 1.26.4）、`D:\项目\CellPainting-Claw-v2\.venv`（无 torch） |
| 权重根 | `E:\cpc_weights`（`CPC_WEIGHTS_ROOT`） |
| 数据 | `E:\bbbc022\cells`（`cells_ca.npy` / `cells_dp.npy` / `cells_o2.npy` / `cellclip_vectors.npy` / `meta.csv`） |
| 官方源码 | `D:\official_src\` |
| GPU | RTX 4060 Laptop 8 GB（注意：可能有其它进程占用，ca_mae 在空闲时 191.7 s、拥挤时曾达 1341 s） |

**ca_mae 的耗时受 GPU 争用影响极大**。看到它从 190 s 变成 1300 s 时，
先怀疑 GPU 被占，而不是代码变慢。