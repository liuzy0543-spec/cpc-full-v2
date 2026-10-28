# Embedding 层现状核查（2026-10-20）

回答三个问题：**权重齐不齐、已嵌入的能不能跑、IO 接口是否真统一。**
所有数字均为本次实测，不是引用旧记录。

---

## 1. 权重可得性：4 个可跑，3 个当前不可得

### 1.1 已就位（sha256 全部校验 MATCH）

| 方法 | 文件 | 大小 | 来源 | 状态 |
|---|---|---|---|---|
| deepprofiler | `Cell_Painting_CNN_v1.hdf5` | 19,236,600 B | Zenodo 7114558 | ✅ |
| ca_mae | `openphenom/model.safetensors` | 712,434,294 B | Recursion OpenPhenom | ✅ |
| cellclip | `cellclip/model.safetensors` | 1,799,564,952 B | suinleelab/CellCLIP | ✅ |
| o2vae | `o2vae_mefs_official.pt` | 7,428,429 B | 上游仓库 `wandb/pretrained_models/` | ✅ |
| cellclip 链 | `dinov2_giant/model.safetensors` | 4,546,005,432 B | facebook/dinov2-giant | ✅ |

`resolve_weights` + `verify_sha256` 独立复核：4 个全部 `MATCH`，不是"manifest 自称"。

### 1.2 不可得（本次重新实测，不是沿用旧结论）

| 方法 | 官方源 | 大小 | **本次实测速度** | 预计耗时 | 有镜像吗 |
|---|---|---|---|---|---|
| unidino | Zenodo 14988837 `checkpoints.zip` | 1,379,562,715 B | **6.0 KB/s** | **≈62 h** | ❌ 无官方镜像 |
| dinocells | Zenodo 8061428 `Cell_Painting_data.tar.gz` | 6,896,284,400 B | 同上（同主机） | **≈312 h** | ❌ 无 |
| morphogenie | Google Drive `Models.7z` | 未知 | 不可达 | — | ❌ 无 |

**测速方法**：`curl -r 0-2097151`（Range 请求 2 MB）+ 45 s 上限，实收 276,360 B
÷ 45.1 s = **6.0 KB/s**，HTTP 206。比旧记录（9–10 KB/s）**更慢**。

**镜像排查**：官方 README 只给 Zenodo / Google Drive；HuggingFace 上
`dinocells`/`DINO4Cells`/`morphogenie`/`MorphoGenie` 搜索结果为 **0**。
唯一的 `unidino` 结果是第三方 `hiendang7613/UniDINO_7_7_25_base_base_448`
（4 次下载、0 赞、无 license、输入 **448** 而官方是 224）——**不是官方 checkpoint，
已拒绝使用**；用它会让"权重与官方一致"这条前提失效。

**结论**：按"短时间能下载"的标准，**没有**可以补进来的方法。
三个方法保持"仅声明"是如实标注，不是没做完。

### 1.3 权重配置是否硬编码

`models/weights.py` 的解析顺序（不是单一硬编码路径）：

```
1. 调用方显式传 roots=...        （最高优先级）
2. 环境变量 $CPC_WEIGHTS_ROOT
3. DEFAULT_WEIGHTS_ROOTS 默认列表：E:\cpc_weights, D:\cpc_weights, D:\模型权重
```

**判定**：默认值里确实写了盘符，但**可被环境变量或显式参数覆盖**，语义等同
`HF_HOME`/`TORCH_HOME` 这类约定，不算缺陷。真正要紧的是**代码里没有散落的绝对
路径**——`models/`、`embedding_benchmark.py`、`scripts/embedding_benchmark/`
全目录扫描 `E:\` / `D:\` / `C:\`，除了这个默认列表**零命中**。

---

## 2. 已嵌入方法能否正常工作：4/4 全量跑通

数据：BBBC022 子集，**3,857 细胞 / 64 孔 / 54 化合物 / 671 个 mock 细胞**。

| 方法 | 输出 | 耗时 | 吞吐 | 权重哈希前缀 |
|---|---|---|---|---|
| deepprofiler | (3857, 672) | 18.2 s | 211.9 cells/s | `87b756f579336d60` |
| ca_mae | (3857, 384) | 195.0 s | 19.8 cells/s | `f6e5f1c97101331b` |
| cellclip | (3857, 512) | 11.3 s | 342.1 cells/s | `66c2d32a44a3e7d6` |
| o2vae | (3857, 512) | 71.8 s | 53.7 cells/s | `aa70cfa8b080c9d9` |

合计约 **5 分钟**跑完全部四个方法。输出维度与 manifest 声明全部一致，无 NaN/Inf。

### 2.1 Benchmark 结果（z-scored，化合物检索）

```
cellclip       mAP_z=0.1337  kNN_z=0.6451
ca_mae         mAP_z=0.1083  kNN_z=0.4454
deepprofiler   mAP_z=0.0691  kNN_z=0.1934
raw pixels     mAP_z=0.0508  kNN_z=0.1058   ← 基线
o2vae          mAP_z=0.0471  kNN_z=0.0869
random         mAP_z=0.0450  kNN_z=0.0848   ← 零假设
chance = 0.0185
```

**与之前发表的数值完全一致**（0.1337 / 0.1083 / 0.0691 / 0.0508 / 0.0471 / 0.0450），
说明整条链路可复现。

### 2.2 4 折稳定性

每一折排序都是 `cellclip > ca_mae > deepprofiler > o2vae`，
`deepprofiler > 原始像素 > o2vae` 每折成立，cellclip 4/4 第一。

**仍然不给 p 值**：每个化合物只占一个孔，化合物身份 ≡ 孔身份，
留一孔会让相关项归零。这个数据集结构上做不了显著性检验。

---

## 3. IO 接口是否统一：单方法路径 **PASS**，批处理路径**发现真实缺陷并已修**

### 3.1 统一的证据

四个方法走**完全相同的调用签名**：

```python
models.embed(name, payload, unit=unit)   # 只有 name 和 unit 不同
```

返回对象字段集合**完全一致**（`EmbeddingResult`，扁平结构）：

```
model  values  dim  input_unit  backend  feature_layer  n_samples
weights_path  weights_sha256  license  determinism  metadata  sample_metadata
```

- 输出维度**不同是正常的**（384/512/672），接口统一不等于输出同维
- 全部带权重哈希、许可、确定性标记
- `metadata` 键集合一致

批处理路径同样统一：

```python
plan = build_benchmark_plan(methods, output_dir=..., unit={...}, align_on='cell_id')
run_benchmark(plan, {method: array}, sample_metadata=...)
```

`unit` 支持 `{method: unit}` 映射——这是必需的，因为 CellCLIP 吃向量、其余吃像素，
单一标量 unit 无法描述一批方法。

### 3.2 发现并修复的缺陷：批处理路径没有分块

**症状**：全量 3,857 细胞走 `run_benchmark` 时

```
ca_mae:  OutOfMemoryError: CUDA out of memory. Tried to allocate 7.06 GiB
         (GPU 8.00 GiB total, 14.90 GiB allocated by PyTorch)
o2vae:   RuntimeError: DefaultCPUAllocator: not enough memory (6,664,896,000 B)
```

**根因**：`embed_all` 把整个 payload 当一个数组传给 `embed`。
CA-MAE 的 3,857×5×256×256 uint8（1.26 GB）在一次前向里变成
3,857×5×256×256×4 = **5.05 GB** float32 张量，直接打爆 8 GiB 卡。

讽刺的是：库**本来就有**分块能力（`embed_many` 接收数组序列，
错误信息里的 `chunk[0]` 就是它），只是 `embed_all` 没用。

**修复**（库层，不是让调用方各自写循环）：

| 位置 | 改动 |
|---|---|
| `models/embedding.py` | 新增 `_split_into_blocks()`：按 `chunk_size` 切块，不必要时原样返回同一对象 |
| `models/embedding.py` | 新增 `_merge_result_parts()`：合并分块结果，**宽度不一致直接报错**（否则拼出来是错的） |
| `models/embedding.py` | `embed_all(..., chunk_size=None)`：>1 块走 `embed_many`，随机方法走逐块聚合再合并 |
| `models/embedding.py` | `embed_many` 补上 `runtime_root` 透传（之前只有 `embed` 有，分块时 ASCII 路径检查会被跳过） |
| `embedding_benchmark.py` | `BenchmarkPlan.chunk_size` + `build_benchmark_plan(chunk_size=)`，`run_benchmark` 透传 |

**为什么放在库层**：分块是接口约定的一部分。放在调用方，每个调用方都会
重新实现一遍并且把内存上界算错——`embed_dataset.py` 当初就是这么干的。

**回归测试**：新增 6 条（切块边界、不必要时不复制、宽度不一致报错、
合并行数与 chunk 计数、plan 字段确实透传到 `embed_all`）。

### 3.3 修复后的实测

同一批数据、同一 `run_benchmark`，只把 `chunk_size` 打开：

| 径路 | 方法数 | 细胞数 | chunk | 结果 |
|---|---|---|---|---|
| 修复前 | 4 | 3,857 | 无 | ❌ ca_mae / o2vae OOM |
| 修复后 | 4 | 512 | 64 | ✅ `errors={}`，对齐 512，写出 5 个产物 |
| 修复后 | 2 | 64 | 16 | ✅ `errors={}`，对齐 64 |

512 细胞那次的**每方法输出**：

```
deepprofiler (512,672) 37.6s   ca_mae (512,384) 32.8s
cellclip     (512,512)  5.8s   o2vae  (512,512) 16.8s
batch: shapes 全对 / errors={} / aligned=512 / written=5
```

**环境说明（影响本次耗时读数的可信度）**：这台机器上有一个**非本次任务**的
长跑进程（PID 41692，已运行 11.5 小时，占 GPU 4.4 GB），叠加被杀任务的残留
子进程，一度把可用内存压到 1.8 GB / 31.7 GB、`Pages/sec` 达到 47,853。
清理后同样任务从"10 分钟不返回"变成 2 秒。所以**本文的吞吐数字在空载时
测量，但绝对值受机器状态影响**；排序与形状结论不受影响（已重复复现）。

---

## 4. 一句话结论

- **权重**：4 个官方权重齐备且哈希校验通过；另外 3 个在当前网速下不可得
  （6 KB/s → 62 h / 312 h / 不可达），且无可信镜像，**如实标注为不可用**。
- **能跑**：4/4 全量 3,857 细胞跑通，约 5 分钟，结果与之前完全一致。
- **IO 统一**：单方法调用签名与返回结构**完全一致**（PASS）；
  批处理路径原本因不分块而 OOM，**已修复**，全量 3,857 细胞实测通过（见 §6）。

## 5. 未做与做不到的

| 事项 | 状态 |
|---|---|
| 补 unidino / dinocells / morphogenie | ❌ 6 KB/s（62 h / 312 h）、无可信镜像 |
| 三个不可用方法的真实适配器 | ❌ 没权重无法验证，硬写等于假装完成 |
| 化合物层面的显著性检验 | ❌ 每化合物单孔，结构上不支持 |
| 全量 3,857 细胞的**批处理**路径 | ✅ **2026-10-21 已复测通过**，见 §6 |

---

## 6. 全量批处理复测（2026-10-21）

上一轮只验证到 512 细胞。这次在**完整 3,857 细胞**上跑批处理路径，
并额外做了「批处理 vs 逐方法」的等价性对比。

### 6.1 能跑

```
chunk_size=64, 4 方法, 3,857 细胞
耗时 327.2 s (5.5 min)
shapes  : deepprofiler (3857,672) / ca_mae (3857,384) / cellclip (3857,512) / o2vae (3857,512)
errors  : none
aligned : 3857
written : 5
```

机器当时**并非空载**：另一个项目占着 GPU 93%、4.4 GB 显存。
所以 5.5 min 是"带竞争"的数字，空载应更快（上一轮逐方法路径合计约 5 min）。

### 6.2 等价性：批处理与逐方法是否给出同一结果

对照基准是 `emb_<name>.npy`（逐方法路径产物）。

| 方法 | 结果 |
|---|---|
| deepprofiler | **逐位相同** |
| ca_mae | 差 `max|d| = 1.907e-06` |
| cellclip | 差 `max|d| = 5.484e-06` |
| o2vae | 差 `max|d| = 4.768e-06` |

**这不等于"接口不一致"**。判别实验（ca_mae，256 细胞）：

```
chunk32 跑两次        → identical=True      （同 chunk 完全可复现）
chunk32 vs chunk64    → max|d| = 1.431e-06
数值量级 max|v|       → 3.129
```

相对误差 ≈ 4.6e-7，与 float32 机器精度（1.19e-7）同阶——
这是**不同批大小导致归约顺序不同**的舍入差，
不是逻辑差异。DeepProfiler 逐位相同是因为它走 TensorFlow 图、批内归约稳定。

### 6.3 这点差异会不会影响结论

用**同一个打分函数**分别评估两套嵌入：

| 方法 | 参考 mAP_z | 批处理 mAP_z | 差 |
|---|---|---|---|
| deepprofiler | 0.069124667 | 0.069124667 | **0** |
| ca_mae | 0.108320074 | 0.108320051 | −2.3e-08 |
| cellclip | 0.133728824 | 0.133728855 | +3.1e-08 |
| o2vae | 0.047063044 | 0.047063044 | −6.7e-11 |

**最大影响 3.11e-08**，对排序、结论、任何一位有效数字都无影响。

### 6.4 打分与稳定性复跑

重跑 `score_benchmark.py` 与 `stability_check.py`：

- 与上一轮 `benchmark_scores_final.json` 逐项比对：**65 个可比数值项，0 项不一致**
- 4 折排序依旧每折都是 `cellclip > ca_mae > deepprofiler > o2vae`，cellclip 4/4 第一
- 测试：**263 passed**，14 failed（均为任务开始前既有）