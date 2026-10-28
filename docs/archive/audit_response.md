# 对 `embedding_audit_report.md` 的逐条核实与修复

> 日期：2026-10-18
> 方法：每条断言都用代码/命令独立复现，**不采信报告本身**
> 结果：**6 个 P0 全部确认为真并已修复**；3 个 P2 文档漂移已修；P1 中 5 项已修，其余记录在案

---

## 0. 总览

| 报告断言 | 独立核实 | 处置 |
|---|---|---|
| P0-1 DeepProfiler 通道顺序错 | ✅ **为真，且比报告更严重** | ✅ 已修 + 重建数据 + 重跑 |
| P0-2 `models.embed_all` 未导出 | ✅ 为真 | ✅ 已修 + 回归测试 |
| P0-3 `save_npz` 丢 `sample_metadata` | ✅ 为真 | ✅ 已修 + 回归测试 |
| P0-4 `summary.json` 的 `finished_at`/`written` 恒空 | ✅ 为真 | ✅ 已修 + 顺序断言测试 |
| P0-5 CellCLIP 放行 4D 像素堆 | ✅ 为真 | ✅ 已修 + 回归测试 |
| P0-6 Bootstrap 点估计落在 CI 外 | ✅ 为真 | ✅ **已撤回**，改为 4 折稳定性 |
| P2 三处 adapter 文档漂移 | ✅ 全部为真 | ✅ 已修 |

**测试**：`204 passed`（原 192，+12 条回归测试）；14 个失败与会话前一致。

---

## 1. P0-1：DeepProfiler 通道顺序 —— 确认，而且有两处错位

### 上游证据（我方独立读取）

```
D:\official_src\1-deepprofiler\README.md:74
  "... trained on 5-channel Cell Painting images (DNA, ER, RNA, AGP, Mito)"

tests\deepprofiler\profiling\test_cell_painting_cnn_v1.py
  L44-48: ch5=DNA, ch4=ER, ch3=RNA, ch1=AGP, ch2=Mito
  L74:    CHANNELS = ["DNA", "ER", "RNA", "AGP", "Mito"]
  L75:    BOX_SIZE = 128
```

### 我方（修复前）

```
models/registry.py:96   DEEPPROFILER_CHANNELS = ('DNA','RNA','ER','AGP','Mito')
demo pipeline_config.json:32-41  channel_map: 1=Mito, 2=AGP, 3=RNA, 4=ER, 5=DNA
```

### 结论：两处独立错位

| | 上游官方 | 我方（修复前） |
|---|---|---|
| **张量顺序** | `DNA, ER, RNA, AGP, Mito` | `DNA, RNA, ER, AGP, Mito`（RNA/ER 互换） |
| **文件映射** | `AGP→ch1, Mito→ch2` | `AGP→ch2, Mito→ch1`（AGP/Mito 互换） |

**报告只指出了第一处，第二处同样存在。**

### 已做的修复

1. `DEEPPROFILER_CHANNELS` → `('DNA','ER','RNA','AGP','Mito')`
2. 新增独立常量 `CELLCLIP_CHANNELS`（来源：JUMP-CP w1..w5），CellCLIP 不再复用 DP 的常量
3. `build_bbbc022_cells.py` 的 `CHANNELS` 改为官方顺序（`CHANNEL_COLUMNS` 保留为列名查找）
4. 新增回归测试：`TestChannelOrderMatchesUpstream`（3 条），含"旧顺序必须被拒"
5. **重建 BBBC022 数据集并重跑全部四个方法**

### 关于 parity 参照系（报告 §2.3，完全成立）

> `bitwise_equal=True` 证明的是"接口层 == 本项目官方树"，**不是**"== 上游训练语义"。

我方已在 `parity_vs_known_result.py` 与 `parity_check_c.md` 的语境中接受这一限定；本次修正在
`registry.py` 的常量注释里写明了这一点。

---

## 2. P0-1 连带发现：DP 归一化也错了（报告 §6.1 提到，我方确认）

官方 `cropping.py:65-67`：

```python
mini = tf.math.reduce_min(crops, axis=[1, 2], keepdims=True)
maxi = tf.math.reduce_max(crops, axis=[1, 2], keepdims=True)
```

`crops` 形状 `(N,H,W,C)`，`axis=[1,2]` 归约 H、W ⇒ **逐通道 min-max**。

我方 `build_bbbc022_cells.py:278` 原来是 `lo, hi = dp_crop.min(), dp_crop.max()`，
即**跨通道联合 min-max**。**已修为逐通道**。

### ★ 这个错误改变了此前的结论

| | 修复前 mAP_z | 修复后 mAP_z |
|---|---|---|
| deepprofiler | 0.0684 | **0.0691** |
| **原始像素基线** | **0.0664** | **0.0508** |
| cellclip | 0.1338 | 0.1337 |
| ca_mae | 0.1083 | 0.1083 |
| o2vae | 0.0471 | 0.0471 |

**此前"DeepProfiler ≈ 原始像素"的结论是归一化错误造成的假象。**
修正后 DeepProfiler **明确高于原始像素 36%**（0.0691 vs 0.0508），
4 折中每一折都成立（0.160/0.130/0.169/0.170 vs 0.110/0.082/0.130/0.113）。

> 这是我上一轮报告中最实质的错误结论，已在本文档与 `benchmark_bbbc022_result.md` 中更正。

---

## 3. P0-2：`models.embed_all` 未导出 —— 确认

```
修复前: hasattr(models, 'embed_all') == False
        AttributeError: module 'cellpaint_pipeline.models' has no attribute 'embed_all'
```

**修复**：在 `models/__init__.py` 的 `__all__` 与 `_declare(...)` 中导出
（并顺带导出 `clear_adapter_cache`）。
**回归测试**：`test_embed_all_is_reachable_on_the_facade`。

---

## 4. P0-3：`save_npz` 丢行身份 —— 确认

```
修复前 npz keys: ['features', 'metadata']     cell_id 保留: False
```

**修复**：`sample_metadata` 每列写为独立 `sample_<key>` 数组，并校验长度一一对应；
`metadata` 里补 `n_samples`、`determinism`、`sample_metadata_keys`。
**回归测试**：`test_save_npz_keeps_per_row_identity`、`test_save_npz_refuses_mismatched_metadata`。

```
修复后 npz keys: ['features', 'metadata', 'sample_cell_id', 'sample_well']
```

---

## 5. P0-4：`summary.json` 的 `finished_at`/`written` 恒空 —— 确认

```
L221: outcome.written = _write_artefacts(outcome)     ← 先写
L222: outcome.finished_at = datetime.now(...)          ← 后赋值
```

`_write_artefacts` 序列化的是同一个对象，所以写出的 summary 里两个字段必为空。

**修复**：先算好 `finished_at` 与 `written`，再写文件；`_write_artefacts` 改为**只报告确实存在的路径**（`is_file()` 校验），不再盲信。
**回归测试**：`test_run_benchmark_sets_finished_at_before_writing`（断言调用顺序）。

---

## 6. P0-5：CellCLIP 放行 4D 像素堆 —— 确认

```
修复前: 4D 像素堆, unit 省略        -> ok=True
        4D 像素堆, unit=channel_embeddings -> ok=True
```

**修复**：
1. `InputUnit.CHANNEL_EMBEDDINGS` 强制 `ndim == 3`
2. `ModelManifest` 新增 `input_feature_dim`；CellCLIP 声明 `1536`
3. `check_compatible` 校验 `shape[-1]`

```
修复后: 4D 像素堆        ok=False ['ndim']
        3D 像素(宽 256)  ok=False ['feature-width']
        正确向量(宽1536) ok=True  []
```
**回归测试**：4 条（4D 无 unit / 4D 有 unit / 3D 错宽 / 声明存在）。

---

## 7. P0-6：Bootstrap 不成立 —— 确认，且比报告说的更严重

报告指出点估计落在 CI 外。我方独立复现并进一步发现：**这个数据集根本做不出合法的成对检验。**

### 三次尝试、三次失败（全部记录在代码注释里）

| 做法 | 失败原因 | 实测 |
|---|---|---|
| **按孔 bootstrap**（有放回重采样后重算 mAP） | 重复行互为最近邻，抬高 AP | 点估计 0.1338，bootstrap 均值 0.1864，CI [0.1615, 0.2176] —— **点在区间外** |
| **留一孔检索** | 每化合物仅 1 孔，留出该孔后数据库里**无相关项** | 64 孔中 53 孔无定义，只剩 11 个 mock 孔 |
| **逐孔 AP（把自身列设为 inf）** | `argsort` 仍返回被屏蔽列，相关项**全部排到最后** | 所有方法 AP 恒为 0.0346–0.0355，无区分 |

### 根本事实

**化合物身份 == 孔身份。** 任何排除自身孔的指标没有相关项；任何保留自身孔的指标测的是"识别孔"而非"识别化学"。
**这份数据在原理上不支持"方法 A 显著优于 B"的检验。**

### 处置

- **撤回 bootstrap 的 CI 与 P 值**，不再引用
- 改用 **4 折按孔划分的稳定性检查**（`stability_check.py`），只报告"排序是否稳定"，**不报告 p 值**
- 在输出 JSON 里显式写入 `no_p_values_because` 与 `rejected_approaches`，避免后来者重蹈

### 4 折稳定性结果（修正后数据）

| fold | n | deepprofiler | ca_mae | cellclip | o2vae | 原始像素 |
|---|---:|---:|---:|---:|---:|---:|
| 0 | 940 | 0.1600 | 0.1964 | 0.2556 | 0.1015 | 0.1102 |
| 1 | 960 | 0.1298 | 0.1865 | 0.2563 | 0.0747 | 0.0823 |
| 2 | 981 | 0.1687 | 0.1943 | 0.2547 | 0.1179 | 0.1297 |
| 3 | 976 | 0.1697 | 0.1922 | 0.2565 | 0.0972 | 0.1126 |

**每一折排序都是** `cellclip > ca_mae > deepprofiler > o2vae`，
且每一折 `deepprofiler > 原始像素 > o2vae`。**first-place: cellclip=4/4。**

---

## 8. P2：三处 adapter 文档漂移 —— 全部确认并修复

| 文件 | 修复前 | 已改为 |
|---|---|---|
| `adapters/o2vae.py:1-28` | "self-trained … 50 epochs on o2-mnist … 32x32" | 官方 MEF 权重、`zdim=512`、64×64、并说明这是形状模型 |
| `adapters/deepprofiler.py:18-25` | `scale='raw'`、`value_range=None`、25000× | `per_crop_minmax`、`[0,1]`、4.5e6、官方通道顺序及两处错位 |
| `adapters/unavailable.py:5-7` | "Four of seven … O2-VAE was never released" | "Three of seven … O2-VAE 的官方 MEF 权重已发布" |

---

## 9. P1：已全部处理

| # | 项 | 处置 | 验证 |
|---|---|---|---|
| 7 | `BenchmarkPlan` 只有一个 `unit`，无法表达逐方法契约 | ✅ **已修**：`unit` 支持 `{method: unit}` 映射，`embed_all` 同步支持 | 4 方法（含 CellCLIP 向量）在同一 plan 内端到端跑通 |
| 8 | `PreprocessSpec` 只声明不执行 | ⚠️ **仍为声明**：预处理在 `scripts/embedding_benchmark/build_bbbc022_cells.py` | 见下方说明 |
| 9 | O2-VAE `strict=False` 未断言 | ✅ **已修**：missing 只允许是 buffer，可学参数缺失即抛 `EmbeddingError(weight-mismatch)` | 实跑 `(2,512)` 通过 |
| 10 | DeepProfiler `by_name=True` 无加载断言 | ✅ **已修**：加载前快照第一个有权重的层，加载后若**逐字节相同**则抛 `weight-load-noop` | 实跑 `(2,672)` 通过 |
| 11 | DINOv2-Giant 无溯源 | ✅ **已修**：`ModelManifest.chain_weights`；`metadata` 透出；向量 JSON 记 `model_sha256` | 哈希 `917d3c47…` 我方独立计算，与报告一致 |
| 12 | `models/` 等未纳入 git | ❌ **未做（需你决定）** | 见 §12 |
| 13 | 脚本在仓库外、硬编码绝对路径 | ✅ **已修**：迁入 `scripts/embedding_benchmark/`，`src/` 由 `__file__` 解析，**零绝对路径** | 在仓库内实跑，复现同一结果 |
| 14 | `embed()` 不能传 `device`/`weights_roots`/构造参数 | ✅ **已修**：三个入口都新增三个参数；缓存键含 device | 测试确认不是 `TypeError`；plan 新增 `device` |
| 15 | `embed_all` 用 `__errors__` 魔法键 | ✅ **已修**：返回 `BatchResult(results, errors)` | 6 条回归测试；`run_benchmark` 已改 |
| 16 | `pyproject.toml` 缺 torch extras、`deepprofiler` 钉 0.3.1 | ✅ **已修**：新增 `embeddings` / `embeddings-all`；改 `0.5.1` | TOML 解析通过 |

**测试**：`212 passed`（P0 阶段 204，P1 阶段 +8）；14 个失败与会话前一致。

### 仍未完成的两项及原因

- **#8 预处理下沉**：`PreprocessSpec` 目前是纯声明，真正做缩放/重排的是
  `scripts/embedding_benchmark/build_bbbc022_cells.py`。下沉为
  `models.prepare(name, raw)` 会把契约层从"校验"变成"执行"，是职责边界变更，
  需要单独一轮设计，不宜夹在 P1 收尾里改。
- **#12 git 纳入**：仓库流程决定，不是代码修复。新增的 14 个文件
  （`models/`、`embedding_benchmark.py`、两个测试、11 个文档、
  `scripts/embedding_benchmark/`、环境 yml）仍是 untracked，
  **需要你确认后我再提交**，我不擅自改版本历史。

### 端到端复核（真实权重，4 细胞 × 4 方法）

```
shapes : {'deepprofiler': (4,672), 'ca_mae': (4,384),
          'cellclip': (4,512), 'o2vae': (4,512)}
errors : {}
summary.finished_at set: True | summary.written: 5
npz keys: ['features','metadata','sample_cell_id','sample_well']
cell_id : ['20585_M04_5_0', …]   aligns with aligned_ids
```

四项 P0 修复在同一次运行里同时生效：逐方法 unit、`BatchResult`、
`sample_metadata` 落盘、`finished_at`/`written` 非空。

---

## 10. 修正后的完整结果

### 10.1 全量 mAP_z（描述性，受孔混杂）

| 方法 | mAP_z | kNN_z |
|---|---|---|
| **cellclip** | **0.1337** | **0.6451** |
| **ca_mae** | 0.1083 | 0.4454 |
| **deepprofiler** | **0.0691** | **0.1934** |
| *原始像素基线* | 0.0508 | 0.1058 |
| o2vae | 0.0471 | 0.0869 |
| *随机投影基线* | 0.0450 | 0.0848 |

### 10.2 排序（4 折稳定）

```
cellclip > ca_mae > deepprofiler > 原始像素 > o2vae ≈ 随机
```

### 10.3 与上一版结论的差异

| 结论 | 上一版 | 本版 |
|---|---|---|
| DeepProfiler vs 原始像素 | **≈ 打平**（0.068 vs 0.066） | **DP 明确高 36%**（0.069 vs 0.051） |
| 统计显著性 | 报告了 bootstrap P 值 | **不报告 p 值**（数据集不支持） |
| 稳定性 | bootstrap CI | 4 折排序稳定（cellclip 4/4 第一） |
| 通道语义 | 未标注参照系 | 已对齐上游训练顺序，旧顺序被拒 |

---

## 11. 仍需完成

| 优先级 | 事项 | 说明 |
|---|---|---|
| **需你确认** | 把新增文件纳入 git | 14 个文件仍 untracked；我不擅自提交 |
| **P1（遗留）** | `PreprocessSpec` 下沉为可执行预处理 | 职责边界变更，需单独设计；目前预处理在 `scripts/` |
| **P2** | 数据集补齐（需 3.58 GB） | 才能做留一孔 CV 与剂量-反应，进而做真正的显著性检验 |
| **P2** | 与论文协议对齐 | 当前是自建化合物检索协议，非 CA-MAE/CellCLIP 论文指标 |

### 已完成（本轮）

- P0 × 6：通道顺序、`embed_all` 导出、`save_npz` 行身份、`summary` 时间戳、
  CellCLIP 契约收紧、bootstrap 撤回
- P1 × 9（除 #8、#12）：逐方法 unit、O2-VAE 加载断言、DP 加载断言、
  DINOv2 溯源、脚本入库去绝对路径、`device`/`weights_roots`/`adapter_kwargs`、
  `BatchResult`、pyproject extras
- P2 × 3：三处 adapter 文档漂移
