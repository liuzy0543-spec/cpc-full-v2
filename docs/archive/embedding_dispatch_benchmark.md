# 统一 Embedding 调度与 Benchmark 层

> 日期：2026-10-10
> 目标：把**有权重、能复现**的 embedding 方法接入优化版 CellPainting-Claw，
> 统一 IO 接口，支持统一调度与 benchmarking
> 状态：**4 个方法已走通统一调度路径**

---

## 1. 接入结果

**一条调用，4 个方法，全部产出嵌入。**

```
methods runnable  : ['deepprofiler', 'ca_mae', 'cellclip', 'o2vae']
plan methods      : ('ca_mae', 'cellclip', 'deepprofiler', 'o2vae')

=== outcome ===
  method        shape         dim   determinism    licence gate
  ca_mae        (3, 384)      384   deterministic  non-commercial
  cellclip      (3, 512)      512   deterministic  no-derivatives
  deepprofiler  (3, 672)      672   deterministic  permissive
  o2vae         (3, 64)       64    deterministic  permissive
  errors          : none
  aligned rows    : 3

  provenance (weights actually hashed):
    ca_mae         sha256=f6e5f1c97101331b1574  layer=encoder_pool
    cellclip       sha256=66c2d32a44a3e7d68deb  layer=CrossChannelFormer CLS
    deepprofiler   sha256=87b756f579336d606fd3  layer=block6a_activation
    o2vae          sha256=1cd6ea8a01b8bb430e76  layer=z_mu

  pairwise RV coefficient (0..1; defined across different widths):
    ca_mae         vs cellclip       0.9982
    ca_mae         vs deepprofiler   0.6600
    ca_mae         vs o2vae          0.9756
    cellclip       vs deepprofiler   0.6669
    cellclip       vs o2vae          0.9611
    deepprofiler   vs o2vae          0.6587
```

> 上面的 RV 数值来自**合成随机输入**，只说明管线通了，**不能解读为方法相似度**。

---

## 2. 统一 IO：一条入口

```python
from cellpaint_pipeline import embedding_benchmark as bench

plan = bench.build_benchmark_plan(
    ['deepprofiler', 'ca_mae', 'cellclip', 'o2vae'],
    output_dir='runs/bench_001',
    align_on='cell_id',          # 跨方法对齐键
)

result = bench.run_benchmark(plan, {
    'deepprofiler': crops_bhwc_f32,   # (N,32,32,5)  原始值
    'ca_mae':       crops_bchw_u8,    # (N,5,256,256) uint8
    'cellclip':     channel_vectors,  # (N,5,1536)   非像素
    'o2vae':        silhouettes,      # (N,1,32,32)  0..1
}, sample_metadata={'cell_id': ids, 'plate': plates})
```

### ★ 为什么 `inputs` 是「按方法名索引的映射」而不是单个数组

**这是本设计最关键的一点。** 4 个方法的输入**根本不同**：

| 方法 | 布局 | dtype | 通道数 | 是像素吗 |
|---|---|---|---|---|
| `deepprofiler` | `BHWC` | float32 | 5 | 是 |
| `ca_mae` | `BCHW` | **uint8** | 5（可变） | 是 |
| `cellclip` | `BCHW` | float32 | 5 | ❌ **是向量** |
| `o2vae` | `BCHW` | float32 | **1** | 是 |

`cellclip` 吃的是**逐通道 DINOv2 向量**，不是图像。若接口设计成
`embed_all(names, crops)`，那么喂给 cellclip 的必然是错的——而它**不会报错**，
只会产出一个看起来正常的嵌入。

所以接口**必须**能表达"每个方法各自的输入"，而不是假装它们共享一个。

### 也支持单数组的简写，但严格

```python
# 只有当 unit 与方法声明的 input_unit 一致时才会跑
result = benchmark_plan_with(unit='single_cell_crop')
```

不一致的方法**被跳过并记录原因**，而不是收到一个会产生错误嵌入的数组：

```
'cellclip' consumes 'channel_embeddings', but the shared payload was
declared as 'single_cell_crop'
```

---

## 3. Benchmark 运行器

### 3.1 三条不变量（在 plan 里显式声明，可打印可审计）

```python
plan.invariants
# ('every method reports weights_sha256, verified against the file',
#  'stochastic methods are repeated and their spread is recorded',
#  'all methods are aligned on one shared row identity before comparison')
```

### 3.2 对齐键

`align_on='cell_id'` 时，运行器**校验每个方法的行数与 `sample_metadata` 一致**，
并取**交集**（不是并集）：

- 一个方法少看了几个细胞，**不能补齐**
- 静默比较不同的细胞集合，正是这里要防的失败

行数不匹配会直接报错，而不是产出一个看似合理的对比。

### 3.3 随机方法

`determinism='stochastic'` 的方法会被**重复 3 次**，取均值作为值，
并把观测到的**最大离散度**记进 `metadata['stochastic_max_spread']`，
上报为 `stability`。

理由：把随机方法的**单次抽样**与确定方法对比，是拿噪声比信号。

### 3.4 合规闸门

每个方法自动分档（`licence_gate`），因为许可差异极大：

| 方法 | 分档 |
|---|---|
| `deepprofiler` | **permissive** |
| `ca_mae` | non-commercial |
| `cellclip` | **no-derivatives**（ND 禁衍生） |
| `o2vae` | permissive（但自训） |
| `unidino` | no-derivatives（上游自相矛盾） |
| `dinocells` | unknown（代码无许可声明） |
| `morphogenie` | copyleft（GPL-3.0） |

> **只有 `deepprofiler` 代码与权重双双宽松**，可直接用于对外发布的对比。

### 3.5 落盘

```
runs/bench_001/
  embeddings/
    ca_mae.npz          ← features + metadata + sample_metadata
    cellclip.npz
    deepprofiler.npz
    o2vae.npz
  summary.json          ← 溯源、形状、稳定性、许可分档、RV
```

`summary.json` 的 `provenance` 段记录每个方法的 `weights_sha256`、
`weights_path`、`license`、`determinism`、`feature_layer`、`dim`。

---

## 4. 权重溯源：真校验，不是照抄

**改动前**：`EmbeddingResult.weights_sha256` 只是把 manifest 里的字符串复制过去，
**从未与文件核对**——那是"溯源表演"，不是溯源。

**改动后**：
- `BaseAdapter.resolve_weights()` 在声明了 sha256 时**默认校验**
- 缺失即拒绝（`WeightsUnavailable`）
- 记忆化：键为 `(path, size, mtime_ns)`，文件被替换会自动失效

实测（CellCLIP 1.8 GB）：
```
first  : 1.963 s     ← 首次流式哈希
cached : 0.000340 s  ← 命中记忆化
```

`o2vae` 的自训权重也补上了 sha256（`1cd6ea8a…`），
这样它的 benchmark 结果同样可归属到**确切的那份 checkpoint**。

---

## 5. 本轮修掉的 4 个真实缺陷

| # | 缺陷 | 危害 | 修法 |
|---|---|---|---|
| 1 | `embed_all` 首版对 384 维与 512 维做逐行余弦 | `ValueError: broadcast (3,384) (3,512)` | 改用 **RV 系数**（定义在 Gram 矩阵上，跨宽度有效） |
| 2 | RV 首版写成 `a @ b.T` | 同样跨宽度失败 | 改走 `K_a = a aᵀ`、`K_b = b bᵀ` |
| 3 | **O2-VAE 在 torch 2.x 下根本加载不了** | `RuntimeError: more than one element of the written-to tensor refers to a single memory location` | e2cnn 把 index buffer 注册成**零步长扩展视图**（`strides=(0,)`）；改为**替换**该 buffer 而非 `copy_` 进去 |
| 4 | `weights_sha256` 从未核对文件 | 溯源失效 | 默认校验 + 记忆化 |

### 缺陷 3 的诊断依据（不是猜测）

```
zero-stride buffers in the model : 3
checkpoint tensors               : 427
  3 of the checkpoint keys land on zero-stride buffers
     model buffer : shape=(24,) strides=(0,)
     checkpoint   : shape=(24,) unique_values=[0]

=== 替换 buffer 后 ===
  load_state_dict: OK
  q_net(2,1,32,32) -> (2, 128)
  latent mean      -> (2, 64)      ← 与 manifest 声明的 output_dim 一致
```

**只有 3 个 buffer 受影响，且 checkpoint 里它们的值恒为 0**，
所以"替换成连续张量"是**正确**而非绕过：目标本来就不含 0 以外的信息。

---

## 6. 我修正的两处自己的过度声明

| 位置 | 原说法 | 实测 | 改为 |
|---|---|---|---|
| `_rv_coefficient` docstring | "invariant to invertible linear reparameterisation" | 各轴**各向异性缩放**使 RV 降到 **~0.72**，不是 1.0 | RV 只对**正交变换**+全局缩放不变 |
| 测试 `test_a_linear_map...` | 断言可逆线性映射得 1.0 | 随机良态矩阵得 **0.847** | 改用**置换矩阵**（正交、可精确表示） |

**为什么要改而不是调松阈值**：RV 的适用边界决定了 benchmark 里
"低 RV"该如何解读。若声称它对一切可逆变换不变，那么一个被各向异性缩放过的
嵌入会被误判为"几何不同"。**这是会误导结论的**。

---

## 7. 测试

新增 `tests/test_embedding_benchmark.py`：**31 个测试，0.26 秒**，
不需要 torch / tensorflow / 权重。

| 组 | 数量 | 覆盖 |
|---|---|---|
| `TestLicenceGate` | 9 | 许可分档 + **"只有 deepprofiler 双双宽松"钉死** |
| `TestBenchmarkPlan` | 5 | 未知方法、空清单、不变量 |
| `TestRvCoefficient` | 7 | 跨宽度、正交不变、**各向异性缩放的实测边界**、退化输入返回 `None` |
| `TestEmbedAllDispatch` | 4 | 缺 payload、**input_unit 不匹配必须跳过而非猜测** |
| `TestWeightHashing` | 5 | 哈希正确性、**缓存随文件变更失效**、已下载方法必须有 sha256 |
| `TestSummarySerialisation` | 2 | JSON 可序列化、RV 说明随结果一起落盘 |

```
tests/test_embedding_benchmark.py   31 passed
tests/test_models_contract.py       30 passed
全量套件                             14 failed, 187 passed
```

**187 - 126 = 61** 为本轮新增测试数；**14 个失败与改动前完全一致**，均为既有问题。

---

## 8. 两树同步

改动落在（逐字节一致）：
- `D:\项目\CellPainting-Claw-v2\`（canonical）
- `D:\cpc_c_v2\`（对拍用）

同步文件：`models/{contract,registry,embedding,weights}.py`、
`models/adapters/{base,o2vae}.py`、`embedding_benchmark.py`。

> **注意**：`cpc_c_v2` 是 C 步骤对拍用的树。本轮新增的 benchmark 层**不被流水线引用**，
> 故不影响此前"官方 ↔ v2 产物逐字节相同"的结论。

---

## 9. 仍然存在的缺口（未解决，明确记录）

| 缺口 | 说明 |
|---|---|
| **`models/` 与官方路径的等价性未验证** | 2516 行代码只有"能跑"的证据；**benchmark 数字的可信度仍取决于此** |
| `unidino` / `dinocells` / `morphogenie` 无适配器 | 注册表已声明契约，但 `adapters/` 里没有模块 |
| `morphogenie` 是 `stochastic` | 官方路径本身随机，用作基准需先决定取后验均值还是固定种子 |
| 无 CLI / 技能入口 | 目前只能从 Python 调用；接入 `cli.py` 与技能目录是独立的一步 |
| 评测指标未定义 | 取决于科学问题，本层只产出对齐矩阵，不猜"哪个方法更好" |

**建议下一步仍是等价性验证**——它决定上面所有数字能不能被引用。
