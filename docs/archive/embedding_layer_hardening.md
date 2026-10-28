# Embedding 接口层：本轮加固记录

> 日期：2026-10-10
> 触发：源码审计发现 3 个 manifest 申报错误，且本层**零测试覆盖**
> 改动文件：`models/contract.py`、`models/registry.py`、`models/embedding.py`、新增 `tests/test_models_contract.py`

---

## 1. 修掉的 3 个申报错误

| # | 方法 | 字段 | 原值 | 现值 | 依据 |
|---|---|---|---|---|---|
| 1 | `unidino` | `order_matters` | `False` | **`True`** | 实测：通道置换使输出变化 `2.109`（`inference_utils.py:29-34` 按输入顺序拼接） |
| 2 | `morphogenie` | `output_dim` | `256` | **`10`** | 实测 `encode_only` 返回 `(4,10)`；宽度是 `c_dim=10`（`:67`），`z_dim=256`（`:68`）是 GAN 潜维 |
| 3 | `unidino` | `scale` | `div65535` | **`none`**（并注明未声明原因） | 全仓 grep `65535/255/*255/Normalize` **无像素缩放代码**；`inference.py:159` 只有 `Resize` |
| 4 | `morphogenie` | `determinism` | （字段不存在） | **`stochastic`** | 实测三次调用差 `2.979`；`:297` 把 `mu_real` 绑到第一个返回值，而 `dvae.py:396` 返回 `(c,mu,logvar)` |

> 第 1 条尤其要紧：`order_matters=False` 会让 `check_compatible` **放行置换过的输入**，
> 属于本层设计出来要拦的那类静默错误——**结果它自身的申报就是错的**。

## 2. 新增的 2 个契约原语

### 2.1 `PreprocessSpec.input_encoding`

```python
input_encoding: Literal['stack', 'tape'] = 'stack'
```

- `stack` = 普通 `(N,C,H,W)`（6/7 个方法）
- `tape` = DINO4Cells 的"胶带"格式 `(H, W*C)`，需 `reshape(H,W,-1,order='F')` 后再丢掩膜通道

**为什么必须显式声明**：`(H,192)` 的 tape 和 `(H,192)` 的普通图**形状相同**。
不声明则 tape 数组会通过所有形状检查，然后静默产出错误嵌入。

配套新增 `_check_tape_shape()`：校验秩为 2、宽度能被 `C+1` 整除、折叠后尺寸等于 `spatial`。
**并在消息里明说"若通道数不同，掩膜会从错误位置被丢弃"**——因为 `W` 与 `C` 无法从形状反推，这部分只能声明不能推断。

### 2.2 `ChannelSpec.channel_count_fixed`

```python
channel_count_fixed: bool = True
```

**为什么与 `names_are_labels` 分开**：两者真正独立。
CA-MAE 是**架构级**通道无关（patch token 上的逐通道 cross-attention，实测 C ∈ {1,4,5,6,11}），
所以对它而言名字是占位符**且数量自由**。但一个模型也可能"不关心身份、却受形状约束"。
原实现把两者绑在一起，导致 CA-MAE 传给 7 通道会被误拒。

`__post_init__` 增加一条不变量：`channel_count_fixed=False` 与 `order_matters=True` 互斥
（数量自由则声明的顺序不可能是要求）。

### 2.3 `ModelManifest.determinism`

```python
determinism: Literal['deterministic', 'stochastic'] = 'deterministic'
```

**为什么上到结果对象**：把随机方法与确定方法各跑一次做对比，是拿噪声比信号。
`EmbeddingResult.determinism` 与 `to_dict()` 都会带上，下游看到 `stochastic` 就知道要重复跑取统计。

## 3. 补上零测试覆盖

新增 **`tests/test_models_contract.py`：30 个测试，0.15 秒，不需要 torch/tensorflow/权重**。

| 测试组 | 数量 | 作用 |
|---|---|---|
| `TestRegistry` | 5 | 7 个方法齐全、注册项结构、未知方法抛类型化错误 |
| `TestManifestConsistency` | 5 | 跨全表的**不变量**（顺序/标签/数量三者一致性、tape 只用于图像模态、spatial 为正） |
| `TestMeasuredDefectsDoNotRegress` | 5 | **把上面 4 个实测缺陷钉死**，含"置换必须被拒"的行为级断言 |
| `TestChannelContract` | 4 | 通道集合/顺序/数量/通道无关 |
| `TestDtypeAndRangeContract` | 5 | dtype 与值域，含 `0..1 ⊂ 0..255` 那个抓不到的情况 |
| `TestInputUnitContract` | 3 | CellCLIP 必须吃向量而非像素 |
| `TestAsciiPathPolicy` | 3 | ASCII 路径策略 |

**测试设计上的一个修正**：最初的 fixture 硬编码 `BCHW`，导致 DeepProfiler（Keras，`BHWC`）误报 3 个失败。
改成 `_crop_for(manifest)` 按**各自声明的布局**构造——因为布局本身就是被测对象，
fixture 硬编码会掩盖真实的布局错误。

## 4. 验证结果

```
tests/test_models_contract.py      30 passed in 0.15s
全量套件                            14 failed, 156 passed   （原 14 failed, 126 passed）
契约自检（26 条）                   all contract checks passed
三个 embedding 适配器               deepprofiler (4,672) / ca_mae (2,384) / cellclip (3,512) 全部 OK
```

**14 个失败与改动前完全一致**，全部是既有问题（`ProjectConfig` NameError、Windows 路径分隔符、
需要 demo 数据的端到端冒烟），**不是本轮引入**。

## 5. 两树同步

改动同时落在：
- `D:\项目\CellPainting-Claw-v2\`（canonical 开发树）
- `D:\cpc_c_v2\`（双树对拍用的临时树）

四个文件逐字节一致。**注意**：`cpc_c_v2` 是 C 步骤对拍用的树，改了它就要重跑对拍才作数
（本轮改的是 `models/`，而流水线**不引用** `models/`，故不影响对拍结论）。

## 6. 权重策略：小数据集自训

用户指示「权重实在不行拿小数据集训练一个吧」。现状与判断：

| 方法 | 官方权重 | 自训可行性 | 判断 |
|---|---|---|---|
| `unidino` | Zenodo 1.38 GB，≈43 h | DINO 自蒸馏，ViT-S，**单通道** | ⚠️ 可行但需 ~1 天；且需构造单通道裁剪数据集 |
| `dinocells` | Zenodo 6.90 GB，≈180 h | DINO 自蒸馏，ViT-S | ⚠️ 同上，且**官方缺 Cell Painting 配置**（通道数不明） |
| `morphogenie` | Google Drive 不可达 | 两阶段（dVAE + ID-GAN） | ❌ 官方实现本身随机 + 需 MATLAB 预处理，**不适合作为基准** |
| `o2vae` | 官方从未发布 | **已完成**（50 epoch，o2-mnist） | ✅ 已有先例 |

**已有一个自训先例**：`o2vae` 的权重就是本项目自训的 50 epoch。
但它**明确标注为"非方法复现"**，只验证接口形状——这个标注方式应当沿用到后续自训权重。

**尚未做的决定**：是否为 `unidino` / `dinocells` 自训。
这需要先确认：自训权重产出的嵌入**不能与论文比较**，只能验证接口；
若目标是 benchmark，自训权重**没有可比性**。因此这属于产品决策，不是技术障碍。

## 7. 下一步（按价值排序）

| 优先级 | 事项 | 理由 |
|---|---|---|
| **P1** | 补"新接口层产出 == 官方路径产出"的等价性验证 | 目前**最大缺口**：2516 行代码与官方等价性未证明 |
| **P2** | 为 `unidino` / `dinocells` 写**真实适配器**（结构先就位） | 结构就位后，权重到手即可跑；也可先用随机权重跑通接口 |
| **P3** | 决定是否自训权重 | 需先明确"benchmark 是否接受自训权重" |
| **P4** | `morphogenie` 若要包含，须改用后验均值并标注非复现 | 官方路径随机，直接用作基准不可信 |
