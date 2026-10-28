# 审查提示词：黑盒功能 + 白盒逻辑 + 软件设计规范

> 版本：2026-10-27
> 对象：`src/cellpaint_pipeline/models/`（v2 四个官方方法 + v3 三个替身方法）
> 用途：交给另一个模型（或另一个自己）独立审查，产出**可复现**的缺陷清单

---

## 0. 你的角色与硬性约束

你是外部审查者。你的产出是**缺陷清单与证据**，不是修改后的代码。

**必须遵守：**

1. **不许改任何文件。** 只读审查。发现问题写进报告，不要顺手修——修了就没法复现你的发现。
2. **每条结论必须带证据**：执行的命令 + 实际输出片段（或 `文件:行号`）。没有证据的结论一律不写。
3. **区分"我不理解"与"这是错的"。** 前者写进"待澄清"，不要伪装成缺陷。
4. **不许把注释当事实。** 注释和 docstring 都是被审对象，它们可能与代码不符——这种不符本身要作为缺陷上报。
5. 报告语言：中文。代码片段保持原样。

**禁止的推断方式：**

- ❌"看起来应该没问题" —— 必须实际跑
- ❌"作者写了会校验，所以应该校验了" —— 必须构造反例验证
- ❌"数字变了就是回归" —— 先对照 §5 的基线

---

## 1. 环境（必须先搭好，否则后面全是误报）

两个解释器，能力不同、失败方式也不同，**两个都要用**：

| 解释器 | 有什么 | 没有什么 |
|---|---|---|
| `D:/ep_bench/venv/Scripts/python.exe` | torch 2.5.1 / TF 2.15.1 / torchvision / e2cnn | — |
| `./.venv/Scripts/python.exe` | numpy / pandas / scipy / 测试栈 | **没有 torch、没有 TF** |

```bash
export PYTHONPATH="D:/项目/CellPainting-Claw-v2/src"
export CPC_WEIGHTS_ROOT="E:/cpc_weights"
export PYTHONIOENCODING=utf-8
```

**先跑这三条，确认环境可用：**

```bash
# 1) 门面能在没有深度学习框架的解释器里导入（如果这条就炸，后面不用做了）
./.venv/Scripts/python.exe -c "import sys, cellpaint_pipeline.models as m; \
print('torch/tf loaded:', len([k for k in sys.modules if k.split('.')[0] in ('torch','tensorflow','keras')]))"

# 2) 七个方法的能力盘点
D:/ep_bench/venv/Scripts/python.exe -c "from cellpaint_pipeline import models; \
[print(p['model'], p['status'], p['weights_found']) for p in models.doctor()]"

# 3) 契约层的 66 项自检（审核器，不是测试）
D:/ep_bench/venv/Scripts/python.exe scripts/embedding_benchmark/audit_round.py --round 0 --device cpu
```

第 1 条必须打印 `torch/tf loaded: 0`。**不是 0 就是缺陷**（P0）：门面一旦拖入框架，
连"只想列一下有哪些方法"的场景都得起一个 4.5 GB 的环境。

---

## 2. 黑盒检查（不看源码，只用公开 API）

只看 `from cellpaint_pipeline import models` 暴露的东西，把库当黑箱。

### 2.1 接口形状：一个方法名换一个 Embedding

| # | 检查 | 怎么验 | 通过判据 |
|---|---|---|---|
| B1 | 七个方法各自可调用 | `models.embed(name, payload)` | 返回对象有 `as_numpy()`、`dim`、`license`、`metadata` |
| B2 | 宽度与声明一致 | 对每个方法比 `result.dim` 与 `models.manifest(name).output_dim` | 相等（`unidino` 必须是 **1920**，不是 384） |
| B3 | 同输入同输出 | 连续调用两次，`np.array_equal` | 逐位相同 |
| B4 | 批量与单张一致 | `embed(x[0:1])` 与 `embed(x[0:5])[0:1]` | 逐位相同 |
| B5 | `embed_many` 与 `embed` 一致 | 两种入口跑同一批 | 逐位相同 |
| B6 | `embed_all` 一次驱动七个 | `assume`/`channels` 按方法传映射 | 七个结果、`errors == {}`、字段集合相同 |
| B7 | 输出有限 | `np.isfinite(v).all()` | True |
| B8 | 溯源标注 | `result.metadata['weights_provenance']` | v2 全 `official`；v3 三个替身必须是 `surrogate` |

**B2 是重点**：契约声明与实际输出不符时，调用方会按声明去 `np.zeros((n, dim))` 拼数组，
错误会一路活到发表的数字里。历史上 `unidino` 就声明过 384 而实际返回 1920。

### 2.2 该拒的必须拒（拒绝也是一种功能）

构造反例，确认**拒绝**而不是"跑出一个看起来合理的数"：

| # | 反例 | 期望 |
|---|---|---|
| B9 | uint8/uint16 原始值喂 `unidino` | 拒，`reason='value-range'` |
| B10 | 256×256 喂 `deepprofiler`（要 32×32） | 拒，`reason='spatial-size'` |
| B11 | 5 通道喂 `morphogenie`（要 3 通道） | 拒，`reason='channel-count'` |
| B12 | v2 剖面下 `embed('unidino', …)` | 拒，`UnknownModel` |
| B13 | `prepare()` 遇到 `source_map=None` | 拒，并说明"调用方已自行准备" |
| B14 | float 值域 0..255 喂声明 [0,1] 的方法 | 拒（防双重缩放） |

**判定"拒得对"的三个条件**：异常类型正确、`reason` 码正确、错误信息里说了**怎么办**。
只说"invalid input"的算缺陷（P2）。

### 2.3 黑盒要验的"不做事"

| # | 检查 | 判据 |
|---|---|---|
| B15 | 不隐式下载 | 断网/把 `CPC_WEIGHTS_ROOT` 指向空目录，必须在毫秒级抛 `WeightsUnavailable`，而不是卡住 |
| B16 | 不静默降级 | 权重缺失 → 异常；不许返回随机初始化的嵌入 |
| B17 | 不改入参 | 调用后原数组 `np.array_equal` 不变 |
| B18 | 不吞异常 | 一个方法失败时，`embed_all(on_error='raise')` 必须抛，不能悄悄少一个结果 |

---

## 3. 白盒检查（读代码，验证逻辑与声明是否一致）

### 3.1 契约字段与实现的对应关系

对**每一个** `ModelManifest` 字段，找出"谁读它、读了之后做什么、如果不读会怎样"：

| 字段 | 要验证的点 |
|---|---|
| `preprocess.source_map` | 每个 `SourceMap` 字面值都有实现？有没有声明了却没人用的（死字段）？`'identity'` 是否真的是恒等？ |
| `preprocess.scale` / `value_range` / `to_dtype` | 三者是否互相一致（如 `scale='div255'` 配 `value_range=(0,1)`）？`prepare` 与"调用方直接喂"两条路径是否都受 `value_range` 约束？ |
| `channels.channel_count_fixed` / `max_channels` | 一个"固定 5 通道"的方法传入 3 通道，是拒还是静默取前 3 个？ |
| `output_dim` / `output_dim_per_channel` | 二者的组合是否有唯一解释？`_check_output_width` 的"倍数容忍"会不会放过本该拒的宽度？ |
| `determinism` | 标 `stochastic` 的方法是否真的做了多次采样聚合？标 `deterministic` 的有没有隐藏的 dropout/随机初始化？ |
| `parity_mode` / `parity_prerequisites` | 是装饰性字段还是真被读？被谁读？ |
| `requires` | 与实际 `import` 是否一致（声明了没 import、import 了没声明，都算） |
| `weights.sha256` | 校验是默认开还是默认关？关着的时候调用方能知道吗？ |
| `ascii_path_required` | 置位的方法在非 ASCII 路径下是拒还是崩？ |

**这一类最常见的缺陷是"声明了但没人读"**——字段看着齐全，实际是文档。逐个 `grep` 使用点，
使用点为零的字段要么删、要么接上，不能留。

### 3.2 关键路径的逻辑推理

| # | 路径 | 要问的问题 |
|---|---|---|
| L1 | `prepare()` | 形状变换（BHWC→BCHW）与 dtype 转换的**顺序**会不会让守卫失效？守卫是在 resize 前还是后？ |
| L2 | `prepare()` | 幂等吗？对已经是 BCHW 的数组再调一次会怎样？ |
| L3 | `_check_output_width` | 允许"任意倍数"的理由成立吗？对每个方法都成立吗？（`ca_mae` 池化后宽度不随通道变 → 倍数规则对它是不是错的） |
| L4 | 适配器缓存 | `_cached_adapter` 以什么为键？换权重文件后缓存会失效吗？（构造：同路径换内容 → 必须报错或重载，不许用旧模型） |
| L5 | 剖面切换 | `set_profile` 是否清空缓存？切换后旧剖面加载的模型会不会被复用？ |
| L6 | 替身权重校验 | 三个替身的 sha256 是否在**每次**加载时校验？只校验文件名不算 |
| L7 | 分块 | `chunk_size` 切分是否保持行序？`_merge_result_parts` 拼接后是否与不分块逐位相同？ |
| L8 | 随机方法聚合 | 重复 N 次后是平均还是取第一个？N 从哪来？`repeats=3` 时 `max(2, repeats)` 的下界有意义吗？ |
| L9 | 错误聚合 | `on_error='collect'` 时 `errors` 的键是否与 `names` 对齐？部分成功的结果里 `sample_metadata` 行数还对得上吗？ |

### 3.3 与官方实现的一致性（只对这些可验）

| 方法 | 可验的东西 | 不可验的 |
|---|---|---|
| `deepprofiler` | 与官方 DP 图逐位对拍（`tests/test_parity_deepprofiler.py`） | — |
| `ca_mae` | 官方只有 `pixels/255`（`normalizer.py:7`）；`prepare` 输出必须与原始 uint8 逐位相等 | — |
| `cellclip` | 官方直接吃预计算向量；预处理计划里不得出现 source-map / resize | — |
| `o2vae` | 64×64 时与官方 `T.Resize(64)` 逐位相等 | 数据分布（官方训练于细胞剪影，这里喂核剪影） |
| `unidino` / `dinocells` / `morphogenie` | **仅接口一致性** | 效果（官方权重不可得，见 §6 豁免） |

审查时**不要**试图给后三个方法找"效果是否与论文一致"的证据，那是不可能完成的，
把它们当"接口能不能驱动"来验。

---

## 4. 软件设计规范检查

这一节不看功能对不对，只看**这个库能不能被别人接手**。

### 4.1 分层与依赖方向

约定：`contract`（纯数据，无框架） → `preprocessing` → `registry` → `embedding` → `adapters`。

| # | 检查 | 怎么验 |
|---|---|---|
| D1 | 依赖不反向 | `contract.py` 不许 import `embedding`/`adapters`/`registry`（`grep "^from\|^import"` 逐个看） |
| D2 | 无循环导入 | `python -c "import cellpaint_pipeline.models.contract"` 单独导入成功 |
| D3 | 纯模块保持纯 | `weights.py` 不许 import numpy/torch（docstring 自己声明了"pure"） |
| D4 | 门面惰性 | §1 第 1 条命令输出 0 |
| D5 | 适配器只做适配 | 适配器里不该出现 benchmark 逻辑、文件命名约定、CLI 参数解析 |

### 4.2 单一职责与可读性

| # | 检查 | 判据 |
|---|---|---|
| D6 | 超长函数 | `>100 行`的函数逐个说明理由；`check_compatible`(201) / `embed_all`(191) / `prepare`(150) 是否该拆 |
| D7 | 重复实现 | 同一件事有没有两处实现？（历史教训：`unit` 曾与 `channels`/`assume`/`masks` 各有一套映射解析） |
| D8 | 命名一致 | 同类概念是否同名？（`roots` vs `weights_roots` vs `weights_root`；`unit` vs `input_unit`） |
| D9 | 死代码 | 无人调用的私有函数、无人读的字段、注册表里 `PENDING` 与 `SURROGATES` 是否还有意义 |
| D10 | 注释与代码一致 | 抽查 10 处"为什么这样做"的注释，逐条验证是否成立 |

**D10 是重点。** 这个库的注释密度很高，这是优点，但注释一旦与代码不符，
比没有注释更糟——它会让下一个审查者相信一个假理由。历史上就出现过
"InstanceNorm 会抵消拉伸、残差 ≤0.7%"，实测是 21%（错了一个数量级）。

### 4.3 错误处理

| # | 检查 | 判据 |
|---|---|---|
| D11 | 异常类型统一 | 只有 `errors.py` 里那几类；`grep "raise [A-Z]"` 不应出现裸 `ValueError`/`RuntimeError` |
| D12 | 都带 `reason` | 每个 `IncompatibleInput` 都有稳定的 reason 码（调用方要能分支） |
| D13 | 错误信息可执行 | 信息里包含"怎么办"（改哪里、放哪个路径、设哪个环境变量） |
| D14 | 异常可编程消费 | 需要的字段是**属性**，不只是拼进 message（历史教训：`WeightsUnavailable.hint` 曾只存在于 `details` 里） |
| D15 | 不用 `except Exception` 兜底 | `grep -n "except Exception"` 逐处看，兜底处必须重新抛出或明确记录 |

### 4.4 测试规范

| # | 检查 | 判据 |
|---|---|---|
| D16 | 每个 manifest 字段有断言 | 逐字段找测试；找不到的就是"未受保护的声明" |
| D17 | 拒绝路径有测试 | §2.2 的 B9–B14 每条都要有对应测试 |
| D18 | 测试不依赖网络 | `grep -n "download\|http" tests/` |
| D19 | 测试能在缺框架的解释器里跳过 | `.venv` 下 `pytest` 应是 skip 而不是 error |
| D20 | 回归测试指向具体缺陷 | 测试名/docstring 里能看出"它防的是哪个 bug" |

---

## 5. 已知基线（数字对不上先查这里，别急着报回归）

| 项 | 基线值 |
|---|---|
| `.venv` 全量测试 | 324 passed / 13 skipped / **14 failed（既有）** |
| `ep_bench` 四个模型测试文件 | 116 passed |
| 审核器（`--device cpu`） | 66 项全过，约 72 秒 |
| v2 mAP_z | `cellclip 0.1337` / `ca_mae 0.1083` / `deepprofiler 0.0691` / `o2vae 0.0471` |
| v3 追加 | `unidino 0.0684` / `morphogenie 0.0574` / `dinocells 0.0460` |
| 宽度 | deepprofiler 672 / ca_mae 384 / cellclip 512 / o2vae 512 / unidino **1920** / dinocells 384 / morphogenie **10** |
| 官方权重 sha256 前缀 | `87b756f5` / `f6e5f1c9` / `66c2d32a` / `aa70cfa8` |
| 替身权重 sha256 前缀 | unidino `a0265184` / dinocells `5e172d8f` / morphogenie `49102997` |

数据集：`E:/bbbc022/cells`，3,857 细胞 / 54 化合物 / 64 孔。

---

## 6. 已批准的豁免（**不要**作为缺陷上报）

| 豁免 | 理由 | 正确的表述 |
|---|---|---|
| v3 三个方法效果与论文不一致 | 官方权重不可得（Zenodo ~10 KB/s ≈ 180 h；另两个未发布），权重为本地自训替身 | "接口一致，效果不可比" |
| o2vae 用核剪影而非细胞剪影 | BBBC022 无可靠胞质染色 | 必须与结论同句出现 |
| 不报 p 值 | 一个化合物恰好一个孔，化合物 ≡ 孔 | 改报 4 折排序稳定性 |
| o2vae 名次不被支持 | 4 折区间 0.0747–0.1179 与随机 0.0683–0.1109 重叠 | 其名次不作为结论 |
| silhouette 全为负 | 实测如此 | 不支持"形成可分离簇" |
| 审核不能与其它 GPU 作业并行 | 8 GiB 卡上会段错误（exit 139）；用 `--device cpu` 规避 | 运行约定，非代码缺陷 |

---

## 7. 缺陷分级

| 级别 | 定义 | 例子 |
|---|---|---|
| **P0** | 会产出**错误数值**且无人察觉，或门面在常规环境不可用 | 静默用错权重、双重缩放、宽度声明与实际不符 |
| **P1** | 该拒的没拒 / 该报的没报 | 值域越界被放过、异常类型错导致 `except` 捕不到 |
| **P2** | 能用但会误用、错得难查 | 错误信息没写怎么办、字段声明了没人读、注释与代码不符 |
| **P3** | 风格与整洁度 | 命名不统一、死代码、函数过长 |

**报告只写 P0/P1 也必须写出证据；P2/P3 允许批量列出但每条一行。**

---

## 8. 输出格式（严格照此提交）

```markdown
# 审查报告 YYYY-MM-DD

## 环境确认
- 解释器 / PYTHONPATH / CPC_WEIGHTS_ROOT
- §1 三条命令的实际输出
- 数据集：细胞数 / 化合物数

## 结论摘要
| 级别 | 数量 |
|---|---|
| P0 | n |
| P1 | n |
| P2 | n |
| P3 | n |

## 缺陷清单
### [P0-1] 一句话标题
- **位置**：`src/.../x.py:123`
- **复现**：
  ```bash
  <可直接粘贴执行的命令>
  ```
- **实际输出**：<片段>
- **期望**：<片段或描述>
- **为什么是缺陷**：<导致什么错误结果>
- **建议方向**：<改哪里；不要贴大段补丁>

## 待澄清（我不理解但不确定是缺陷）
- ...

## 已确认正常的部分（含你实际跑过的证据）
- ...
```

---

## 9. 交付前的自检

1. 每条缺陷都能用你写的命令**复现**吗？（换一个 shell 重跑一遍）
2. 有没有把 §5 的基线值当成回归？
3. 有没有把 §6 的豁免当成缺陷？
4. P0/P1 是否都说明了"会导致什么错误结果"，而不只是"这里不好"？
5. 有没有在报告里留下对源码的修改？（不应该有）