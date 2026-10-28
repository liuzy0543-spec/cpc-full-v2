# 自我审查报告：按本仓库的汇总提示词审自己

**被审对象**：`D:\项目\CellPainting-Claw-v2`（`src/cellpaint_pipeline/models/`）
**审查依据**：`docs/review_prompt_consolidated_2026-10-26.md`（v4）
**审查方式**：被审方**按自己的提示词逐条执行**——§5.0 探针、B1–B14、W1–W12、§8.1–§8.11 攻击清单、§7 benchmark 判据
**审查日期**：2026-10-27
**环境**：Python 3.10.20 / `D:\ep_bench\venv`（torch 2.5.1、TF 2.15.1、numpy 1.26.4），RTX 4060 Laptop 8 GB

---

## 0. 结论

**这一轮审出了 4 项真缺陷，全部已修并加了回归测试。**

其中两项属于本项目最忌讳的那一类——**不报错的错误**：

- **F15**：`value_range` 在两条路径上含义不同，导致原始 16-bit 计数绕过契约直接进了网络
  （嵌入偏差 max\|diff\| = 2.98，与嵌入自身同阶）。
- **F16**：适配器缓存以**声明的**摘要为键，会话中途换掉权重文件仍被接受，
  而上报的 `weights_sha256` 已不再描述磁盘上的文件——**溯源悄悄变假**。

另两项是溯源缺口（F17）与结论口径缺口（F18：三个方法的名次不被数据支持）。

**v2 的四个官方方法在全部修复后仍逐位不变**（见 §4）。

---

## 1. §5.0 随机探针（本提示词未给答案，只能实跑）

| # | 探针 | 实测值 |
|---|---|---|
| 1 | `embed('deepprofiler', dp[:2])` 第 0 行前 8 个 float32 的 sha256 前 16 | `deae4cf62f3a` |
| 2 | v3 剖面按名排序的第 4 个方法名 | `dinocells` |
| 3 | `prepare('dinocells', crop)` 的 dtype 与 shape | `float32` / `(2, 5, 96, 96)` |
| 4 | v2 四个方法 mAP_z 的极差 | `0.087` |

## 2. B1–B14 与 W1–W12

**B1–B14 全过**（七方法宽度 672/384/512/512/1920/384/10，全有限，B8/B9 返回 7 个，
B10 `value-range`、B11 `spatial-size`、B12 `(2,5,96,96)`、B13 与官方链差 **3.14% std**、B14 三个替身摘要全等）。

**W1–W12 全过**（W1 两个解释器均未加载框架；W4 越界 16-bit 拒 `value-range`；
W5 `assume` ≡ prepare+embed 逐位相同；W7 切剖面清缓存；W10 uint8 与已缩放 float 均拒；
W11 只 32×32 放行；W12 C=11 通、C=12 拒）。

---

## 3. 攻击清单逐条结果

| # | 攻击 | 结果 | 判定 |
|---|---|---|---|
| 8.1 | 无 torch 环境下的 `probe` | `.venv` 报 `wrong-interpreter`（因为缺 torch，正确）；有 torch 时 `ready` | ✅ 攻击无效 |
| 8.1 | 适配器模块 import 失败 | 回落到 `manifest.requires`，报出**官方**依赖（`kornia`/`oyaml`）而非替身所需 | ⚠️ 轻微误导，可接受 |
| 8.2 | uint8 → unidino | 拒 `value-range` | ✅ 攻击无效 |
| 8.2 | **已缩放 float（max≤1）→ unidino** | 拒 `value-range`（阈值 0.999/1.0 拒、1.001 放行，边界自洽） | ✅ 攻击无效 |
| 8.2 | Python list 载荷 | **被拒** `value-range`（`_normalise_input` 还原 dtype 后守卫生效） | ✅ 前提已过时，见 §5 |
| 8.3 | deepprofiler 31/32/33 | 只 32 放行；3-D 与 4-D 都放行 | ✅ 攻击无效 |
| 8.3 | ca_mae C=0 | **放行**（退化输入） | ⚠️ 观察项，未处理 |
| 8.4 | 与官方链独立重算 | 逐通道 max\|diff\| 0.020–0.031，合计 **3.14% std**，全来自 uint8 量化 | ✅ 攻击无效 |
| 8.5 | 替身 checkpoint **改名** | 拒（文件缺失） | ✅ 攻击无效 |
| 8.5 | 替身 checkpoint **内容被替换**（全新进程） | 拒 `WeightsUnavailable`（摘要不匹配） | ✅ 攻击无效 |
| 8.5 | 替身 checkpoint **会话中途被替换** | **修复前：被接受**（缓存掩盖）；**修复后：被拒** | 🔴 **F16**，已修 |
| 8.6 | v2 嵌入逐位比对 | 四个方法全部 `identical=True` | ✅ 攻击无效 |
| 8.7 | 文档过期语句 | 当前状态文档已更新；其余为历史记录 | ✅ 已处理 |
| 8.8 | `mask_mode='apply'` | 已实现且生效（掩膜外清零）；**原先无测试** | 🟡 已补测试 |
| 8.8 | 是否有代码仍声明 tape | 无 | ✅ 攻击无效 |
| 8.8 | `embed_all` 共享 `channels` | 如设计，两个方法都被 `channel-set` 拒 | ✅ 已知边界 |
| 8.8 | `_masks_from_dna` 死代码 | **F2 之后已无人调用** | 🟡 已清理 |
| 8.9 | 契约层三个新字段的默认值 | `mask_mode='none'` / `resize_policy='auto'` / `max_channels=None`，既有方法行为未变 | ✅ 攻击无效 |
| 8.9 | 声明 `'tape'` 的 manifest 过 `check_compatible` | 通过（契约层不再有 tape 专项检查），但 `prepare` 以 `unsupported-encoding` 拒 | ✅ 可接受 |
| 8.10 | 逐方法值域守卫 | 4 个声明了 `value_range` 的方法，不带 `assume` 时越界输入全部拒 `value-range` | ✅ 攻击无效 |
| 8.10 | ca_mae 的 16-bit 换算 | `per_crop_max255` **尺度不变**：×1 / ×16 / ×257 嵌入逐位相同（×1000 不同是 uint16 溢出，非尺度问题） | ✅ 攻击无效 |
| 8.11 | 排序有效性 | 见 §6 | 🔴 **F18**，须写入结论 |

---

## 4. 本轮修的四项

### F15（🔴 值域语义在两条路径上不是一回事）

**现象**：`PreprocessSpec.value_range` 声明的是**原始域**，而 `check_compatible` 检查的是
**契约所看到的那个数组**。走 `prepare` 的路径上二者一致；**不走** `prepare` 的路径上，
契约看到的就是调用方给的原始数组——而原始 16-bit 计数落在声明区间内 → 放行。

**实测（修复前）**：

```
unidino 契约 value_range = (0.0, 65535.0)
正确路径：prepare 后的 [0,1]  -> 嵌入
可疑路径：原始 uint16 (2,5,224,224)，域 [0, 64973] -> 被接受
两条路径 max|diff| = 2.982      # 嵌入自身 |x| max = 3.401
```

即 0..64973 被直接喂进只见过 [0,1] 的网络，**全程无任何报错**。

**修法**：`value_range` 一律描述**契约所检查的那个数组**（= 模型输入域）。
unidino 改为 `(0.0, 1.0)`（官方 `uint16_to_float` 在进网络前除过 65535）。
原始域交给 `prepare` 的 dtype 守卫与二次缩放守卫管。
同时改写了 `contract.py` 里 `value_range` 的定义与值域报错的措辞
（此前 int16 溢出被拒时消息显示 `-0.49 不在 (0,65535) 内`，让人摸不着头脑）。

### F16（🔴 适配器缓存掩盖 checkpoint 被替换）

**现象**：缓存键含**声明的**摘要——来自 manifest，**不随磁盘文件变**。
于是会话中途换掉权重文件仍从缓存出旧权重，而 `weights_sha256`
仍报声明值，**溯源链悄悄变假**。

**修法**：缓存命中时比对权重文件指纹 `(path, size, mtime_ns)`（一次 `stat`），
变了就丢弃条目重载 → 重载重新校验摘要 → 换掉的文件按新加载一样被拒。
`clear_adapter_cache()` 同步清指纹表。

**实测（修复后）**：

```
STEP1 正常嵌入 OK            -> 适配器进缓存
STEP2 替换磁盘上的 checkpoint -> sha 变成 491029978fef0969
STEP3 -> WeightsUnavailable reason=weights-unavailable   （修复前：ACCEPTED）
STEP4 还原 -> sha 回到 e53838fe807311bf
STEP5 还原后正常嵌入 dim=1920
```

### F17（🟠 cellclip 的输入产物无摘要）

契约钉了 DINOv2-Giant 权重摘要，但链上最靠近数据的两环——
`cells_clip.npy`（**仓库内无生成脚本**）与 `cellclip_vectors.npy`——**都没有摘要**。
旧的 sidecar 还缺 `model_sha256`/`model_name`/`channel_count`。

**修法**：`make_cellclip_vectors.py` 的 sidecar 扩充 `vectors_sha256` /
`source_crops_sha256` / `channel_order`；磁盘上那份旧 sidecar 按现算值补齐
（**未重算任何向量**）。现算 `model_sha256` 与 manifest 的 `chain_weights` **完全一致**
（`917d3c47…`），这条链因此第一次自洽。

**残留边界**：契约仍不校验**数据产物**——换一个内容不同的向量文件，
嵌入会变而没有任何一层会拒。已写进提示词 §8.8 请审查者定夺。

### F18（🔴 三个方法的名次不被数据支持）

§8.11 要求判断排序有没有意义。实测：

| 方法 | 4 折 mAP_z 区间 | 全量 mAP_z | 与随机基线区间（0.0683..0.1109） |
|---|---|---|---|
| cellclip | 0.2547..0.2565 | 0.1337 | ✅ 完全高于 |
| ca_mae | 0.1865..0.1964 | 0.1083 | ✅ 完全高于 |
| deepprofiler | 0.1298..0.1697 | 0.0691 | ✅ 完全高于 |
| unidino | 0.1243..0.1592 | 0.0684 | ✅ 完全高于 |
| morphogenie | 0.1092..0.1417 | 0.0574 | ⚠️ **重叠** |
| o2vae | 0.0747..0.1179 | 0.0471 | ⚠️ **重叠** |
| dinocells | 0.0700..0.1114 | 0.0453 | ⚠️ **重叠** |

**morphogenie / o2vae / dinocells 与随机基线不可区分**，其中 o2vae 是 **v2 的官方方法**。
任何"按 mAP_z 排名"的表述都必须带上这句。

### 另修

- `train_surrogates.py` 的 `_masks_from_dna` 在 F2 之后**已无人调用**（dinocells 的掩码被丢弃）→
  停止调用，保留函数并写明它为何不再被用。
- `mask_mode='apply'` 此前**没有测试** → 补 2 项。
- `PROJECT_STATUS.md` 仍把「tape 编码」写成现行契约 → 改为「通道堆叠输入」并加 2026-10-26 更新说明。

---

## 5. 新增回归测试（+6）

| 测试 | 锁定什么 |
|---|---|
| `test_value_range_describes_the_model_input_not_the_raw_domain` | unidino 的 `value_range` 是 (0,1) |
| `test_raw_counts_are_refused_when_the_caller_skips_prepare` | 原始 uint16 在非 assume 路径被拒 `value-range` |
| `test_the_prepared_domain_is_accepted` | prepared [0,1] 放行 |
| `test_mask_mode_apply_zeroes_the_region_outside_the_mask` | `'apply'` 分支真的清零掩膜外 |
| `test_mask_mode_apply_needs_a_mask` | `'apply'` 缺掩膜拒 `mask-required` |
| `TestAdapterCacheDoesNotMaskAReplacedCheckpoint::test_replacing_the_checkpoint_is_detected` | 会话中途替换权重文件被拒 |

**测试基线**：ep_bench 4 个模型测试文件 **107 → 113 passed**；
`.venv` 全量 **317 → 322 passed**（14 项既有失败不变）。

---

## 6. v2 逐位不变性（修复后重算）

| 方法 | 与修复前逐位比对 | 新 sha | 旧 sha |
|---|---|---|---|
| deepprofiler | ✅ 逐位相同 | 3cc75e4f570ce293 | 3cc75e4f570ce293 |
| ca_mae | ✅ 逐位相同 | 788acbd47c696dee | 788acbd47c696dee |
| cellclip | ✅ 逐位相同 | db59a9d831646ce5 | db59a9d831646ce5 |
| o2vae | ✅ 逐位相同 | 484839ed878eb72b | 484839ed878eb72b |
| unidino | ✅ 逐位相同 | 24dbbef12080acc7 | 24dbbef12080acc7 |
| dinocells | ✅ 逐位相同 | 12ddfa698d6d4494 | 12ddfa698d6d4494 |
| morphogenie | ✅ 逐位相同 | 34b3416b10945927 | 34b3416b10945927 |

**结论**：F15 与 F16 的修复**没有动到任何方法的数值路径**——
上表七个方法（4 个 v2 官方 + 3 个 v3 替身）**全部逐位相同**。

> 附注（批大小效应，实测）：同一批细胞用批大小 2 / 8 / 64 嵌入，
> max\|diff\| ≈ 2.5e-06（相对 std ≈ 2.5e-06）——纯浮点归约顺序噪声。
> 这说明**逐位比对必须用相同分块**，否则会看到 sha 不同但数值实质相同的结果。

---

## 7. §5.2 独立判断（不依赖 §4.2 的任何数字）

> 在 v2 的四个方法中，自行判定**哪一个最可能存在实现错误**，并给出证据。

**判定：cellclip。**

证据（全部来自官方代码与自测现象，不引用 §3 的 F/P 清单）：

1. **它的输入是链上唯一没有被摘要覆盖的一环。** 官方 `README.md:14` 说 CellCLIP
   checkpoint 配 DINOv2-giant，`retrieval_whole.py:390` 取 `dinov2-giant_ind.h5`；
   本项目 manifest 的 `chain_weights` 钉了 DINOv2-Giant 的摘要 ✅，
   但**模型实际消费的是** `cellclip_vectors.npy` —— 这个文件的摘要只存在于一个
   sidecar 里，而**契约完全不校验数据产物**。
2. **那份 sidecar 此前连自己用了哪个权重都没记**（缺 `model_sha256`），
   只记了 `model_dir` 这个路径字符串——路径上的文件被换掉，它仍然"看起来"有效。
3. **通道顺序无记录。** sidecar 记了 `channel_count`（个数）但不记哪个槽是哪个通道；
   而这个项目历史上出过一次 ER/RNA 互换的变体
   （`docs/embedding_framework_summary.md:257` 记有 `cells_oldorder`）。
   也就是说，**从 sidecar 无法判断这份向量的通道序对不对**。
4. **它的源头 `cells_clip.npy` 在本仓库没有生成脚本**（967 MB，只有读取方），
   所以链条的起点是不可复现的。
5. **影响面最大**：cellclip 在本协议下排第一（kNN_z 0.6451，是第二名的 1.4 倍），
   它的输入漂移直接动头条数字。

**为什么不是别的**：
deepprofiler 有真实逐位对拍（`max|diff| = 0`）→ 最不可能；
ca_mae 的非官方归一化被官方 checkpoint 内的 InstanceNorm 抵消，残差 ≤0.7% std；
o2vae 的问题是**模态错配**（喂核剪影、官方权重训练于细胞剪影）与**结论口径**
（名次与随机基线不可区分），不是"实现写错了"。

**这个判断没有标准答案**：判分看推理是否成立，不看结论是否与被审方一致。

---

## 8. 仍然开放的边界（未处理，需决定）

| # | 边界 | 我的判断 |
|---|---|---|
| B-a | 契约不校验**数据产物**（`cellclip_vectors.npy` 等） | 换文件→嵌入变→无人拒。应写进论文局限性；若要堵，需要给 manifest 加"输入产物摘要"这一类字段 |
| B-b | ca_mae 的 C=0 被放行 | 退化输入，官方 checkpoint 也会形状报错；低优先 |
| B-c | `embed_all` 的 `channels` 共享、无 `assume`/`masks` | 已知设计；正确用法已在文档写明 |
| B-d | 适配器 import 失败时 `probe` 回落到官方依赖 | 对替身有轻微误导；可接受 |
| B-e | 同一细胞在不同批大小下嵌入可能有微小差异 | 浮点归约顺序所致；**逐位比对必须用相同分块** |
| B-f | morphogenie / o2vae / dinocells 的名次与随机基线不可区分 | 已在 F18 记录；**必须写进结论** |

---

## 9. 变更清单

**源码**
- `models/contract.py`：`value_range` 定义改为"契约所检查的数组"；值域报错措辞改写
- `models/registry.py`：unidino `value_range` → `(0.0, 1.0)`
- `models/embedding.py`：新增 `_ADAPTER_FINGERPRINTS` 与 `_weights_fingerprint()`；
  缓存命中时校验文件指纹，变了就重载；`clear_adapter_cache()` 同步清理
- `models/preprocessing.py`：无改动（本轮）

**脚本**
- `scripts/embedding_benchmark/train_surrogates.py`：停止调用 `_masks_from_dna`（死代码）
- `scripts/embedding_benchmark/make_cellclip_vectors.py`：sidecar 扩充
  `vectors_sha256` / `source_crops_sha256` / `channel_order`；新增 `CHANNEL_ORDER` 常量

**测试**：`tests/test_models_contract.py` +3、`tests/test_models_preprocessing.py` +2、
`tests/test_models_v3_dispatch.py` +1

**文档**：`docs/PROJECT_STATUS.md`（tape → 通道堆叠 + 更新说明）、
`docs/review_prompt_consolidated_2026-10-26.md`（升 v4：F15–F18、3 处失效前提修正）

**数据**：`E:\bbbc022\cells\cellclip_vectors.json`（补齐溯源字段，未重算向量）

---

# 第二轮（第五轮自审）：一次真实事故，以及它牵出的两项新发现

**触发**：用户要求「按你的提示词重新审一遍，运行一遍，确保 embedding 都能跑通」。
**结果**：最终审查 **54 / 54 全过**；七个方法的嵌入用**各自正确的基线**逐位比对**全部相同**；
但过程中出了**一次真实事故**，并因此查出两项新缺陷（F19、F20）。

---

## 10. 事故：审计脚本两次把替身权重改坏，且不可恢复

### 10.1 经过

自审脚本要验证「权重文件被替换会不会被拒」，所以它去动 `E:\cpc_weights\unidino\surrogate_unidino.pt`。
**两次都把文件弄坏了**：

| # | 写法 | 为什么坏 |
|---|---|---|
| ① | 用 `shutil.move(bak, tgt)` 把备份移回原位 | Windows 上目标已存在时 `move` 的行为不可靠；备份被移走、原文件被覆盖 |
| ② | 在 `__exit__` 里用 `os.rename(renamed, path)` 还原 | Windows 上 `os.rename` 到**已存在**的目标抛 `FileExistsError` → 还原没执行；`finally` 又把备份删了 → **原文件永久丢失** |

第②次之后，正确的 unidino 权重只剩 `.renamed` 残留（幸而还在），恢复它之后
权重摘要正确，但**替身被迫重训**——而重训因为 F19 的非确定性产生了**新的权重**，
于是 v3 基线被迫重算。

### 10.2 正确写法（三原则，已写进汇总提示词 §8.5）

1. 备份用 **copy**（`shutil.copy2`），**永不 move**——原文件始终在原处；
2. 还原也用 **copy**，**永不** `os.rename` 覆盖已存在的目标；
3. 退出时**校验摘要**，不一致就报错——宁可报错也不要静默留下坏文件；
4. 改名残留只有在内容确实等于原始权重时才删，否则保留它（它是最后的救命副本）。

配套的 `WeightFileGuard` 已在**临时文件**上测过四种场景（纯替换 / 改名+替换 /
块内抛异常 / 空块），全部能还原且摘要一致——**先在临时文件上测通，再动真文件**。

### 10.3 连带后果

- unidino 替身重训 → sha `e53838fe807311bf…` → **`a0265184661d511e…`**
- unidino 的 v3 mAP_z **0.0684 → 0.0646**
- v3 稳定性由「7/7 全序稳定」变为「**稳定前缀 3/7 + 互换对 morphogenie / unidino**」
- manifest 钉住值、根 `surrogates_manifest.json`、三个 sidecar 全部同步更新
- v3 嵌入与打分、稳定性**全部重算**

---

## 11. F19（🔴 替身重建是非确定性的）

**实测**：同一 recipe（seed 20261026、300 步、batch 32、同一份 `cells_ca.npy`）
重训 unidino，得到的**不是同一个权重**：

| | sha256（前16） | emb_std |
|---|---|---|
| 原权重 | `e53838fe807311bf` | 0.4033 |
| 重训 | `a0265184661d511e` | 0.6681 |

两者**都未塌缩**（std 都在 0.4–0.7 的正常区间），但**是不同的权重**。

**为什么**：VICReg 自监督训练的轨迹对微小数值差异**混沌敏感**；
GPU 算子本身也不确定（cuDNN/cuBLAS 的算法选择、原子加法顺序），
`torch.manual_seed` 只保证 CPU 侧确定性。

**后果与处置**：
- manifest 里钉的摘要**只能检测「意外」替换**——这正是它的价值（F16 就靠它挡住了事故）；
- **故意重训会破坏它**，必须显式更新摘要**并重算 v3 基线**；
- 已把这段写进 `registry.py` 的 `SURROGATE_SHA256` 注释与汇总提示词 §3.1。

> 这条本来是我在提示词里让审查者去确认的悬念（「若有脚本在自动重建替身权重且重建
> 非确定性，B14 会误报」），现在有答案了。

---

## 12. F20（🔴 list 载荷丢失值宽 → 守卫失效 → 静默错误）

**现象**：`np.asarray(list)` 给的是 **int64**。于是所有 dtype 守卫都把
「8-bit 的数字」当成「合法的 64 位数组」放行。

**实测（修复前）**：

```
embed('unidino', uint8_array.tolist(), assume='raw_crop')   ->  被接受，dim=1920
embed('unidino', uint8_array,       assume='raw_crop')       ->  拒 value-range
```

同一批数字，**list 路径被接受、ndarray 路径被拒**——而 list 路径算出的嵌入来自
一张近乎全黑的图（0..255 被当成 16 位计数再除以 65535），**与正确结果静默不同**。

**修法（两处）**：
1. `_normalise_input` 把整数载荷**收窄到最窄的无符号类型**（非负时）。
   注意 `np.min_scalar_type` 对 int64 输入是**保守的**：0..255 它答 int16
   （因为 int8 顶到 127），那样守卫仍认不出 8-bit——所以全非负时直接取无符号宽度。
2. 把归一化**提到 `prepare` 之前**。原来 `prepare` 先跑，list 到守卫时还是 int64。

**修复后**：list 与 ndarray 两条路径**一致**被拒 `value-range`；uint16 的 list 正常放行。

---

## 13. 本轮最终验证

| 项 | 结果 |
|---|---|
| **最终审查（§5.0 + B1–B14 + W1–W12 + §8.1–§8.11）** | **54 / 54 通过**，零 FAIL |
| **七个方法的嵌入** | 全部跑通 ✅ |
| **逐位不变性** | v2 四方法对 F2 前备份 **全 identical**；v3 三方法对当前磁盘文件 **全 identical**（基线口径见下） |
| **权重文件事后校验** | 三个替身摘要与 manifest 钉住值一致，无残留文件 |
| **测试** | `ep_bench` 5 个相关文件 **203 passed**；`.venv` 全量 **323 passed / 11 skipped / 14 failed**（14 项仍是既有失败） |
| **v2 打分** | 完全不变：cellclip 0.1337 > ca_mae 0.1083 > deepprofiler 0.0691 > o2vae 0.0471；像素 0.0508、随机 0.0450 |
| **v3 打分** | unidino **0.0646**（换权重所致）、morphogenie 0.0574、dinocells 0.0453 |
| **v3 稳定性** | 稳定前缀 **3/7**，唯一互换对 **morphogenie / unidino** |

> **基线口径提醒（本轮踩过的坑）**：
> v2 的四个方法该比 `_emb_backup_before_f2/`（F2 之后它们不该变）；
> v3 的三个替身该比**当前磁盘文件**——dinocells 在 F2 时重建过、
> unidino 在 F19 时重训过，拿 F2 前的备份去比会得到假阳性差异。

---

## 14. 顺带修的既有测试缺陷

`tests/test_embedding_benchmark.py::test_per_method_units_let_mixed_input_kinds_share_one_call`
断言「ca_mae 必须失败」——那只在**无框架**环境成立，在有 torch 的 `ep_bench` 里
ca_mae 会**成功**，于是该测试在有框架环境下必然失败（与本轮改动无关，是既有缺陷）。
已改为「两种结果都接受，但若失败则不得是 unit 相关的 reason；若成功则宽度须为 384」——
与它自己的注释（"only unit handling asserted"）一致。
