# Embedding 审核循环

自动化执行：**跑全部方法 → 审核 → 问题写 md → 修复 → 重跑**，直到审核通过。
停止标准写在 `docs/verify_prompt_e2e_2026-10-27.md`；判定标准由
`scripts/embedding_benchmark/audit_round.py` 的六个闸门机器执行。

## 怎么跑

```bash
# 一轮：重算全部产物 + 审核（产物过期或代码刚改过时用这个）
bash scripts/embedding_benchmark/audit_loop.sh 1

# 只审核，不重算（调试审核器时用）
SKIP_PIPELINE=1 bash scripts/embedding_benchmark/audit_loop.sh 1

# 审核器单独跑
D:/ep_bench/venv/Scripts/python.exe scripts/embedding_benchmark/audit_round.py --round 6
```

### ⚠️ 审核不能与其它 GPU 作业并行

审核的 G1 会**同时**加载七个方法的权重（含 711 MB 的 ca_mae 与 4.5 GB 的
DINOv2-Giant 依赖路径）。8 GiB 卡上若另有作业占着显存，进程会直接
**段错误**（exit 139）而不是抛 `OutOfMemory`，且此时一条闸门结果都不会写出。

实测：2026-10-27 19:33 的第 10 轮在 `make_cellclip_vectors.py` 占着
7484/8188 MiB 时崩在 G1 中途；同一份代码在显存空闲时通过。
**跑审核前先确认 `nvidia-smi` 显存空闲**，或直接钉到 CPU：

```bash
python scripts/embedding_benchmark/audit_round.py --round 10 --device cpu
```

CPU 上整轮约 **72 秒**（GPU 空闲时约 65 秒），比等显存划算得多。

`audit_loop.sh` **不修任何东西**：能自己修输入的循环可以靠放宽闸门"通过"。
修复必须由人（或下一轮的你）读 findings 后动手。

## 六个闸门

| 闸门 | 内容 | 为什么单独列 |
|---|---|---|
| **G1** runs | 七个方法返回 `(N,dim)`、有限、宽度符合契约、带权重摘要 | "跑出数"最容易被当成"跑对" |
| **G2** dispatch | `embed_all` 一次调用驱动七个（`assume`/`channels` 按方法传）；无误报；结果字段集合一致；provenance 正确 | 统一 IO 的核心承诺 |
| **G3** guards | uint8→unidino 拒 `value-range`、256→deepprofiler 拒 `spatial-size`、v2 够不到替身、`probe` 不假阴性、dinocells 丢 mask | 守卫缺失是静默错误 |
| **G4** determinism | 同输入两次调用逐位相同 | 否则 benchmark 不可复现 |
| **G5** artefacts | benchmark/stability/IO 产物存在；**v2 四个 mAP_z 钉死** | 官方方法路径不许漂移 |
| **G6** parity | 四个官方方法各自与上游对拍或给出实测等价证据 | 这是"效果一致"唯一可验的部分 |

**G6 的四条证据**（不是"看起来一样"）：

| 方法 | 证据形式 | 判据 |
|---|---|---|
| deepprofiler | 与官方 DP 图逐位对拍（`tests/test_parity_deepprofiler.py`） | 测试必须过 |
| ca_mae | 声明的 `source_map` 必须是 `identity`，且 `prepare` 输出与原始 uint8 **逐位相等** | 官方只有 `pixels/255` |
| cellclip | 预处理计划里**不得**出现 `source-map` / `resize` | 官方直接喂预计算向量 |
| o2vae | 剪影进出一律二值；**64×64 时与官方 `T.Resize(64)` 逐位相等** | 官方 `config_mefs.py:28` 就是 `T.Resize(64)` |

**v3 三个替身不在 G6 范围内**，这是目标里写明的豁免：官方权重不可得，只能证明
接口能驱动它们，不能证明效果与论文一致。任何把它们的效果说成"与论文一致"的报告
都是错的。

## 轮次记录

| 轮 | 结果 | 阻塞项 | 性质 |
|---|---|---|---|
| 01 | FAIL | 4 | **全是审核器自身的 bug**（`embed_all` 载荷形状、3 个对拍误用 `prepare`） |
| 02 | FAIL | 1 | 暴露出**真问题**：ca_mae 的 `per_crop_max255` |
| 03 | PASS | 0 | 修掉审核器的布局比较 bug 后通过（本轮不重算产物） |
| 04 | PASS | 0 | 重算全部产物后仍通过 |
| 05 | PASS | 0 | 改注释后冒烟 |
| 06 | PASS | 0 | **补齐 o2vae 的独立对拍**（此前只有行为断言） |
| 07 | PASS | 0 | G2 改为测**统一调用形式**（不再替调用方手工 `prepare`），66 项 |
| 08 | PASS | 0 | e2e 重算后复跑（七个嵌入与上一轮逐位相同） |
| 09 | PASS | 0 | 修 `unidino.output_dim`（384→1920）后复跑 |
| 10 | PASS | 0 | `unit`/`channels`/`assume`/`masks` 合并为同一解析器；本轮在 **CPU** 上跑 |

### 第 06 轮：o2vae 的对拍是补出来的

第 04 轮之前，o2vae 这条门断言的是"128×128 块复制掩码缩到 64×64 应与原图一致"。
**这条判据是我自己发明的，比官方更严**：官方 `T.Resize(64)` 是双线性，
对 128×128 掩码会糊出灰度（实测 `max|diff|` 0.4375），并不等于原图。

现在改成直接与官方变换对比：

| 情形 | 官方 `T.Resize(64)` | 本项目 | 判定 |
|---|---|---|---|
| 64×64（数据集实际尺寸） | 恒等，保持二值 | 恒等，保持二值 | **逐位相等** ✅ |
| 128×128 | 双线性糊成灰度 | 双线性后按 0.5 重新二值 | 阈值化后**逐像素一致率 100%**，只有灰度层不同 |

即：差异只在"是否保留灰度"，不在掩码本身；且数据集原生 64×64，这条路径从未被触发。
写成 `blocking=False` 的信息项，判据里如实地既不说它是缺陷，也不说它"与官方一致"。

### 第 02 轮：唯一一个真实缺陷

**现象**：`ca_mae` 声明 `source_map='per_crop_max255'`（把每个裁剪按自身最大值拉伸到
0..255），官方 `normalizer.py:7` 只有 `pixels / 255.0`。**这是编外步骤。**

**原文档理由**："checkpoint 内的 InstanceNorm 会抵消拉伸，残差仅 ≤0.7% std。"

**实测结论**：这条理由**半对，而错的是要害那一半**。
（4 个裁剪、384 维、关掉拉伸后缩放输入）

| 输入缩放 | cosine | `max|diff|` | 占输出 std |
|---|---|---|---|---|
| ×0.8 | 0.99970 | 0.0568 | **21.0%** |
| ×0.5 | 0.99948 | 0.0697 | **25.8%** |

即 InstanceNorm 保住的是**方向**，不是**数值**。原文的 0.7% 把量级低估了约一个数量级。

**它为什么一直没出事**：BBBC022 全部 **3,857** 个裁剪的 per-crop max **都是 255**，
所以拉伸在这份数据上是**恒等变换**——从来没真的触发过。

**修复**：新增 `SourceMap='identity'`（显式"不改值"），ca_mae 改用它。

**修复的代价**：零。重算后四个官方方法嵌入与旧备份**逐位相等**
（`ca_mae` max\|diff\| = 0.0），v2 打分仍是 `0.1337 / 0.1083 / 0.0691 / 0.0471`。

**顺带修正**：`adapters/ca_mae.py` 的 docstring、`registry.py` 的注释、
`review_prompt_consolidated` §3.4 都已改为上面的实测表，不再有"0.7%"。

### 其余三轮的教训

01 轮的 4 个阻塞项**全是审核器自己写错**：把 `embed_all` 的载荷当成 kwargs 字典、
对 cellclip/o2vae 硬调 `prepare`（它们声明的流水线不该被这样调用）。
这说明审核器本身也要被审——一个写错的闸门会报出不存在的缺陷，
而**误报比漏报更贵**，因为它会引导出错误的"修复"。

03 轮的阻塞项同样是审核器的 bug：`prepare` 返回 BCHW，我拿 BHWC 去比。

## 本轮达标状态

| 标准 | 状态 | 证据 |
|---|---|---|
| 七方法全部跑通、宽度/有限性正确 | ✅ | G1，28 项检查（六闸门共 66 项）|
| 统一 IO 一次调度七个、字段一致 | ✅ | G2 + `io_uniformity_v3_e2e.json` 7/7 |
| 四个官方方法与官方代码一致 | ✅ | G6 四条证据 |
| v2/v3 benchmark 跑通、v2 稳定可复现 | ✅ | G5，四个数字钉死 |
| v3 效果与论文一致 | ⚠️ **不适用** | 替身权重，明确豁免 |

**最后一次全流程**：2026-10-27 17:47:20 结束，七步全 `exit=0`（总耗时约 10 分钟）；
随后审核轮次 09 `PASS (66 checks, 0 blocking)`。

**测试基线**（本轮更新，因新增 1 个测试）：

| 环境 | 结果 |
|---|---|
| `.venv` 全量 | 324 passed / 13 skipped / 14 failed |
| `ep_bench` 全量 | 334 passed / 15 failed |
| `ep_bench` 4 个模型文件 | 116 passed（新增 2 条统一调度测试）|

## 已知仍未解决（不属于本轮目标）

1. **o2vae 域偏移**：官方权重训练于细胞剪影（`config_mefs.py:19` → `data/mefs/scaled/cell`），
   这里喂的是核剪影。与官方"一致"仅指代码路径，不指数据分布。
2. **16→8 转换上游未定义**：仓库里没有 16-bit 到 8-bit 的转换代码，
   所以 ca_mae 对 16-bit 输入只有"拒绝"这一个诚实选项。
3. **o2vae 名次不被数据支持**：其折区间 0.0747–0.1179 与随机基线 0.0683–0.1109 重叠。
4. **cellclip 的数据产物无摘要**：`cellclip_vectors.npy` 被换掉，契约不会报错。