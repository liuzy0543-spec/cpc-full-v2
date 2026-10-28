# 修改说明（交给执行模型）

**这份文档是自包含的。** 你不需要之前的对话，也不需要读别的文档就能开始。
它告诉你：这个项目要干什么、现在到哪一步、哪里错了、怎么改、改完怎么验。

**阅读顺序**：§1 目的 → §2 现状 → §3 问题 → §4 改法 → §5 顺序 → §6 审查 → §7 禁止。
如果你只有十分钟，读 §3 的 P0 和 §5。

---

## 1. 这个项目要干什么

### 1.1 一句话

把 Cell Painting 领域若干个 embedding 方法的**输入输出统一成一个接口**，
做到"调用一个方法名就得到 embedding"：

```python
from cellpaint_pipeline import models
emb = models.embed('ca_mae', crops)        # -> EmbeddingResult, .as_numpy() -> (N, 384)
```

### 1.2 为什么要有这个

这些方法的官方实现互不相容：有的吃 32×32 裁片，有的吃 256×256；有的吃 5 通道像素，
有的吃每通道预计算的 DINOv2 向量；有的吃核剪影而不是强度图；输出宽度从 10 到 1920。
不统一的话，**每做一次 benchmark 就要重写一遍数据加载**，而且每次都可能把通道顺序
搞错——那种错误不会报错，只会给出一个看起来正常但毫无意义的嵌入。

### 1.3 硬约束（这是整个项目的判据，不是建议）

> **输入与官方版本一致的情况下，中间产物和对应输出也应该一致。**
> 该用到 CellProfiler 的地方就要真跑一遍，以官方为准。

推论，全部不可违反：

1. **以官方为准**。任何"我们觉得更好"的归一化/预处理，都必须明确标注为"有意差异"，
   不得表述为"与官方一致"。
2. **中间产物可对拍**。预处理后的张量要能和官方代码在同一输入上逐位比较。
3. **拒绝优于猜测**。契约判断不了的情况，抛带 reason code 的类型化错误，
   不要静默猜一个。
4. **静默错误是头号敌人**。通道顺序反了、8bit 当 16bit 用、磁带方向反了——
   这些都会正常返回一个数，只是数是错的。

### 1.4 两个版本（**不要合并它们**）

| 版本 | 方法数 | 权重 | 能干什么 |
|---|---|---|---|
| `v2`（默认） | 4 | 全部官方 | **可发表**的 benchmark |
| `v3` | 7 | 4 官方 + 3 替身 | 只验证**接口能驱动全部方法** |

```python
models.set_profile('v3')   # 切到 7 个
models.set_profile('v2')   # 切回默认
```

`v3` 多出的三个（unidino / dinocells / morphogenie）用的是**本地训练的替身权重**
（`weights_provenance='surrogate'`）。它们的接口是真的，**分数不是论文的**。
任何把它们当官方方法报告的结论都是错的。

---

## 2. 现在到哪一步了

### 2.1 已建成的模块

| 文件 | 作用 |
|---|---|
| `models/contract.py` | 契约：`ModelManifest` / `ChannelSpec` / `PreprocessSpec` / `WeightsSpec` / `check_compatible()`（纯函数，不 import torch/TF） |
| `models/registry.py` | 注册表：`ADAPTERS`(v2 的 4) / `SURROGATES`(v3 的 3) / `PENDING`(取消的 3) / `set_profile()` / `describe()` / `probe()` |
| `models/preprocessing.py` | 执行 `PreprocessSpec`：通道重排、resize、`source_map` 数值映射、布局、dtype |
| `models/embedding.py` | 入口 `embed()` / `embed_all()`：契约检查→预处理→适配器→宽度校验→结果 |
| `models/adapters/*.py` | 7 个适配器 + `base.py` + `_surrogate.py`（替身共享 ViT） |
| `scripts/embedding_benchmark/*` | 生成嵌入、打分、稳定性、逐位对拍、替身训练 |

### 2.2 已验证到什么程度

**黑盒 7/7 通过**（输出宽度逐项吻合，无 NaN/Inf）：

```
deepprofiler 672 | ca_mae 384 | cellclip 512 | o2vae 512
unidino 1920 | dinocells 384 | morphogenie 10
```

**v2 benchmark**（BBBC022，3857 细胞，54 化合物，chance 0.0185，mAP_z）：

```
cellclip 0.1337 > ca_mae 0.1083 > deepprofiler 0.0691 > o2vae 0.0471
原始像素 0.0508    随机 0.0450
```

**v3 追加**（替身，mAP_z）：unidino 0.0684、morphogenie 0.0574、dinocells 0.0460。
稳定性：稳定前缀 5/7，唯一跨折互换对 `dinocells / o2vae`。

**逐位对拍**：只有 deepprofiler 做过实测（`max|diff| = 0.000000e+00`，样本仅 4 细胞）。
其余 6 个是"对照官方公式的推理验证"，**不是实测对拍**——报告里必须区分这两类。

**测试基线**：`.venv`（无 torch）300 passed / 9 skipped / 14 failed（14 项为既有失败，
全在 CLI/交付/编排冒烟层，与 embedding 无关）；`ep_bench`（有 torch）4 个模型测试文件 90 passed。

### 2.3 已做过的审查

`docs/review_report_2026-10-26.md` 里 14 项新发现（F1–F14）+ 8 条对审查提示词的意见（P1–P8）。
**F2 / F1 / F3 / F4 / P3 已由第二方独立实测复现，确认成立。** 见 §3。

---

## 3. 问题在哪里

按优先级。每条给：现象 / 证据 / 根因 / 影响。

### P0-1 ｜ F2：dinocells 把"磁盘存储格式"当成了"模型输入格式" 🔴

**现象**：官方 dinocells 在磁盘上把通道叠成一条"磁带" `(h, w*c)`；读进来后
`fold_channels` **展开**成 `(h, w, c)`、**丢掉 mask 通道**，再转成 `(C, h, w)` 喂给
`in_chans=num_channels` 的多通道 ViT。我们反过来：把 `(H,W,C)`+mask **折成磁带**
`(H, W*(C+1))` 直接喂一个单通道宽图 ViT。

**证据**（全部可复现）：

```
D:\official_src\6-dinocells\data_utils\cellpainting_dataset.py:23-26
    # Expected input image shape: (h, w * c)     <- 磁盘存的是磁带
    # Output image shape: (h, w, c)               <- 展开
    output = np.reshape(image, (image.shape[0], channel_width, -1), order="F")
    output = output[:, :, 0:-1]                   # drop: 丢掉 mask
:86   image = fold_channels(image, channel_width)  # 默认 mode="drop"
main_dino.py:369  /  run_get_features.py:88   in_chans=args.num_channels
config.yaml:12    num_channels: 4

本项目：manifest.preprocess.input_encoding = 'tape'
        prepare() 输出 = (96, 576)   <- 单通道宽图
        官方模型输入  = (5, 96, 96)  <- 多通道堆叠
```

补充实测：`_to_tape` 与官方 `fold_channels` **确实互逆**（`(96,96,5)`+mask →
`(96,576)` → 还原后逐位相等 `True`）。**"互逆"正是问题所在**：我们把上游的
**存储格式**实现对了，然后把它当成了**模型输入格式**。

**根因**：`preprocessing._to_tape`（`preprocessing.py:263-303`）的 docstring 写
"Mirrors the upstream `fold_channels` inverse"——这句话本身没错，错在调用方向。

**影响**：
- **不影响** v2 的官方方法数字，也**不影响**已跑出的 v3 替身分数（替身训练和推理都在
  磁带上，自洽）。
- **影响**契约本身：它是"将来换回官方权重"的唯一依据。
- 附带：`tape-batch` 拒绝（`preprocessing.py:272-278`）是本地编码的产物，不是官方约束——
  官方可以成批（`batch_size_per_gpu: 24`）。
- 替身模型比官方**多看了一个 mask 通道**，官方是 `mode="drop"` 丢掉的。

**注意**：有一种说法是"换权重后会**静默**产生错误嵌入"。这句**不成立**——
官方 5 通道 patch_embed 形状 `(384,16,16,5)` 与我们 1 通道 `(384,16,16,1)` 不兼容，
`load_state_dict` 会直接报错。真正的缺陷是"契约把官方几何描述错了"。

### P0-2 ｜ F1：`probe()` / `runnable()` 对 dinocells 报假阴性 🔴

**现象**：`probe('dinocells')` 说不可用（缺 `oyaml`），但 `models.embed('dinocells', ...)`
成功返回 `(1, 384)`。

**证据**：

```
manifest.requires            = ('torch', 'kornia', 'oyaml')   # 描述的是【官方】实现
probe.runnable               = False
probe.status                 = 'wrong-interpreter'
probe.missing_dependencies   = ['oyaml']
models.embed('dinocells', …) = (1, 384)                        # 实际可用
runnable()                   = 6 个方法（漏了 dinocells）
```

**根因**：`requires` 描述的是官方实现，而实际跑的是替身（`adapters/dinocells.py:47`
`requires_frameworks=('torch',)`）。`embed()` 走适配器自检，**从不校验 `manifest.requires`**；
`probe()` 只看 `manifest.requires`。两条判断路径不一致。

**影响**：诊断面不可信。用户按 `doctor()` 的提示会以为方法坏了，或者反过来以为能用。

### P0-3 ｜ o2vae 论文出处缺失 + 输入域偏移

**现象一**：`registry.py` 与文档都记"本仓库未记录论文出处"。
**补**：Burgess J, Nirschl JJ, Zanellati M-C, Lozano A, Cohen S, Yeung-Levy S.
*Orientation-invariant autoencoders learn robust representations for shape profiling of
cells and organelles.* **Nature Communications 15, 1022 (2024)**,
doi:`10.1038/s41467-024-45362-4`；代码 `github.com/jmhb0/o2vae`；Zenodo `10.5281/zenodo.10206848`。
（`mefs` = Mouse Embryonic Fibroblasts，不是任何缩写展开。）

**现象二（更要紧）**：官方 `config_mefs.py:19` 的 `data_dir` 指向 `scaled/cell`，
即**官方权重训练于细胞剪影**；本项目喂的是**核剪影**。这是推理期的域偏移，
嵌入分布与训练分布不一致。

### P0-4 ｜ ca_mae 的归一化定性不能写"与官方一致"

官方做法只有 `pixels.float()/255`（`normalizer.py:4-7`）+ checkpoint 内
`nn.InstanceNorm2d(None, affine=False)`（`huggingface_mae.py:88-91`），
**没有任何逐图拉伸，也没有 16→8 的转换代码**。
本项目用 `per_crop_max255`（逐样本全局最大值拉伸到 255）+ `scale='div255'`。

这是**源域选择**，不是官方行为。它之所以在数值上几乎无影响，是因为
`InstanceNorm2d` 对逐通道正仿射不变，拉伸被抵消（实测残差仅来自 uint8 量化网格，
max 0.0143 ≈ 输出 std 的 0.7%）。

**可以写**："拉伸被官方 checkpoint 内的 InstanceNorm 抵消，实测残差 ≤0.7% std"。
**不能写**："与官方一致"。

### P1-5 ｜ F3：unidino 喂 uint8 静默变黑 🟠

`div65535` 要求输入是 uint16。manifest 没声明值域，而 `contract.py:571-586` 的
dtype 守卫只在 `to_dtype='uint8'` 时触发（unidino 是 `float32`），所以 8bit 输入直接穿过。

实测：uint8 → prepared `max=0.00383`；uint16 → `max=0.98305`；**比值 257.0**。
也就是喂 uint8 得到一张近乎全黑的图，然后正常返回一个嵌入。

**这正是 §1.3 第 4 条说的头号敌人。**

### P1-6 ｜ F4：deepprofiler 输入尺寸无守护 🟠

`manifest.spatial = None`（**注意：审查提示词文档里写成 `(32,32)` 是错的，以代码为准**），
所以 `prepare()` 不 resize。喂 256×256 会穿透契约，在 Keras 内部抛裸
`ValueError: expected input_1 to have shape (32,32,5) but got (256,256,5)`。

**违反**"一切拒绝都用带 reason code 的类型化错误"。

**关于 `32` 这个数字的出处（别搞错）**：官方裁片尺寸**不是常量**，来自项目配置
`deepprofiler/imaging/cropping.py:125` `box_size = self.config["dataset"]["locations"]["box_size"]`
（官方自带 `tests/files/config/test.json` 里是 **16**）。所以我们 manifest 里的
`(32,32)` **不是**"官方规定 32"，而是**本仓库发货的那个 checkpoint 期望的尺寸**——
依据是它的 `input_1` 形状 `(32,32,5)`（上面那条 Keras 报错原文即可证实）。
写文档时必须按这个口径写。

### P1-7 ｜ F5 / F5b：morphogenie 的表述与引用失真 🟠

- `registry.py:465-467`、`:496`、`preprocessing.py:47` 都称 MATLAB 居中步骤
  "**不在仓库**"。**不成立**：`ImagePreProcessing/MultipleCells/CPA/GetCentImages.m`
  就在官方仓库内。缺的是 **MATLAB 运行时 + Cellpose + 原始数据**。
- 而且居中是 **200×200**（`:26` `Crop=[Centerx-100, Centery-100, 200, 200]`），
  256×256 的缩放发生在 Python 侧（`gan_training/utils.py:352-357`）。
- **F5b 张冠李戴**：`adapters/morphogenie.py:5-8` 与 `_surrogate.py:82-83` 引用
  `dVAE/model.py` 并称其返回三元组 `(recon, mu, logvar)`。实际：
  - 推理路径是 **`gan_training/models/dvae.py`**：`:393-396` 返回 `(c, mu, logvar)`，
    `:409-413` `c = reparametrize(mu, logvar)`（采样）。
  - `dVAE/model.py` 是 **legacy 训练脚本**，用 `z_dim`，返回**四元组**。
  - 结论（10 维瓶颈、官方 stochastic）**不受影响**，但引用会误导人。

### P1-8 ｜ F6：unidino 权重来源不可核 + notes 陈旧 🟠

- `registry.py:368` `source='unidino/checkpoints.zip'`——这个文件名在官方仓库
  **不存在**（全树 grep 无命中）。官方是 Zenodo `10.5281/zenodo.14988837`，
  脚本里 `run_dino_inference.sh:2` 用 `--ckpt ./checkpoints/uniDINO.pth`。
- `registry.py:394-395` 的 notes 自称"scale 未声明、官方未写明"，与同文件
  `:364` 的 `scale='div65535'` 和官方 `source/image_ops.py:14-20` 直接矛盾。

### P1-9 ｜ F7：ca_mae 放行官方处理不了的通道数 🟠

官方 `huggingface_mae.py:75` `max_in_chans=11`，pos_embed 按 11 生成；
本项目 `channel_count_fixed=False`。实测 `check_compatible` 对 C=20 返回 `ok=True`。
这类输入会在 checkpoint 内部形状报错，而不是在契约层被拒绝。

### P2-10 ｜ 其余（低优先，但都要处理）

| # | 问题 | 改法 |
|---|---|---|
| **F10** | 3 个替身在 manifest 里 `sha256=''`，实际值只在结果运行时出现 | 把已知摘要钉进 manifest（见 §4 F10） |
| **F8** | `embed_all` 的 `channels` 是所有方法共享的单一值 | 文档写明限制；短期不加参数 |
| **F9** | `embed_all` 无 `assume=` / `masks=` | 同上；正确用法是"先 `prepare` 再 `embed_all`" |
| **F11** | 替身场景下 `manifest.requires` 语义不统一 | 由 P0-2 的改法统一解决 |
| **F12** | `embed(skip_contract_check=True)` 可绕过契约 | 设计内，但要在文档里列为"高危开关" |
| **F13** | `chain-required` 只由 `prepare()` 抛，`check_compatible()` 从不产生 | 修正审查文档的判定标准（不是代码问题） |
| **F14** | ca_mae 论文标题错配 + "Valence Labs" 无依据 | 见 §4 F14 |

### P2-11 ｜ 对审查提示词文档本身的修正（P1–P8）

`docs/review_prompt_v2_v3_2026-10-26.md` 需要改这 8 处：

| # | 位置 | 改成 |
|---|---|---|
| P1 | §2.3 输出块 `sha=` | 标为 `weights_sha256[:12]`（不是嵌入数组哈希） |
| P2 | §3.2 deepprofiler `spatial=(32,32)` | 实际 `spatial=None`，并补 F4 后果 |
| P3 | §3.2 dinocells `norm_loader` "逐图逐通道 min-max" | 实际是**全图全局** min-max（`np.min(2D, axis=(0,1))` 是 0 维标量，已实测） |
| P4 | §3.2 morphogenie "不在仓库" | 源码在仓库内，缺运行时；居中是 200×200 |
| P5 | §3.1 / §3.5 G1 | 填入 o2vae 完整引用（§3 P0-3） |
| P6 | §3.2 cellclip "应以 `chain-required` 拒绝" | `check_compatible` 不产生该 reason，实际是 `ndim`/`feature-width`/`channel-count` |
| P7 | §3.2 deepprofiler 通道序历史缺陷 | ✅ 正确，保留 |
| P8 | §2.4 B4 | 明确"先 `models.prepare()` 预制备 + 传 `unit` 映射"，因为 `embed_all` 无 `assume`/`masks` |

另：§3.1 论文表里 ca_mae 标题归属错配，见 §4 F14。

---

## 4. 怎么改

**逐条给改法、验收测试、不能破坏什么。行号以本文档写作时为准，改前请先确认。**

### 4.1 改 F2（最重要，且会导致重训）

**决策**：改成官方几何（**推荐**），而不是"保留磁带 + 声明差异"。

理由：契约是"换回官方权重"的唯一依据。留着错的契约，等于给未来埋一个必然踩的坑。
代价是替身要重训、v3 数字要重跑——这是可接受的，数字随时能重跑，契约不能。

**改动**

1. `registry.py` dinocells manifest：
   ```python
   preprocess=PreprocessSpec(
       ...
       input_encoding='stack',        # 'tape' -> 'stack'
       # 官方 fold_channels 默认 mode="drop" 丢掉 mask；把它显式声明，别隐式
       mask_mode='drop',
       spatial=(96, 96),
       batch_layout='BCHW',
   )
   ```
2. `PreprocessSpec` **新增**字段
   `mask_mode: Literal['drop', 'apply', 'none'] = 'none'`
   （**当前不存在，需要你加**），并在 docstring 里写清语义
   （对应官方 `fold_channels` 的 `mode`）、默认值保持 `'none'` 以免影响其它方法。
3. `preprocessing.py`：删掉 `_to_tape` 的折叠逻辑，`input_encoding='stack'` 时
   **不折叠**，按 `mask_mode='drop'` 丢掉 mask；同时删掉 `tape-batch` 拒绝
   （官方可批）。
4. `adapters/dinocells.py`：ViT 用 `spatial=(96,96)`、`in_chans=5`（不再是
   `tape_spatial(...)` 的 1 通道宽图）。
5. `_surrogate.tape_spatial` 变为无人使用；保留但标注"仅 PENDING 契约参考"，
   或直接删除（删除更干净，但要同步改 `_surrogate.py` 的 `__all__` 与文档）。
6. **重训替身**：`python scripts\train_surrogates.py --methods dinocells --steps 300 --batch 32`
   （`_build_inputs` 里 dinocells 分支要去掉磁带构造，改成 `(N,5,96,96)` 堆叠）。
7. 重跑 `build_v3_surrogates.py --force` 与 v3 打分。

**验收测试**

```python
man = registry.manifest('dinocells')
assert man.preprocess.input_encoding == 'stack'
assert man.preprocess.mask_mode == 'drop'
prep = preprocessing.prepare(man, crops_bhwc_of_5_channels, masks=mask)
assert prep.shape == (N, 5, 96, 96)          # 不是 (N, 96, 576)
# 批处理不再被拒
assert prep.shape[0] == N > 1
```

并加一条**回归测试**：`prep` 与"手工按官方 `fold_channels` 展开"的结果逐位相等。

**不能破坏**：v2 的四个方法（deepprofiler / ca_mae / cellclip / o2vae）的数字必须
**一分不变**。改完立刻重跑 v2 打分与既有 `benchmark_scores_v2` 对比。

### 4.2 改 F1（诊断假阴性）

**改动**：让 `probe()` / `runnable()` 问**实际要跑的适配器**需要什么，而不是问
manifest 里为官方实现写的 `requires`。

```python
# registry.py 的 probe 逻辑
entry = _BY_NAME[name]
try:
    needs = entry.adapter_class.requires_frameworks      # 类属性，无需 import 框架
except AttributeError:
    needs = manifest.requires
```

`requires_frameworks` 是类属性，模块级不 import torch，所以可以在无框架解释器里读。

**验收**

```python
assert models.probe('dinocells')['runnable'] is True      # 装 torch 时
assert 'dinocells' in models.runnable()                    # 不再是 6 个
```
并在**没有** `kornia`/`oyaml` 的环境里确认它仍然报 runnable=True（因为替身不需要它们）。

**不能破坏**：`probe()` 仍须在无 torch/TF 的解释器里可调用（契约层不得引入框架 import）。

### 4.3 改 F3（unidino 静默变黑）

**改动**（两处都要）：

1. manifest 声明值域：`value_range=(0, 65535)`。
2. `contract.py` 的 dtype 守卫**不再只看 `to_dtype`**：当 `scale='div65535'` 时，
   输入 dtype 必须能表达 65535。若给 uint8，抛
   `IncompatibleInput(reason='value-range')`，并在 message 里写明"需要 uint16 或已缩放到
   65535 的值域"。

**验收**

```python
with pytest.raises(IncompatibleInput):   # reason='value-range'
    models.embed('unidino', uint8_crops, assume='raw_crop')
models.embed('unidino', uint16_crops, assume='raw_crop')   # 正常
```
并加一条**数值回归**：uint16 输入的 prepared `max ≈ 0.98`，与现在一致。

**不能破坏**：现有 v3 的 unidino 嵌入是用 `uint16*257` 生成的，改完必须逐位不变
（否则说明你顺手改了别的）。

### 4.4 改 F4（deepprofiler 尺寸守护）

**决策**：加**类型化拒绝**，不要静默 resize。

理由：静默 resize 会改变中间产物，而官方是 `crop_and_resize`，我们的 resize
不一定等价——那会引入一个**新的**静默差异，比现在的问题更糟。

**改动**

1. manifest 声明 `spatial=(32, 32)`——**依据是这个 checkpoint 的 `input_1` 形状
   `(32,32,5)`，不是"官方规定 32"**（官方尺寸来自 `config.dataset.locations.box_size`，
   官方 test 配置是 16）。若 `PreprocessSpec` 没有"声明期望尺寸但不自动缩放"的开关，
   **新增**一个（如 `resize_policy: Literal['auto','refuse'] = 'auto'`，**当前不存在**）。
2. `check_compatible()` 在尺寸不符时抛
   `IncompatibleInput(reason='spatial-size')`，message 里写明期望 `(32,32)` 与实际。

**验收**

```python
with pytest.raises(IncompatibleInput):   # reason='spatial-size'
    models.embed('deepprofiler', crops_256, assume='raw_crop')
models.embed('deepprofiler', crops_32, assume='raw_crop')   # 正常
```
**错误类型必须是 `IncompatibleInput`，不能是 Keras 的 `ValueError`。**

**不能破坏**：`cells_dp.npy`（32×32）路径的嵌入逐位不变。

### 4.5 改 F5 / F5b（morphogenie 表述与引用）

**改动**

1. 把 `registry.py:465-467`、`:496`、`preprocessing.py:47` 的"不在仓库"改成：
   > 源码在官方仓库内（`ImagePreProcessing/MultipleCells/CPA/GetCentImages.m`），
   > 缺的是 MATLAB 运行时、Cellpose 与原始数据。居中裁剪为 200×200
   > （`:26`），256×256 缩放发生在 Python 侧（`gan_training/utils.py:352-357`）。
2. 引用改为 `gan_training/models/dvae.py:393-396`（返回 `(c, mu, logvar)`）与
   `:409-413`（`reparametrize`）；标注 `dVAE/model.py` 是 legacy 训练脚本（四元组）。

**不能破坏**：`output_dim=10`、`determinism` 语义（官方 stochastic / 替身 deterministic）、
以及 `matlab_centring` 仍然**拒绝**（`reason='unsupported-preprocess'`）。
拒绝是对的，只是**理由**要改。

### 4.6 改 F6（unidino 权重来源）

```python
source='unidino/uniDINO.pth',                    # 官方脚本里的文件名
# 官方分发：Zenodo 10.5281/zenodo.14988837
```
并删掉 notes 里"scale 未声明"那句（与 `scale='div65535'` 矛盾）。

### 4.7 改 F7（ca_mae 通道数上限）

给 `ChannelSpec` **新增**字段 `max_channels: int | None = None`（**当前不存在**）：

```python
max_channels=11,     # 官方 huggingface_mae.py:75 max_in_chans=11
```
`check_compatible()` 在 `channel_count_fixed=False` 且通道数 > `max_channels` 时抛
`IncompatibleInput(reason='channel-count')`。

**验收**：C=20 → 拒绝；C∈{1,4,6,11} → 通过；C=5（本项目实际用的）→ 通过。

### 4.8 改 F10（钉住替身摘要）

把三个已知 sha256 写进 manifest：

```
unidino      e53838fe807311bf04bce00e790f9f1f1aa0854731be0f0bb06a5e01e09e6975
dinocells    5475ec885ae6b7d9d4afa6762801f9f00068e7c847b9d048bf3dd5c5ff0de9e5
morphogenie  491029978fef0969ed7ed56abaaf189faf34f6abc936c13e1487e877d41dc86d
```

⚠️ **dinocells 因为 F2 会重训，它的摘要一定会变**。重训后重新计算并更新；
不要直接抄上面这个旧值。

### 4.9 改 F14（论文归属）

- arXiv **2309.16064** = *Masked Autoencoders are Scalable Learners of **Cellular
  Morphology***（NeurIPS 2023）
- arXiv **2404.10242** = *Masked Autoencoders **for Microscopy** are Scalable Learners of
  **Cellular Biology***（CVPR 2024）
- 删掉 `adapters/ca_mae.py:5` 的 "Valence Labs"（官方仓库与两篇论文中均无依据，
  作者单位仅 Recursion）

### 4.10 改 P0-3 / P0-4（o2vae 与 ca_mae 的定性）

- **o2vae**：把 §3 P0-3 的完整引用写进 manifest 与文档。
  输入域偏移**必须**二选一处理：改喂细胞剪影，或在论文里明确声明"域外推理"。
  顺带记一笔：论文正文写 `d = 128`，而发布权重实测 `zdim=512`——**以代码/权重为准**，
  这条差异要写进补充材料，否则审稿人会以为我们算错。
- **ca_mae**：所有"与官方一致"的表述改为
  "拉伸被官方 checkpoint 内的 InstanceNorm 抵消，实测残差 ≤0.7% std"。

### 4.11 改 P1–P8（审查提示词文档）

按 §3 P2-11 的表逐条改 `docs/review_prompt_v2_v3_2026-10-26.md`。

---

## 5. 修改顺序与依赖

```
1. F1（诊断）        ← 先做，否则后面你无法判断"能不能跑"
2. F3 / F4 / F7      ← 都是契约守卫，互不干扰，可并行
3. F6 / F5 / F5b / F14 / P1–P8   ← 纯文档与元数据，零风险
4. F10               ← 等 F2 重训后才知道 dinocells 的新摘要
5. F2                ← 会触发重训，放最后
6. 重训 dinocells 替身 → 重跑 v3 嵌入 → 重跑 v3 打分与稳定性
7. P0-3 / P0-4       ← 与代码无关，但决定"能不能发表"，必须做
```

**依赖关系**：

- **F2 之前不要动 v3 基线**。F2 完成后 v3 的三行数字**会变**，
  旧基线作废，必须以新跑的结果为准。审查者**不得**把这种变化当作回归。
- F10 依赖 F2（dinocells 摘要会变）。
- F1 建议先做：它决定了你在别的修复后能否信任 `doctor()`。

**每一步之后都要**：跑 §6.1 的黑盒 + 回归，确认 v2 四个方法的数字**一分不变**。

---

## 6. 改完怎么审查

**审查要求与上一轮相同：黑盒 → 白盒 → benchmark，不许跳段、不许只跑部分方法。**

### 6.1 阶段一：黑盒（先做，七个必须全测）

**通过标准只有四条**（不看分数）：

1. 返回而非抛异常
2. `as_numpy()` 是二维 `(N, dim)` 浮点数组
3. 全部有限
4. `dim` 与 manifest 一致

**七个方法的输入构造**（数据在 `E:\bbbc022\cells`，权重根 `E:\cpc_weights`）：

```python
models.set_profile('v3')
block = np.asarray(np.load(r'E:\bbbc022\cells\cells_ca.npy', mmap_mode='r')[:2])
bhwc  = np.moveaxis(block, 1, -1).astype('uint16') * 257      # uint16，F3 之后这是必须的
mask  = (block[:, 0] > block[:, 0].mean(axis=(1, 2), keepdims=True)).astype('uint8')

models.embed('deepprofiler', np.load(...+'\\cells_dp.npy')[:2], assume='raw_crop')   # 32x32
models.embed('ca_mae',       bhwc, assume='raw_crop')                                 # 256x256
models.embed('cellclip',     np.load(...+'\\cellclip_vectors.npy')[:2])               # (N,5,1536)
models.embed('o2vae',        np.load(...+'\\cells_o2.npy')[:2])                       # 剪影
models.embed('unidino',      bhwc, assume='raw_crop')                                 # 224x224
models.embed('dinocells',    bhwc[0], masks=mask[0], assume='raw_crop')               # 改后仍单细胞
models.embed('morphogenie',  block[:, :3].astype('float32') / 255.0,
             channels=('C0','C1','C2'))
```

**期望宽度**：672 / 384 / 512 / 512 / 1920 / 384 / 10。

**本次新增的黑盒项**（上一轮没有）：

| # | 检查 | 期望 |
|---|---|---|
| B8 | `probe('dinocells')['runnable']` | **True**（F1 修好了） |
| B9 | `runnable()` 返回 7 个 | 不是 6 个 |
| B10 | uint8 喂 unidino | **抛 `IncompatibleInput`**（F3 修好了） |
| B11 | 256×256 喂 deepprofiler | **抛 `IncompatibleInput(reason='spatial-size')`**，不是 Keras `ValueError`（F4） |
| B12 | `prepare(dinocells)` 输出形状 | `(N,5,96,96)`，且 N>1 不报错（F2） |
| B13 | dinocells 与官方 `fold_channels` 展开结果 | **逐位相等** |
| B14 | `weights_sha256`（替身） | 与 manifest 钉住的值一致（F10） |

### 6.2 阶段二：白盒

**逐方法对照论文与官方代码**，源码在 `D:\official_src\`，权重在 `E:\cpc_weights`。
完整对照表见 `docs/review_prompt_v2_v3_2026-10-26.md` §3.1（改完 P1–P8 后）。

**本次必须重点复核的点**：

| 检查 | 判定 |
|---|---|
| dinocells 模型输入是 `(C,H,W)` 且 mask 被丢 | 与 `cellpainting_dataset.py:23-39` + `in_chans=num_channels` 一致 |
| `_to_tape` 是否已删除/未使用 | `grep -r "_to_tape\|tape_spatial" src/` |
| `input_encoding='stack'` | manifest 与实际 prepared 形状一致 |
| unidino 值域守卫 | `contract.py` 在 `scale='div65535'` 时检查 dtype |
| deepprofiler 尺寸守卫 | 拒绝是 `IncompatibleInput`，不是框架异常 |
| `probe()` 的依赖来源 | 读适配器类，不读 manifest.requires |
| morphogenie 引用行号 | `gan_training/models/dvae.py:393-396`，不是 `dVAE/model.py` |
| ca_mae 论文归属 | 2309.16064 与 2404.10242 各自标题正确 |

**三态标注**：✅ 一致 / ⚠️ 有意差异（引用编号）/ ❌ 不一致（**必须给官方行号 + 本项目行号 + 差异**）。

### 6.3 阶段三：benchmark

```powershell
python scripts\embedding_benchmark\embed_dataset.py      --cells-dir E:\bbbc022\cells --force
python scripts\embedding_benchmark\build_v3_surrogates.py --force
python scripts\embedding_benchmark\score_benchmark.py    --profile v2 --output ...\v2_after_fix.json
python scripts\embedding_benchmark\score_benchmark.py    --profile v3 --output ...\v3_after_fix.json
python scripts\embedding_benchmark\stability_check.py    --profile v3 --folds 4
```

**判据（这两条是本次审查的核心）**：

1. **v2 四个方法必须与修改前逐位一致**：
   ```
   cellclip 0.1337 > ca_mae 0.1083 > deepprofiler 0.0691 > o2vae 0.0471
   原始像素 0.0508    随机 0.0450
   ```
   **任何偏差都是回归**——本次修改不动 v2 的任何路径。
2. **v3 三个替身的数字会变**（dinocells 重训 + F2 几何改变）：
   - 旧值 `unidino 0.0684 / morphogenie 0.0574 / dinocells 0.0460` **作废**
   - 新值当作**新的基线**，并在文档里写明"基线已于 <日期> 因 F2 修复而更新"
   - **不得**把变化描述为"性能下降/提升"——权重和几何都换了

**必须随表附带**（一条都不能省）：

1. 不给 p 值（每化合物单孔，化合物身份 ≡ 孔身份）
2. 替身行必须标注
3. 不可与论文数字比较
4. o2vae 是形状模型 + 域偏移
5. 稳定性报"稳定前缀 + 互换对"，不只报布尔
6. silhouette 全为负 → **"嵌入形成可分离簇"这个说法在本数据上不成立**
7. mock-AUC 除少数外都在 0.47–0.54，随机基线 0.4996 → **不宜用于方法排序**

### 6.4 交付物

一份 `docs/review_report_after_fix_<日期>.md`，含：

1. 黑盒表（七行 + B8–B14）
2. 白盒表（逐方法 × 逐检查点，三态）
3. F1–F14 与 P1–P8 的**逐条关闭确认**
4. benchmark 表（v2 逐位对比 + v3 新基线）
5. 结论三档：v2 能否发表 / v3 是否仅接口验证 / 哪些结论数据不支持

---

## 7. 禁止事项

| 禁止 | 原因 |
|---|---|
| 只跑部分方法就下结论 | 用户明确要求"务必全部测试" |
| 把替身分数当方法性能 | 权重不是官方的 |
| 给 p 值或"显著优于" | 单孔数据结构上不成立 |
| 用"看起来对"代替运行 | 本项目的缺陷都是静默型的（通道序 / 磁带方向 / 8bit 当 16bit） |
| 跳过黑盒直接白盒 | 会把"跑不起来"误诊成"实现错" |
| 声称除 deepprofiler 外某方法已与官方逐位一致 | 只有那一个有实测对拍 |
| 为了"通过测试"而放宽断言 | 契约的价值在于拒绝 |
| 把 F2 修复引起的 v3 数字变化当作回归 | 权重和几何都换了，不可比 |
| 未经核实就照抄本文档的行号 | 行号会漂移，改前先确认 |

---

## 8. 交接检查清单

开工前确认你能回答：

- [ ] 这个项目为什么需要统一接口？（§1.2）
- [ ] 硬约束是什么？哪四条推论？（§1.3）
- [ ] v2 和 v3 的区别？为什么不能合并？（§1.4）
- [ ] F2 的官方证据在哪三处？（§3 P0-1）
- [ ] 为什么"磁带互逆"反而是问题？（§3 P0-1）
- [ ] 改 F2 会让哪些数字作废？为什么不算回归？（§5、§6.3）
- [ ] 为什么 F1 要先修？（§5）
- [ ] v2 的数字在本次修改中必须是什么状态？（§6.3 第 1 条）

收工时，以下必须为真：

- [ ] 七方法黑盒全过，宽度 672/384/512/512/1920/384/10
- [ ] B8–B14 全过
- [ ] v2 四个方法数字**逐位不变**
- [ ] v3 三个替身数字已更新并注明基线变更原因
- [ ] F1–F14、P1–P8 逐条有结论（修好 / 有意保留 / 转下一轮，且写明理由）
- [ ] 没有任何一条结论超出数据支持范围