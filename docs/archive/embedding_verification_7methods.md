# 7 个 Embedding 方法：与官方论文/源码的一致性验证

> 日期：2026-10-10
> 官方源码：`D:\official_src\{1-deepprofiler,2-camae-openphenom,3-cellclip,5-unidino,6-dinocells,7-morphogenie}` + `E:\cpc_weights\o2vae_repo`
> 我方实现：`D:\cpc_c_v2\src\cellpaint_pipeline\models\`
> 探针脚本：`D:\项目\cellpainting claw复现\_probe\verify_*.py`

---

## 0. 验证方法与边界（先说清楚）

用户要求：「拿它们的示例简单跑一下，能运行出期望结果就可以」。

因此本轮的判据是**架构契约可运行**，而非**权重复现论文指标**。分两档：

| 档 | 含义 | 覆盖 |
|---|---|---|
| **A 档：真实权重端到端** | 官方权重 + 官方/我方代码，实际产出嵌入 | 4 个 |
| **B 档：官方代码架构验证** | 跑**官方原代码**（随机权重），验证输入输出契约与申报维度 | 3 个 |

**B 档能证明什么、不能证明什么**：
- ✅ 能证明：官方代码确实吃这种输入、出这种维度、走这条通路
- ❌ 不能证明：训练后的表示质量、论文报告的指标

**随机权重是有意为之**：要验的是「输入什么、输出什么」，而不是「学得好不好」。

---

## 1. 总表

| # | 方法 | 档 | 官方产物 | 我方产物 | 一致 |
|---|---|---|---|---|---|
| 1 | DeepProfiler | **A** | `(2, 672)` | `(4, 672)` | ✅ |
| 2 | CA-MAE / OpenPhenom | **A** | `(2, 384)` | `(2, 384)` | ✅ |
| 3 | CellCLIP | **A** | `(3, 512)` | `(3, 512)` | ✅ |
| 4 | O2-VAE | **A** | `(N, 64)` | `(N, 64)` | ✅ |
| 5 | uniDINO | **B** | `(2, 3, 1920)` = 5×384 | 未实现 | ⚠️ 见 §3 |
| 6 | DINO4Cells | **B** | `(2, 384)` | 未实现 | ✅ 维度一致 |
| 7 | MorphoGenie | **B** | `(4, 10)` | 未实现 | ❌ 我方 manifest 错 |

---

## 2. A 档：4 个真实权重端到端（已通过）

| 方法 | 权重 | 官方/我方输出 | 决定性 |
|---|---|---|---|
| DeepProfiler | `Cell_Painting_CNN_v1.hdf5` 19,236,600 B | `(4, 672)` | `array_equal=True`, maxdiff `0.000e+00` |
| CA-MAE | `openphenom/model.safetensors` 712,434,294 B | `(2, 384)` | `array_equal=True` |
| CellCLIP | `cellclip/model.safetensors` 1,799,564,799 B | `(3, 512)` | `array_equal=True` |
| O2-VAE | `o2vae_50ep.pt` 9,297,256 B（**自训**） | `(N, 64)` | — |

> ⚠️ O2-VAE 的权重是**本项目自训的 50 epoch**（官方从未发布权重），故它只验证**接口形状**，
> **不是**论文复现。

---

## 3. B 档：3 个官方架构验证（本轮新增）

### 3.1 DINO4Cells —— ✅ 维度与官方代码一致

**探针**：`verify_dinocells_arch.py`（跑官方 `archs/vision_transformer.py`）

```
arch        in_chans  out shape         embed_dim   params
vit_tiny    4         (2, 192)          192         5.6M
vit_small   4         (2, 384)          384         21.8M
vit_small   5         (2, 384)          384         21.9M
vit_base    4         (2, 768)          768         86.0M
```

- 官方 CLS 取值行：`archs/vision_transformer.py:216` → `return x[:, 0]`
- 我方 manifest `output_dim=384` → **与 vit_small 一致** ✅

**发现的问题（我方）**：`scale` 申报可能与官方不符。

官方 `data_utils/file_dataset.py:77-82`：
```python
77:  def norm_loader(path, training=True):
79:      img = img.astype(float)
80:      img -= np.min(img, axis=(0, 1))
81:      img *= 255 / np.max(img, axis=(0, 1))
82:      return img.astype(np.uint8)
```
这是**逐通道 min-max 归一到 0-255**，而 `config.yaml:81-82` 又有 `self_normalize: True`。
我方 manifest 写的是 `scale='per_image_selfnorm'` —— **方向对，但官方实际是「min-max 到 uint8」+「self_normalize」两步**，我方只用了一个标签概括，**精度不足**。

**另一个缺口**：仓库自带的 `config.yaml` 是 **HPA 场景（`num_channels: 4`）**。Cell Painting 的
checkpoint 只在 `README.md:211-213` 列出，**仓库里没有任何 Cell Painting 的配置**，
所以 Cell Painting 版的 `num_channels` **在代码里查不到**（论文层面应为 5）。

### 3.2 uniDINO —— ⚠️ 架构验证通过，但我方 manifest 有**真 bug**

**探针**：`verify_unidino_arch.py`（跑官方 `source/inference_utils.py:11` + `source/models.py:17`）

```
model: vit_small, in_channels=1, patch_size=16, embed_dim=384     params 21.5M
input crops shape: (2, 5, 3, 1, 224, 224)
forward_inference output: (2, 3, 1920)
  expected (batch, crops, channels*384) = (2, 3, 1920)   MATCH: True
per-(channel,crop) embedding from the bare ViT: (30, 384)
  -> 5 channels x 384 = 1920, matches last dim: True
```

官方拼接行 `inference_utils.py:29-34`：
```python
29:  batch_of_embeddings = rearrange(
30:      batch_of_embeddings,
31:      "(batch ch crops) h -> batch crops (ch h)",
```

**验证结论**：`384 × C` 确实是 `rearrange` **把逐通道嵌入拼接**出来的，与论文一致 ✅

#### ❌ 我方 manifest 的错误（必须修）

我方 `registry.py` 对 `unidino` 声明 `order_matters=False`。**实测这是错的**：

**探针**：`verify_unidino_order.py`
```
channel permutation applied : [3, 0, 4, 2, 1]
max |plain - permuted|      : 2.109090e+00
outputs bitwise identical   : False
VERDICT: ORDER MATTERS
```

**原因**：拼接 `(ch h)` 是**按输入顺序**拼的，换顺序 ⇒ 换向量。声称"顺序无关"会让
`check_compatible` 放行一个**静默错误**的输入——正是这一层本该拦住的那类错误。

**修正**：`order_matters=True`。

#### ⚠️ 一个查不到的申报

我方 manifest 写 `scale='div65535'`。我 grep 了 uniDINO 全仓（`65535`、`/255`、`*255`、
`ToTensor`、`Normalize`、`astype`），**没有找到任何向 65535 归一化的代码**。
官方 `inference.py:152-177` 构造的 transform 列表在 jumcp/cell-health 分支下都是空的
（只有 `Resize`）。所以**这个 scale 申报在官方代码里无据可查**——可能是从论文或数据说明来的，
但我**没有验证到**。

**修正建议**：要么标注来源，要么降级为「未验证」。

#### 附带更正我自己的一个误判

我起初怀疑 `inference_utils.py:35` 的 `batch_of_embeddings[~labels] = np.nan` 会崩。
**实测它能正常工作**（`verify_unidino_masking.py`：case 2 → 1920 个 NaN）。

```
=== case 1: all labels True ===  OK  shape=(2, 3, 1920)
=== case 2: some labels False ===  OK  shape=(2, 3, 1920)   NaN count: 1920
```

**我错了**，那不是 bug。记录在此以免误导。

### 3.3 MorphoGenie —— ❌ 发现官方代码的**元组顺序错位**，导致嵌入是随机的

这是本轮最重要的发现。

**探针**：`verify_morphogenie_arch.py`（跑官方 `gan_training/dvae/dvae.py`）

#### 证据链

官方调用方 `MorphoGenie_Test.py:297,299`：
```python
297:  mu_real, logvar, c_real = fvae(x_true, encode_only=True)
299:  mu_real_All.append(mu_real.detach().cpu())     # ← 这个被当成嵌入
```

官方返回方 `gan_training/dvae/dvae.py:393-396`：
```python
393:  def forward(self, x, c, encode_only, decode_only):
394:      if encode_only:
395:          c, mu, logvar = self._encode(x)
396:          return c, mu, logvar                      # ← 顺序是 (c, mu, logvar)
```

官方 `_encode` `dvae.py:408-413`：
```python
410:  mu = distributions[:, :self.c_dim]
411:  logvar = distributions[:, self.c_dim:]
412:  c = reparametrize(mu, logvar)                     # ← ★ 从 N(0,1) 采样
413:  return c, mu, logvar
```

`dvae.py:20-23`：
```python
20:  def reparametrize(mu, logvar):
21:      std = logvar.div(2).exp()
22:      eps = std.data.new(std.size()).normal_()      # ← ★ 随机噪声
23:      return mu + std*eps
```

#### 实测结果

```
FactorVAE256(c_dim=10, nc=3), params 0.79M
因子返回三个张量形状: (4,10) (4,10) (4,10)

first value  identical across 3 calls : [False, False]    ← 随机
second value identical across 3 calls : [True, True]      ← 确定
max |first_call - second_call| = 2.979007
```

#### 结论

**名字与实际绑定错位**：调用方解包顺序是 `mu_real, logvar, c_real`，而返回顺序是 `(c, mu, logvar)`。
于是：

| 调用方的名字 | 实际是 | 性质 |
|---|---|---|
| `mu_real`（被当嵌入持久化） | **`c` = 重参数化采样** | **随机** |
| `logvar`（被丢弃） | **`mu` = 后验均值** | **确定** |
| `c_real`（被丢弃） | **`logvar`** | — |

**即：官方持久化到 `Mix_Latent.csv` 的嵌入是非确定性的**——同一输入两次运行结果不同
（实测差值最大 2.979）。

**对 benchmark 的含义**：这是**致命**的。用它做基准，每次跑出的数字都不一样。
若要复现，**必须改成取第二个返回值（真 `mu`）**。

#### 我方 manifest 的错误（必须修）

我方声明 `output_dim=256`，但**实测输出是 `(4, 10)`**：

- 嵌入宽度 = **`c_dim = 10`**（`MorphoGenie_Test.py:67`），**不是 256**
- `z_dim=256`（`:68`）是 **GAN 生成器的潜维**，与这个嵌入无关

**修正**：`output_dim=10`。

#### 通道数的不确定性

- `MorphoGenie_Test.py:65` → `--nc default=3`（**RGB**）
- 论文是 Cell Painting（5 通道），但**官方例子里没有 5 通道的演示**
- 我方 manifest 写 `channels=('C0','C1','C2')` —— 与官方默认一致，但**论文场景未被官方代码证实**

---

## 4. 我方 manifest 的 3 个错误汇总（本轮实测发现）

| # | 方法 | 字段 | 现在写的 | 应为 | 危害 |
|---|---|---|---|---|---|
| 1 | `unidino` | `order_matters` | `False` | **`True`** | **静默错误**：换通道顺序不报错，嵌入错 |
| 2 | `morphogenie` | `output_dim` | `256` | **`10`** | 契约与实际不符，下游按 256 处理会错 |
| 3 | `unidino` | `scale` | `div65535` | **无据可查** | 可能误判输入值域 |

**这 3 个错误有一个共同点：都在「我没有真实权重、也没读官方代码时」写下的。**
其中 #1 正是这一层设计出来要拦的那类静默错误——**结果它自身的申报就是错的**。

---

## 5. 诚实标注：哪些没有验证

| 项 | 状态 |
|---|---|
| 4 个方法的**真实权重端到端** | ✅ 已验证 |
| 3 个方法的**官方架构契约** | ✅ 已验证（随机权重） |
| 3 个方法的**真实权重** | ❌ 未获取（见下） |
| **论文报告的指标/表示质量** | ❌ **完全未验证**（本轮不在范围） |
| uniDINO `scale=div65535` 的来源 | ❌ 未找到官方依据 |
| DINO4Cells Cell Painting 版 `num_channels` | ❌ 官方仓库无配置可查 |
| MorphoGenie 5 通道能否工作 | ❌ 官方例子只有 3 通道 |

**权重缺口**：

| 方法 | 权重 | 大小 | 障碍 |
|---|---|---|---|
| uniDINO | `checkpoints.zip` | 1,379,562,715 B | Zenodo ~9 KB/s → **≈43 h**（已下 5.7 MB 后停滞） |
| DINO4Cells | `Cell_Painting_data.tar.gz` | 6,896,284,400 B | Zenodo ~10 KB/s → **≈180 h** |
| MorphoGenie | `Models.7z` | 未知 | **Google Drive 本机不可达** |

---

## 6. 复现命令

```powershell
$py = 'D:\ep_bench\venv\Scripts\python.exe'
$env:PATH = 'C:\Users\DELL\miniconda3\envs\torch_gpu\Library\bin;' + $env:PATH

# B 档三个架构验证
& $py 'D:\项目\cellpainting claw复现\_probe\verify_dinocells_arch.py'
& $py 'D:\项目\cellpainting claw复现\_probe\verify_unidino_arch.py'
& $py 'D:\项目\cellpainting claw复现\_probe\verify_unidino_order.py'
& $py 'D:\项目\cellpainting claw复现\_probe\verify_unidino_masking.py'
& $py 'D:\项目\cellpainting claw复现\_probe\verify_morphogenie_arch.py'
& $py 'D:\项目\cellpainting claw复现\_probe\verify_pending_preprocess.py'
```

**依赖**（本轮新装）：`kornia`（DINO4Cells）、`pycytominer` + `pytorch-lightning`（uniDINO）。
