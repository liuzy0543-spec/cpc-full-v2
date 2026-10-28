# 真实数据集基准：四方法公平配置

> 日期：2026-10-18
> 目标：用真实数据集、官方权重，让 4 个方法可比
> 状态：**权重与预处理全部就位；数据集下载中（约 4 小时）**

---

## 1. 数据集：BBBC022

选它的理由：本机唯一能拿到的、**5 通道 + 有化合物标签**的真实 Cell Painting 数据。

| 项 | 实测 |
|---|---|
| 板数 | 20 |
| 每板孔数 | 381 |
| 每孔位点 | 9 |
| **化合物** | **1594** |
| **浓度** | 766 |
| 孔角色 | `compound` 57,492 / `mock` 11,520 |
| 通道 | ER, Hoechst(DNA), Mito, Ph_golgi(AGP), Syto(RNA) |
| 单图下载 | http://data.broadinstitute.org/bbbc/BBBC022/ |

**标签文件**：`D:\bbbc022\BBBC022_v1_image.csv`（37 MB，69,012 行）
含 `SOURCE_COMPOUND_NAME` / `CPD_MMOL_CONC` / `BROAD_ID` / `ASSAY_WELL_ROLE`。

**注意**：该 CSV 有若干行字段数异常（SMILES 内含逗号），必须
`engine='python', on_bad_lines='skip'` 解析，实测丢掉 108 行。

### 下载策略
每板按通道打成 5 个 zip（w1–w5），每个约 1.38 GB，合计 **6.92 GB**。
zip 内是**乱序**的全部视野（实测首个 zip 含 K05/L22/G01/J11/E16/M21/O09/L04 等），
所以**必须 5 个都下**才能凑齐一个视野的 5 个通道。

```
已下 0.43 GB / 6.92 GB    速率 ~479 KB/s    预计剩余 ~4.0 小时
后台任务: pwsh-da65768a9ba44d4b
```

---

## 2. DINOv2-Giant：已下载并验证

CellCLIP 不吃像素，吃 `(N, 5, 1536)` 的逐通道 DINOv2-Giant 向量。本轮补齐。

| 项 | 实测 |
|---|---|
| 权重 | `E:\cpc_weights\dinov2_giant\model.safetensors` **4,546,005,432 B** |
| 来源 | **ModelScope**（`AI-ModelScope/dinov2-giant`），**7 MB/s** |
| 架构 | `hidden_size=1536, patch_size=14, layers=40, heads=24` ✔ 与 CellCLIP 需要的 1536 一致 |
| 处理器 | `crop_size=224`，`do_resize` 被上游关掉（`convert_npz_to_avg_emb.py:520`） |

**为什么换源**：`hf-mirror.com` 把 DINOv2 重定向到 `cas-bridge.xethub.hf.co`（xet 存储），
本机**连接被重置**（curl 35）；`.bin` 变体同样走 xet。ModelScope 直连可用且快 13 倍。

### 实测链路（已验证）
```
synthetic crops (2, 224, 224, 5) uint8
  -> DINOv2-Giant per-channel
  -> vectors (2, 5, 1536) float32   range [-4.740, 7.215]
  -> models.embed('cellclip', vectors, unit='channel_embeddings')
  -> (2, 512)  deterministic=True   sha=66c2d32a44
```

**这是 CellCLIP 第一次拿到真实向量**；此前演示用的都是随机噪声。

---

## 3. ★ O2-VAE：找到官方权重，纠正一个重要错误

**此前判断"官方从未发布权重"是错的。**

官方仓库 `wandb/pretrained_models/` 下有**被 git 跟踪**的预训练模型（commit `4c70d58`）：

| 文件 | 大小 | 用途 |
|---|---|---|
| `model_mefs.pt` | 7,428,429 B | **MEF 真实细胞**（小鼠胚胎成纤维细胞） |
| `model_o2_mnist.pt` | 24,393,468 B | o2-mnist 演示 |

MEF 数据也随仓库发布（`data/mefs/`，36 MB）：
**25,000 训练 + 6,598 测试**个单细胞，128×128，DAPI + Phalloidin。

### 纠正清单

| 项 | 之前（错误） | 现在（实测） |
|---|---|---|
| 权重 | 我自训 50 epoch 的**手写数字**模型 | **官方 `model_mefs.pt`** |
| `output_dim` | 64 | **512**（`configs/config_mefs.py:46`；checkpoint `q_net.fully_net.3.bias` 为 (1024,)） |
| `spatial` | (32, 32) | **(64, 64)**（`config_mefs.py:28,33` 把 128 resize 到 64） |
| 解码器 | 5 层 `[192,96,96,48,48]` | **6 层 `[192,96,96,48,48,48]`** |

### 加载完整性校验（关键，不能用 strict=False 蒙混）
```
learnable params      : 68
learnable MISSING     : 0          ← 所有可学参数都加载了
buffers total/missing : 308 / 188
missing but derived   : 176        ← e2cnn 的 basis-expansion，构造时重算
unexpected non-derived: 0
```
缺的 188 个 buffer 是 e2cnn 的 `_basisexpansion`/`indices`/`expanded_bias`/`filter`，
由架构**确定性重算**，不含学习信息。**可学参数 0 缺失**才是判定标准。

实测：`(3, 1, 64, 64) -> (3, 512)`，deterministic=True。

---

## 4. 四个方法的最终配置

| 方法 | 官方权重 | sha256 | 维度 | 原生输入 |
|---|---|---|---|---|
| `deepprofiler` | ✅ CP CNN v1 | `87b756f5…` | 672 | `(N,32,32,5)` float [0,1] |
| `ca_mae` | ✅ OpenPhenom-S/16 | `f6e5f1c9…` | 384 | `(N,5,256,256)` uint8 |
| `cellclip` | ✅ CellCLIP | `66c2d32a…` | 512 | `(N,5,1536)` **DINOv2 向量** |
| `o2vae` | ✅ **MEF** | `aa70cfa8…` | 512 | `(N,1,64,64)` 二值剪影 |

`doctor()` 实测：**4 个全部 `ready`**。

---

## 5. 公平性：做到了什么、没做到什么

### 已消除的不公平
| 之前的问题 | 现在 |
|---|---|
| CellCLIP 吃随机噪声 | 吃真实 DINOv2-Giant 向量 |
| O2-VAE 是手写数字模型 | 官方 MEF 真实细胞权重 |
| 合成高斯团块当数据 | BBBC022 真实显微图像 |
| 各方法看不同细胞 | 同一分割、同一 `cell_id` |
| 分辨率不对 | 各方法按其训练尺度（32/256/224/64） |

### 仍然存在的不对称（必须披露，不能掩盖）

**O2-VAE 是形状模型，不是强度模型。**

它训练于 MEF 的**细胞剪影**（二值），而 BBBC022 没有可靠可分割的胞质染色，
所以只能给它**细胞核剪影**。核轮廓 ≠ 全细胞轮廓，这是它的模态决定的，
不是处理偷懒。**任何把它与强度模型放在同一张表里的结论，都必须带这句。**

**其他三个方法之间是可比的**：都是 5 通道强度模型，吃同一批细胞的真实像素。

---

## 6. 数据准备管线（已写好，待数据）

`_probe/build_bbbc022_cells.py`：
1. 直接从 5 个 zip 里按标签文件名取图（不用先解压 7 GB）
2. 在 Hoechst 上做 Otsu + 距离分水岭分割，面积过滤
3. 同一批细胞产出**四套表示**：32 / 256 / 224 / 64
4. 输出 `cells_{dp,ca,clip,o2}.npy` + `meta.csv`（含 compound/浓度/角色）+ `dataset_summary.json`

`_probe/make_cellclip_vectors.py`：
```
crops (N,224,224,5) uint8 -> DINOv2-Giant 逐通道 -> (N,5,1536) float32
```
**已验证可用**（见 §2）。

---

## 7. 还差什么

| 项 | 状态 |
|---|---|
| BBBC022 下载（6.92 GB） | ⏳ **~4 小时**，后台任务 `pwsh-da65768a9ba44d4b` |
| 跑分割 → 四套输入 | 待下载完成 |
| 跑 DINOv2 → CellCLIP 向量 | 待分割完成 |
| 四方法统一调度 + 评分 | 待上一步 |

**评分方案**（数据到位后执行）：
- 用 `compound` 做类别、`concentration` 做剂量
- 指标：同化合物孔级检索 mAP、kNN 化合物分类、剂量-反应单调性
- 对照：随机投影基线（已验证落在 chance，说明指标校准正常）

---

## 8. 复现

```powershell
# 1) 数据集（后台）
& 'D:\项目\cellpainting claw复现\_probe\download_real_data.ps1'

# 2) 建单细胞数据集
D:\ep_bench\venv\Scripts\python.exe 'D:\项目\cellpainting claw复现\_probe\build_bbbc022_cells.py' `
  --zips D:\bbbc022\BBBC022_v1_images_20585w1.zip D:\bbbc022\BBBC022_v1_images_20585w2.zip `
         D:\bbbc022\BBBC022_v1_images_20585w3.zip D:\bbbc022\BBBC022_v1_images_20585w4.zip `
         D:\bbbc022\BBBC022_v1_images_20585w5.zip `
  --labels D:\bbbc022\BBBC022_v1_image.csv --output-dir D:\bbbc022\cells --max-cells 3000

# 3) CellCLIP 向量
D:\ep_bench\venv\Scripts\python.exe 'D:\项目\cellpainting claw复现\_probe\make_cellclip_vectors.py' `
  --crops D:\bbbc022\cells\cells_clip.npy --model-dir E:\cpc_weights\dinov2_giant `
  --output D:\bbbc022\cells\cellclip_vectors.npy --device cpu
```
