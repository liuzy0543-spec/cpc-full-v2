# 审查报告：Embedding 统一框架 v2 / v3

**审查对象**：`D:\项目\CellPainting-Claw-v2`（对拍树 `D:\cpc_c_v2`）
**审查提示词**：`docs/archive/review_prompt_v2_v3_2026-10-26.md`
**审查日期**：2026-10-26
**审查环境**：Python 3.10.20 / `D:\ep_bench\venv`（torch 2.5.1、TF 2.15.1、numpy 1.26.4），GPU RTX 4060 Laptop 8 GB
**权重根**：`E:\cpc_weights` ｜ **数据**：`E:\bbbc022\cells` ｜ **官方源码**：`D:\official_src\`

---

## 0. 结论速览

| 阶段 | 结果 |
|---|---|
| **阶段一 黑盒** | ✅ **7/7 方法全部跑通**，输出宽度与形状全部吻合，无 NaN/Inf，B1–B7 全过 |
| **阶段二 白盒** | ⚠️ 官方四方法逐项对照**基本成立**；但 **dinocells 的磁带方向与官方相反**（唯一结构性缺陷），另有 14 项新发现 |
| **阶段三 benchmark** | ✅ v2/v3 数值与提示词 §4.4 基线**逐项一致**（7 个方法 + 2 基线 + 稳定性前缀与互换对全部吻合） |

**三档结论（详见 §5）**

- **v2 可发表性**：⚠️ 有条件 —— 4 个官方方法的实现与官方代码逐行核对通过，DeepProfiler 逐位一致已实测复现；但 `o2vae` 存在**输入域偏移风险**（喂核剪影、官方权重训练于细胞剪影），且 `ca_mae` 的归一化并非官方做法（虽被官方 checkpoint 内的 InstanceNorm 抵消）。发表前须处理 §5.1 的 P0/P1 事项（G1 已补齐；G2/G3/G4 须在论文中显式区分与修正表述）。
- **v3 替身定位**：✅ **仅可用于接口验证**。3 个替身的权重、架构、预处理均与官方不同，任何表格中必须标注 `surrogate`。
- **数据不支持的结论**：❌ 任何"某方法优于某方法"的跨方法性能结论（自建协议、单孔设计）；❌ 除 DeepProfiler 外任何"已与官方逐位一致"的声明；❌ 替身分数的方法学解读。

---

## 1. 阶段一：黑盒测试

### 1.1 黑盒记录表（提示词 §2.6）

| 方法 | 能跑 | dim | shape | dtype | 有限 | weights sha256（前16） | provenance | 异常/原因 |
|---|---|---|---|---|---|---|---|---|
| deepprofiler | ✅ | 672 | (2, 672) | float32 | ✅ | `87b756f579336d60` | official | — |
| ca_mae | ✅ | 384 | (2, 384) | float32 | ✅ | `f6e5f1c97101331b` | official | — |
| cellclip | ✅ | 512 | (2, 512) | float32 | ✅ | `66c2d32a44a3e7d6` | official | — |
| o2vae | ✅ | 512 | (2, 512) | float32 | ✅ | `aa70cfa8b080c9d9` | official | — |
| unidino | ✅ | **1920** | (2, 1920) | float32 | ✅ | `e53838fe807311bf` | **surrogate** | — |
| dinocells | ✅ | 384 | **(1, 384)** | float32 | ✅ | `5475ec885ae6b7d9` | **surrogate** | — |
| morphogenie | ✅ | **10** | (2, 10) | float32 | ✅ | `491029978fef0969` | **surrogate** | — |

**七行全部通过，零异常。** 宽度序列 672 / 384 / 512 / 512 / 1920 / 384 / 10 与提示词 §2.3 及 §7 基线**逐项吻合**。

> **`sha` 列口径澄清（提示词需修订 P1）**：§2.3 输出块的 `sha=` 是**权重文件 sha256 的前 12 位**，不是嵌入数组的哈希。实测佐证：deepprofiler 嵌入数组哈希 = `a05475038143`，而权重 sha256 = `87b756f579336d60…`；四个官方 + 三个替身的**权重** sha256 与 §7 表**7/7 全等**。若按"结果哈希"理解会误报 7/7 不一致。

> **dinocells 的 N=1 属正确行为**：`bhwc[0]` 是单条磁带，磁带宽度承载通道索引、无批次轴，故返回 `(1, 384)`。与提示词 §2.3 说明一致。

### 1.2 接口性质（B1–B7）

| # | 检查 | 判定 | 实测证据 |
|---|---|---|---|
| B1 | 七个结果对象字段集合相同 | ✅ **通过** | 12 个键完全一致：`backend, determinism, dim, feature_layer, input_unit, license, metadata, model, n_samples, shape, weights_path, weights_sha256` |
| B2 | 每个结果带 `weights_sha256` | ✅ **通过** | 7/7 非空（**位于 `to_dict()` 顶层，不在 `metadata` 内** —— 提示词 §2.4 表述为"每个结果是否带"，若按 `metadata['weights_sha256']` 取会误判为缺失） |
| B3 | `weights_provenance` 4 official + 3 surrogate | ✅ **通过** | 完全吻合 |
| B4 | `embed_all` 一次跑七个 | ✅ **通过**（有前提，见下） | 映射式 `embed_all(names, {方法:载荷}, unit={...})` → 7/7 全部返回，errors 为空 |
| B5 | v2 下 `unidino` 抛 `UnknownModel` | ✅ **通过** | `UnknownModel: Unknown embedding model 'unidino'. Available: deepprofiler, ca_mae, cellclip, o2vae` |
| B6 | 同一输入连跑两次一致 | ✅ **通过** | 7/7 `identical=True, max|d|=0.0`（含 morphogenie 替身 —— 官方 morphogenie 为 stochastic，替身为 deterministic，**两者不可混比**，已在 manifest 声明） |
| B7 | `pending()` 返回三个已取消契约 | ✅ **通过** | `['dinocells', 'morphogenie', 'unidino']` |

**B4 的前提（提示词需修订 P8）**：`embed_all` **不提供** `assume=` / `masks=` 参数，`channels=` 也是**所有方法共享的单一值**。因此 §2.3 的代码块**不能原样**用于 B4；必须先用 `models.prepare()` 逐方法预制备，再以映射 + `unit` 映射调用。实测：传 `channels=('C0','C1','C2')`（morphogenie 的通道名）会使 **5/7 方法被 `channel-set` 拒绝**。

### 1.3 测试基线复现

| 环境 | 提示词 §7 基线 | 本次实测 | 判定 |
|---|---|---|---|
| `ep_bench`（有 torch，4 个模型测试文件） | 90 passed | **90 passed** | ✅ 一致 |
| `.venv`（无 torch，全量） | 300 passed / 9 skipped / 14 failed | **300 passed / 9 skipped / 14 failed** | ✅ 一致 |

14 项既有失败**全部位于** `test_data_access_smoke` / `test_deepprofiler_profile_env_smoke` / `test_delivery_smoke` / `test_orchestration_smoke` / `test_public_api_dispatch_smoke` / `test_public_branding_smoke`，即 CLI/交付/编排冒烟层，**与 embedding 层无交集**。

---

## 2. 阶段二：白盒检查

### 2.1 官方锚点对照表（逐方法 × 逐检查点）

#### deepprofiler（官方权重）

| 检查点 | 官方位置 | 本项目位置 | 判定 |
|---|---|---|---|
| 通道顺序 | `1-deepprofiler/README.md:74` "5-channel Cell Painting images (DNA, ER, RNA, AGP, Mito)" | `registry.py:122` `DEEPPROFILER_CHANNELS = ('DNA','ER','RNA','AGP','Mito')` | ✅ 一致（历史缺陷 DNA/RNA/ER 互换已修复） |
| 逐裁片 min-max | `deepprofiler/imaging/cropping.py:65-67` `reduce_min/max(axis=[1,2])` → `(crops-mini)/(maxi-mini+eps)` | `preprocessing.py:234-237` `per_crop_minmax_bhwc`（`axis=(1,2)` 逐样本逐通道） | ✅ 一致（epsilon 形式略异，数值等价） |
| **输入尺寸** | 官方 crop 图 `crop_and_resize(..., [32,32])` | `registry.py` `spatial=None`（**非提示词所称 `(32,32)`**） | ⚠️ 见 **F4** |
| 输出层 | `block6a_activation` → 672 | `registry.py` `feature_layer='block6a_activation'`, `output_dim=672`, `output_dim_alternatives=(1280,)`（pool5） | ✅ 一致 |
| **逐位一致性** | 官方 `deepprofiler` 包 demo 产物 `1.npz` | `scripts/embedding_benchmark/parity_vs_official_dp.py` | ✅ **实测复现**：case A `bitwise_equal=True, max\|diff\|=0.000000e+00`；case B（原始像素）`max\|diff\|=4.481230e+06`（契约拦截）；契约守卫 "旧通道序被拒 / 16-bit 原始输入被拒" 均 **PASS** |

> 对拍样本量提示：官方 demo 仅 2 孔 × 2 细胞 = **4 个细胞**。逐位一致成立，但样本极小。

#### ca_mae / OpenPhenom（官方权重）

| 检查点 | 官方位置 | 本项目位置 | 判定 |
|---|---|---|---|
| 输入尺寸/通道/dtype/范围 | `test_huggingface_mae.py:25-26` `size=(2,C,256,256)`, `dtype=torch.uint8`；`vit.py:184` `img_size=256` | `registry.py` `spatial=(256,256)`, `to_dtype='uint8'`, `value_range=(0,255)`, `batch_layout='BCHW'` | ✅ 一致 |
| 通道数 | `huggingface_mae.py:75` `max_in_chans=11`；官方测试 `C ∈ {1,4,6,11}` | `registry.py` `channel_count_fixed=False` | ⚠️ 见 **F7**（契约放行 C=20，官方会形状报错） |
| **归一化** | `normalizer.py:4-7` `pixels.float()/255`；`huggingface_mae.py:88-91` `nn.InstanceNorm2d(None, affine=False)`；**官方无任何逐图拉伸、无 16→8 转换代码** | `preprocessing.py:238-240` `per_crop_max255`（逐样本 H,W,C 全局最大值拉伸）+ `registry.py` `scale='div255'` | ⚠️ **有意差异**：官方未规定 16-bit→8-bit 的做法，本项目的拉伸是**源域选择**而非官方行为；但官方 checkpoint 内的 `InstanceNorm2d` 对逐通道正仿射不变，故拉伸本身被完全抵消（理论残差 1e-15），实测残差仅来自 uint8 量化网格（max 0.0143 ≈ 输出 std 的 0.7%）。**不可表述为"与官方一致"** |
| 输出维度与池化 | `huggingface_mae.py:275` `latent = X[:,1:,:].mean(dim=1)`（弃 CLS，patch 均值）；`:31` `384*C if return_channelwise else 384` | `adapters/ca_mae.py:152` 直接调用官方 `predict()`；`registry.py` `output_dim=384`, `output_dim_per_channel=384` | ✅ 一致（复用官方 `predict`，池化行为按构造相同） |
| 论文归属 | 官方 `README.md:3-9` **分列两篇**：`2309.16064`（NeurIPS 2023，标题 "…are Scalable Learners of **Cellular Morphology**"）、`2404.10242`（CVPR 2024，标题 "Masked Autoencoders **for Microscopy** are Scalable Learners of **Cellular Biology**"）；作者单位仅 Recursion | 提示词 §3.1 把 "…Cellular Morphology" 标题挂在 `2404.10242` 上；`adapters/ca_mae.py:5` 写 "Recursion + **Valence Labs**" | ❌ **F14**：提示词论文标题错配；`Valence Labs` 在官方仓库与两篇论文中**均无依据** |

#### cellclip（官方权重）

| 检查点 | 官方位置 | 本项目位置 | 判定 |
|---|---|---|---|
| 输入是通道向量不是像素 | `README.md:30` `input_dim=1536`；`helper.py:169/185-186` `vision_width = input_dim` | `registry.py` `input_unit=CHANNEL_EMBEDDINGS`, `input_feature_dim=1536` | ✅ 一致 |
| 5×1536 → 512 的聚合 | `src/clip/model.py:443-511` `CrossChannelFormer`：`:467` 逐通道可学习嵌入 `channel_embed (5,1536)` → `:499-501` 12 层 Transformer → `:503` 取 CLS → `:508-509` `proj (1536,512)` | `adapters/cellclip.py:50-56/155-161/163-177` 同名同形状模块 | ✅ 一致（**不是 pooling**，是逐通道可学习嵌入 + Transformer + 线性投影） |
| 输出维度 512 | `configs/model_config.py:60` `embed_dim: 512`；`model.py:478` `proj (1536,512)` | `registry.py` `output_dim=512`；`adapters/cellclip.py:161` | ✅ 一致 |
| 链式权重 DINOv2-Giant | `README.md:14` "CellCLIP checkpoint with DINOv2-giant"；`retrieval_whole.py:390` `dinov2-giant_ind.h5` | `registry.py:289-292` `chain_weights=(('facebook/dinov2-giant', '917d3c47…'),)` | ✅ 一致（要求成立）；⚠️ 但 `facebook/dinov2-giant` 这个 **id 在官方仓库中不存在**，1536 亦无字面量，二者均为项目固化值 |
| 权重 sha256 | **官方仓库无任何 sha256/checksum 声明**（`grep sha256\|checksum\|md5` 无命中） | `registry.py` 自证 `66c2d32a…` + chain 内 DINOv2-Giant sha | ⚠️ 超出官方的可追溯机制，无官方值可核对 |
| 像素直喂被拒 | — | 实测 `check_compatible`：像素 `(2,5,256,256)` → `('ndim',)`；`(5,256,256)` → `('channel-count','feature-width')`；`(2,5,256)` → `('feature-width',)`；`(2,5,1536)` → 接受 | ✅ 拒绝有效，但 ❌ **F13**：`check_compatible` **从不产生 `chain-required`**（该 reason 只由 `prepare()` 抛出），提示词 §3.2 的判定标准不准确 |
| 通道顺序 | `preprocess_images_jumpcp.py:78-83` 规范序 DNA, ER, RNA, AGP, Mito；`:94` `order=[3,2,0,4,1]` | `registry.py:131` `CELLCLIP_CHANNELS = ('DNA','ER','RNA','AGP','Mito')`, `order_matters=True` | ✅ 一致（引用行号精确命中） |

#### o2vae（官方权重）

| 检查点 | 官方位置 | 本项目位置 | 判定 |
|---|---|---|---|
| 输入 resize 到 64 | `o2vae_repo/configs/config_mefs.py:28` `T.Resize(64)`、`:33` `transform_test=T.Resize(64)` | `registry.py` `spatial=(64,64)`；`preprocessing.py:382-390` resize + 恢复二值 | ✅ 一致 |
| 潜变量维度 512 | `config_mefs.py:46` `zdim=512`；`:47` `channels=1` | `registry.py` `output_dim=512` | ✅ 一致 |
| **特征层 z_mu** | `o2vae_repo/models/vae.py:148-149` `z_mu, _ = torch.split(self.q_net(x), self.zdim, dim=1); return z_mu` | `registry.py` `feature_layer='z_mu'` | ✅ 一致 |
| **权重形状交叉验证** | `o2vae_mefs_official.pt`：`q_net.fully_net.3.bias` 形状 **(1024,)** = 2×zdim；`p_net.fc.0.weight` 形状 **(48,512)**（解码器输入维 = zdim） | `registry.py` `output_dim=512` | ✅ **实测双重交叉验证通过** |
| 剪影语义（缺 mask 拒） | 官方 `config_mefs.py:19` `data_dir=…/data/mefs/scaled/cell` → 官方权重训练于**细胞**剪影 | `preprocessing.py:369-375` 缺 mask → `reason='mask-required'`（实测） | ✅ 拒绝正确；但 ⚠️ **F3b**：项目喂**核**剪影，官方权重训练于**细胞**剪影 → 推理期域偏移，嵌入分布与训练分布不一致 |
| 上游仓库依赖 | `E:\cpc_weights\o2vae` → `o2vae_repo`（`jmhb0/o2vae` clone，HEAD `4c70d58`），含 `models/` 可导入包 | `registry.py` `requires_upstream_repo=True`, `repo_source='o2vae'` | ✅ 一致；`probe()` 报 `repo_found=True` |
| **论文出处** | Burgess, Nirschl, Zanellati, Lozano, Cohen, Yeung-Levy. *Orientation-invariant autoencoders learn robust representations for shape profiling of cells and organelles.* **Nature Communications 15:1022 (2024)**, doi:[10.1038/s41467-024-45362-4](https://doi.org/10.1038/s41467-024-45362-4)；代码 [github.com/jmhb0/o2vae](https://github.com/jmhb0/o2vae)；Zenodo 10.5281/zenodo.10206848 | 提示词 §3.1 记为"**本仓库未记录论文出处**" | ❌ **G1 已可关闭**（见 §3） |

#### unidino（替身）

| 检查点 | 官方位置 | 本项目位置 | 判定 |
|---|---|---|---|
| 架构 / patch_size | `5-unidino/inference.py:95` `patch_size=16`、`:97` `arch="vit_small"`；`source/dino/vision_transformer.py:353-366` `vit_small(depth=12, embed_dim=384)` | `adapters/unidino.py:61` `patch_size=16`；`_surrogate.py:91` `embed_dim=384`；`unidino.py:49` `depth=4` | ✅ 规格一致；⚠️ 深度 4 vs 12 = **D1**（已声明） |
| **逐通道独立前向** | `source/inference_utils.py:19-21` `rearrange(batch_of_crops, "batch ch crops one h w -> (batch ch crops) one h w")` → 同一份 1 通道权重跑每个通道；`:29-34` 按 **ch 升序**拼成 `batch crops (ch h)` = 384×C | `adapters/unidino.py:90` `reshape(B*C,1,H,W)` → `:93-95` 分批前向 → `:96-97` 通道升序块拼接 | ✅ **一致**（1920 = 384×5） |
| 预处理 div65535 | `source/image_ops.py:14-20` `img = img / (2**16 - 1)` | `preprocessing.py:241-242` `arr / 65535.0` | ✅ 数值等价；⚠️ 顺序不同：官方 **scale→crop**，本项目 **resize→scale**（线性缩放下数值等价） |
| **uint8 输入** | 官方亦不检查 dtype（假定上游 TIFF 为 16-bit） | `registry.py` 未声明 `value_range`；`contract.py:571-586` 的 dtype 守卫只在 `to_dtype='uint8'` 时触发 | ❌ **F3**：喂 uint8 → `div65535` 后只剩 1/257（实测 prepared 输入 max ≈ 0.0039），**静默变黑**，契约不拦 |
| 通道顺序敏感 | `inference_utils.py:29-34` 块按 ch 升序排列 → 置换通道即置换块 | `registry.py` `order_matters=True` | ✅ 一致（有官方代码依据） |
| 权重来源 | 官方：Zenodo doi:10.5281/zenodo.14988837；`run_dino_inference.sh:2` `--ckpt ./checkpoints/uniDINO.pth`；`source/dino/utils.py:98-109` 取 `state_dict["teacher"]` | `registry.py:368` `source='unidino/checkpoints.zip'` | ❌ **F6**：`checkpoints.zip` 在官方仓库**不存在**（全树 grep 无命中）；manifest notes 自称"scale 未声明、官方未写明"，与同文件 `scale='div65535'` 及官方 `image_ops.py:19` 直接矛盾（陈旧文案） |

#### dinocells（替身）

| 检查点 | 官方位置 | 本项目位置 | 判定 |
|---|---|---|---|
| **磁带方向** | `6-dinocells/data_utils/cellpainting_dataset.py:23-47` `fold_channels(image, channel_width, mode="drop")`：`np.reshape(image, (H, channel_width, -1), order="F")` 后 `output[:, :, 0:-1]` **丢弃 mask 通道**，返回 `ToTensor` 后的 **(C,H,W)** 堆叠张量；`:77-93` `SingleCellDataset.__getitem__` 在 transform **之前**调用它 | `preprocessing.py:263-303` `_to_tape`：把 (H,W,C) + mask 折成 **(H, W\*(C+1))** 磁带并送入模型；`registry.py` `input_encoding='tape'` | ❌ **F2（核心缺陷）**：官方方向是 **磁带 → 堆叠（弃 mask）→ 模型**；本项目方向是 **堆叠 + mask → 磁带 → 模型**。Fortran order 用法本身 ✅ 正确（实测本地 `_to_tape` 与官方 `fold_channels` 互逆，C order 会打乱通道 ✅） |
| 官方归一化 | `data_utils/file_dataset.py:77-82` `norm_loader`：`img -= np.min(img, axis=(0,1))`；`img *= 255/np.max(img, axis=(0,1))` → uint8。**`np.min(2D, axis=(0,1))` 是全图标量**（实测），非"逐图逐通道" | `preprocessing.py:245-248` `per_sample_selfnorm` = **逐样本逐通道 z-score**（`ddof=0`） | ⚠️ **P3 + F2b**：提示词与 `registry.py:322-325` 均把 `norm_loader` 描述为"逐图逐通道 min-max"，**与 numpy 语义不符**。本地 z-score 与官方 `self_normalize`（`utils/augmentations.py:460-468`：`m=x.mean((-2,-1))`, `s=x.std((-2,-1), unbiased=False)`, `x/=(s+1e-7)`）**语义一致**；且官方全局 min-max 是逐通道 z-score 的前置仿射变换，代数上被抵消（1e-15 量级），故本地实现**等价**于官方链（仅余 uint8 量化差） |
| 磁带形状 / 批语义 | 官方模型输入是 (C,H,W) 堆叠 → **天然可批**（`batch_size_per_gpu: 24`） | `preprocessing.py:272-278` 对 batch 抛 `reason='tape-batch'` | ❌ **F2 的下游后果**：批拒绝源于本地非官方磁带编码，非官方约束 |
| 模型输入宽度 | `config.yaml` `arch: vit_tiny`, `patch_size: 16`, `num_channels: 4`（HPA 变体）；`README.md:207-213` CP 权重 `DINO_cell_painting_base/small_checkpoint.pth`（ViT-B 768 / ViT-S 384）；README 未给出 CP 裁片尺寸与通道数 | `registry.py` `spatial=(96,96)`, `output_dim=384`, `output_dim_alternatives=(768,)` | ✅ 宽度档位一致；⚠️ 96×96 与 5 通道在官方代码中**无出处**（manifest notes 已自陈） |
| 替身架构 | 官方 CP 用 ViT-S/ViT-B | `adapters/dinocells.py` 替身为 `vit_small` | ⚠️ **D2**（已声明） |

#### morphogenie（替身）

| 检查点 | 官方位置 | 本项目位置 | 判定 |
|---|---|---|---|
| 输出宽度 | `7-morphogenie/MorphoGenie_Test.py:67` `--c_dim default=10`、`:68` `--z_dim default=256` | `registry.py` `output_dim=10` | ✅ 一致（嵌入宽 = c_dim = 10，**不是 256** ✅ 提示词正确） |
| 持久化的是哪个张量 | `gan_training/models/dvae.py:393-396` `forward` 返回 `(c, mu, logvar)`；`:409-413` `c = reparametrize(mu, logvar)`（抽样噪声）；`MorphoGenie_Test.py:297` `mu_real, logvar, c_real = fvae(x_true, encode_only=True)` → 因返回序为 `(c,mu,logvar)`，`mu_real` 实际绑定到 **`c`（随机样本）**；`:340` 落盘 `Mix_Latent.csv` | `registry.py` `feature_layer='latent (reparameterised sample -- see notes)'`, `determinism='stochastic'`（PENDING 契约）；v3 替身 `registry.py:619-622` 改写为 `deterministic` | ✅ 一致（官方 stochastic，替身返回 mu 并已声明 deterministic = **D3**） |
| c_dim 与 z 分布 | `MorphoGenie_Test.py:227-229` `get_zdist('gauss', args.c_dim)` + `zdist`（`configs/cells_650.yaml:30-32` `dim: 256`）；`gan_training/config.py:43` `z_dim = z_dist.dim + c_dim` = **266** | `registry.py` `output_dim=10` | ✅ 一致；⚠️ `--z_dim 256` 是**死参数**，生成器实际 latent = 256+10 |
| MATLAB 居中 | **`ImagePreProcessing/` 就在官方仓库内**：`MultipleCells/CPA/GetCentImages.m:26` `Crop=[Centerx-100, Centery-100, 200, 200]`（**200×200**，非 256×256）、`:31-32` `circshift` + `imcrop`、`:38` `AdjBGSub = uint8((B1-Submin)/(Submax-Submin)*255)`；256×256 缩放在 Python 侧 `gan_training/utils.py:352-357` `transforms.Resize((image_size,image_size))` + `Normalize((0.5,)*3,(0.5,)*3)` | `registry.py:465-467` `source_map='matlab_centring'`；`preprocessing.py:361-367` 抛 `reason='unsupported-preprocess'`；`registry.py:496` 与 `preprocessing.py:47` 均称该步骤"**不在仓库**" | ⚠️ 拒绝本身 ✅ 正确且安全；但 ❌ **F5**：**代码就在仓库内**（缺的是 MATLAB 运行时 + Cellpose + 原始数据），"不在仓库"的表述**过强**，削弱了"不可复现"的论证；且 256×256 属 Python 侧。官方另有逐 crop min-max→uint8 与 `Normalize(0.5)`，manifest 却声明 `scale='none'`、无值域 → 即使日后放开拒绝，契约也不足以描述官方输入 |
| 引用文件消歧 | 推理路径是 `gan_training/models/dvae.py`（`c_dim`、三元组）；`dVAE/model.py:384-394` 是 legacy 训练脚本（用 `z_dim`，返回 **四元组** `(x_recon, mu, logvar, z.squeeze())`） | `adapters/morphogenie.py:5-8`、`_surrogate.py:82-83` 引用 `dVAE/model.py` 并称其返回 `(recon, mu, logvar)` + 采样 | ❌ **F5b**：张冠李戴。**结论（10 维瓶颈、官方 stochastic）不受影响**，但引用会误导审查者 |

### 2.2 横切检查项（W1–W8）

| # | 检查 | 判定 | 实测证据 |
|---|---|---|---|
| W1 | `check_compatible()` 不 import torch/TF | ✅ **通过** | `.venv`（无框架）与 `ep_bench`（有框架）两个解释器中，调用后 `sys.modules` 中 `torch`/`tensorflow` **均为空** |
| W2 | manifest `output_dim` 与实际一致；不一致被 `output-width` 拒绝 | ✅ **通过** | 7/7 实际宽度与声明一致；`_check_output_width` 对 383 / 999 / 0 均抛 `EmbeddingError reason='output-width'`，对 384 接受 |
| W3 | 通道顺序错时真的报错 | ✅ **通过（附边界说明）** | 声明错序 → `IncompatibleInput reason='channel-order'`（deepprofiler、unidino 均实测）；**不声明 names 的置换无法检测**，实测嵌入改变（max\|diff\|=1.6328）且不报错 —— 这是契约的固有限制（只能查调用方声明过的信息），应在文档中明示 |
| W4 | 16-bit 喂 8-bit 方法 | ⚠️ **部分通过** | 16-bit 直喂 `ca_mae` → `IncompatibleInput`（`spatial-size` + `value-range`）✅；16-bit 经 `prepare` → 被重标定为 uint8 后接受（设计如此）✅；但 **256×256 喂 deepprofiler → Keras 内部裸 `ValueError`**（`expected input_1 to have shape (32,32,5) but got (256,256,5)`），**非类型化 `IncompatibleInput`** → **F4** |
| W5 | `assume='raw_crop'` ≡ 手动 prepare + embed | ✅ **通过** | deepprofiler / ca_mae / unidino / dinocells 四项 **逐位相同**（`max\|diff\| = 0.0`） |
| W6 | 权重 sha256 与实际文件相符 | ⚠️ **部分通过** | 4/4 官方权重 **声明值 = 实测值**（`verify_sha256` 全 MATCH）✅；3/3 替身在 manifest 中 **未固定 sha256**（`declared=''`），实际值仅在结果运行时记录 → **F10** |
| W7 | 切 profile 清空适配器缓存 | ✅ **通过** | 切到 v2 后缓存键数 4 → **0**；切回 v3 仍为 0（未重建） |
| W8 | `describe()` 带 `weights_provenance` | ✅ **通过** | 7/7 均带，4 official + 3 surrogate |

---

## 3. 缺口确认（G1–G4）

| # | 缺口 | 状态 | 说明 |
|---|---|---|---|
| **G1** | o2vae 论文/官方仓库未记录 | ✅ **可关闭** | 已补齐： Burgess J, Nirschl JJ, Zanellati M-C, Lozano A, Cohen S, Yeung-Levy S. *Orientation-invariant autoencoders learn robust representations for shape profiling of cells and organelles.* **Nature Communications 15, 1022 (2024)**, doi:[10.1038/s41467-024-45362-4](https://doi.org/10.1038/s41467-024-45362-4)；代码 [github.com/jmhb0/o2vae](https://github.com/jmhb0/o2vae)；Zenodo 10.5281/zenodo.10206848。**注意 `mefs` = Mouse Embryonic Fibroblasts，不是 "Microscopy-Enhanced Feature Space"。** 另发现两处需在论文正文写明的风险：**①** 论文正文写 `d = 128` 而发布权重实测 `zdim=512`（以代码/权重为准）；**②** 官方 `data_dir` 指向 `scaled/cell`（**细胞**剪影）且无单独的 nucleus 权重，而本项目喂**核**剪影 → 域偏移 |
| **G2** | 只有 deepprofiler 做过真实逐位对拍 | ⚠️ **仍然存在，但本次已实测复现** | `parity_vs_official_dp.py` 本次运行：`bitwise_equal=True, max\|diff\|=0.000000e+00`（**成立**）。但样本仅 4 个细胞；其余 6 个方法仍是"对照官方公式的推理验证"，**报告中必须继续区分这两类** |
| **G3** | morphogenie 的 `matlab_centring` 永远拒绝 | ⚠️ **仍存在，且理由需修订** | 拒绝行为 ✅ 正确；但"不在仓库"**不成立**（`.m` 源码在 `ImagePreProcessing/` 内，缺的是 MATLAB 运行时 + Cellpose + 原始数据），且居中是 **200×200**、256×256 在 Python 侧 → **F5** |
| **G4** | ca_mae / o2vae 未用官方 demo 输入实测对比 | ⚠️ **仍然存在** | ca_mae：官方无 16→8 规定、无 demo 预处理产物可比，仅能靠 checkpoint 内 `InstanceNorm2d` 的仿射不变性做等价性论证（已做，见 §2.1）；o2vae：官方数据为 MEFs，与 BBBC022 不同源，**无法用官方 demo 输入对比** |

---

## 4. 阶段三：benchmark

### 4.1 协议与数据

- 数据：BBBC022 子集，**3,857 细胞 / 64 孔 / 54 化合物 / 671 mock / 11 mock 孔**，chance = 0.0185（脚本自报，与提示词 §4.1 一致）
- 任务：化合物检索（同化合物细胞应邻近）；指标 kNN / mAP / silhouette / mock-AUC，均给 z-scored 版
- 七个方法的嵌入**全部用 `--force` 从头重算**（未复用既有产物），产物：3857×672 / 3857×384 / 3857×512 / 3857×512 / 3857×1920 / 3857×384 / 3857×10

**嵌入耗时**（RTX 4060 8 GB，期间有其他进程占卡）：

| 方法 | 形状 | 耗时 | 吞吐 |
|---|---|---|---|
| deepprofiler | (3857, 672) | 14.7 s | 262.2 cells/s |
| ca_mae | (3857, 384) | 1341.4 s | 2.9 cells/s |
| cellclip | (3857, 512) | 17.7 s | 218.1 cells/s |
| o2vae | (3857, 512) | 148.2 s | 26.0 cells/s |
| unidino / dinocells / morphogenie | (3857, 1920/384/10) | 见 `v3_surrogate_embeddings.json` | — |

### 4.2 v2 结果（4 个官方权重方法 + 2 基线）

| 方法 | kNN | kNN_z | mAP | **mAP_z** | sil_z | mock-AUC_z | rank |
|---|---|---|---|---|---|---|---|
| cellclip | 0.6248 | 0.6451 | 0.1318 | **0.1337** | −0.0454 | 0.4763 | 23.4 |
| ca_mae | 0.4286 | 0.4454 | 0.1001 | **0.1083** | −0.1365 | 0.5861 | 2.0 |
| deepprofiler | 0.1960 | 0.1934 | 0.0701 | **0.0691** | −0.0341 | 0.5935 | 34.7 |
| o2vae | 0.0843 | 0.0869 | 0.0473 | **0.0471** | −0.1424 | 0.5054 | 2.9 |
| *baseline 原始像素* | 0.1084 | 0.1058 | 0.0503 | **0.0508** | −0.0655 | 0.5227 | 19.8 |
| *baseline 随机投影* | 0.0837 | 0.0848 | 0.0450 | **0.0450** | −0.0141 | 0.4996 | 348.8 |

### 4.3 v3 结果（追加 3 个替身）

| 方法 | kNN | kNN_z | mAP | **mAP_z** | sil_z | mock-AUC_z | rank |
|---|---|---|---|---|---|---|---|
| cellclip | 0.6248 | 0.6451 | 0.1318 | **0.1337** | −0.0454 | 0.4763 | 23.4 |
| ca_mae | 0.4286 | 0.4454 | 0.1001 | **0.1083** | −0.1365 | 0.5861 | 2.0 |
| deepprofiler | 0.1960 | 0.1934 | 0.0701 | **0.0691** | −0.0341 | 0.5935 | 34.7 |
| **unidino** 🔶surrogate | 0.2365 | 0.2320 | 0.0681 | **0.0684** | −0.1739 | 0.5842 | 2.2 |
| **morphogenie** 🔶surrogate | 0.1361 | 0.1281 | 0.0580 | **0.0574** | −0.7161 | 0.5402 | 1.0 |
| o2vae | 0.0843 | 0.0869 | 0.0473 | **0.0471** | −0.1424 | 0.5054 | 2.9 |
| **dinocells** 🔶surrogate | 0.0817 | 0.0814 | 0.0460 | **0.0460** | −0.1891 | 0.4772 | 3.7 |
| *baseline 原始像素* | 0.1084 | 0.1058 | 0.0503 | **0.0508** | −0.0655 | 0.5227 | 19.8 |
| *baseline 随机投影* | 0.0837 | 0.0848 | 0.0450 | **0.0450** | −0.0141 | 0.4996 | 348.8 |

> 🔶 **替身行（unidino / morphogenie / dinocells）不是这三个方法的论文性能**，权重是本地训练的 surrogate，架构、预处理均与官方不同（D1–D4）。脚本自身也在排名行尾打了 `<-- SURROGATE (not this method's published performance)` 标记。

### 4.4 与提示词 §4.4 基线的逐项核对

| 项 | 提示词基线 | 本次实测 | 判定 |
|---|---|---|---|
| v2 排序 | cellclip 0.1337 > ca_mae 0.1083 > deepprofiler 0.0691 > o2vae 0.0471 | 完全相同（4/4） | ✅ **一致** |
| 两基线 | 原始像素 0.0508、随机 0.0450 | 0.0508 / 0.0450 | ✅ **一致** |
| v3 追加 | unidino 0.0684、morphogenie 0.0574、dinocells 0.0460 | 0.0684 / 0.0574 / 0.0460 | ✅ **一致** |
| 稳定性前缀 | 5/7（cellclip > ca_mae > deepprofiler > unidino > morphogenie） | **5/7**，前缀逐项相同 | ✅ **一致** |
| 唯一互换对 | dinocells / o2vae | **dinocells / o2vae** | ✅ **一致** |

**逐折排序（4 折 × 按孔切分）**

```
fold 0: cellclip > ca_mae > deepprofiler > unidino > morphogenie > o2vae    > dinocells
fold 1: cellclip > ca_mae > deepprofiler > unidino > morphogenie > dinocells > o2vae
fold 2: cellclip > ca_mae > deepprofiler > unidino > morphogenie > o2vae    > dinocells
fold 3: cellclip > ca_mae > deepprofiler > unidino > morphogenie > o2vae    > dinocells
```

`order_stable=False`，`stable_prefix_size=5`，`unstable_pairs=['dinocells','o2vae']`，`first_place_counts: cellclip=4`（其余全 0）。

### 4.5 必须随表附带的 caveat（提示词 §4.3，本次全部适用）

1. **不给 p 值** —— 每化合物只有单孔，化合物身份 ≡ 孔身份，留一孔 CV 无相关项。（脚本在 `no_p_values_because` 字段中亦记录了这一点）
2. **替身行必须标注** —— v3 表中 unidino / dinocells / morphogenie 不是论文性能。
3. **不可与论文数字比较** —— 协议是自建的（BBBC022 子集 + 自定义化合物检索任务）。
4. **o2vae 是形状模型** —— 吃剪影，与化合物表型任务不匹配（其 mAP_z 0.0471 甚至低于原始像素基线 0.0508，符合预期）。
5. **稳定性要报"稳定前缀 + 互换对"**，不能只报一个布尔值 —— 本报告已按此呈现（5/7 + dinocells/o2vae）。
6. **【补充】silhouette 一律为负** —— 7 个方法 + 2 基线的 sil_z 全为负值，morphogenie 低至 −0.7161。这说明在该协议下**不存在任何形成簇结构的嵌入**；若论文中声称"嵌入形成可分的细胞群簇"，本数据不支持。mAP/kNN 的相对差异是有的，但"簇结构"这一说法不成立。
7. **【补充】mock-AUC 不区分方法** —— 除 deepprofiler(0.5935)/ca_mae(0.5861)/unidino(0.5842) 外，多数在 0.47–0.54 徘徊；`baseline_random` 为 0.4996，说明 mock-AUC 在此协议下几乎不携带信号，不宜作为方法优劣证据。

### 4.6 产物

| 文件 | 说明 |
|---|---|
| `E:\bbbc022\cells\benchmark_scores_v2_review.json` | v2 打分（本次重算） |
| `E:\bbbc022\cells\benchmark_scores_v3_review.json` | v3 打分（本次重算） |
| `E:\bbbc022\cells\stability_v3_review.json` | 4 折稳定性（本次重算） |
| `E:\bbbc022\cells\emb_*.npy` | 七个方法的嵌入（本次 `--force` 重算） |

---

## 5. 结论

### 5.1 v2 能否发表 —— ⚠️ **有条件可发表**

**支持发表的证据**

- 阶段一 7/7 方法全部由统一接口驱动，输出宽度、形状、有限性、provenance 全部合规；W1–W8 中 6 项完全通过。
- 四个官方权重方法的实现与 `D:\official_src\` 的官方代码**逐行对照通过**：DeepProfiler 通道序（README:74）、逐裁片 min-max（cropping.py:65-67）、输出层与 672/1280 双宽度；CA-MAE 的 256×256/uint8/384 与复用官方 `predict()`；CellCLIP 的 `CrossChannelFormer` 模块名、形状、12 层 Transformer、`proj (1536,512)` 逐项吻合；O2-VAE 的 `zdim=512`、`Resize(64)`、`z_mu` 特征层、`q_net.fully_net.3.bias=(1024,)` 交叉验证通过。
- **DeepProfiler 逐位对拍已实测复现**：`bitwise_equal=True, max|diff|=0.000000e+00`（G2 中唯一成立的一项）。
- 测试基线与提示词 §7 完全一致（ep_bench 90 passed；.venv 300/9/14），且 14 项既有失败与 embedding 层无交集。

**发表前必须处理**

| 优先级 | 事项 | 出处 |
|---|---|---|
| P0 | **G1 关闭**：把 o2vae 的论文/仓库/DOI 写进 manifest 与文档 | §3 |
| P0 | **o2vae 输入域偏移**：官方权重训练于**细胞**剪影（`config_mefs.py:19` `scaled/cell`），本项目喂**核**剪影。要么改喂细胞剪影，要么在论文中明确声明这是域外推理 | §2.1 / §3 |
| P0 | **ca_mae 归一化定性**：`per_crop_max255` 不是官方做法（官方只有 `/255` + `InstanceNorm2d`）。虽被仿射不变性抵消（实测残差 ≈ 输出 std 的 0.7%），但**不得表述为"与官方一致"** | §2.1 |
| P1 | **F4**：deepprofiler `spatial=None` 导致 256×256 输入穿透契约、在 Keras 内部抛未分类 `ValueError` | 附录 A |
| P1 | **F2**：`dinocells` 契约方向修正（虽当前用替身权重，但契约是"未来换回官方权重"的唯一依据） | 附录 A |
| P1 | **F1**：`doctor()` / `runnable()` 对 dinocells 报假阴性，诊断面不可信 | 附录 A |
| P1 | **G2/G4 必须在论文中区分**：除 DeepProfiler 外，其余方法均为"对照官方公式的推理验证"，不是实测对拍 | §3 |

**结论**：四个官方方法的**实现质量足以支撑可发表**，但当前文档层面对 o2vae（论文缺失 + 域偏移）与 ca_mae（归一化定性）的陈述尚不足以支撑发表。补齐 P0 三项后可发表。

### 5.2 v3 的替身是否只用于接口验证 —— ✅ **是，且只能如此**

- 3 个替身的权重来自 `train_surrogates.py`（VICReg 自监督、仅 mock 细胞），**不是官方权重**（官方 Zenodo/Google Drive 均未取得）。
- 架构差异已全部声明并在本次复核确认：D1 unidino depth 4 vs 官方 12；D2 dinocells `vit_small` vs 官方 CP 的 `vit_tiny`/`vit_base`；D3 morphogenie 返回 mu（deterministic）vs 官方重参数化采样（stochastic）；D4 dinocells 的 mask 由 DNA 通道 Otsu 阈值生成，非 CellProfiler 分割。
- 本次 v3 打分与既有基线**逐位一致**，说明替身链路是**确定性可复现**的 —— 这正是接口验证需要的性质。
- ⚠️ **必须警惕的一处**：替身 `morphogenie` 的 sil_z = **−0.7161**，远低于其余所有方法（−0.03 ~ −0.19）。这不是性能信息，而是"10 维瓶颈 + 确定性均值 + 3 通道缩放裁片"这一组合在该协议下的分布特征。若有人把它当作"morphogenie 表现很差"的证据，是误读。

### 5.3 数据不支持的结论 —— ❌ 以下均为**数据不支持**

| 结论 | 为何不支持 |
|---|---|
| "cellclip 显著优于 ca_mae / deepprofiler" | 协议自建 + 单孔设计，无 p 值、无重复孔；只能表述为"在本协议下 mAP_z 更高" |
| "某方法优于论文报告的性能" | 协议（BBBC022 子集 + 化合物检索 + 自定义 z-score）与论文协议不同，无可比性 |
| "除 DeepProfiler 外某方法已与官方逐位一致" | G2 仍然成立 —— 那 6 个是推理验证，不是实测对拍 |
| "替身分数反映 unidino/dinocells/morphogenie 的性能" | 权重不是官方的（D1–D4） |
| "嵌入形成了可分的细胞群簇" | **7 个方法 + 2 基线的 silhouette 全为负**（§4.5-6） |
| "mock-AUC 可用于方法排序" | 基线随机投影为 0.4996，该指标在本协议下几乎无信号 |
| "o2vae 适合化合物表型任务" | 形状模型 + 域偏移；mAP_z 0.0471 低于原始像素基线 0.0508 |
| "dinocells 与 o2vae 谁更好" | 唯一跨折互换对，本数据无法分辨 |

### 5.4 审查者对提示词本身的意见（8 处）

| # | 提示词位置 | 问题 | 建议 |
|---|---|---|---|
| P1 | §2.3 输出块 `sha=` | 是**权重 sha256** 前缀，不是嵌入数组哈希；未标注口径，易导致审查者误报 7/7 不一致 | 改标注为 `weights_sha256[:12]` |
| P2 | §3.2 deepprofiler「manifest `spatial=(32,32)`」 | 实际是 **`spatial=None`** | 更正；并补上 F4 的后果 |
| P3 | §3.2 dinocells「`norm_loader`：逐图逐通道 `-min; *=255/max`」 | `np.min(img, axis=(0,1))` 在 2D 上是**全图标量**，非逐通道（numpy 语义） | 更正为"全图全局 min-max → uint8" |
| P4 | §3.2 morphogenie「MATLAB 居中，**不在仓库**」 | `.m` 源码**在仓库内**（`ImagePreProcessing/`），缺的是 MATLAB 运行时 + Cellpose + 原始数据；且居中是 **200×200**，256×256 在 Python 侧 | 更正；G3 的表述也需相应调整 |
| P5 | §3.1 / §3.5 G1 | o2vae 论文可补 | 直接填入 Burgess et al. 2024 Nat Commun 的完整引用 |
| P6 | §3.2 cellclip「`check_compatible` 应以 `chain-required` 拒绝」 | `check_compatible` 从不产生该 reason（实际是 `ndim`/`feature-width`/`channel-count`）；`chain-required` 只由 `prepare()` 抛出 | 更正判定标准 |
| P7 | §3.2 deepprofiler 通道序历史缺陷 | ✅ **正确** | 保留（官方 README:74 确认 DNA, ER, RNA, AGP, Mito） |
| P8 | §2.4 B4「`embed_all(names, payload)` 一次跑七个」 | `embed_all` 无 `assume`/`masks`，`channels` 为共享值；§2.3 的代码块**不能原样**用于 B4 | 明确要求"先 `models.prepare()` 预制备 + 传 `unit` 映射" |

另：§3.1 论文表中 ca_mae 的标题归属错配 ——「…are Scalable Learners of **Cellular Morphology**」属 **arXiv 2309.16064**；**arXiv 2404.10242** 的标题是「Masked Autoencoders **for Microscopy** are Scalable Learners of **Cellular Biology**」。

---

## 附录 A：新发现清单（提示词未覆盖）

共 **14 项**新发现（提示词 §3.4 的 D1–D4 之外）。按严重度排序：

| # | 严重度 | 发现 | 证据 | 影响 |
|---|---|---|---|---|
| **F2** | 🔴 高 | **dinocells 的磁带方向与官方相反**。官方 `data_utils/cellpainting_dataset.py:23-47` `fold_channels(image, channel_width, mode="drop")` 把磁盘磁带 **折回 (C,H,W) 堆叠张量并丢弃 mask 通道**，再送入模型（`:77-93` `SingleCellDataset.__getitem__` 在 transform 之前调用它）；本项目 `preprocessing.py:263-303` 反向把 (H,W,C)+mask **折成磁带**送入模型 | 官方源码逐行；Fortran order 用法本身正确（本地 `_to_tape` 与官方 `fold_channels` 实测互逆） | 契约描述的是**非官方**的模型输入几何。当前用替身权重，不影响已发表数字；但契约是"未来换回官方权重"的唯一依据，**换权重后会静默产生错误嵌入**。连带：`tape-batch` 批拒绝（官方可批，`batch_size_per_gpu: 24`）也是本地编码的产物而非官方约束 |
| **F1** | 🔴 高 | **`probe()` / `runnable()` / `doctor()` 对 dinocells 报假阴性**。manifest `requires=('torch','kornia','oyaml')`（沿用官方契约），但替身适配器只需 `torch`（`adapters/dinocells.py:47` `requires_frameworks=('torch',)`），且 `embed()` 走适配器自检、**从不校验 manifest.requires** | 实测：`doctor()` → `dinocells: runnable=false, status='wrong-interpreter', missing_dependencies=['oyaml']`，而 `models.embed('dinocells', …)` **成功返回 (1,384)**；`runnable()` 只返回 6 个 | 诊断面不可信。用户按 `doctor()` 指引会误以为该方法"不可用" |
| **F3** | 🟠 中 | **unidino 喂 uint8 静默变黑**。`div65535` 要求输入为 uint16，但 manifest 未声明 `value_range`，`contract.py:571-586` 的 dtype 守卫只在 `to_dtype='uint8'` 时触发（本方法 `to_dtype='float32'`） | 实测：uint8 输入 → 制备后 max ≈ 0.0039（= 1/257）；官方 `source/image_ops.py:14-20` 亦不检查 dtype | 静默产生近黑输入 → 嵌入无意义但不报错。正是提示词 §6 所列的"静默缺陷"类型 |
| **F4** | 🟠 中 | **deepprofiler 输入尺寸未被契约守护**。manifest `spatial=None`（提示词 §3.2 称 `(32,32)`，不符），`prepare()` 因此不 resize | 实测：喂 256×256 → Keras 内部裸 `ValueError: expected input_1 to have shape (32, 32, 5) but got array with shape (256, 256, 5)` | 违反本项目"一切拒绝都用带 reason code 的类型化错误"的既定原则；调用方拿不到可定位的错误 |
| **F5** | 🟠 中 | **morphogenie"官方居中不在仓库内"的表述不成立**。`ImagePreProcessing/MultipleCells/CPA/GetCentImages.m` **就在官方仓库内**；缺的是 MATLAB 运行时 + Cellpose + 原始数据。且 MATLAB 居中裁剪是 **200×200**（`:26` `Crop=[Centerx-100, Centery-100, 200, 200]`），256×256 缩放在 Python 侧 `gan_training/utils.py:352-357`；MATLAB 侧还做了逐 crop min-max→uint8（`:38`） | 官方源码逐行 | ①"不可复现"的论证被削弱（代码在，只是运行时不在）；② `registry.py:465/496` 与 `preprocessing.py:47` 的描述失实；③ manifest 声明 `scale='none'` 且无值域，即使日后放开拒绝也不足以描述官方输入 |
| **F6** | 🟠 中 | **unidino 权重来源不可核 + manifest notes 陈旧**。`registry.py:368` `source='unidino/checkpoints.zip'` 在官方仓库**不存在**（全树 grep 无命中；官方为 Zenodo doi:10.5281/zenodo.14988837 + `SSL_data/checkpoints`，脚本内为 `uniDINO.pth`）；`registry.py:394-395` notes 自称「scale 未声明、官方未写明」，与同文件 `registry.py:364` 的 `scale='div65535'` 及官方 `image_ops.py:19` 直接矛盾 | 官方仓库检索 | 契约元数据失真 |
| **F7** | 🟠 中 | **ca_mae 契约放行官方 checkpoint 无法处理的通道数**。官方 `huggingface_mae.py:75` `max_in_chans=11`，pos_embed 按 11 模态生成；本项目 `channel_count_fixed=False` | 实测：`check_compatible` 对 C=20 的 uint8 数组返回 `ok=True`（codes 为空） | 放行的输入会在 checkpoint 内部形状报错，而非在契约层被拒 |
| **F8** | 🟡 低 | **`embed_all` 无 per-method `channels`**。`channels` 是所有方法共享的单一值 | 实测：传 `('C0','C1','C2')`（morphogenie 的通道名）→ **5/7 方法被 `channel-set` 拒绝** | 七方法异构调度只能靠"都不传 channels"这一条窄路 |
| **F9** | 🟡 低 | **`embed_all` 无 `assume=` / `masks=`** | `inspect.signature` 实测：参数列表为 `names, inputs, unit, channels, sample_metadata, runtime_root, on_error, device, weights_roots, adapter_kwargs, chunk_size, kwargs` | §2.3 的代码块不能原样用于 B4（见提示词 P8） |
| **F10** | 🟡 低 | **替身权重未固定 sha256**。4 个官方权重在 manifest 中声明并校验通过（`verify_sha256` 4/4 MATCH）；3 个替身 `manifest.weights.sha256` 为空串 | 实测 W6：`declared=''` vs `actual=e53838fe807311bf` 等 | 替身 checkpoint 被替换不会被前置检查发现（实际值仍出现在结果 `weights_sha256` 中） |
| **F11** | 🟡 低 | **`manifest.requires` 字段在替身场景下语义不统一**：morphogenie 只写 `('torch',)`、unidino 写 `('torch','timm','einops','pytorch_lightning')`、dinocells 写 `('torch','kornia','oyaml')` —— 三者一致地描述**官方**实现，而非替身 | `doctor()` 输出 | `requires` 字段不能作为"这个方法在本机需要什么"的可靠依据（F1 的根因之一） |
| **F12** | 🟡 低 | **`embed(skip_contract_check=True)` 可完全绕过契约** | `inspect.signature` 实测：参数存在，默认 `False` | 默认关闭，属设计内；但应与 `skip_contract_check` 同名的其他开关一起在文档中列为"高危" |
| **F13** | 🟡 低 | **`chain-required` 永远不会由 `check_compatible` 产生**。该 reason 只出现在 `preprocessing.py:110`（`plan()` 步骤名）与 `:352`（`prepare()` 抛错） | 实测：像素喂 cellclip → codes 为 `('ndim',)` / `('channel-count','feature-width')` / `('feature-width',)` | 提示词 §3.2 的判定标准不准确（见 P6） |
| **F14** | 🟡 低 | **论文引用错配**：ca_mae 的两篇 arXiv 标题在提示词中互换；`adapters/ca_mae.py:5` 的 "Recursion + **Valence Labs**" 在官方仓库与两篇论文中**均无依据**（作者单位仅 Recursion） | `2-camae-openphenom/README.md:3-9` | 归属表述失实 |

### 附：另一处需要在论文中写明的 o2vae 风险

官方 `model_mefs.pt` 的 `p_net.fc.0.weight` 形状 **(48, 512)** → 官方 mefs 权重 `zdim=512`（mu 512 / logvar 512）；但论文正文 Methods 写 "embedding dimensions **d = 128** for all models trained in this paper"。**以代码/权重为准 = 512**（本项目配置正确），但这一"论文正文 vs 发布权重"的不一致应写入论文的补充材料，否则审稿人会认为是本项目算错。

---

## 附录 B：复现命令与产物

### B.1 环境

```powershell
$env:PATH = 'C:\Users\DELL\miniconda3\envs\torch_gpu\Library\bin;' + $env:PATH
$env:PYTHONPATH = 'D:\项目\CellPainting-Claw-v2\src'
$env:CPC_WEIGHTS_ROOT = 'E:\cpc_weights'
D:\ep_bench\venv\Scripts\python.exe   # py 3.10.20 / torch 2.5.1 / TF 2.15.1 / numpy 1.26.4
```

### B.2 执行过的命令

```powershell
# 阶段一（黑盒）
python -m pytest tests\test_models_v3_dispatch.py tests\test_models_profiles.py `
                tests\test_models_contract.py tests\test_models_preprocessing.py -q
#   -> 90 passed
python -m pytest tests -q          # 在 .venv（无 torch）下
#   -> 14 failed, 300 passed, 9 skipped

# 阶段二（白盒）
python scripts\embedding_benchmark\parity_vs_official_dp.py
#   -> A. demo order + per-crop min-max  bitwise_equal=True  max|diff|=0.000000e+00
#      契约守卫：旧通道序被拒 PASS / 16-bit 原始输入被拒 PASS
# 逐点对照通过阅读 D:\official_src\{1-deepprofiler,2-camae-openphenom,3-cellclip,5-unidino,6-dinocells,7-morphogenie}
# 与 E:\cpc_weights\o2vae_repo (= jmhb0/o2vae, HEAD 4c70d58)

# 阶段三（benchmark）
python scripts\embedding_benchmark\embed_dataset.py --cells-dir E:\bbbc022\cells --force
python scripts\embedding_benchmark\build_v3_surrogates.py --cells-dir E:\bbbc022\cells --force
python scripts\embedding_benchmark\score_benchmark.py --cells-dir E:\bbbc022\cells --profile v2 `
    --output E:\bbbc022\cells\benchmark_scores_v2_review.json
python scripts\embedding_benchmark\score_benchmark.py --cells-dir E:\bbbc022\cells --profile v3 `
    --output E:\bbbc022\cells\benchmark_scores_v3_review.json
python scripts\embedding_benchmark\stability_check.py --cells-dir E:\bbbc022\cells --profile v3 --folds 4 `
    --output E:\bbbc022\cells\stability_v3_review.json
```

### B.3 本次新增/变更的文件

| 文件 | 说明 |
|---|---|
| `docs/review_report_2026-10-26.md` | **本报告**（新建） |
| `E:\bbbc022\cells\emb_{deepprofiler,ca_mae,cellclip,o2vae,unidino,dinocells,morphogenie}.npy` | 七个方法的嵌入，`--force` 重算 |
| `E:\bbbc022\cells\benchmark_scores_v2_review.json` / `benchmark_scores_v3_review.json` / `stability_v3_review.json` | 打分与稳定性产物 |

**审查过程未修改仓库源码任何一行。**
