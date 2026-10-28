# scripts/embedding_benchmark

这个目录按**一件事**分工（不是按"一次实验"）。先看你要做什么，再看表。

背后那份结果说明在
[`docs/benchmark_bbbc022_result.md`](../../docs/benchmark_bbbc022_result.md)，
接口契约在 [`docs/embedding_interface.md`](../../docs/embedding_interface.md)。

## 我要跑起来 / 看结果

| 脚本 | 用途 | 耗时 |
|---|---|---|
| `run_e2e.sh` | **全流程**：重算 v2+v3 嵌入 → IO 校验 → 打分 → 稳定性；日志写 `E:/bbbc022/cells/_e2e_run.log` | 8–25 分钟（ca_mae 受 GPU 争用影响最大：空闲 190 s，拥挤曾到 1341 s） |
| `audit_round.py` | **审核**：六道闸门，输出 `docs/audit/round_NN_findings.md` | ~1 分钟 |
| `audit_loop.sh` | 串起上面两个：跑 → 审核 → 不通过就停下等人修 | — |

## 我要重建数据集

| 脚本 | 用途 |
|---|---|
| `fetch_bbbc022_plate.sh` | 下载 plate 20585 的 5 个**完整**分通道 zip。本地旧副本被截断（0.57–0.78 GB vs 应有 ~1.38 GB），只有 72 个视野凑齐 5 通道，这是数据集只有 3.9k 细胞的原因 |
| `salvage_partial_zips.py` | 从**截断的** zip 里按本地文件头抢救可读图像（截断 zip 没有中央目录，标准解压会整体失败） |
| `build_bbbc022_cells.py` | 分割 + 裁剪，一次产出四套表示（`cells_{dp,ca,clip,o2}.npy` + `meta.csv`）。按**每视野配额**分配，不是全局截断 |
| `make_cellclip_vectors.py` | CellCLIP 的输入：crops → DINOv2-Giant 逐通道 → `(N,5,1536)`。**慢（19,285 次前向，RTX 4060 约 45 分钟）且不能跳**：CellCLIP 不读像素，喂噪声会得到一个"跑通了但无意义"的结果 |

### 实测用过的大数据集配方（11,657 细胞 / 378 孔 / 314 化合物）

```bash
# 1) 取完整 zip
bash scripts/embedding_benchmark/fetch_bbbc022_plate.sh          # 5 x 1.44 GB, 各 3457 张

# 2) 只保留本板标签：否则每视野配额会被其它 19 个板的孔稀释
python -c "import pandas as pd; \
d = pd.read_csv('E:/bbbc022/BBBC022_v1_image.csv', engine='python', on_bad_lines='skip'); \
d[d['Image_Metadata_PlateID'].astype(str).str.strip() == '20585'].to_csv('E:/bbbc022/labels_20585.csv', index=False)"

# 3) 建细胞（--max-cells 12000 时每视野上限 31 个，实际产出 11,657）
python scripts/embedding_benchmark/build_bbbc022_cells.py \
    --zips E:/bbbc022/_full/BBBC022_v1_images_20585w{1,2,3,4,5}.zip \
    --labels E:/bbbc022/labels_20585.csv \
    --output-dir E:/bbbc022/cells_full --max-cells 12000

# 4) CellCLIP 的输入向量（58,285 次前向；RTX 4060 + 显卡争用下约 2.3 小时）
python scripts/embedding_benchmark/make_cellclip_vectors.py \
    --crops E:/bbbc022/cells_full/cells_clip.npy \
    --model-dir E:/cpc_weights/dinov2_giant \
    --output E:/bbbc022/cells_full/cellclip_vectors.npy --device cuda --batch-size 2

# 5) 其余全部步骤走同一条链，只换数据集目录
CELLS_DIR=E:/bbbc022/cells_full bash scripts/embedding_benchmark/run_e2e.sh
```

**为什么步骤 2 不能省**：`build_bbbc022_cells.py` 的每视野配额是
`max_cells // len(rows)`，而 `rows` 来自标签文件。用全.csv 时 `rows` 有 6.9 万条
（20 个板都在内），配额掉到 1，于是只有本板的 3,457 个视野各出 1 个细胞。
历史上这个坑表现为"细胞数看着正常、孔数从 65 掉到 26"。

## 我要训练替身权重（v3 的三个方法）

| 脚本 | 用途 |
|---|---|
| `train_surrogates.py` | VICReg 自监督，**只用 mock 细胞、不用化合物标签**，产出三个替身 checkpoint |
| `build_v3_surrogates.py` | 用替身权重跑 embedding，写 `emb_{unidino,dinocells,morphogenie}.npy` |

## 我要评测

| 脚本 | 用途 |
|---|---|
| `embed_dataset.py` | v2 四个官方方法 → `emb_*.npy` |
| `score_benchmark.py` | mAP / kNN / silhouette / mock-AUC，`--profile {v2,v3}` |
| `stability_check.py` | 4 折按**孔**划分，报告稳定前缀与跨折互换对 |
| `verify_io_uniformity.py` | 七方法输出形状/字段/权重摘要是否一致，`--profile {v2,v3}` |

## 我要对拍官方实现

| 脚本 | 用途 |
|---|---|
| `parity_vs_official_dp.py` | 与官方 DeepProfiler 逐位对拍 |

## 我要检查文档

| 脚本 | 用途 |
|---|---|
| `check_doc_links.py` | `docs/` 下 markdown 链接是否还有效（移动文档后必跑） |

---

## 这些脚本刻意不做的两件事

**不报 p 值。** 本子集里每个化合物恰好占一个孔，所以化合物身份与孔身份是同一个变量。
排除查询自身所在的孔之后就没有相关项了（实测 64 个孔里 53 个无定义）；
保留它测的是"认孔"而不是化学。三种重采样方案都试过且都失败——各自哪里错了写在
`stability_check.py` 的 docstring 里。改为报告**排序**是否能跨折存活。

**不把绝对 mAP 当准确率。** 原始像素与随机基线用同一批裁剪打分，可解释的只有"相对它们的比较"。
注意：o2vae 的 4 折区间（0.0747–0.1179）与随机基线（0.0683–0.1109）**重叠**，
它的名次不被本数据支持。

## 约定

1. **所有脚本必须 profile-aware**：默认 `v2`，`--profile v3` 才包含三个替身。
2. **替身输出必须标注 `surrogate`**，不得与官方权重的方法混进同一张结论表。
3. 脚本只报告、不自我修复：审核脚本若改动被测对象，等于自己给自己发许可证。
4. 新增脚本先在本文件登记，否则下一个人找不到它。

## Requirements

`pip install -e ".[embeddings-all]"` — 或者只要评分那一半（不需要深度学习框架）：
`numpy pandas scipy`。