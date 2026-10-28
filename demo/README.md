# demo/ —— 两个可以直接看的示例

这里放了两份**真实跑过**的示例，不是示意图。数据量很小（2 视野 / 4 细胞 / 8 个样本细胞），
目的是让你看清"输入长什么样、每一步产出什么、分数从哪来"。

| demo | 目录 | 证明什么 | 规模 |
|---|---|---|---|
| **一体** | [`integrated/`](integrated/RUN.md) | 19 个 skill 从原始图一路跑到 DeepProfiler 特征：CellProfiler 测量 → pycytominer → 分割 → 深度特征 | 2 视野 / 4 细胞 / 1 个化合物 |
| **embedding** | [`embeddings/`](embeddings/RUN.md) | 7 个 embedding 方法一次调用跑通，并用同一套 benchmark 调度对齐比较 | 8 个细胞 |

> 本树带 7 个方法，其中 3 个是本地替身权重（已在各处标注）。

## 目录

```
demo/
├── integrated/                 # demo 1：一体流程
│   ├── RUN.md                  # 逐步说明 + 绝对路径
│   ├── raw/                    # 原始数据：2 视野 × 8 通道 tiff
│   ├── artifacts/              # 01–19 每一步的产物（CP 表、parquet、masks、DP 特征）
│   └── report.json             # 这一轮每个 skill 的耗时/产物/mode
├── embeddings/                 # demo 2：embedding + benchmark
│   ├── RUN.md
│   ├── data/sample_inputs.npz  # 原始数据：8 个细胞的四种输入格式
│   ├── artifacts/emb_*.npy     # 输出结果：每个方法一个矩阵
│   ├── benchmark/              # benchmark 调度产物 + report
│   └── report.json             # 每个方法的形状/维度/溯源/耗时
├── backend/                    # skill 用的 CellProfiler 后端（pipeline、load data、捆绑测量表）
└── data/sample_inputs.npz      # verify_all.py 用的同一份 8 细胞样本
```

## 怎么验证这两个 demo 是真的

```bash
python verify_all.py
```

它会重新跑一遍 19 个 skill + 全部 embedding 方法 + benchmark，结果写到
`verify/verify_report.json`。`demo/integrated/report.json` 和
`demo/embeddings/report.json` 是构建 demo 时的同一套代码产出的记录，可以对照着看。

## 关于体积

- DeepProfiler 跑完会在项目里存一份 checkpoint 副本，权重不进仓库，**已删除**；
- `data-download` 真下载的两个 illum `.npy`（各 4.45 MB）也只保留 `download_manifest.json`；
- 其余产物（CP 测量表、parquet、masks、DP 特征）体积小，全部保留，便于直接查看。

## 关于产物里的路径

我们**自己写**的三份 JSON（`integrated/report.json`、`embeddings/report.json`、
`embeddings/benchmark/*.json`）里所有路径都是**相对本树**的，换机器 clone 后可以直接解析。

由外部工具直接产出的文件保持原样，因此**里面会带构建机的绝对路径**，属于原始证据：

- `artifacts/04-cellprofiler-measurements/*.csv`、`artifacts/12-segmentation/cellprofiler_masks/*.csv`
  （CellProfiler 写的 `FileName_*` / `PathName_*`）
- `artifacts/*/pipeline_skill_manifest.json`、`artifacts/15-dp-export/images/field_metadata.csv`
- `artifacts/16-dp-project/inputs/metadata/index.csv`

这些是"工具确实跑过"的痕迹，不做改写。要换成本机路径请重跑 `python verify_all.py`。
