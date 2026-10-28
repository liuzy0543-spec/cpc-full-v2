# demo · 一体流程（19 个 skill 全跑）

这一份是**从头到尾**的一次真实运行：CellProfiler 实际执行、pycytominer 实际聚合、
分割实际产出 mask、DeepProfiler 用官方权重实际前向。

```bash
cd D:\release\cellpainting-claw-v2
python verify_all.py --skills          # 重跑一遍（会把下面所有产物重新生成）
```

或者直接看已经跑好的结果：

```
demo/integrated/
├── raw/BR00000001/*.tiff     # 投入数据：2 视野 × 8 通道
├── artifacts/                # 产出数据：按步骤编号
└── report.json               # 每一步的 ok / 耗时 / mode / 产物路径
```

## 投入数据

`raw/BR00000001/` 下 16 个 tiff（2 视野 × 8 通道，64×64），命名遵循 CPJUMP1：
`r{row}c{col}f{site}p01-ch{n}sk1fk1fl1.tiff`。
通道映射：ch1=Mito、ch2=AGP、ch3=RNA、ch4=ER、ch5=DNA、ch6=Brightfield、ch7=HighZBF、ch8=LowZBF。

## 每一步的脚本 / 模型 / 产物

| # | skill | 实际做什么 | 产物（相对 `artifacts/`） |
|---|---|---|---|
| 01 | `data-inspect-availability` | 探测 S3/Quilt/cpgdata 适配器可用性 | `01-data-inspect/data_access_summary.json` |
| 02 | `data-plan-download` | 生成下载计划 | `02-data-plan/download_plan.json` |
| 03 | `data-download` | 真实下载 2 个对象（unsigned S3） | `03-data-download/downloads/download_manifest.json` |
| 04 | `cp-extract-measurements` | **启动 CellProfiler 4.2.7** 跑 `CPJUMP1_analysis_without_batchfile_406.cppipe` | `04-cellprofiler-measurements/*.csv` |
| 05 | `cp-build-single-cell-table` | 合并 CP 表 | `05-single-cell-table/single_cell.csv.gz` |
| 06–09 | `cyto-*` | pycytominer 聚合/注释/归一化/特征选择 | `06-aggregate/` … `09-feature-select/`（parquet） |
| 10 | `cyto-summarize-classical-profiles` | 汇总 + PCA | `10-classical-summary/`（summary、top features、pca png） |
| 11 | `cp-prepare-segmentation-inputs` | 生成分割 load-data | `11-seg-load-data/load_data_for_segmentation.csv` |
| 12 | `cp-extract-segmentation-artifacts` | **再次启动 CellProfiler** 跑 mask-export 管线 | `12-segmentation/cellprofiler_masks/`（Image/Cells/Nuclei/Cytoplasm CSV + labels/ + outlines/） |
| 13 | `cp-generate-segmentation-previews` | 合成预览图 | `13-seg-previews/sample_previews_png/` |
| 14 | `crop-export-single-cell-crops` | 按 mask 裁单细胞 | `14-single-cell-crops/masked/` |
| 15 | `dp-export-deep-feature-inputs` | 组织 DeepProfiler 输入 | `15-dp-export/images/`、`locations/` |
| 16 | `dp-build-deep-feature-project` | 生成可运行项目 | `16-dp-project/inputs/` |
| 17 | `dp-run-deep-feature-model` | **DeepProfiler 前向**（TF，官方 `Cell_Painting_CNN_v1.hdf5`，层 `block6a_activation`） | `16-dp-project/outputs/*/features/*.npz` |
| 18 | `dp-collect-deep-features` | 收集特征成表 | `18-dp-collect/deepprofiler_single_cell.parquet` 等 |
| 19 | `dp-summarize-deep-features` | 汇总 + PCA（672 维） | `19-dp-summary/` |

## 这一轮的数字

| 指标 | 值 |
|---|---|
| CellProfiler 版本 | 4.2.7（`C:\CellProfiler\CellProfiler.exe`） |
| CP 测量表列数 | 约 1,900 列（Image/Cells/Nuclei/Cytoplasm） |
| 分割结果 | 2 视野 / 4 个细胞核 / 4 个细胞 |
| DeepProfiler 特征 | 672 维，4 个细胞，2 个孔 |
| 每一步耗时 | 见 `report.json`（CP 两步各约 35 s，DP 前向约 22 s） |

## 关于 CellProfiler

两条硬约束是这台机器上撞出来的，脚本里已经处理：

1. **CP 子进程的 cwd 必须在 CP 安装盘上**，否则启动即崩（`ntpath.relpath` 抛
   `path is on mount 'C:', start on mount 'D:'`）。脚本把它钉在 `C:\`。
2. **仓库路径不能含非 ASCII**，否则 CP 读图报 `Test for access to directory failed`。
   本树在 `D:\release\cellpainting-claw-v2`，是纯 ASCII。

没有装 CellProfiler 时，04 会退回"复用捆绑测量表"（`mode=bundled-demo-outputs`），
12 会退回内置原生分割（`mode=native-fallback`）——两种降级都会写进 `report.json` 的 `mode`。
