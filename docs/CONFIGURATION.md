# 配置指南：从零到 `ALL CHECKS PASSED`

发布树里所有可配置项都在三处：**环境变量**、`configs/*.json`、以及权重目录布局。
本文按"先能跑 → 再换成自己的数据 → 再调参"的顺序说明。

## 1. 三步跑起来

```bash
cd D:\release\cellpainting-claw-v2
pip install -e .[deepprofiler,embeddings]      # 或使用已有环境
set CPC_WEIGHTS_ROOT=E:\cpc_weights
set CELLPROFILER_EXECUTABLE=C:\CellProfiler\CellProfiler.exe     # 可选，缺省会自动探测
python verify_all.py
```

`verify_all.py` 会依次做三件事，并把结果写到 `verify/verify_report.json`：

1. 跑 19 个 skill（有 CellProfiler 就用真的，没有就标注降级模式）；
2. 跑本树声明的全部 embedding 方法；
3. 跑 benchmark 调度。

## 2. 环境变量

| 变量 | 必填 | 默认 | 说明 |
|---|---|---|---|
| `CPC_WEIGHTS_ROOT` | 是（无权重时用不了方法） | 自动探测若干目录 | 权重根；覆盖而非追加 |
| `CELLPROFILER_EXECUTABLE` | 否 | 探测 PATH / `C:\CellProfiler\CellProfiler.exe` | 指到 `CellProfiler.exe` |
| `CPC_EMBEDDING_PROFILE` | 否 | v1 树 `v2`，v2 树 `v3` | 可用值：`v2`/`official`（4 个方法）、`v3`/`all`（7 个） |
| `KMP_DUPLICATE_LIB_OK` | 建议 | — | `TRUE`，torch + MKL numpy 共存 |
| `HTTPS_PROXY` | 否 | — | 仅 GitHub / PyPI 需要 |

## 3. 权重目录布局

```
$CPC_WEIGHTS_ROOT/
├── deepprofiler/Cell_Painting_CNN_v1.hdf5
├── openphenom_pkg/openphenom/            # 目录本身是 importable 包
│   ├── __init__.py, huggingface_mae.py, ...
│   └── model.safetensors
├── cellclip/model.safetensors
├── o2vae/o2vae_mefs_official.pt
├── unidino/surrogate_unidino.pt          # 替身，仅 v3/all 可见
├── dinocells/surrogate_dinocells.pt
└── morphogenie/surrogate_morphogenie.pt
```

`ca_mae` 特殊：`openphenom_pkg/openphenom` 这个**目录本身**是一个 Python 包（上游
`save_pretrained` 布局），适配器会把它的**父目录**加到 `sys.path` 再 `import openphenom.huggingface_mae`。

## 4. `configs/project_config.demo.json` 逐字段

| 字段 | 作用 | 换成自己数据时怎么改 |
|---|---|---|
| `project_name` | 仅标识 | 随意 |
| `python_executable` | 后端脚本用哪个解释器（空=当前） | 一般留空 |
| `profiling_backend_root` | CP profiling 后端目录 | 指到自己的后端 |
| `profiling_backend_config` | 后端自己的 `pipeline_config.json`（相对后端目录） | 改里面的 `paths` |
| `segmentation_backend_root` / `segmentation_backend_config` | 同上，分割后端 | 同上 |
| `workspace_root` | 该项目的运行时根目录 | 改成绝对路径 |
| `default_output_root` | 默认输出根 | 同上 |
| `deepprofiler_export_root` / `deepprofiler_project_root` | DP 导出/项目目录 | 同上 |
| `deepprofiler_runtime.*` | 实验名、box size、batch、特征层、checkpoint 名 | 按需 |
| `data_access.*` | S3 gallery / Quilt / cpgdata 配置 | 换 bucket 或关掉不用的适配器 |
| `cellprofiler.*` | 用哪个 `.cppipe` 模板 | 换模板或自定义路径 |

后端自己的 `pipeline_config.json` 里 `paths` 才是真正决定文件读写的字段
（`raw_images_dir`、`cellprofiler_output_dir`、`*_table_csv`、`single_cell_output_csv_gz` 等），
相对路径是相对**后端根目录**解析的。

## 5. 换自己的数据

1. 原始图按 CPJUMP1 命名：`r{row}c{col}f{site}p01-ch{n}sk1fk1fl1.tiff`（8 通道）。
2. 写 `data/metadata/image_manifest.csv`、`plate_map.csv`、`load_data_with_illum.csv`。
3. `configs/pipeline_config.json` 的 `paths.raw_images_dir` 指到图目录。
4. 跑 `cp-extract-measurements`（会自动跑 CP），再跑 05–10 的 pycytominer 链。
5. 分割同理：`data/source_load_data.csv` + `configs/segmentation_config.json`。

## 6. embedding 层配置

```python
from cellpaint_pipeline import models
models.set_profile('v3')                  # 或 'all'；'v2'/'official' 是 4 个方法
print(models.declared())
result = models.embed('ca_mae', crops)    # 输入契约由 manifest 决定
```

- 换权重：`models.embed(..., weights_roots=[r'D:\my_weights'])`
- 分批：`models.embed_all(names, inputs, chunk_size=256)`
- 单方法失败不中断：`on_error='collect'`

## 7. benchmark 配置

```python
from cellpaint_pipeline import embedding_benchmark as bench, models
plan = bench.build_benchmark_plan(models.declared(), output_dir='bench',
                                  unit={'cellclip': 'channel_embeddings'},
                                  assume={'ca_mae': 'raw_crop'},
                                  channels={'morphogenie': ('C0', 'C1', 'C2')})
result = bench.run_benchmark(plan, inputs, sample_metadata={'row_ids': ids})
```

`unit` / `assume` / `channels` 都支持"一个值给所有方法"或"按方法名给映射"。

## 8. 提交前自检清单

- [ ] `python verify_all.py` → `ALL CHECKS PASSED`
- [ ] `verify/verify_report.json` 里 `skills` 全 ok、`benchmark.errors` 为空
- [ ] 没有把权重、`.hdf5`、下载的数据文件提交进仓库
- [ ] 路径没有非 ASCII 字符（CellProfiler 限制）
