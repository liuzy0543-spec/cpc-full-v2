# CellPainting-Claw v2 —— 完全体

这一棵树把三件事合到了一起，并且**已经在本机实跑验证**：

1. **官方 19 个 skill**（数据访问 / 经典 profiling / 分割 / DeepProfiler 深度特征）
2. **训练 + 数据 + 配置 + 实验** 的完整工作流（presets 编排、DeepProfiler 训练、CellProfiler 管线定义）
3. **统一的 embedding 层**：4 个官方权重方法 + 3 个替身方法，一套 `models.embed()` 接口 + benchmark 调度

```bash
python verify_all.py          # 一条命令验证全部（本树实测见文末）
```

---

## 1. 本树与另一棵树的唯一区别

| | 本树（v2） | 说明 |
|---|---|---|
| embedding profile | `v3（别名 all）` | 写在 `src/cellpaint_pipeline/models/_release_profile.py` |
| 可用方法 | 7 个 | `deepprofiler`、`ca_mae`、`cellclip`、`o2vae`、`unidino`、`dinocells`、`morphogenie` |
| 其余代码 | 与另一棵树逐字节相同 | 差异只在这一个常量与文档 |

环境变量 `CPC_EMBEDDING_PROFILE` 可覆盖（接受 `v2`/`official` 或 `v3`/`all`）。
未识别的值不会让包不可导入，只会回退到 `v2`。

---

## 2. 19 个 skill（已逐个实跑）

单位：秒；`mode` 是这一步实际走的实现路径。

| # | skill | 产物 | 实跑结果 |
|---|---|---|---|
| 01 | `data-inspect-availability` | `data_access_summary.json` | OK，降级适配器记在 `degraded_adapters` |
| 02 | `data-plan-download` | `download_plan.json` | OK |
| 03 | `data-download` | `downloads/` + manifest | OK，**真实下载 2 个对象**（S3，unsigned） |
| 04 | `cp-extract-measurements` | Image/Cells/Nuclei/Cytoplasm CSV | OK，`mode=bundled-demo-outputs` |
| 05 | `cp-build-single-cell-table` | `single_cell.csv.gz` | OK，4 细胞 / 16 列 |
| 06 | `cyto-aggregate-profiles` | `aggregated.parquet` | OK，2 孔 |
| 07 | `cyto-annotate-profiles` | `annotated.parquet` | OK |
| 08 | `cyto-normalize-profiles` | `normalized.parquet` | OK |
| 09 | `cyto-select-profile-features` | `feature_selected.parquet` | OK |
| 10 | `cyto-summarize-classical-profiles` | summary + PCA 图 | OK |
| 11 | `cp-prepare-segmentation-inputs` | `load_data_for_segmentation.csv` | OK |
| 12 | `cp-extract-segmentation-artifacts` | masks/labels/outlines + CSV | OK，`mode=native-fallback`（见 §6） |
| 13 | `cp-generate-segmentation-previews` | `sample_previews_png/` | OK，2 张预览 |
| 14 | `crop-export-single-cell-crops` | crops + `single_cell_manifest.csv` | OK，4 个裁剪 |
| 15 | `dp-export-deep-feature-inputs` | `images/`, `locations/` | OK，2 视野 |
| 16 | `dp-build-deep-feature-project` | 可运行的 DP 项目 | OK |
| 17 | `dp-run-deep-feature-model` | `outputs/<exp>/features/*.npz` | OK，**真实前向**（TF，block6a_activation） |
| 18 | `dp-collect-deep-features` | parquet/csv.gz + manifest | OK，4 细胞 / 2 孔 |
| 19 | `dp-summarize-deep-features` | summary + PCA | OK，672 维特征 |

链路是**用上一步返回的 `primary_outputs` 串起来**的，不靠猜文件名 —— 这就是"统一 IO"的实际含义。

---

## 3. embedding 层

```python
import numpy as np
from cellpaint_pipeline import models

vec = models.embed('ca_mae', crops)        # 一次调用，方法名即入口
print(models.manifest('ca_mae').output_dim, vec.as_numpy().shape)
```

| 方法 | 输入单元 | 输出维度 | 权重来源 |
|---|---|---|---|
| `deepprofiler` | 32×32×5 裁剪 | 672 | 官方 `Cell_Painting_CNN_v1.hdf5` |
| `ca_mae` | 像素裁剪（自动缩放） | 384 | 官方 OpenPhenom-S/16（非商用 EULA） |
| `cellclip` | 每通道 1536 维向量 | 512 | 官方 CellCLIP |
| `o2vae` | 核剪影 64×64 | 512 | 官方 O2-VAE（MEFS） |
| `unidino` | 像素裁剪 | 1920 | **替身**（本地自训） |
| `dinocells` | 像素裁剪 | 384 | **替身**（本地自训） |
| `morphogenie` | 3 通道、上游已居中 | 10 | **替身**（本地自训） |

三个替身方法（`unidino` / `dinocells` / `morphogenie`）的权重是**本地自训替身**，
`weights_provenance='surrogate'` 会一路带到结果元数据里，不会混进官方成绩。

### 统一 IO 的约定

- 输入按 manifest 声明：像素裁剪（BHWC/BBHW 双向自适配）、通道向量、核剪影；
- `assume='raw_crop'` 表示走声明的预处理，`None` 表示已经准备好的值；
- 通道名、顺序、值域、空间尺寸都在 manifest 里，方法之间不共享硬编码；
- `models.declared()` 就是本树的可用集合，`embed_all(..., on_error='collect')` 批量跑。

### benchmark 调度

```python
from cellpaint_pipeline import embedding_benchmark as bench, models

plan = bench.build_benchmark_plan(output_dir='bench')   # 默认全部可跑方法
result = bench.run_benchmark(plan, inputs, sample_metadata=metadata)
print(result.shapes, result.errors)
```

`run_benchmark` 一次调用覆盖所有方法：按行身份对齐、逐方法记录权重 sha256、
随机方法重复采样并记录散度、许可证分级，最后落盘可比较的产物。

---

## 4. 训练 / 数据 / 配置 / 实验

| 环节 | 入口 | 说明 |
|---|---|---|
| 数据访问 | `data-*` 三个 skill + `data_access` 模块 | S3 gallery（unsigned）、Quilt、cpgdata 索引；缺适配器时记 `degraded_adapters` 而不是整体失败 |
| 配置 | `configs/project_config.*.json` | 项目配置即契约：workspace、后端、DP 运行时、data_access |
| 训练/推理 | `dp-run-deep-feature-model`、`deepprofiler` CLI | demo 用官方 Cell_Painting_CNN_v1 权重做**特征抽取**；训练同一入口换 config |
| 实验编排 | 8 个 preset：`run-pipeline-preset --preset <key>` | `profiling-only`、`segmentation-only`、`deepprofiler-export`、`deepprofiler-full`、`full-pipeline`、`full-pipeline-with-data-plan`、`data-access-plan`、`data-access-download` |

---

## 5. 环境

已验证的解释器：conda 环境 `cellpainting-claw`（Python 3.10.12）。

| 依赖 | 版本 | 用途 |
|---|---|---|
| numpy / pandas / scikit-learn / scikit-image | 1.26.4 / 2.3.3 / 1.7.2 / 0.22 | 全流程 |
| pycytominer | 已装 | 06–09 |
| deepprofiler + tensorflow | 0.5.1 / 2.10.1 | 17–19 |
| torch / torchvision | 2.5.1 / 0.20.1 | embedding |
| timm / transformers / safetensors | 1.0.30 / 4.46.1 / 0.8.0 | ca_mae、cellclip |
| e2cnn / torchgeometry | 0.2.3 / 0.1.2 | o2vae |
| boto3 / quilt3 / cpgdata | 1.43 / 8.0 / 0.6 | data-* |

三个必须知道的坑（都已在代码里处理，但部署时要清楚）：

1. **TF 与 torch 不能同进程**：两个框架各带一套 OpenMP/MKL，同进程会让
   `torch/lib/fbgemm.dll` 加载失败（WinError 127）。`verify_all.py` 因此把
   embedding / benchmark 放到子进程跑 —— 这不是验证脚本的取巧，生产里也建议分开。
2. **`KMP_DUPLICATE_LIB_OK=TRUE`**：torch 的 `libiomp5md` 与 MKL numpy 的 `libomp`
   会争用 OpenMP 运行时，不设这个变量进程会直接 abort。`verify_all.py` 已设置。
3. **PATH 不要混装别的环境**：把 `torch_gpu/Library/bin` 之类加进 PATH 会让
   `numpy.linalg.svd` 崩在 DLL 初始化（0xc06d007f），表现为"随机崩溃"。

权重目录用 `CPC_WEIGHTS_ROOT` 指定，默认探测 `E:\cpc_weights` 等位置。

---

### 从哪棵树导入，要说清楚

本机的 conda 环境里装的是**开发树**的 editable 包（`D:\项目\CellPainting-Claw-v2\src`），
所以直接在发布树目录里 `python -c "import cellpaint_pipeline"` 有可能导入到开发树而不是本树。
两种正确用法：

```bash
python verify_all.py          # 它会把本树的 src 放到 sys.path 最前，再跑全部检查
# 或者
pip install -e .              # 把本树装成当前环境的包
```

判断当前导入的是哪棵树：`python -c "import cellpaint_pipeline; print(cellpaint_pipeline.__file__)"`。

## 6. 与官方的功能对应，以及一处明确的替代

| 官方做法 | 本树做法 | 是否等价 |
|---|---|---|
| CellProfiler 跑 profiling | 有 CellProfiler 就跑；否则复用捆绑的 CP 测量表 | 是（`mode=bundled-demo-outputs`） |
| CellProfiler 跑分割 | 有 CellProfiler 就跑；否则用内置原生分割（Otsu + 分水岭 + 细胞体扩张） | **产物契约等价，像素级不等价**（`mode=native-fallback`） |
| DeepProfiler 抽取深度特征 | 同一入口、同一官方权重、同一输出层 | 是 |
| 19 个 skill 的输入输出契约 | 逐项对齐 | 是 |
| 新增 embedding 方法 | 官方没有，属新增层 | — |

原生分割不是桩：它真的做阈值、距离变换、分水岭、细胞体扩张，产出
labels/outlines + `Image.csv`/`Cells.csv`/`Nuclei.csv`/`Cytoplasm.csv`，
demo 上得到 2 视野 / 4 细胞，与捆绑 CP 表的行数一致。但它**不能**冒充 CellProfiler
的数值结果，所以每处结果都带 `mode`，且文档写明。

---

## 7. 本机实测结果

```bash
python verify_all.py
```

| 检查 | 结果 |
|---|---|
| 19 个 skill | **19/19 OK** |
| embedding 方法 | ****7/7 OK**** |
| benchmark 调度 | ****7 个方法，0 错误**** |

报告：`verify/verify_report.json`（含每个 skill 的耗时、产物路径、mode）。
原始运行产物：`verify/run/`。

---

## 8. 已知限制（不粉饰）

1. **demo 是 2 孔 / 4 细胞的小样本**，用于证明链路通，不代表统计结论；
   `cyto-*` 在 2 行数据上算出的相关性没有意义（部分指标会是 NaN/常数）。
2. **`data-download` 需要网络**；受限网络下会失败，这是外部依赖，不是代码问题。
   验证时用 `max_files=2` 限定下载量。
3. 三个替身方法的分数**不是**这些方法的发表性能。
4. `o2vae` 吃核剪影，官方权重训练于细胞剪影，属域外推断 —— 见 `docs/` 里的评分说明。
## 9. demo/（两份真实运行，不是示意图）

| demo | 目录 | 内容 |
|---|---|---|
| **一体** | `demo/integrated/` | 原始 2 视野 × 8 通道 tiff → **真 CellProfiler** 测量 → pycytominer → **真 CellProfiler** 分割 → DeepProfiler 672 维特征；`report.json` 记录每步耗时/产物/mode |
| **embedding** | `demo/embeddings/` | 8 个细胞的四种输入格式 → 7 个方法各一个 `.npy` → benchmark 调度产物 |

各自的说明：`demo/README.md`、`demo/integrated/RUN.md`、`demo/embeddings/RUN.md`。

## 10. 本机绝对路径与后续配置

- `docs/ABSOLUTE_PATHS.md`：这台机器上实测通过的全部绝对路径（解释器、CellProfiler、权重、
  数据、环境变量），以及换机器要替换哪几项；
- `docs/CONFIGURATION.md`：从零到 `ALL CHECKS PASSED` 的配置顺序、`project_config` 逐字段说明、
  权重目录布局、换自有数据的具体步骤、提交前自检清单。

关键取值（写死在文档里，不写死在代码里）：

```
Python          C:\Users\DELL\miniconda3\envs\cellpainting-claw\python.exe   (3.10.12)
CellProfiler    C:\CellProfiler\CellProfiler.exe                                 (4.2.7)
权重根           E:\cpc_weights
发布树           D:\release\cellpainting-claw-v2
```

## 11. CellProfiler 的真实状态

本机**已经装了** CellProfiler 4.2.7（`C:\CellProfiler`，PyInstaller 冻结版）。
两棵树的最新一轮验证里，04 和 12 两步都是 `mode=cellprofiler`：

| skill | mode | 耗时 |
|---|---|---|
| `cp-extract-measurements` | `cellprofiler` | ~36–41 s |
| `cp-extract-segmentation-artifacts` | `cellprofiler` | ~33–37 s |

两个路径相关的坑（脚本已处理，换机器仍要注意）：

1. CellProfiler 用 `os.path.relpath(cwd)` 找自身资源，**cwd 必须与安装盘同盘**，
   否则启动即崩（`path is on mount 'C:', start on mount 'D:'`）；
2. **仓库路径不能含非 ASCII 字符**，否则 CP 读图报 `Test for access to directory failed`。
   两棵发布树都在 `D:\release\` 下，满足这个条件。

没有 CellProfiler 的机器上，04 退回捆绑测量表、12 退回内置原生分割，`mode` 会如实标注。

