# 本机绝对路径清单（改动前先看这里）

这份文档把**这台机器上实测通过**的绝对路径全部列出来。发布树里的代码不写死这些路径，
它们只出现在：环境变量、`configs/*.json`、以及本目录的文档里。换机器时照 §6 逐项替换即可。

> 采集时间：2026-10-28　平台：Windows-11-10.0.26200-SP0　Python：3.10.12

## 1. 解释器与外部程序

| 用途 | 绝对路径 | 版本 | 说明 |
|---|---|---|---|
| 主运行环境 | `C:\Users\DELL\miniconda3\envs\cellpainting-claw\python.exe` | 3.10.12 | 装了 pycytominer / deepprofiler / TF / torch / pycytominer 等 |
| CellProfiler | `C:\CellProfiler\CellProfiler.exe` | 4.2.7 | 冻结版（PyInstaller one-dir），只能 CLI 调用 |
| 开发树（源码） | `D:\项目\CellPainting-Claw-v2` | — | 发布树的来源，改代码改这里再同步 |
| 官方副本（只读参考） | `C:\cpc-repo-v1` | — | 含官方 demo 后端脚本，如 `07_run_official_cellprofiler.py` |

## 2. 权重

根目录：`E:\cpc_weights`（由 `CPC_WEIGHTS_ROOT` 指定）

```
  _backup_20261026/
  _speedtest/
  _v3/
  cellclip/
  deepprofiler/
  dino4cells/
  dinocells/
  dinov2_giant/
  morphogenie/
  o2vae/
  o2vae_repo/
  openphenom_pkg/
  surrogates_manifest.json
  unidino/
```

解析顺序：`roots=` 显式参数 → `CPC_WEIGHTS_ROOT` → 内置默认（`E:\cpc_weights`、
`D:\cpc_weights`、`D:\模型权重`、`~/.cellpainting-weights`）。环境变量是**覆盖**不是追加。

## 3. 发布树与 demo

| 树 | 根目录 | demo |
|---|---|---|
| v1（4 个官方权重方法） | `D:\release\cellpainting-claw-v1` | `D:\release\cellpainting-claw-v1\demo\` |
| v2（4 官方 + 3 替身） | `D:\release\cellpainting-claw-v2` | `D:\release\cellpainting-claw-v2\demo\` |

每棵树内部固定布局（相对路径，可随树移动）：

| 相对路径 | 内容 |
|---|---|
| `src/cellpaint_pipeline/` | 主包：skills、workflows、models |
| `configs/project_config.demo.json` | demo 项目配置 |
| `demo/backend/profiling_backend/` | CellProfiler profiling 后端（pipeline、load data、捆绑测量表） |
| `demo/backend/segmentation_backend/` | CellProfiler 分割后端（mask-export pipeline） |
| `demo/integrated/` | demo 1：一体流程（原始图 → CP → pycytominer → 分割 → DeepProfiler） |
| `demo/embeddings/` | demo 2：embedding（原始样本、7 种输出、benchmark） |
| `verify/verify_report.json` | 最近一次 `verify_all.py` 的报告 |

## 4. 数据

| 用途 | 绝对路径 | 规模 |
|---|---|---|
| demo 原始图（2 视野 × 8 通道） | `D:\release\cellpainting-claw-v2\demo\integrated\raw\BR00000001\` | 16 个 tiff，64×64 |
| demo 单细胞样本（4 种输入格式） | `<tree>\demo\embeddings\data\sample_inputs.npz` | 8 个细胞 |
| 完整 BBBC022 小数据集 | `E:\bbbc022\cells` | 3,857 细胞 / 54 化合物（评测用，**不随仓库分发**） |

## 5. 环境变量（实测生效的取值）

| 变量 | 本机取值 | 作用 |
|---|---|---|
| `CPC_WEIGHTS_ROOT` | `E:\cpc_weights` | 权重根目录 |
| `CPC_EMBEDDING_PROFILE` | 不设（v1 默认 `v2`，v2 默认 `v3`） | 覆盖 embedding 方法集合 |
| `CELLPROFILER_EXECUTABLE` | `C:\CellProfiler\CellProfiler.exe` | 直接指定 CellProfiler；不设则按 PATH / 常见目录探测 |
| `KMP_DUPLICATE_LIB_OK` | `TRUE` | torch 与 MKL numpy 的 OpenMP 共存 |
| `HTTPS_PROXY` | `http://127.0.0.1:7890` | 仅访问 GitHub/PyPI 时需要；S3 直连 |

## 6. 换一台机器要改什么

1. **Python**：新建环境，装 `pip install -e .[deepprofiler,embeddings]`（或直接复用 `cellpainting-claw` 环境的包清单）。
2. **CellProfiler**：装 4.2.x，把 `CELLPROFILER_EXECUTABLE` 指到 `CellProfiler.exe`。
   注意两个硬约束，否则会看到致命错误：
   - **工作目录不能与 CellProfiler 安装盘不同**（本机 CP 在 C:，所以脚本把子进程 `cwd` 钉在 `C:\`）；
   - **仓库路径不能含非 ASCII 字符**，否则 CP 读图时报 `Test for access to directory failed`。
3. **权重**：把权重按 `<root>/<model>/<file>` 放好，`CPC_WEIGHTS_ROOT` 指过去。
4. **数据**：`configs/project_config.demo.json` 里的 `workspace_root`、`default_output_root`、
   `deepprofiler_export_root`、`deepprofiler_project_root` 默认是相对配置文件的 `../demo/...`，
   指向自有数据时改成绝对路径。
5. **跑一遍**：`python verify_all.py`，要求 `ALL CHECKS PASSED`。

## 7. 已知的路径相关错误（按报错查表）

| 报错 | 原因 | 处理 |
|---|---|---|
| `ValueError: path is on mount 'C:', start on mount 'D:'` | CellProfiler 用 `os.path.relpath(cwd)` 找资源，cwd 与安装盘不同 | 脚本已把 CP 子进程 `cwd` 钉到安装盘根 |
| `Test for access to directory failed` | 仓库路径含中文，CP 的缓存检查失败 | 把树放在纯 ASCII 路径（本机 `D:\release\...` 就是） |
| `torch/lib/fbgemm.dll ... WinError 127` | TF 与 torch 同进程 | 分进程运行（`verify_all.py` 已隔离） |
| `numpy.linalg.svd` 崩在 `0xc06d007f` | PATH 里混入其它环境的 MKL/TBB | 只保留本环境的 `Library\bin` |
| `OMP: Error #15 ... libiomp5md.dll` | torch 与 MKL numpy 双 OpenMP | 设 `KMP_DUPLICATE_LIB_OK=TRUE` |
