# C — 双树产物一致性验证结果

> **结论：官方树与 v2 树（重构版）的 DeepProfiler 产物逐字节相同，max|diff| = 0.000e+00。**
> 日期：2026-10-09

---

## 1. 实验设计

关键前提：**只有 `src/` 不同**，其余全部相同。

| 项 | 设置 |
|---|---|
| 官方树 | `D:\cpc_c_official`（`Seniorious123/CellPainting-Claw` @`59df377`，42 个 `src/*.py`，单体 `cli.py`+`skills.py`） |
| v2 树 | `D:\cpc_c_v2`（`liuzy0543-spec/cpc-v2` @`612aa79`，85 个 `src/*.py`，`ports/registry/errors/capabilities` + `cli/`+`skills/` 包） |
| 数据 | 同一份 CellProfiler 掩膜与原始图像（从参考树整份复制） |
| 权重 | 同一份 `E:\cpc_weights\deepprofiler\Cell_Painting_CNN_v1.hdf5` |
| 配置 | 同一份 `configs/project_config.run.json`：`box_size=32`、`batch_size=4`、`feature_layer=block6a_activation`、`checkpoint_url=""`（**关闭隐式下载**） |
| 命令 | 完全相同：`python -m cellpainting_claw run-deepprofiler-pipeline --config ... --output-dir demo/workspace/outputs/dp_run` |
| 输出 | 每次都先删除 `dp_run`，两次运行互不污染 |

---

## 2. 结果

### 2.1 嵌入层（语义层，**必须逐位相同**）

| well | 官方 sha256[:16] | v2 sha256[:16] | 数组相等 | max\|diff\| | 文件字节 |
|---|---|---|---|---|---|
| A01 | `63c3c8a95e83fd95` | `63c3c8a95e83fd95` | ✅ True | **0.000e+00** | 6,308 vs 6,308 |
| A02 | `f265191f8360bb07` | `f265191f8360bb07` | ✅ True | **0.000e+00** | 6,318 vs 6,318 |

* 形状均为 `(2, 672)`
* `locations` 数组同样相等
* **连 `.npz` 文件的 sha256 都相同** —— 比"语义一致"更强

### 2.2 元数据层

```
A01: 0 个键不同
A02: 0 个键不同
```

> 这一项**比预期更好**。早前（v10 §1.3）观察到官方树自身重跑会产生 24 个
> `*_Illum*` 元数据键差异，成因是"是否跑过分割的照明校正阶段"。
> 本次两棵树消费的是**同一份上游产物**，所以该变量被消除，元数据也完全一致。

### 2.3 下游表

| 表 | 形状 | 列名一致 | 672 个特征列 max\|diff\| |
|---|---|---|---|
| `deepprofiler_single_cell.parquet` | (4, 718) vs (4, 718) | ✅ | **0.000e+00** |
| `deepprofiler_well_aggregated.parquet` | (2, 677) vs (2, 677) | ✅ | **0.000e+00** |

### 2.4 清单

```
official   feature_count=672  cell_count=4  well_count=2
v2         feature_count=672  cell_count=4  well_count=2
```

---

## 3. 判定

> ### ✅ **语义输出完全一致，且达到逐字节相同**
>
> 官方树与重构树在**同一输入、同一权重、同一配置**下产生
> **完全相同的嵌入、相同的下游表、相同的文件哈希**。

这满足用户的核心要求：**「输入与官方一致 → 中间产物与输出也必须一致」**。

---

## 4. 复现所需的两个环境前置（本轮踩到，已修）

1. **`python_executable` 必须指向一个真实存在、且装有 DeepProfiler 栈的解释器。**
   参考树里写的是 `D:/项目/CellPainting-Claw/.venv/Scripts/python.exe`——
   那棵树在清库时已删除。症状是 `WinError 2`（"系统找不到指定的文件"）。

2. **子进程需要 `torch_gpu\Library\bin` 在 PATH 上。**
   `ep_bench\venv` 由 conda `torch_gpu` 派生，其 numpy / tensorflow 轮子依赖该目录的 DLL。
   直接 `python -m` 调用时不会自动带上，症状是
   `ImportError: DLL load failed while importing _multiarray_umath`。

3. **`deepprofiler` 的 PyPI 包必须完整。** 踩坑记录：
   * `pip install --no-deps <sdist>` 会**只装 2 个文件**（上游 `setup.py` 写的是
     `packages=["deepprofiler"]`，**不含子包** `dataset/imaging/learning`），
     症状是 `ModuleNotFoundError: deepprofiler.dataset`；
   * `v0.3.1` 的 `setup.py` 还把依赖钉在 `tensorflow==2.5.*`（PyPI 已无该版本），
     所以只能 `--no-deps`，而这又回到上一个问题；
   * **可行路径**：`pip install deepprofiler==0.5.1`（PyPI 版，26 个 `.py`，含全部子包）。
   * 另需 `comet_ml`（deepprofiler 运行期依赖）、`pyarrow`（parquet 写出）。

---

## 5. 复现命令

```powershell
# 环境
#   python_executable = D:/ep_bench/venv/Scripts/python.exe
#   该 venv 需具备: numpy==1.26.4 skimage tensorflow==2.15.1 efficientnet==1.1.1
#                   deepprofiler==0.5.1 comet_ml pyarrow torch
$env:PATH = 'C:\Users\DELL\miniconda3\envs\torch_gpu\Library\bin;' + $env:PATH
$env:TF_CPP_MIN_LOG_LEVEL = '3'

foreach ($tree in 'cpc_c_official','cpc_c_v2') {
    $root = "D:\$tree"
    Set-Location $root
    $env:PYTHONPATH = "$root\src"
    & 'D:\ep_bench\venv\Scripts\python.exe' -m cellpainting_claw run-deepprofiler-pipeline `
        --config configs/project_config.run.json `
        --output-dir demo/workspace/outputs/dp_run
}

# 比对
& 'D:\ep_bench\venv\Scripts\python.exe' 'D:\项目\cellpainting claw复现\_probe\compare_c.py'
```

两个脚本：`_probe\run_c_both.py.ps1`（双侧运行）、`_probe\compare_c.py`（分层比对）。
