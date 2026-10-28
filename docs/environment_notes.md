# 环境配置记录（v2 树）

> 记录本机两套解释器的分工、各自踩过的坑，以及**为什么必须有两套**。
> 日期：2026-10-09

---

## 1. 两套解释器，各司其职

| 用途 | 解释器 | 关键依赖 | 状态 |
|---|---|---|---|
| DeepProfiler（Keras 图） | conda env `cellpainting-claw` | tensorflow 2.10.1 + efficientnet 1.1.1 + deepprofiler | ✅ |
| 绘图 / 聚合 / 其余 | `D:\项目\CellPainting-Claw-v2\.venv` | numpy 2.5.3, pandas 2.3.3, scikit-image 0.26, matplotlib 3.10.9 | ✅ |
| CA-MAE / CellCLIP / O2-VAE | `D:\ep_bench\venv` | torch 2.5.1+cu124, timm, transformers 4.46.1 | ✅ |

**为什么不能合成一套**：见 §3。这是 Windows + conda 的 DLL 搜索路径问题，不是配置疏漏。

---

## 2. 官方 environment.yml 在 Windows 上不可满足

用 anaconda.org API 逐包核验：**所有 CellProfiler 相关 pin 都没有 win-64 构建**。

| 包 | subdir 分布 | win-64 |
|---|---|---|
| `bioconda/cellprofiler` | linux-64:4, noarch:3, osx-64:3 | **0** |
| `bioconda/cellprofiler-core` | noarch:12 | **0** |
| `bioconda/python-bioformats` | noarch:6 | **0** |
| `conda-forge/python-javabridge` | linux-64:2, osx-64:2 | **0** |
| `conda-forge/cellprofiler` | 未在该频道发布 | — |

**架构上的正解**：CellProfiler for Windows 以独立冻结应用发布
（本机 `C:\CellProfiler\CellProfiler.exe`，PyInstaller + 内置 Python 3.8），
而本项目**本来就用子进程调它、从不 import**。于是产出
`environment/cellpainting-claw.win64.environment.yml`：移除那 5 个包，其余 pin 逐字保留。

### 2.1 另外两个失效的 pin

| pin | 问题 | 处置 |
|---|---|---|
| `deepprofiler==0.3.1` | **已从 PyPI 下架**（只剩 0.5.1） | 改从官方 git tag `v0.3.1` 装（tarball，经 gh-proxy） |
| `scikit-image==0.20.0` | 在本机**原生崩溃**（exit `0xC0400003`，无 traceback） | 提到 0.22.0（conda 侧）/ 0.26.0（venv 侧） |

---

## 3. ★ 根因：conda 环境的 `DLLs` / `Library\bin` 会毒化原生扩展加载

### 3.1 现象

在 `cellpainting-claw` conda 环境中，**任何原生渲染都会硬崩**（exit `-1066598273` = `0xC0400003`）：

```
matplotlib  → plt.subplots() 后 savefig/draw 一律崩（png/pdf/svg 都崩，无文字也崩）
scikit-image → import skimage 直接崩，无任何输出
```

### 3.2 排查（逐条证伪）

| 假设 | 结论 |
|---|---|
| freetype 版本太旧 | ❌ `ft2font.pyd` **根本不 import freetype**（PE 导入表只有 CRT + python310 + KERNEL32）——静态链接 |
| `ft2font.pyd` 损坏 | ❌ 与可用环境的副本**逐字节相同**（sha256 一致，979,456 B） |
| 字体缓存损坏 | ❌ 清空 `MPLCONFIGDIR` 仍崩 |
| 我加的 `sitecustomize.py` 导致 | ❌ 重命名后仍崩；`python -S` 也崩 |
| matplotlib 版本问题 | ❌ 换成 PyPI 3.10.9 仍崩 |
| matplotlib 自身的 pyd 依赖 conda DLL | ❌ `_path`/`_image`/`_qhull`/`_c_internal_utils` 的导入表**同样只有 CRT + python310 + 系统 DLL** |

### 3.3 定位

**同一个 `python.exe`**（`envs\cellpainting-claw\python.exe`），
**相同的 site-packages 内容**，只差一个 `pyvenv.cfg`：

```
conda env 直跑            → 崩   (prefix == base_prefix == envs\cellpainting-claw)
由该 python 建的 venv 跑  → 正常 (prefix == venv，base_prefix == envs\cellpainting-claw)
```

两者 `sys.path` 前四项完全相同、`os.add_dll_directory` 都为空、`PATH` 都不含任何
`Library\bin`。唯一差别是 **prefix 身份**：以 conda env 为 `sys.prefix` 时，
其 `DLLs\` 与 `Library\bin\` 参与原生模块加载，其中的 DLL 与 matplotlib /
scikit-image 的扩展**二进制不兼容**；以 venv 为 `sys.prefix` 时则绕过。

**这也是 `scikit-image 0.20.0` 崩、升到 0.22.0 就不崩的原因**——不是版本本身，
而是哪个二进制被优先加载。

### 3.4 处置

绘图 / 聚合类任务改用**由 base conda 建的 venv**
（`D:\项目\CellPainting-Claw-v2\.venv`），原生渲染正常。

```powershell
C:\Users\DELL\miniconda3\python.exe -m venv D:\项目\CellPainting-Claw-v2\.venv
D:\项目\CellPainting-Claw-v2\.venv\Scripts\python.exe -m pip install -i https://pypi.tuna.tsinghua.edu.cn/simple `
    matplotlib scikit-image pandas scikit-learn tifffile pyyaml pillow pyarrow pytest boto3 "fire==0.7.1"
D:\项目\CellPainting-Claw-v2\.venv\Scripts\python.exe -m pip install -e . --no-deps
```

**代价（必须知道）**：base conda 的 Python 上拿不到 `tensorflow==2.15.1`
（PyPI 只提供 2.20+），所以这个 venv **没有 TF**，DeepProfiler 仍走 conda 环境。
`tf` 与现代 CuDNN 的兼容面也更窄，不宜同时压两套。

---

## 4. venv 的依赖漂移（相对官方 pin）

| 包 | 官方 pin | venv 实际 | 说明 |
|---|---|---|---|
| numpy | 1.26.4 | **2.5.3** | base conda Python 上无 1.26.4 轮子；`pycytominer` 也强制 2.x |
| scipy | 1.11.4 | **1.18.1** | 同上 |
| scikit-image | 0.20.0 | **0.26.0** | 0.20.0 在该环境原生崩溃 |
| pandas | 2.3.3 | 2.3.3 | ✅ 一致 |
| scikit-learn | 1.7.2 | 1.7.2 | ✅ 一致 |

> ⚠️ **`pycytominer==1.4.0` 的 sdist 装不上**：其 `pyproject.toml` 含非 UTF-8 字节，
> 元数据生成阶段 `UnicodeDecodeError`。必须用 `--only-binary :all:` 走轮子。

---

## 5. 测试基线（三个环境对照）

```
official 树 (干净)          : 13 failed, 118 passed
v2 树 + conda env           : 13 failed, 126 passed   （绘图测试原生崩溃，未计入）
v2 树 + 新 venv             : 14 failed, 126 passed   ← 绘图测试已修好
```

**14 个失败全部与本次改动无关**，分三类：

| 类别 | 数量 | 例 |
|---|---|---|
| **既有代码缺陷** | 1 | `public_api.py:470` `NameError: ProjectConfig` —— 名字只在 `TYPE_CHECKING` 下导入却在运行时使用。**不 import `models` 也复现**，是 v2 自带缺陷 |
| Windows 路径分隔符 | 2 | `assertEqual(payload['output_path'], '/tmp/listing.json')` → Windows 归一化成 `'\\tmp\\listing.json'` |
| DeepProfiler 冒烟 / 交付冒烟 | ~11 | 需要 demo 数据与 checkpoint 的端到端步骤 |

**修好的那一项**：`tests/test_profile_summaries_smoke.py` 之前**原生崩溃**（整个 pytest 进程死掉、
后续测试全不跑），换 venv 后正常通过。

**顺带修掉的自身缺陷**：`models/adapters/{deepprofiler,o2vae,unavailable}.py` 曾被
PowerShell 批量改写引入 **UTF-8 BOM**，导致 `tests/test_layering.py::test_no_upward_imports`
（用 `ast.parse` 扫源文件）`SyntaxError: invalid non-printable character U+FEFF`。已全部剥离。
