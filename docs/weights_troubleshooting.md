# 权重缺失 / 路径不一致：怎么定位，改哪里

> 适用：`models.embed()` 报 `WeightsUnavailable`、`MissingDependency`，或 `doctor()` 显示不 ready
> 结论先行：**永远先跑 `models.doctor()` 和 `models.probe(name)`，不要靠猜。**
> 它们会直接告诉你"是权重没找到、还是解释器不对、还是依赖没装"，以及**已经找到的绝对路径**。

---

## 1. 三步定位

### 第 1 步：全局体检

```python
from cellpaint_pipeline import models
for row in models.doctor():
    print(row['model'], row['status'], row['weights_found'], row['weights_detail'])
```

`doctor()` 返回一个 **list**（不是 dict），每项字段：

| 字段 | 含义 |
|---|---|
| `model` | 方法名 |
| `status` | `ready` / `wrong-interpreter` / `needs-weights` / `needs-deps-and-weights` / `needs-upstream-repo` |
| `runnable` | 当前解释器下真能跑吗 |
| `requires` / `manifest_requires` | 声明需要的框架 vs 实际缺的 |
| `missing_dependencies` | 具体缺哪个模块 |
| `weights_found` | **权重文件是否找到**（与依赖无关） |
| `weights_detail` | **解析到的绝对路径**——路径不一致时，看这里和你的实际路径差在哪 |
| `repo_found` / `repo_detail` | 需要上游仓库的方法（若有） |

### 第 2 步：单个方法细看

```python
print(models.probe('ca_mae'))
```

输出示例（权重在、但解释器没有 torch）：

```python
{'model': 'ca_mae', 'runnable': False, 'status': 'wrong-interpreter',
 'requires': ['torch'], 'manifest_requires': ['torch','transformers','einops'],
 'missing_dependencies': ['torch'],
 'weights_found': True,
 'weights_detail': 'E:\\cpc_weights\\openphenom_pkg\\openphenom\\model.safetensors',
 'repo_found': None, 'repo_detail': None, 'license': 'Recursion Non-Commercial EULA'}
```

**关键区分**：`weights_found=True` 而 `status=wrong-interpreter` ⇒ **不是权重问题**，
是解释器没装框架。这种情况下改权重路径怎么改都没用。

### 第 3 步：直接看它试过哪些路径

```python
from cellpaint_pipeline.models.errors import WeightsUnavailable
try:
    models.embed('ca_mae', uint8_bhwc, assume='raw_crop')
except WeightsUnavailable as e:
    print(e.model)     # 'ca_mae'
    print(e.probed)    # 它实际去过的每一个路径（tuple）
    print(e.hint)      # 该怎么修
```

`e.probed` 是**最重要的信息**：路径不一致时，它精确告诉你"我在哪找、没找到"，
你只需要把文件放过去，或者把根目录指过去。

实测（把根目录指向空目录）：

```
probed: E:\nope_not_here\openphenom_pkg\openphenom\model.safetensors
hint  : Place the file under one of ['E:\\nope_not_here'] or set CPC_WEIGHTS_ROOT.
```

---

## 2. 搜索顺序（权威）

`src/cellpaint_pipeline/models/weights.py:102` 的 `resolve_weights()`：

1. **`roots=` 参数**（显式传入永远最高优先）
   → `models.embed(..., weights_roots=('D:/my_weights',))`
2. **环境变量 `CPC_WEIGHTS_ROOT`**（设了就**只**找它，不再看默认列表）
3. **`DEFAULT_WEIGHTS_ROOTS`**（`weights.py:52`，按顺序）
   ```
   E:\cpc_weights
   D:\cpc_weights
   D:\模型权重
   ~/.cellpainting-weights
   ```

**注意第 2 条的语义**：设了 `CPC_WEIGHTS_ROOT` 就是**覆盖**，不是追加。
所以"本机权重散在两个盘"的情况不能用环境变量，要用 `weights_roots=(...)`。

## 3. 目录布局规则

候选名由 `_candidate_names()`（`weights.py:87`）生成，顺序：

1. `root/<manifest.weights.source>` —— `source` 可以带子目录，如 `openphenom_pkg/openphenom/model.safetensors`
2. 若 `source` 里**不含** `/`：再试 `root/<model>/<source>`

本机七个方法期望的位置（实测解析结果）：

| 方法 | `weights.source` | 当前解析到 |
|---|---|---|
| `deepprofiler` | `Cell_Painting_CNN_v1.hdf5` | `E:\cpc_weights\deepprofiler\Cell_Painting_CNN_v1.hdf5` |
| `ca_mae` | `openphenom_pkg/openphenom/model.safetensors` | `E:\cpc_weights\openphenom_pkg\openphenom\model.safetensors` |
| `cellclip` | `cellclip/model.safetensors` | `E:\cpc_weights\cellclip\model.safetensors` |
| `o2vae` | `o2vae/o2vae_mefs_official.pt` | `E:\cpc_weights\o2vae\o2vae_mefs_official.pt` |
| `unidino` | `unidino/surrogate_unidino.pt` | `E:\cpc_weights\unidino\surrogate_unidino.pt` |
| `dinocells` | `dinocells/surrogate_dinocells.pt` | `E:\cpc_weights\dinocells\surrogate_dinocells.pt` |
| `morphogenie` | `morphogenie/surrogate_morphogenie.pt` | `E:\cpc_weights\morphogenie\surrogate_morphogenie.pt` |

**最省事的做法**：照"当前解析到"那一列建目录，把文件按名字放进去即可。
`deepprofiler` 因为 `source` 不带斜杠，放在根下**或** `deepprofiler/` 子目录都能找到。

---

## 4. 改哪里（按推荐顺序）

| 场景 | 改什么 | 位置 |
|---|---|---|
| 只是本机路径不同 | **设环境变量**，不动代码 | `export CPC_WEIGHTS_ROOT=E:/cpc_weights` |
| 多个盘、多个根 | **调用时传 `weights_roots`** | `models.embed(name, ..., weights_roots=('E:/cpc_weights','D:/w2'))` |
| 想永久加一个默认根 | 改默认元组 | `src/cellpaint_pipeline/models/weights.py:52` `DEFAULT_WEIGHTS_ROOTS` |
| 某个方法的文件名/子目录不同 | 改该方法的声明 | `src/cellpaint_pipeline/models/registry.py` 里该条的 `WeightsSpec(source=...)` |
| 权重内容不对（版本不符） | 重新下载**官方**文件 | sha256 校验失败会明说期望值与实际值 |
| 替身权重被换掉 | 更新期望哈希 | `registry.py` 的 `SURROGATE_SHA256` |

**不要**为了跑通去改 `sha256`。校验失败意味着"这个文件不是声明的那份"，
把期望值改成实际值等于把安检关掉——应该换文件。

---

## 5. 五个常见坑

### 坑 1：把"解释器不对"当成"权重缺失"

`status='wrong-interpreter'` 且 `weights_found=True` ⇒ 权重没问题。
本项目有两个解释器：

| 解释器 | 能跑 |
|---|---|
| `D:/ep_bench/venv/Scripts/python.exe`（torch+TF） | 全部七个 |
| `./.venv/Scripts/python.exe`（无 torch/TF） | 只跑测试与纯计算部分 |

在 `.venv` 下 `doctor()` 会显示四个方法 `wrong-interpreter`，这是**预期行为**，不是缺陷。

### 坑 2：设了 `CPC_WEIGHTS_ROOT` 之后默认路径全失效

它是**覆盖**而非追加（`weights.py:119-123`）。设成 `E:/cpc_weights` 之后，
`D:\cpc_weights` 里的东西就再也找不到了。

### 坑 3：路径里的中文

`manifest.ascii_path_required` 置位的方法（CellProfiler 相关路径）在非 ASCII 路径下会失败，
报错是个看不懂的 `OSError`。权重根本身就放在 `E:\cpc_weights`（纯 ASCII）就是为了这个。
`D:\模型权重` 在默认列表里，但相关方法不保证能用。

### 坑 4：只校验文件名，不校验内容

`verify_sha256` 默认**不**在普通加载时开启（避免每次都哈希几 GB）。
所以"文件在、但不是我期望的那份"这种错误，只有显式 `verify=True` 或替身加载路径才会被抓到。
排查"效果突然不对"时，先手动核一次哈希：

```python
from cellpaint_pipeline.models.weights import verify_sha256
from pathlib import Path
print(verify_sha256(Path(r'E:/cpc_weights/o2vae/o2vae_mefs_official.pt'))[:16])
# 应与 registry 里该方法的 weights.sha256 前缀一致：aa70cfa8
```

官方权重前缀：`deepprofiler 87b756f5` / `ca_mae f6e5f1c9` / `cellclip 66c2d32a` / `o2vae aa70cfa8`
替身前缀：`unidino a0265184` / `dinocells 5e172d8f` / `morphogenie 49102997`

### 坑 5：以为它会自己下载

**它不会，永远不会。** `embed()` 绝不隐式下载（`weights.py` 模块 docstring 第 2 段）。
上游 DeepProfiler 会回退到硬编码 Zenodo URL，本机实测 9 KB/s，1.4 GB 要 43 小时——
所以这里选择了"立刻抛错并告诉你放哪"。

---

## 6. 一条命令自检

```bash
PYTHONPATH="D:/项目/CellPainting-Claw-v2/src" CPC_WEIGHTS_ROOT="E:/cpc_weights" \
D:/ep_bench/venv/Scripts/python.exe -c "
from cellpaint_pipeline.models import registry
from cellpaint_pipeline import models
registry.set_profile('v3')
bad = 0
for r in models.doctor():
    ok = r['status'] == 'ready'
    bad += 0 if ok else 1
    print('%-13s %-22s %s' % (r['model'], r['status'], r['weights_detail'] or '-'))
print('不 ready 的方法数:', bad)
"
```

输出 `不 ready 的方法数: 0` 才说明权重与依赖都到位。
若不为 0，按 §1 第 2 步看 `probe()` 的 `missing_dependencies` 与 `weights_detail`，
再按 §4 决定改环境变量还是改声明。