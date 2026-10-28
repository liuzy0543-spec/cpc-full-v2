# 内聚耦合审查路线图

本路线图用于**独立复核**重构版相对官方源码在**内聚与耦合**上的变化。
所有数字都可复现，不含主观判断。

---

## 一、三个仓库地址

| 角色 | GitHub | 本次审查的提交 |
|---|---|---|
| **官方原版** | **https://github.com/Seniorious123/CellPainting-Claw** | **`59df377`** |
| **我们的 v1** | **https://github.com/liuzy0543-spec/cpc-v2** | **`07408de`** |
| **我们的 v2** | **https://github.com/liuzy0543-spec/cpc-promote-v2** | **`828d544`** |

> **v1 与 v2 的 `src/` 逐字节相同**（各 72 个 `.py`，0 差异）。
> 差异只在提交历史：v2 多 10 个提交（提示词演进记录）。
> **所以内聚耦合的结论对两者同时成立，测任意一棵即可。**

### 1.1 取代码

```bash
git clone https://github.com/Seniorious123/CellPainting-Claw.git official
cd official && git checkout 59df377 && cd ..

git clone https://github.com/liuzy0543-spec/cpc-v2.git ours-v1
cd ours-v1 && git checkout 07408de && cd ..

git clone https://github.com/liuzy0543-spec/cpc-promote-v2.git ours-v2
cd ours-v2 && git checkout 828d544 && cd ..
```

> ⚠️⚠️ **必须全新克隆到干净目录，不要复用本机已有的仓库。**
>
> 2026-10-07 复核时实测发现：本机已有 4 份官方克隆，**每一份都带未提交改动**，
> 其中 `D:\项目\cellpainting claw remake v1\github-src` 里实际装的是**重构版代码**
> （19 项变更、`skills.py` 已删、`cli/` 已拆包、HEAD 虽显示 `59df377`）。
> **用它跑官方基线会得到完全错误的数字，且不会有任何报错。**
>
> 克隆后必须逐个确认：
>
> ```bash
> for d in official ours-v1 ours-v2; do
>   echo "$d HEAD=$(git -C $d rev-parse HEAD) dirty=$(git -C $d status --porcelain | wc -l)"
> done
> ```
>
> 三个 `dirty` 都必须是 **0**，`HEAD` 分别是 §一 表里的三个 sha。
>
> 另注：`git clone` 偶发因网络抖动中途失败（表现为目录存在但 `git rev-parse` 报
> `Needed a single revision`），此时删掉重来即可，不要在半成品目录里继续操作。

**审查 v1 与 v2 的 `src/` 是否真的一致**（这是结论能否合并的前提）：

```bash
diff -r ours-v1/src ours-v2/src && echo 'IDENTICAL'
# 预期：无任何输出。实测提交数 155 vs 165（v2 多 10 个），py 文件各 72 个。
```

---

## 二、内聚指标：怎么测

### 2.1 度量定义

| 指标 | 定义 |
|---|---|
| **模块行数** | 每个 `.py` 的行数（含空行与注释） |
| **函数行数** | `ast.FunctionDef` / `AsyncFunctionDef` 的 `end_lineno - lineno + 1` |
| **<100 行模块数** | 行数小于 100 的模块个数（越多越碎、越易读） |

### 2.2 复现命令

```bash
# 依赖：Python 3.11+（标准库 ast 即可，无需第三方）
python docs/measure.py            # 仓库自带，v6.1 起就在
#   files / lines / longest_function / largest_file 与 §2.3 一致，可直接用。
#   ⚠️ 但 avg_function_lines 与 §2.3 的 20.7 不是一个口径，别拿它对表：
#     measure.py 报 32.9，用的是「模块总行数 ÷ 函数数」（472 个函数，含嵌套/闭包）；
#     §2.3 的 20.7 用的是 §2.2 手工版的「函数体 end_lineno-lineno+1 的均值」。
#     两者差12 点纯粹因为 import、类体、注释、空行进了后者的分子。
#   2026-10-07 修正：measure.py 内部曾把 avg_function_lines 实现成了后者那个公式，
#     属命名与实现不符；现已拆为 avg_function_lines 与 lines_per_function 两个字段。
```

> ⚠️ **`src_root` 必须传 `src/cellpaint_pipeline`，不是 `src`。**
> 传 `src` 会把 `src/cellpainting_claw`、`src/cellpainting_skills` 也算进去，
> 得到 **42 / 72** 个模块，与 §2.3 的 35 / 65 对不上。
>
> | 传入 | 官方模块 | 我们模块 | 官方 <100 行 | 我们 <100 行 |
> |---|---|---|---|---|
> | `src` | 42 | 72 | 11 | 20 |
> | **`src/cellpaint_pipeline`** | **35** | **65** | **5** | **14** |
>
> **§2.3 与 §3.3 的全部数字只在 `src/cellpaint_pipeline` 口径下成立。**

**手工版**（不依赖仓库脚本，便于交叉验证）：

```python
import ast, statistics
from pathlib import Path

def cohesion(src_root):   # 调用：cohesion("src/cellpaint_pipeline")
    sizes, fns = {}, []
    for p in Path(src_root).rglob('*.py'):
        if '__pycache__' in str(p):
            continue
        text = p.read_text(encoding='utf-8', errors='replace')
        sizes[p.relative_to(src_root).as_posix()] = len(text.splitlines())
        for n in ast.walk(ast.parse(text)):
            if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef)) and n.end_lineno:
                fns.append(n.end_lineno - n.lineno + 1)
    return {
        'modules': len(sizes),
        'max_module_lines': max(sizes.values()),
        'modules_under_100': sum(1 for v in sizes.values() if v < 100),
        'max_function_lines': max(fns),
        'avg_function_lines': round(statistics.mean(fns), 1),
    }
```

### 2.3 期望结果

| 指标 | 官方 | 我们的 | 方向 |
|---|---|---|---|
| 模块数 | 35 | **65** | — |
| **最大模块行数** | **1646** | **911** | ✅ −45% |
| **最长函数行数** | **855** | **201** | ✅ −76% |
| **平均函数行数** | **26.7** | **20.7** | ✅ −22% |
| **<100 行模块数** | **5** | **14** | ✅ +180% |

> ⚠️ **口径提醒**：911 是 `segmentation_native.py`（官方同名文件 **899**，**实际 +12 行**）。
> 1646 的 `skills.py` 是**被删除拆包**，不是「缩到 911」。**这两个数字是「最大值」的前后对比，不是同一文件的前后对比。**
>
> **2026-10-07 更正**：此处原写「+8 行」，算术有误，899 → 911 是 **+12**。
> 最长函数同理：855 的 `main` 被消除，新的最长 201（`run_end_to_end_pipeline`）**在官方就是 201 行且未被修改**。

**「201 行未被修改」已被函数级sha256 实证**（复核必做的一步）：

```python
# 两棵树各取 orchestration.py 里 run_end_to_end_pipeline 的函数体，算 sha256
# 实测结果：官方 = 我们 = 1357eec403bb3b88d6f6999c0156df5c6b8248137c6699afa66547f6df049be9
```

即该函数**逐字节一致**。该文件的唯一差异是 `ProjectConfig` 的导入位置
（官方在运行时导入，我们的移入 `TYPE_CHECKING`），函数体本身未动。

```bash
#顺带核对被拆包/被消除的文件是否真如描述
wc -l official/src/cellpaint_pipeline/segmentation_native.py   # 899（我们 911，+12）
ls official/src/cellpaint_pipeline/skills.py                   # 应存在，1646 行（我们已拆为 skills/ 包 9 个模块）
```

---

## 三、耦合指标：怎么测

### 3.1 度量定义

| 指标 | 定义 |
|---|---|
| **扇出 Ce** | 本模块 import 了几个内部模块 |
| **扇入 Ca** | 有几个内部模块 import 了本模块 |
| **平均/最大扇出** | 所有模块的 Ce 的均值 / 最大值 |
| **最大扇入** | 所有模块的 Ca 的最大值 |
| **循环依赖** | 内部导入图的强连通分量（Tarjan SCC），大小 > 1 即环 |

### 3.2 ⚠️ 关键口径：必须用**运行时**导入图

重构版把 **17 个模块**的类型标注导入移到了 `if TYPE_CHECKING:` 下 —— **这些不产生运行时依赖**。
如果用静态 AST 统计（把 `TYPE_CHECKING` 里的也算上），会得出**错误结论**。

**正确的做法**：统计时跳过 `if TYPE_CHECKING:` 块内的行号范围。

```python
import ast
from collections import Counter, defaultdict
from pathlib import Path

def coupling(src_root):
    src = Path(src_root)
    edges = defaultdict(set)
    for p in src.rglob('*.py'):
        if '__pycache__' in str(p):
            continue
        rel = p.relative_to(src).with_suffix('').as_posix()
        name = (rel[:-9] if rel.endswith('/__init__') else rel).replace('/', '.').rstrip('.')
        tree = ast.parse(p.read_text(encoding='utf-8', errors='replace'))
        # 收集 TYPE_CHECKING 块覆盖的行号 —— 这些不算运行时依赖
        guarded = set()
        for n in ast.walk(tree):
            if isinstance(n, ast.If) and isinstance(n.test, ast.Name) and n.test.id == 'TYPE_CHECKING':
                for b in n.body:
                    for ln in range(b.lineno, (b.end_lineno or b.lineno) + 1):
                        guarded.add(ln)
        for n in ast.walk(tree):
            tgt = None
            if isinstance(n, ast.ImportFrom) and n.module and n.module.startswith('cellpaint_pipeline'):
                tgt = n.module
            elif isinstance(n, ast.Import):
                for a in n.names:
                    if a.name.startswith('cellpaint_pipeline'):
                        tgt = a.name
            if not tgt or n.lineno in guarded:
                continue
            short = tgt.replace('cellpaint_pipeline.', '').replace('cellpaint_pipeline', '(root)')
            if short != name:
                edges[name].add(short)
    fan_out = {m: len(d) for m, d in edges.items()}
    fan_in = Counter()
    for m, deps in edges.items():
        for d in deps:
            fan_in[d] += 1
    return {
        'modules': len(fan_out),
        'edges': sum(fan_out.values()),
        'avg_fan_out': round(sum(fan_out.values()) / max(len(fan_out), 1), 2),
        'max_fan_out': max(fan_out.values()),
        'max_fan_in': max(fan_in.values()) if fan_in else 0,
    }
```

> ⚠️ **上面这段代码有一个盲点**：它把导入目标截成短名（`short`），
> 于是「包 `__init__` ↔ 包内模块」这类**间接环会被 `short != name` 掩盖**。
> 本次复核用**完整点号路径**重跑过，结论一致（0 环），但严格复核时建议用完整路径。
> 项目里的 `docs/layers.py` 就是按完整路径判层的，可以互补。

**循环依赖**用 Tarjan SCC；项目里还有一个现成的分层检查：

```bash
python docs/layers.py      # 只允许向下层 import，向上即失败
python -m pytest tests/test_layering.py -q
```

### 3.3 期望结果

| 指标 | 官方 | 我们的（运行时） | 判定 |
|---|---|---|---|
| 模块数 | 35 | 65 | — |
| 导入边数 | 88 | 134 | — |
| **平均扇出** | **2.51** | **2.06** | ✅ 更低 |
| **最大扇出** | **12** | **11** | ✅ 更低 |
| **最大扇入** | **25** | **21** | ✅ 更低 |
| **循环依赖** | **0** | **0** | ✅ 持平 |

> ⚠️ **平均扇出有两个口径，必须说清分母**：
>
> | 分母 | 官方 | 我们 |
> |---|---|---|
> | **全部模块数**（§3.3 上表用这个） | 88 ÷ 35 = **2.51** | 134 ÷ 65 = **2.06** |
> | **有出边的模块数**（§3.2 代码用这个） | 88 ÷ 27 = **3.26** | 134 ÷ 44 = **3.05** |
>
> **两种口径下我们都优于官方**，但数字不同。
> §3.2 的代码把 `len(fan_out)`（有出边的模块）当分母，跑出来是 **3.26 / 3.05**；
> 上表用的是全部模块数。**复核时请对号入座，不要混用。**

> **边数 88 → 134 不是变差。** 模块数从 35 涨到 65（+86%），边的增速（+52%）低于模块增速，
> 所以**每模块的平均依赖反而更少**。**判断耦合要看均值和最大值，不要只看边数总数。**

### 3.4 一个容易被误读的点：`config` 的高扇入

```
config    Ca=21   Ce=0
```

`Ce=0` 说明它是**纯叶子**（只被依赖、不依赖别人）—— 这正是**稳定依赖原则**要的形状。

> ⚠️ **两个口径必须一起报，只报运行时会被反驳**：
>
> | `config` 扇入 | 官方 | 我们 |
> |---|---|---|
> | **运行时**（跳过 `TYPE_CHECKING`） | 25 | **21** |
> | **静态声明**（含 `TYPE_CHECKING`） | 25 | **37** |
> | 静态总边数 | 88 | **153** |
>
> 官方源码里 `TYPE_CHECKING` 模块数 = **0**，所以它两个口径相同（88 / 25）。
> 我们有 **23 个** `TYPE_CHECKING` 模块，其中 18 个守的是 `config`。
>
> **所以准确的说法是**：「**运行时依赖图更松**（25→21），但**声明层面的依赖反而增加**（25→37）」。
> 增加的原因是 `skills.py` 拆成 9 个模块时新增了引用 `ProjectConfig` 的文件。
> （复核者独立测得静态扇入 **39**，与这里的 37 差 2，属统计口径差异，方向一致。）
高扇入本身不是缺陷；**它的代价只在于「改它的公开方法会波及 21 个地方」**。

**本次已做的改善**：官方 25 → 我们 21。做法是把 **17 个**只在类型标注里用 `ProjectConfig`、
从不调用其方法**的模块，改为在 `TYPE_CHECKING` 下导入。

---

## 四、确认「最新代码是否已在 GitHub 上」

### 4.1 一条命令

```bash
git ls-remote https://github.com/liuzy0543-spec/cpc-v2 refs/heads/main
git ls-remote https://github.com/liuzy0543-spec/cpc-promote-v2 refs/heads/main
```

### 4.2 期望值（写作本次审查基准）

| 仓库 | 远端 `main` 应为 | 该提交信息 |
|---|---|---|
| `cpc-v2` | **`1c0dde9567f017d79fbc9707d21d0c39b1cc55f4`** | `Fix a mislabelled metric and record the coupling work` |
| `cpc-promote-v2` | **`92bc4b95b9bf7469ac94fc0852708d4be3c6d7ec`** | 同上 |

> **2026-10-07 更新**：此处原写 `07408de` / `828d544`。两仓库各自又多了一个提交
> （只动 `CHANGELOG.md` 与 `docs/measure.py`，**`src/` 未变**）。
>
> **因此 §5 第 8 条的判据不要写成「远端 HEAD 等于某个固定 sha」** —— 那会在每次追加文档提交后误判失败。
> 正确的判据是下面这条 **tree 哈希**，它只随 `src/` 等实际内容变化：
>
> ```bash
> git rev-parse HEAD^{tree}
> # 期望（2026-10-07 基准）: 36fc4cde1ace9a0db9b1a054a645161da830f1a7
> ```
>
> **两个子树哈希（2026-10-07 基准，三个提交点上完全不变）**：
>
> | 路径 | 哈希 |
> |---|---|
> | `src` | `9d4ac260dba2c14f79306a086bdc1b0d119a2a47` |
> | `src/cellpaint_pipeline` | `857f36d6ae079eb3dcd9ee65ab639c7dad809e45` |
>
> ⚠️ 本行原写「`src/cellpaint_pipeline` 保持 `9d4ac260…`」，**路径张冠李戴** ——
> `9d4ac260` 是 `src` 那一层的哈希。两个都列在这里。

### 4.3 网络不通时的替代核验（本次实测有效）

```powershell
# 走 GitHub API，比 git ls-remote 更耐网络抖动
Invoke-RestMethod -Uri 'https://api.github.com/repos/liuzy0543-spec/cpc-v2/commits/main' `
  -Headers @{'User-Agent'='audit'} | Select-Object -ExpandProperty sha

# 顺带确认新增文件确实推上去了
$t = Invoke-RestMethod -Uri 'https://api.github.com/repos/liuzy0543-spec/cpc-v2/git/trees/main?recursive=1' `
  -Headers @{'User-Agent'='audit'}
$t.tree.path | Where-Object { $_ -in 'docs/layers.py','tests/test_layering.py' }
```

**本次实测结果**：

```
cpc-v2           远端 HEAD = 07408debc846404410a467262cde4dfed59f28cd   ✅ 与本地一致
cpc-promote-v2   远端 HEAD = 828d5449ed8a18696392a1257ec3c4cf0045251e   ✅ 与本地一致
两仓库均含 docs/layers.py、tests/test_layering.py，git 跟踪文件数 242
```

> ⚠️ **`git ls-remote` 在本机常因系统代理设置失败**
> （`ProxyEnable=1` 指向 `127.0.0.1:7890`，而该端口已关闭）。
> 绕过方式：设 `NO_PROXY=*`，或直接用上面的 API 方式。

---

## 五、审查清单（按顺序做）

| # | 步骤 | 通过标准 |
|---|---|---|
| 1 | clone 三个仓库到**干净目录**，`git checkout` 到对应提交 | 官方在 `59df377`，v1 在 `07408de`，v2 在 `828d544`，且三者 `git status --porcelain` 均为空（见 §1.1 警告） |
| 2 | `diff -r ours-v1/src ours-v2/src` | 无输出（完全相同） |
| 3 | 跑内聚度量（§2.2，**`src_root` 传 `src/cellpaint_pipeline`**） | 最大模块 ≤911、最长函数 ≤201、<100 行模块 ≥14 |
| 4 | 跑耦合度量（§3.2，**必须跳过 TYPE_CHECKING**） | 平均扇出 ≤2.06 量级（全部模块口径）、最大扇出 ≤11、最大扇入 ≤21 |
| 5 | 跑 `python docs/layers.py` | 输出 `layering OK - no upward imports`（v1、v2 都应是这句） |
| 6 | 跑 `pytest tests/test_layering.py` | `1 passed` |
| 7 | 循环依赖（Tarjan SCC） | 两棵树都是 **0** |
| 8 | 核验 GitHub远端（§4） | 远端 HEAD 等于上表两个 sha |
| 9 | 抽查「未修改」声明（§2.3） | `run_end_to_end_pipeline` 函数体 sha256 两树相同 |
| 10 | 查证环的修复来源（§六） | 能找到 `0d475d9 "Break the cli <-> cli.app import cycle"` |

> 2026-10-07 实测：清单 1–8 **全部通过**，另补做9–10 也通过。
> 脚本与逐项结果见 `D:\项目\cpc-audit-2026\`（`复核报告-实测结果.md`）。

---

## 六、口径与已知边界（**复核时请一并核对**）

| 项 | 说明 |
|---|---|
| **必须用运行时图** | 静态 AST 会把 `TYPE_CHECKING` 的导入算上，得出与 §3.3 不符的结论 |
| **最大值 ≠ 同一文件** | 1646→911 是两个不同文件；855→201 是两个不同函数（详见 §2.3 提醒） |
| **边数不是耦合** | 88→134 是模块数 +86% 的结果；看均值与最大值 |
| **高扇入不等于坏** | `config` 的 `Ce=0` 是稳定叶子，符合稳定依赖原则 |
| **本次只动导入，未动逻辑** | **17 个模块**的 `ProjectConfig` 移到 `TYPE_CHECKING`（边界口径见下） |
| **分层规则是新增的** | 官方无此约束；`cli ↔ cli.app` 那个环正是拆分时无约束引入的（修复提交见下） |
| **17 vs 18 的口径边界** | 官方 `skills.py` 拆成 9 个模块后，`skills.dispatch` 是否计入"迁移"取决于判定方式。计入则 18，不计则 17。**属口径边界，非错误** |
| **不存在"5 个被还原"** | 官方有 **25 个**模块运行时导入 `ProjectConfig`。除 17 个被迁移外，其余 8 个（`delivery`/`evaluation`/`mcp_server`/`mcp_tools`/`orchestration`/`presets`/`workflows.orchestration` 等）**在我们的树里仍是运行时导入，从未被改动** |

### 6.1 `cli ↔ cli.app` 环的修复有据可查

这个环不是口头描述，v1 仓库里有专门的修复提交：

```bash
git show --stat 0d475d9      # v1 仓库内
# commit 0d475d90f64feff68588806444fcec8dc8791aab
#    Break the cli <-> cli.app import cycle
#    cli/__init__.py | 13 +++++++++++++-

git show 0d475d9 -- src/cellpaint_pipeline/cli/__init__.py
# - from cellpaint_pipeline.cli.app import COMMAND_HANDLERS, build_parser, main
# + # 改为通过 PEP 562 __getattr__ 惰性解析这三个名字
```

**成因**（与 §六 描述一致）：`cli/__init__.py` 直接导入 `cli.app`，
而 `cli.app` 又`from cellpaint_pipeline.cli import commands`，两者互导。
官方是单文件 `cli.py`，**没有这个环**。

**验证当前无环**：用完整点号路径跑 Tarjan SCC（绕开 §3.2 短名截断的盲点），
对 `4b8a3ec`（拆分后）→ `0d475d9`（修环）→ `07408de`（当前）三份快照，环数均为 0。
即**最终提交里环已彻底消除**，`07408de` 与官方一样是 0 环。

---

## 七、配套的功能与性能证据（同一提交）

| 项 | 结果 |
|---|---|
| 19 技能 × 3 棵树 | **全部 19/19 通过** |
| 产物比对 | **402 逐字节一致 / 6 差异 / 0 仅部分树有** |
| **REAL-DIFFERENCE** | **0** |
| 6 处差异 | 2 处缓存根修复 + 3 处清单增量披露 + 1 处 `Run_Timestamp`（时间戳，非内容） |
| 冷启动导入 | 官方 **67.2 ms** → 我们的 **35.7 ms**（原始记录，单次） |
| 冷启动导入（2026-10-07 独立复测，3 次取均值） | 官方 **62.2 ms** → 我们的 **40.3 ms**（**−35%**） |
| 重计算步骤 | `cp-extract-measurements` **196.3 / 196.8 / 198.5 s**（三树差异 <3%） |
| 测试集 | 192 张 TIFF = 12 孔 × 2 位点 × 8 通道 |

> 冷启动的复测口径：`PYTHONPATH=<树>/src python -c "import time;t=time.perf_counter();import cellpaint_pipeline.cli;print((time.perf_counter()-t)*1000)"`，
> 取 3 次均值。注意测 `cellpaint_pipeline.cli`（真实冷启动路径），不是顶层 `cellpaint_pipeline`
> ——后者只有约 10ms，两树持平，看不出差别。
