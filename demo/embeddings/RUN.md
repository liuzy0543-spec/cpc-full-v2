# demo · embedding（7 个方法 + benchmark 调度）

一份最小但完整的 embedding 示例：原始输入、每个方法的输出、以及把多个方法放在
同一个调用里比较的 benchmark 调度。

```bash
cd D:\release\cellpainting-claw-v2
python verify_all.py --embeddings --benchmark
```

产物：

```
demo/embeddings/
├── data/sample_inputs.npz      # 投入数据：8 个细胞的四种输入格式
├── artifacts/emb_<method>.npy  # 产出数据：7 个方法各一个矩阵
├── benchmark/                  # benchmark 调度的产物
└── report.json                 # 每个方法的形状/维度/来源/耗时
```

## 投入数据

`sample_inputs.npz` 里是同一批 8 个细胞，按各方法需要的格式各存一份：

| 键 | 形状 | dtype | 给谁 |
|---|---|---|---|
| `ca` | (8, 5, 256, 256) | uint8 | 像素类方法（转 BHWC uint16 后 `assume='raw_crop'`） |
| `dp` | (8, 32, 32, 5) | float32 | `deepprofiler`（它自己的输入规格） |
| `o2` | (8, 1, 64, 64) | float32 | `o2vae`（核剪影） |
| `cellclip` | (8, 5, 1536) | float32 | `cellclip`（每通道一个向量） |
| `cell_ids` | (8,) | str | 行身份，用于 benchmark 对齐 |

## 产出数据（7 个方法）

| 方法 | 输出 | 维度 | 来源 | 耗时 |
|---|---|---|---|---|
| `deepprofiler` | `emb_deepprofiler.npy` (8, 672) | 672 | official | 11.35s |
| `ca_mae` | `emb_ca_mae.npy` (8, 384) | 384 | official | 14.87s |
| `cellclip` | `emb_cellclip.npy` (8, 512) | 512 | official | 4.63s |
| `o2vae` | `emb_o2vae.npy` (8, 512) | 512 | official | 2.94s |
| `unidino` | `emb_unidino.npy` (8, 1920) | 1920 | surrogate | 0.67s |
| `dinocells` | `emb_dinocells.npy` (8, 384) | 384 | surrogate | 0.23s |
| `morphogenie` | `emb_morphogenie.npy` (8, 10) | 10 | surrogate | 0.04s |

三个替身方法（`unidino` / `dinocells` / `morphogenie`）的权重是本地自训，
`report.json` 里 `weights_provenance` 会标成 `surrogate`。

## benchmark 调度

`demo/embeddings/benchmark/benchmark_report.json` 是一次**单次调用覆盖全部方法**的结果：

- 每个方法都记录了输出形状与权重 sha256；
- 所有方法按 `row_ids` 对齐后才比较（对齐失败会先报错，不会静默错位）；
- 结果里带 `agreement`/RV 系数，用来判断两个方法是否在说同一件事。

跑法（`verify_all.py --benchmark` 做的就是这件事）：

```python
from cellpaint_pipeline import embedding_benchmark as bench, models
plan = bench.build_benchmark_plan(
    models.declared(), output_dir='demo/embeddings/benchmark',
    unit={'cellclip': 'channel_embeddings'},
    assume={'ca_mae': 'raw_crop', 'unidino': 'raw_crop', 'dinocells': 'raw_crop'},
    channels={'morphogenie': ('C0', 'C1', 'C2')},
)
result = bench.run_benchmark(plan, inputs, sample_metadata={'row_ids': ids})
```

`unit` / `assume` / `channels` 都可以"一个值给全部"或"按方法名给映射"——
这是七个方法输入契约不同（像素/向量/剪影/已居中）时唯一的出路。

## 这一轮的数字

| 指标 | 值 |
|---|---|
| 方法数 | 7 |
| 输出维度 | deepprofiler=672、ca_mae=384、cellclip=512、o2vae=512、unidino=1920、dinocells=384、morphogenie=10 |
| benchmark 错误数 | 0 |
| 单方法耗时 | 见 `report.json`（CPU） |
