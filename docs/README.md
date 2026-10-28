# docs/ 怎么读

这个目录同时承载两样东西，先分清楚，否则会找不到入口。

## 1. Sphinx 站点（项目本体文档）

`conf.py`、`requirements.txt`、`_static/`、`api/`、`cli/`、`introduction/`、
`installation/`、`quick_start/`、`skills/`、`openclaw/` 属于站点构建，入口是
[`index.md`](index.md)。不要往这些子目录里塞散装报告。

## 2. embedding 统一框架（当前工作）

按用途分四类，**只有第一类是"现在有效"的**：

| 文件 | 作用 |
|---|---|
| [`embedding_interface.md`](embedding_interface.md) | **接口契约**：七个方法的输入单位、形状、归一化、报错码 |
| [`verify_prompt_e2e_2026-10-27.md`](verify_prompt_e2e_2026-10-27.md) | **验证提示词**：V1–V15 断言、实测基线、可复现判据 |
| [`review_prompt_blackbox_whitbox_2026-10-27.md`](review_prompt_blackbox_whitbox_2026-10-27.md) | **审查提示词（当前版）**：黑盒功能 + 白盒逻辑 + 软件设计规范三类检查，含分级、豁免与输出格式 |
| [`weights_troubleshooting.md`](weights_troubleshooting.md) | **权重定位手册**：`doctor()`/`probe()`/`WeightsUnavailable` 三步定位，搜索顺序、目录布局、改哪里 |
| [`review_prompt_consolidated_2026-10-26.md`](review_prompt_consolidated_2026-10-26.md) | 审查提示词（上一版，保留作对照） |
| [`audit/README.md`](audit/README.md) | **审核循环**：六道闸门、轮次记录、每轮 findings |
| [`benchmark_bbbc022_result.md`](benchmark_bbbc022_result.md) | 基准结果 |
| [`PROJECT_STATUS.md`](PROJECT_STATUS.md) | 项目状态与历史 |
| [`real_dataset_setup.md`](real_dataset_setup.md) | 数据集构建方法与已知限制 |
| [`environment_notes.md`](environment_notes.md) | 解释器、权重路径、GPU 注意项 |

## 3. 归档（`archive/`）

历史报告、被取代的提示词、早期审计与 roadmap 都在
[`archive/`](archive/) 里。它们保留是为了追溯"某条结论当时是怎么来的"，
**不作为当前行为的依据**。读到 archive 里的数字时，先确认它是否已被
[`PROJECT_STATUS.md`](PROJECT_STATUS.md) 或 [`audit/README.md`](audit/README.md)
覆盖。

## 维护约定

1. 写新报告先问："它取代了哪一份？" 取代就把旧的移进 `archive/`，不要两份并存。
2. 文档里出现的每个数字都要能追到一次运行；追不到就标"未独立验证"。
3. 结论涉及官方实现时，必须写清对拍方式（逐位 / 等价推导 / 仅代码路径一致）。
4. v3 的三个替身方法**永远**标注 `surrogate`，不得与官方权重的方法混在同一张结论表里。
5. 移动文件后跑一遍链接检查（`docs/README.md` 的校验脚本在 `scripts/embedding_benchmark/check_doc_links.py`）。