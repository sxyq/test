# 保留版本清单

本目录只保留当前记录中每条仍有归因或线上价值的技术分支最高分/最高优先级版本。

| 分支 | 保留版本 | 状态 | 依据 |
|---|---|---|---|
| CPU metadata / output boundary | V30-CPU-METADATA-sxyq | 已线上回收，归因保留 | 62.19199 / 72.19888s |
| result collection | V32-CHUNKED-COLLECT-sxyq | 已构建待 A/B，单变量归因 | docs 记录为该分支候选 |
| clean output stack | V38-CLEAN-OUTPUT-STACK-sxyq | 已线上回收 | 62.47426 / 70.98915s |
| silent runner | V42-SILENT-RUNNER-sxyq | 当前线上最佳 | 62.7338 / 69.87686s |
| selected output head | V82-SELECTED-HEAD-SILENT-sxyq | 被 V84/V99 覆盖，源码归因保留 | 本地校验通过 |
| zero pred skip | V84-ZERO-PRED-SKIP-sxyq | 被 V99 覆盖，源码归因保留 | 本地校验通过 |
| pred-route H2D skip | V99-PRED-ROUTE-H2D-SKIP-sxyq | 已线上回收 | 62.40069 / 71.30446s |
| single-user causal SDPA | V101-SINGLE-USER-CAUSAL-SDPA-sxyq | 低风险线上 A/B 候选 | 本地 gate 通过，待线上 |

已删除内容：其它历史 probe、低分版本、中间版本、重复 package_root、未进入正式索引的 V103 实验包。
