# GRAB 赛题资料调研汇总

> 生成日期：2026-06-06  
> 赛题：百度 2026 CTI 生成式推荐广告排序推理性能优化  
> 目标：围绕「保证 AUC/PCOC 的前提下极致优化推理速度」整理可参考资料，覆盖帖子、开源/闭源项目、论文、期刊、会议、官方文档等，不低于 100 条。

## 调研口径

本目录把资料分成 8 类：

1. 官方/赛题/本地上下文
2. 生成式推荐与序列推荐
3. Attention 与长序列推理
4. MoE 路由与专家推理
5. 量化、混合精度与低精度 kernel
6. Embedding 与推荐系统推理
7. 编译、CUDA Graph、kernel fusion 与系统工程
8. 工程项目、帖子、博客与实践资料

每条资料都按同一张表记录：

- `类型`：论文、会议/期刊、官方文档、开源项目、帖子/博客、闭源/商业产品、赛题资料等。
- `相关性`：说明它和当前 GRAB 模型的哪一部分有关。
- `可落地方向`：落到 `infer.py` 或提交策略时可能对应的实验方向。
- `优先级`：`P0` 表示已被当前成绩验证或高度相关；`P1` 表示值得作为下一阶段主线；`P2` 表示适合做辅助/备选；`P3` 表示启发意义为主。

## 当前结论

当前本地分数记录显示，`V16-FP16EMB-sxyq` 仍是最佳已验证基线：`score_all=58.82749`，`latency=86.6182s`，且 AUC/PCOC 稳定。调研表因此不再把“泛泛 attention 优化”放在首位，而是优先围绕：

- FP16/Embedding 带宽：已经被 V14/V16 实测证明有效。
- 静态化执行与 kernel launch 减少：CUDA Graph、torch.compile、Torch-TensorRT、Triton fusion。
- 推荐模型 embedding lookup / embedding bag / cache / quantization。
- MoE grouped GEMM、expert batching、router 稳定性，但避免纯 PyTorch 动态 indexing 重写。
- 更高风险的 INT8/INT4、稀疏 attention、Top-1/专家剪枝，应独立开关验证，不和主线混在一个提交里。

主表见：[research_table.md](research_table.md)

