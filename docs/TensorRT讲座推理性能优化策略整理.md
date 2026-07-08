# TensorRT 讲座推理性能优化策略整理

> 整理日期：2026-07-08  
> 重构方式：按图片顺序逐页审核，而不是按主题粗分  
> 适用范围：`/Users/sunyiyang/Desktop/Project/Baidu  GRAB/submission` 当前各版本方法线的后续推理优化决策  
> 资料来源：  
> 1. 现场拍照图片逐张人工核对  
> 2. 录音转写补充的工程方法论  
> 3. NVIDIA / PyTorch / ModelOpt / Netron 官方资料交叉校对

## 0. 先说结论

这组材料讲得最好的地方，不是某个单一加速技巧，而是把推荐推理优化拆成了 4 条线：

1. **先定位热点**：Nsight Systems、时间线、同步、短 kernel、H2D/D2H 重叠。
2. **先减无效计算**：padding、冗余 embedding lookup、动态 shape 抖动。
3. **再选部署路径**：Torch-TensorRT、ONNX/TensorRT、原生 TensorRT。
4. **最后做低精度和工具链闭环**：PTQ/QAT、GraphSurgeon、Polygraphy、验证证据链。

对当前 GRAB 项目最有价值的不是“现在就全量上 TensorRT”，而是：

- 把 `V139` 作为 attention 对照主线；
- 把 `V169/V176` 作为结构压缩主线；
- 把动态 shape、ONNX 导出、数值回归、profiling 流程补成一条真正可复现的验证链。

## 0.1 必须补充的违规边界

这份文档原本更偏“如何优化”，但要真正作为长期 `infer.py` 研究底座，还必须补上“哪些内容即使更快也不能做”。

下面这组边界来自你补充的规则截图，应视为当前项目的**硬约束清单**。后续所有 profiling、TensorRT、compile、shape bucket、剪枝、量化、kernel 优化，都必须先过这一关。

### 规则 1：禁止在推理路径对输入采样 / 截断

以下做法都应直接判为违规：

- 只跑部分 batch、subset、sampling 过的数据；
- 用环境变量或逻辑限制 `max_batches` / `limit_batches` / 早停 batch；
- 在 `infer` 中把 `max_ctx_len`、有效历史长度、人群数、候选数主动改小；
- 把线上完整输入换成“更容易跑的小输入”。

这条对当前项目的实际含义是：

1. 不能把“本地轻量测试逻辑”搬进正式提交版 `infer.py`；
2. 任何只跑前若干 batch 的逻辑只能存在于研究脚本里，不能出现在 submission；
3. 动态 shape 优化的目标是**更高效地处理原输入**，不是把原输入缩小。

### 规则 2：禁止结构性裁剪组网

命中任一条都应判为违规：

- Transformer 层数变少，例如 `< 8 layer`、ZeroLayer、直接跳层；
- 删除或绕过完整的 attention 主路径，例如 `qkv_proj` / `out_proj` / `scaled_dot_product_attention`；
- 删除或跳过完整的一层 SMoE，例如整层 `expert` 或 `gate` 被拿掉；
- 改变有效 expert 数，或把 `Top-K` 从 `K=2` 改小；
- 删除 `RepEncoder` 里的完整大层，例如大 Embedding、LayerNorm、Linear 整层缺失；
- 直接在 forward 里写类似 `if layer_idx == 1: return x` 这种跳层捷径。

这条需要特别区分两种情况：

- **允许的减算量**：保持数学语义基本不变的精度、kernel、编译、layout、融合、低秩近似、受控结构压缩；
- **不允许的减算量**：直接删层、跳层、减 K、减有效 expert、绕过主干 forward。

所以后续如果继续做剪枝，也必须坚持：

1. 只做**细粒度、可解释、可验证**的压缩；
2. 不把“少算”偷偷变成“少跑整段网络”；
3. 所有结构改动都要先问一句：是否让完整前向语义发生了离散变化。

### 规则 3：禁止破坏 forward 拓扑 / 输入完整性

以下行为应视为违规：

- 剪枝或压缩后不再保持数学等价或近似等价；
- 为了提速修改输入掩码语义，例如改 `slot_mask`、把合法 mask 变成更稀疏的 mask；
- 为了提速修改 `RepEncoder` 输入语义；
- 让 embedding 某些表、某些 slot 在没有数据驱动依据的情况下被静态跳过；
- 用“默认置零”“默认常量”“伪造 mask/zero slot”替代真实输入处理。

这条对当前 TensorRT / ONNX / compile 路线尤其重要：

- 做 graph rewrite 可以；
- 做 plugin 可以；
- 做 operator fusion 可以；
- 但不能因为改图方便，就把原始输入语义悄悄简化。

### 规则 4：禁止 hack / 欺骗评测脚本

这类行为命中任一条都应直接视为作弊：

- 篡改计时，例如 monkey-patch `time`、`perf_counter`、`torch.cuda.Event`、`cuda.synchronize`；
- 把真正应该计时的计算挪到计时区外，例如把大部分 forward 放到 `load_model` 或预处理阶段；
- 用大规模 dummy 预热、cache 预分配等方式，实质上提前算完正式 batch；
- 提前离线生成 `logit_cache`、`pred_cache`、`logid -> prob` 字典，再在 infer 中直接查表；
- 用并行流、后台线程、异步 I/O 把该轮 forward 结果偷运到计时区外；
- 直接输出静态/随机/预写好的 `predict.txt`，或用小模型替换正式模型输出。

这条对后续实验的直接约束是：

1. `warmup` 可以做，但必须是工程上正常的首次编译/首次 kernel 触发，不是“先替正式数据跑一遍”；
2. cache 可以做，但只能做**不改变评测语义**的正常 runtime cache，而不是离线偷算；
3. 任何“把计算搬出计时区”的优化，都要先默认怀疑其违规风险。

### 当前鼓励的合规优化方向

截图里同时明确了，下面这些方向是鼓励做的，只要不破坏前面 4 条：

- 精度层：`fp16` / `bf16` / `int8` / `fp8` / `W8A8` / `W4A16`
- 算子层：Triton / CUDA kernel、FlashAttention、PagedAttention、fused gate / MoE / MLP / LayerNorm
- 推理层：`torch.compile`、CUDA Graph、多 stream 并行
- 内存层：KV cache 复用、预分配、pinned memory、零拷贝
- 调度层：batch grouping、padding 优化、pack / varlen attention、动态 batch
- MoE 层：expert grouped GEMM、topk 优化、expert 并行（前提是有效 expert 数和 `K=2` 不变）
- I/O 层：异步加载、prefetch、`predict` batch 重排
- 低改等价重写：A/B fused、权重预处理等

所以这份文档后面的所有“建议”，都应默认理解成：

**在不碰违规红线的前提下，尽可能把同一份完整 forward 跑得更快。**

## 1. 图片顺序与审核口径

本次共收到 15 张图片，其中 **第 10 张与第 8 张为重复页**，所以实际按 **14 个唯一页面** 处理。

每页都按以下 4 个维度审核：

1. **页面主旨**：这页到底在说什么。
2. **官方资料校对**：讲法是否与现有官方资料一致，哪里需要修正。
3. **对 GRAB 的具体意义**：这页内容如何映射到当前项目。
4. **逐页建议**：这页对应的下一步动作。

---

## 2. 逐页审核

### 第 1 页：长序列与 Attention

**页面主旨**

- 问题特征：
  - 用户行为序列长度差异大
  - padding 带来无效计算
  - attention 中间 tensor 大
  - `mask`、`bias`、`head dimension`、`dtype` 影响 kernel 选择
- 优化方向：
  - 统计长度分布和有效 token 占比
  - 用 packed / jagged layout 减少 padding
  - 选择合适的 attention kernel

**官方资料校对**

- 这页方向整体是对的。  
  PyTorch 官方教程已经把 `torch.jagged` / nested tensor、`scaled_dot_product_attention`、`torch.compile()` 放在同一组 transformer building blocks 里，明确指出它们适合处理变长序列并减少 padding/masking 成本。  
  参考：
  - [PyTorch Transformer building blocks](https://docs.pytorch.org/tutorials/intermediate/transformer_building_blocks.html)
  - [scaled_dot_product_attention](https://docs.pytorch.org/docs/stable/generated/torch.nn.functional.scaled_dot_product_attention.html)
  - [torch.nested / jagged](https://docs.pytorch.org/docs/stable/nested.html)

**审核结论**

- 页面判断 **准确**。
- 但需要补一条现实约束：  
  packed / jagged layout 适合 **PyTorch / custom kernel 路线**，未必能直接平移到你当前的 ONNX / TensorRT 导出链路中。

**对 GRAB 的意义**

- 当前本地历史已经说明：
  - `V139` 的 flash-attn 路线有效；
  - `permute` 这类 eager 小优化不是主要收益点；
  - 真正值得做的是减少 padding、稳定 kernel 选择。

**逐页建议**

1. 先统计真实序列长度分布，而不是先写 packed 代码。
2. 对 `V139` / `V169` 建一个长度桶分析表：
   - `seq_len` 分布
   - 平均 padding 比例
   - 热点长度段
3. 在 PyTorch 主线上先验证：
   - 变长 attention 是否适合 `flash_attn_varlen`
   - 是否值得试 packed/jagged 表达
4. 在 TensorRT 路线上，不要先承诺 packed/jagged；先验证 ONNX 导出和动态 shape profile。

### 第 2 页：Embedding 优化

**页面主旨**

- 问题特征：
  - 稀疏 ID 随机访问
  - 表规模大，局部性不稳定
  - 中间 tensor、concat/pooling 带来额外搬运
- 优化方向：
  - ID 去重
  - 缩窄 key/index 类型
  - 融合 lookup、dequant、pooling、concat

**官方资料校对**

- 这页的核心判断也是对的：embedding 侧优化重点在 **访存和中间结果消除**，而不是纯算力。  
  PyTorch 官方 `EmbeddingBag` 文档明确说明，它可以在不显式生成中间 embedding tensor 的情况下直接做聚合。  
  参考：
  - [torch.nn.EmbeddingBag](https://docs.pytorch.org/docs/stable/generated/torch.nn.EmbeddingBag.html)
  - [torch.nn.functional.embedding_bag](https://docs.pytorch.org/docs/stable/generated/torch.nn.functional.embedding_bag.html)
  - [quantized EmbeddingBag](https://docs.pytorch.org/docs/stable/generated/torch.ao.nn.quantized.EmbeddingBag.html)

**审核结论**

- 页面判断 **准确**。
- 但“融合 lookup + dequant + pooling + concat”这一条，落地难度比页面表达高很多：
  - PyTorch eager 中不一定自然得到收益；
  - TensorRT 对稀疏 embedding 这类路径也不是天然强项；
  - 更像 custom kernel / Triton / 专项 embedding 框架问题。

**对 GRAB 的意义**

- 当前仓库里 `V24-SPARSE-EMB` 说明 embedding 聚合融合方向有价值；
- 但历史主线告诉我们：embedding 不是当前最容易直接换分的入口，attention 和结构压缩收益更直接。

**逐页建议**

1. 先做静态盘点：
   - 当前索引是否可以从 `int64` 缩到 `int32`
   - 高频重复 ID 占比有多大
2. 本地优先做低风险验证：
   - `EmbeddingBag` / fused slot aggregation 的收益复核
   - 不直接承诺“自定义 kernel 一定更快”
3. TensorRT 路线中，embedding 不应作为第一批攻坚点；先攻 attention / FFN / 动态 shape。

### 第 3 页：Nsight Systems 端到端时间线分析

**页面主旨**

- 看 GPU timeline 是否有明显空洞
- 看 H2D / D2H 是否与计算重叠
- 看是否存在频繁同步、显存分配
- 看是否存在大量短 kernel
- 按耗时排序 kernel，定位热点

**官方资料校对**

- 这页与 NVIDIA 官方 profiling 方法完全一致。  
  官方文档把 profiling 解释成 “measure first, then optimize”，并强调 timeline、CUDA API 时间、queue time、kernel time、NVTX ranges 的作用。  
  参考：
  - [TensorRT Best Practices](https://docs.nvidia.com/deeplearning/tensorrt/latest/performance/best-practices.html)
  - [Nsight Systems User Guide](https://docs.nvidia.com/nsight-systems/UserGuide/index.html)
  - [Nsight Systems Analysis Guide](https://docs.nvidia.com/nsight-systems/AnalysisGuide/index.html)

**审核结论**

- 页面判断 **非常准确**。
- 这是整套材料里最应该立刻落地到本项目流程里的部分。

**对 GRAB 的意义**

- 你当前大量版本已经说明：
  - “凭感觉改代码”代价很高；
  - compile、padding、dispatch、反量化、同步开销都曾经被误判过。

**逐页建议**

1. 后续每次做新实验前，先用同一套 profile 模板采一次：
   - CPU preprocessing
   - H2D
   - model forward
   - D2H
2. 重点建立 5 个观察项：
   - GPU 空泡比例
   - H2D/D2H 与 compute 重叠度
   - `cudaMemcpy` / sync 频率
   - top kernels
   - alloc/free 热点
3. 如果没有 profile，不要再新增“纯工程猜测版”。

### 第 4 页：赛题约束和限制

**页面主旨**

- 优化目标：完整 CTR / 广告排序推理链路
- 约束条件：
  - 模型效果
  - 官方计时口径
  - 输入 shape
  - batch
  - 显存
  - 软件版本
- 风险提醒：
  - 本地单点 benchmark 变快，不等于官方得分一定变好

**官方资料校对**

- 这页本质不是 TensorRT 文档，而是实验治理原则。  
  但它和 TensorRT 官方 best practices 的 “benchmarking is how you turn a model into trustworthy numbers” 是一致的。  
  参考：
  - [TensorRT Best Practices](https://docs.nvidia.com/deeplearning/tensorrt/latest/performance/best-practices.html)

**审核结论**

- 页面判断 **准确，而且对当前项目尤其重要**。

**对 GRAB 的意义**

- 这页就是你当前项目最容易踩坑的地方：
  - 本地环境和 AI Studio V100 不同
  - Apple MPS 和 CUDA 不同
  - 官方评测只计 `forward + sigmoid`
  - 打包、shape、环境、版本都可能改变结论

**逐页建议**

1. 所有实验必须记录：
   - 硬件
   - dtype
   - 输入 shape / batch
   - warmup 是否计时
   - 软件版本
2. “本地更快”只能作为筛选信号，不能直接当 leaderboard 结论。
3. 文档与版本评估必须始终区分：
   - 本地 feasibility
   - 云端真实性能

### 第 5 页：动态 Shape 与运行时开销

**页面主旨**

- 问题特征：
  - 输入 shape 波动大
  - 不同 batch 序列长度差距大
  - kernel 执行效率和 buffer 分配开销不稳定
  - CPU 前后处理 overhead 重
- 优化方向：
  - shape bucket
  - 提前预热多种 shape
  - 算子融合

**官方资料校对**

- 这页与 TensorRT 动态 shape 官方思路一致。  
  TensorRT 官方对 dynamic shape 的核心建议是：用 optimization profiles 管理 shape 范围；在运行时明确设置 shape；必要时预分配/复用内存。  
  参考：
  - [Dynamic Shapes: Core Concepts](https://docs.nvidia.com/deeplearning/tensorrt/latest/inference-library/dynamic-shapes-basics.html)
  - [TensorRT Runtime Tutorial](https://docs.nvidia.com/deeplearning/tensorrt/latest/getting-started/quick-start-runtime-tutorial.html)

**审核结论**

- 页面判断 **准确**。
- 但 “shape bucket + 预热” 在当前项目里的优先级，应该高于 “算子融合”。

**对 GRAB 的意义**

- 这页几乎就是对 `V142`、`V149`、`V166` 的失败复盘：
  - compile / CUDA graph 不是错在概念
  - 错在 shape 没收敛、warmup 没控制、padding 浪费大

**逐页建议**

1. 第一优先级：统计 `seq_len` 和 batch shape 分布。
2. 第二优先级：设计有限 bucket，而不是全动态。
3. 第三优先级：只在 bucket 收敛后讨论 compile / TensorRT profile / CUDA Graph。
4. 不要先写复杂 kernel，再回头解释 shape 混乱。

### 第 6 页：TensorRT 是什么

**页面主旨**

- TensorRT 是从训练模型到多平台部署的推理优化引擎
- 给出 Optimizer -> Runtime 的整体图
- 提到核心能力：
  - 混合精度
  - 层/张量融合
  - kernel 自动调优
  - 时序融合
  - 动态张量内存
  - 多流并行

**官方资料校对**

- 这页和 TensorRT 官方定义基本一致。  
  官方文档明确把 TensorRT 定义为面向 NVIDIA GPU 的推理优化 SDK，并把 mixed precision、dynamic shapes、runtime API、性能优化工作流作为核心能力。  
  参考：
  - [TensorRT Documentation](https://docs.nvidia.com/deeplearning/tensorrt/latest/index.html)
  - [Architecture Overview](https://docs.nvidia.com/deeplearning/tensorrt/latest/architecture/architecture-overview.html)
  - [TensorRT Quick Start](https://docs.nvidia.com/deeplearning/tensorrt/latest/getting-started/quick-start-guide.html)

**审核结论**

- 页面判断 **准确**。
- “时序融合”这一页更适合理解为 TensorRT builder/runtime 在全图和 tactic 层面的调优能力，而不是 magic black box。

**对 GRAB 的意义**

- 对你当前项目，TensorRT 不是“任何东西都能自动变快”；
- 它更适合承接：
  - attention kernel 选择
  - FFN / GEMM 低精度和 tactic 选择
  - dynamic shape profile

**逐页建议**

1. 把 TensorRT 当作“编译式推理工具链”，不是单个 op 优化器。
2. 第一批目标不要是“全模型一把转”；
3. 第一批目标应是验证：
   - attention 主路径
   - 结构压缩后的 FFN / GEMM
   - 动态 shape profile 可行性

### 第 7 页：TensorRT 11.0 新特性

**页面主旨**

- 原生 MoE 与多 GPU 支持
- 架构精简，`IPluginV3` 统一自定义内核接口
- 无缝接入 Hugging Face 与开源 PyTorch 模型
- 升级 Torch-TensorRT 前端

**官方资料校对**

- 这页方向是对的，但有两点需要降级表述：

1. **`IPluginV3` 统一插件接口**  
   这是有官方资料明确支撑的。  
   参考：
   - [IPluginV3 / Plugin API](https://docs.nvidia.com/deeplearning/tensorrt/latest/inference-library/plugins-cpp.html)

2. **“原生 MoE 与多 GPU 支持”**  
   这更接近 TensorRT 生态和 TensorRT-LLM / multi-device execution 的增强，不应理解成“你的任意自定义 CTR-MoE 直接零改造吃到收益”。  
   参考：
   - [Architecture Overview](https://docs.nvidia.com/deeplearning/tensorrt/latest/architecture/architecture-overview.html)

3. **“无缝接入 Hugging Face / 开源 PyTorch 模型”**  
   这在大模型生态里更常见，但对当前 CTR 自定义模型，仍然要看 operator coverage、导出稳定性、路由逻辑。

**审核结论**

- 页面判断 **大方向正确**。
- 但对当前 GRAB 项目，要避免误解成“TensorRT 11 对自定义 CTR-MoE 已经开箱即用”。

**对 GRAB 的意义**

- 最有价值的不是 MoE 新特性本身，而是：
  - 插件接口统一
  - PyTorch 入口更成熟
  - 多设备/多流执行思想更明确

**逐页建议**

1. 本项目不要把 “原生 MoE 支持” 当成短期承诺。
2. 真正值得用的是：
   - `IPluginV3` 作为未来兜底方案
   - Torch-TensorRT / ONNX 路线的可行性评估
3. 如果未来 TensorRT 主线卡在自定义 op，再考虑 plugin。

### 第 8 页：TensorRT 加速实践（编译期与运行期）

**页面主旨**

- 把 TensorRT 拆成：
  - 编译期：算子编译、融合、图优化
  - 运行期：反序列化与执行
- 路径是：`ONNX Graph -> TensorRT 编译期 -> TensorRT Engine -> TensorRT 运行期 -> 高性能推理服务`

**官方资料校对**

- 这页和官方 workflow 一致。  
  官方 Quick Start 也是 “export -> select precision -> convert -> deploy”。  
  参考：
  - [TensorRT Quick Start](https://docs.nvidia.com/deeplearning/tensorrt/latest/getting-started/quick-start-guide.html)
  - [Architecture Overview](https://docs.nvidia.com/deeplearning/tensorrt/latest/architecture/architecture-overview.html)

**审核结论**

- 页面判断 **准确**。

**对 GRAB 的意义**

- 你当前项目过去很多优化都在 eager runtime 层瞎抠；
- 这一页的价值是提醒：真正高收益的事情，很多发生在 build phase，而不是 run phase。

**逐页建议**

1. 后续如果做 TensorRT，先把“build 期优化空间”想清楚：
   - shape profiles
   - precision
   - tactic / cache
   - 插件/图改写
2. 不要只盯着 runtime。

### 第 9 页：易用性与定制化，如何选择部署路径

**页面主旨**

- 三条路径：
  - `Torch-TensorRT`
  - `TRT Model Convert`
  - `TensorRT 原生（Native API / ONNX -> TensorRT）`
- 强调从“易用优先”到“极致定制”

**官方资料校对**

- 现有官方资料里，对部署路径的更稳妥表述是：
  - Torch-TensorRT：高层 PyTorch 入口，可 fallback
  - ONNX / `trtexec`：更低开销，但要求 operator support
  - 原生 Network API：最灵活，成本也最高  
  参考：
  - [TensorRT Quick Start](https://docs.nvidia.com/deeplearning/tensorrt/latest/getting-started/quick-start-guide.html)
  - [Additional Resources](https://docs.nvidia.com/deeplearning/tensorrt/10.x.x/reference/additional-resources.html)

**审核结论**

- 页面判断 **基本准确**。
- 但“TRT Model Convert”更像讲者对中间转换方案的工程口语化分类；正式文档里你更应该按：
  - Torch-TensorRT
  - ONNX conversion
  - manual Network API
  来理解。

**对 GRAB 的意义**

- 当前项目最现实的优先顺序：
  1. 先试 ONNX 导出是否稳定；
  2. 若 ONNX 卡得厉害，再看 Torch-TensorRT；
  3. 只有主干收益已经明确时，才考虑原生 Network API / plugin。

**逐页建议**

1. 对当前 GRAB，不建议第一步直接走原生 TensorRT。
2. 最务实的顺序是：
   - `V139` / `V169` 小样本导出 ONNX
   - 用 Netron / Polygraphy 检查
   - 再决定 Torch-TensorRT 还是 ONNX/TensorRT

### 第 10 页：与第 8 页重复

**审核结论**

- 这页与第 8 页是重复图。
- 文档中不再重复展开。

### 第 11 页：TensorRT 运行期 / `IExecutionContext`

**页面主旨**

- 运行期重点：
  - plan 反序列化
  - 显存 / 资源分配
  - `IExecutionContext`
  - 多实例并行执行
  - 预分配输入缓冲区，加入任务队列完成推理

**官方资料校对**

- 这页和官方 runtime API 高度一致。  
  官方文档明确指出 execution context 封装执行状态；dynamic shape 下要设置输入 shape；输入/输出显存需要明确管理；多 execution context / 多 stream 可用于并发。  
  参考：
  - [TensorRT Runtime Tutorial](https://docs.nvidia.com/deeplearning/tensorrt/latest/getting-started/quick-start-runtime-tutorial.html)
  - [ICudaEngine / createExecutionContext](https://docs.nvidia.com/deeplearning/tensorrt/latest/_static/c-api/classnvinfer1_1_1_i_cuda_engine.html)
  - [Optimizing TensorRT Performance](https://docs.nvidia.com/deeplearning/tensorrt/latest/performance/optimization.html)

**审核结论**

- 页面判断 **准确**。

**对 GRAB 的意义**

- 如果未来你的模型真走到 TensorRT runtime，这页决定的是服务侧收益，而不是模型结构收益。
- 当前项目短期内不需要自己写复杂服务，但必须提前知道：
  - context 数量
  - profile 数量
  - stream 并发
  - 预分配策略
  会影响最终吞吐/延迟。

**逐页建议**

1. 未来不要只测单 context、单 stream。
2. 如果进入服务化验证，必须同时看：
   - 单请求延迟
   - 多 context 并发吞吐
   - 显存占用上界

### 第 12 页：进阶特性：动态 Shape、Plugin、量化与更多

**页面主旨**

- 动态 shape
- Refitting
- 自定义 plugin
- Algorithm Selector
- Timing Cache
- INT8 量化

**官方资料校对**

- 这页大方向对，但有一个非常关键的“过时点”必须明确：

1. **Algorithm Selector**
   - 旧思路没错；
   - 但 TensorRT 官方现在明确说明：旧版 `IAlgorithmSelector` 已经 deprecated，建议使用 **editable timing cache** 来复现和控制 tactic 选择。  
   参考：
   - [Algorithm Selection and Reproducible Builds](https://docs.nvidia.com/deeplearning/tensorrt/10.x.x/inference-library/precision-control.html)

2. **Refitting**
   - 官方明确支持；
   - 但 refit 会影响某些融合，需要细粒度控制。  
   参考：
   - [Refitting an Engine](https://docs.nvidia.com/deeplearning/tensorrt/latest/inference-library/refitting-engines.html)

3. **Timing Cache**
   - 是非常值得承接的；
   - 尤其适合你这种频繁 build / 多版本试验场景。

4. **INT8**
   - 官方支持 PTQ / QAT；
   - 但并不保证任何模型都自动变快、自动保持精度。  
   参考：
   - [Quantization Workflows](https://docs.nvidia.com/deeplearning/tensorrt/latest/inference-library/quantized-types-workflows.html)

**审核结论**

- 页面判断 **正确，但需要更新术语**：
  - `Algorithm Selector` -> 优先理解为 **editable timing cache / reproducible build**

**对 GRAB 的意义**

- 当前项目最值得吸收的不是“所有高级特性都试”，而是：
  1. dynamic shape profile
  2. timing cache
  3. plugin 兜底方案
  4. PTQ/QAT 的正确顺序

**逐页建议**

1. 文档和实验里不再把 `IAlgorithmSelector` 当主推荐方案。
2. 如果进入 TensorRT 构建实验：
   - 优先保留 timing cache 文件
   - 优先做可复现 build
3. refit 只在确实需要“同结构换权重”时再考虑。

### 第 13 页：ModelOpt：低精度工具链

**页面主旨**

- Quantization：降计算和带宽压力
- PTQ：后训练量化
- QAT：精度不够时再训练补偿
- Pruning / Distillation：减小模型或训练小学生模型
- Export / Deployment：导出后用 TensorRT / Polygraphy 部署验证

**官方资料校对**

- 这页和现有官方文档高度一致。  
  NVIDIA Model Optimizer 文档现在明确把：
  - PTQ
  - QAT
  - Pruning
  - Distillation
  放在同一套工具链里。  
  参考：
  - [Model Optimizer Documentation](https://nvidia.github.io/Model-Optimizer/)
  - [Quantization Workflows](https://docs.nvidia.com/deeplearning/tensorrt/latest/inference-library/quantized-types-workflows.html)

**审核结论**

- 页面判断 **准确**。
- 也是与你当前项目方法库最有共鸣的一页。

**对 GRAB 的意义**

- 你本地已经有：
  - FFN pruning
  - SVD
  - INT8 尝试
- 所以这页不是“新方向”，而是把现有零散尝试统一到更正规的工具链框架中。

**逐页建议**

1. 不要继续做 eager INT8 试错。
2. 低精度路线必须改成：
   - `FP16/BF16 baseline`
   - `PTQ`
   - `敏感层保护/混合精度`
   - `QAT（仅当 PTQ 不够）`
3. 剪枝 / SVD / Distillation 后续要和部署链一起评估，而不是单独看模型侧。

### 第 14 页：调试与验证工具链：ONNX -> TensorRT

**页面主旨**

- `Netron`：看 graph、shape、算子、常量、Q/DQ、子图
- `ONNX GraphSurgeon`：删节点、改输入输出、替换子图、插 plugin 节点
- `Polygraphy`：比 ONNX Runtime 与 TensorRT 输出、设 `atol/rtol`

**官方资料校对**

- 这页是准确的，而且工具链搭配很标准：
  - Netron：官方站点就是模型可视化工具  
    [Netron](https://netron.app/)
  - ONNX GraphSurgeon：NVIDIA 官方 ONNX 改图工具  
    [ONNX GraphSurgeon API](https://docs.nvidia.com/deeplearning/tensorrt/latest/api/onnx-graphsurgeon-api.html)
  - Polygraphy：NVIDIA 官方跨框架调试/比较工具  
    [Polygraphy API](https://docs.nvidia.com/deeplearning/tensorrt/latest/api/polygraphy-api.html)

**审核结论**

- 页面判断 **准确**。
- 这页对你当前项目的实际价值非常高，因为它刚好补上了你现在最缺的“导出后怎么证明没坏”。

**对 GRAB 的意义**

- 你现在最需要的不是“先转 TensorRT”，而是先建立：
  - PyTorch -> ONNX
  - ONNX Runtime 对齐
  - TensorRT 对齐
  的数值闭环。

**逐页建议**

1. 先把这三件工具固定成流程：
   - `Netron` 看图
   - `GraphSurgeon` 修图
   - `Polygraphy` 做数值对比
2. 任何 TensorRT 实验都必须附：
   - 改前图
   - 改后图
   - ORT vs TRT 误差对比

### 第 15 页：验证：验收标准与证据链

**页面主旨**

- Tensor 一致性：以 PyTorch 为基线，对比 ONNX Runtime / TensorRT 差异（`atol/rtol`）
- 端到端效果：速度和精度平衡
- 可复现性：多次实验取稳定结果
- 证据链模板：
  - Baseline
  - Hypothesis
  - Change
  - Evidence
  - Decision

**官方资料校对**

- 这页和 TensorRT 官方 best practices、Polygraphy 思路完全一致：  
  先 measure，再 optimize，再 measure；同时保留 accuracy validation。  
  参考：
  - [TensorRT Best Practices](https://docs.nvidia.com/deeplearning/tensorrt/latest/performance/best-practices.html)
  - [Polygraphy API](https://docs.nvidia.com/deeplearning/tensorrt/latest/api/polygraphy-api.html)

**审核结论**

- 页面判断 **非常准确**。
- 这是当前文档里最应该直接转成你项目实验模板的一页。

**对 GRAB 的意义**

- 你当前版本多、历史长、分支杂，最缺的不是新想法，而是统一验收标准。

**逐页建议**

1. 后续每个版本必须固定填表：
   - Baseline：环境、模型、shape、dtype、原始结果
   - Hypothesis：为什么这次会更快/更稳
   - Change：代码、配置、导出、profile 变化
   - Evidence：延迟、AUC/PCOC、显存、profile、数值对齐
   - Decision：保留 / 回退 / 进入下一轮
2. 没有证据链的实验，不进入主线。

---

## 3. 逐页审核后的总建议

### 3.1 当前最值得吸收的页面

如果按“对当前项目的实际作用”排序，优先级最高的是：

1. 第 3 页：Nsight Systems 时间线分析  
2. 第 5 页：动态 shape 与运行时开销  
3. 第 14 页：ONNX -> TensorRT 调试工具链  
4. 第 15 页：验收标准与证据链  
5. 第 1 页：长序列与 attention  
6. 第 13 页：ModelOpt 低精度工具链

### 3.2 当前不应被误读的页面

最容易被误读的是：

1. 第 7 页：TensorRT 11.0 新特性  
   不要理解成“当前自定义 CTR-MoE 可直接无脑吃到 MoE 原生支持”

2. 第 9 页：部署路径选择  
   不要直接把口语里的 `TRT Model Convert` 当成正式文档分类

3. 第 12 页：Algorithm Selector  
   现阶段应优先理解成 editable timing cache / reproducible build，而不是继续围绕旧接口设计方案

### 3.3 对 GRAB 的最终落地顺序

结合本地版本现状，最务实的顺序仍然是：

1. 用 `V139` 做 attention 主线对照
2. 用 `V169 / V176` 做结构压缩主线
3. 先做：
   - shape 分布统计
   - profiling
   - PyTorch -> ONNX 导出
   - ORT / TRT 数值回归
4. 再做：
   - FP16 TensorRT baseline
   - timing cache
   - profile 配置
5. 最后才讨论：
   - PTQ / QAT
   - plugin
   - 原生 TensorRT network API

### 3.4 当前已经落实到版本里的内容

到 `V185-V177-STACKED-SUBMIT-sxyq` 为止，这份策略文档里已经被真正落实的优化，不再只是“建议”，而是已经进入正式 submission 主线：

1. **结构减算优先，而不是删层/跳层**
   - 采用 `FFN prune25 + SVD rank64`
   - 属于受控结构压缩，不是违规的跳层、减层、减 `Top-K`

2. **先修校准，再谈速度**
   - 采用基于线上锚点的 `logit bias recalibration`
   - 证明结构压缩后的主要风险之一确实是常数级校准漂移

3. **动态 shape 下优先选择更稳的 compile 路线**
   - 默认 `torch.compile(mode=max-autotune-no-cudagraphs, dynamic=True)`
   - 并保留 warmup fallback，而不是强行押注脆弱的 cudagraph

4. **把热点模块而不是全模型纳入编译**
   - `seq_encoder` 继续编译
   - `rep_encoder` 也进入可控 compile 范围
   - 输出头保持 eager，避免为小模块付出无意义编译开销

5. **运行时搬运与输入管线继续收敛**
   - CUDA 路径启用 `pin_memory`
   - 属于文档里明确鼓励的 I/O / 内存层优化

6. **SVD 构建路径做数值稳定化**
   - 分解在 `FP32` 中完成，再 cast 回目标 dtype
   - 这不是改图捷径，而是为了降低低精度分解的数值风险

7. **I/O 层的异步 prefetch 已进入候选实现**
   - 在 `V186-ASYNC-PREFETCH-sxyq` 中，增加了基于 CUDA stream 的 batch 预取
   - 目标是把 `non_blocking` H2D 与当前 batch 计算重叠
   - 这正对应文档里鼓励的 `async loading / prefetch / runtime overlap`
   - 但线上结果已经说明：**实现合规不等于收益成立**，当前官方环境下这条线没有跑赢 `V185`

8. **shape bucket 目前仍是“待验证方向”，没有默认写死进主线**
   - 原因不是这条建议不对，而是当前代码路径已经是 `varlen` attention，不是重 padding 路径
   - 在这条实现里，若只按用户长度强行重排 batch，未必会自然改善 `total_tokens` 形状收敛
   - 所以这部分下一步应先做 batch-shape 统计与 compile 证据，而不是直接默认打开

9. **batch grouping / predict batch 重排 已进入候选实现**
   - 在 `V187-BALANCED-BATCH-sxyq` 中，不改变 `50 user/batch` 上限，只重排“哪些用户被分到同一批”
   - 目标不是减少样本，而是拉平每批 `total_tokens`
   - 本地小样本统计里，这条线已经把 `batch token std` 从 `102.97` 压到 `38.88`
   - 但线上结果 `69.85911 / 39.33983s / 0.75649 / 1.06429` 说明它没有超过 `V185`
   - 后续进一步核对代码路径后确认：如果线上目录里存在 `cached_batches/shard_*.pt`，原版 `V187` 仍会优先命中缓存分支，`Dataset/make_collate_fn` 里的 batch 重排不一定真正进入执行路径

10. **线上 batch 组织优化必须先拿回执行路径控制权**
   - 因为 `cached_batches` 会绕过 `CSV -> Dataset -> DataLoader -> make_collate_fn`
   - 所以新增 `V188-FORCE-DATALOADER-sxyq`，默认忽略 `cached_batches`，强制走自建 batch 路径
   - 线上结果 `70.00710 / 38.70555s / 0.75649 / 1.06429` 说明这一步不是空修复：拿回执行路径后，`V187` 的调度层收益确实开始显现
   - 但它仍然没有超过 `V185`，说明仅靠“强制走 DataLoader”还不够，后续必须继续补 `shape bucket + runtime overlap`

11. **下一步必须把 shape 治理和 runtime overlap 合并验证**
   - 因为 `V188` 已经证明 dataset/collate 路径重新生效
   - 所以继续新增 `V189-BUCKET-PREFETCH-sxyq`：
     - batch 仍保持 `50 user/batch`
     - 先按 `token_total` / `max_len` 做更紧的 batch 顺序治理
     - 再把 median-size batch 放到 warmup 起点附近，降低 compile 首批形状偏置
     - 同时把 CUDA async prefetch 接回这条已生效的 DataLoader 路径
   - 这一步对应的正是文档前面强调的：`shape bucket + 预热` 优先级高于继续盲目押注单点 compile 开关

12. **本地轻量验证已经说明：shape 治理不能乱重排**
   - 在 `V189` 的本地代理对比里，median-first 重排会明显破坏相邻 batch 的 `max_len` 连续性
   - 因此继续新增 `V190-SHAPE-STABLE-sxyq`：
     - 保留 `V188` 的 forced DataLoader 路径
     - 保留 balanced batching
     - 但把 batch 执行顺序显式固定为 `max_len` / `token_total` 优先的 shape-stable 顺序
     - 不再默认叠加 `V189` 的 median-first 重排与 prefetch
   - 这更符合文档前面反复强调的原则：先保证真实执行路径和 shape 连续性，再讨论 runtime overlap

13. **`V190` 进一步改成激进的端到端 CPU batch 构造优化版**
   - 在不改变模型结构、不改变输入规模、不截断、不采样的前提下，把还能安全落地的工程项继续压进 `V190`
   - 默认开启：
     - `GRAB_V188_USE_CACHED_BATCHES=0`：强制走 `CSV -> Dataset -> DataLoader -> make_collate_fn`
     - `GRAB_V190_SHAPE_STABLE_BATCH_ORDER=1`：固定 `max_len` / `token_total` 优先的 batch 执行顺序
     - `GRAB_V190_PRECOMPUTE_USER_RECORDS=1`：在 Dataset 初始化阶段预先完成用户内排序、`pred_mask` 构造和 record 汇总，避免 `__getitem__` 重复排序
     - `GRAB_V190_FAST_COLLATE=1`：重写 collate 聚合路径，减少 Python 层逐字段拼接和重复 membership 检查
   - 同时把 `max_sign_id` 统计合并进加载主循环，去掉原先构造 Dataset 后再扫一遍全部 sign 的额外 pass
   - 本地轻量验证结果：
     - `python3 -m py_compile submission/V190-SHAPE-STABLE-sxyq/infer.py` 通过
     - 默认开关检查为 `cache=False / shape_order=True / precompute=True / fast_collate=True`
     - 与 `V188` 首个 DataLoader batch 做 `userid/logid/label/pred_mask/user_offsets/1..28 slots` 共 33 项对齐，无 mismatch
     - 同一本地样本的 DataLoader 代理迭代从 `V188 6.4470s` 降到 `V190 4.3831s`，覆盖 `20` 个 batch、`296284` 条样本
   - 这部分属于端到端链路里的 CPU 输入管线优化，不是模型近似压缩，也不是评测路径规避
   - 但需要明确：该收益只是本地轻量代理数据，最终是否能压低线上 `38.7s` 级 latency 仍需正式提交确认

## 4. 官方资料索引

下面这些是本次重构时实际参考的一手资料：

- TensorRT 总览  
  https://docs.nvidia.com/deeplearning/tensorrt/latest/index.html

- TensorRT 架构总览  
  https://docs.nvidia.com/deeplearning/tensorrt/latest/architecture/architecture-overview.html

- TensorRT Quick Start  
  https://docs.nvidia.com/deeplearning/tensorrt/latest/getting-started/quick-start-guide.html

- TensorRT Runtime Tutorial  
  https://docs.nvidia.com/deeplearning/tensorrt/latest/getting-started/quick-start-runtime-tutorial.html

- TensorRT Dynamic Shapes  
  https://docs.nvidia.com/deeplearning/tensorrt/latest/inference-library/dynamic-shapes-basics.html

- TensorRT Best Practices  
  https://docs.nvidia.com/deeplearning/tensorrt/latest/performance/best-practices.html

- TensorRT Performance Optimization / Multi-Streaming  
  https://docs.nvidia.com/deeplearning/tensorrt/latest/performance/optimization.html

- TensorRT Plugin API / `IPluginV3`  
  https://docs.nvidia.com/deeplearning/tensorrt/latest/inference-library/plugins-cpp.html

- TensorRT Refit  
  https://docs.nvidia.com/deeplearning/tensorrt/latest/inference-library/refitting-engines.html

- TensorRT Quantization Workflows  
  https://docs.nvidia.com/deeplearning/tensorrt/latest/inference-library/quantized-types-workflows.html

- TensorRT ONNX GraphSurgeon  
  https://docs.nvidia.com/deeplearning/tensorrt/latest/api/onnx-graphsurgeon-api.html

- TensorRT Polygraphy  
  https://docs.nvidia.com/deeplearning/tensorrt/latest/api/polygraphy-api.html

- NVIDIA Model Optimizer  
  https://nvidia.github.io/Model-Optimizer/

- Nsight Systems User Guide  
  https://docs.nvidia.com/nsight-systems/UserGuide/index.html

- Nsight Systems Analysis Guide  
  https://docs.nvidia.com/nsight-systems/AnalysisGuide/index.html

- PyTorch Nested / Jagged / Transformer Building Blocks  
  https://docs.pytorch.org/tutorials/intermediate/transformer_building_blocks.html

- PyTorch `EmbeddingBag`  
  https://docs.pytorch.org/docs/stable/generated/torch.nn.EmbeddingBag.html

- Netron  
  https://netron.app/
