# GRAB 赛题资料调研表

> 条目数：120  
> 说明：本表优先服务于后续实验决策，不是传统参考文献表。`本地` 表示仓库已有 PDF/笔记或赛题文档；`外部` 表示公开网页、论文页、项目页或工程文档。

| ID | 分类 | 类型 | 资料/项目 | 来源 | 相关性 | 可落地方向 | 优先级 |
|---:|---|---|---|---|---|---|---|
| 1 | 官方/赛题 | 赛题页 | 百度 2026 CTI 生成式推荐广告排序推理性能优化 | [AI Studio competition](https://aistudio.baidu.com/competition/detail/1461) | 直接定义任务、指标、提交约束 | 以 AUC/PCOC 合规为硬约束，优化 latency | P0 |
| 2 | 官方/赛题 | Baseline | 官方 baseline 项目 | [AI Studio project](https://aistudio.baidu.com/projectdetail/10186630) | 原始 `infer.py` 和评测入口来源 | 对照模型结构、提交包结构和运行路径 | P0 |
| 3 | 官方/赛题 | 数据集 | 2026 CTI data | [AI Studio dataset](https://aistudio.baidu.com/dataset/detail/375013/file) | 测试 batch 与用户行为序列来源 | 研究 cached shard、序列长度分布、batch 形态 | P0 |
| 4 | 官方/赛题 | 模型权重 | 2026 CTI model | [AI Studio model](https://aistudio.baidu.com/modelsdetail/45703/space) | 原始 checkpoint 来源 | 保持模型参数不变，研究加载和 dtype | P0 |
| 5 | 官方/赛题 | 本地文档 | 项目 README | 本地 `README.md` | 汇总模型架构、baseline 性能、提交方式 | 后续版本必须对照 baseline 指标 | P0 |
| 6 | 官方/赛题 | 本地文档 | 项目详细说明 | 本地 `docs/project_info.md` | 赛题数据、指标、警告和下载方式 | 明确不可修改组网与评测时间限制 | P0 |
| 7 | 官方/赛题 | 本地记录 | 提交记录、分数与论文方法索引 | 本地 `docs/提交记录_分数_方法_论文索引.md` | 当前最重要的实验真相源 | 每个新方向先查是否已失败/已验证 | P0 |
| 8 | 官方/赛题 | 本地代码 | `infer.py` 当前实现 | 本地 `infer.py` | 实际可提交推理入口 | 所有资料最后都要映射到该文件可改点 | P0 |
| 9 | 官方/赛题 | 本地工具 | `tools/validate_submission.py` | 本地工具 | 验证 zip 根目录结构 | 每次打包前检查 `infer.py/build_env/requirements` | P0 |
| 10 | 官方/赛题 | 本地记录 | V16-FP16EMB-sxyq | 本地 `versions/03_quantization/V16-FP16EMB-sxyq/` | 当前最佳成绩和最低风险基线 | 后续实验优先基于 V16 单变量修改 | P0 |
| 11 | 生成式推荐 | 论文 | GRAB: 百度推荐广告生成式排序模型技术实践 | [arXiv PDF](https://arxiv.org/pdf/2602.01865) / 本地 `docs/论文/GARB论文.pdf` | 赛题模型背景和工业优化线索 | 理解 GRAB 原始设计、M-FALCON、混合精度 | P0 |
| 12 | 生成式推荐 | 论文/会议 | Actions Speak Louder than Words: Trillion-Parameter Sequential Transducers for Generative Recommendations | [arXiv](https://arxiv.org/abs/2402.17152) / 本地 PDF | HSTU 和生成式推荐长序列建模 | 对照当前 Transformer/历史序列处理 | P0 |
| 13 | 生成式推荐 | 开源项目 | Meta generative-recommenders | [GitHub](https://github.com/facebookresearch/generative-recommenders) | HSTU 参考实现和推荐序列建模工程 | 参考 jagged tensor、attention backend、数据布局 | P1 |
| 14 | 生成式推荐 | 论文 | Scaling Generative Recommendations with Context Parallelism on HSTU | 本地 `literature/02_GenerativeRecommendation/` | 长序列推荐和上下文并行 | 判断 context parallel 对单 V100 赛题是否适用 | P2 |
| 15 | 生成式推荐 | 论文 | EARN: Efficient Inference Acceleration for LLM-based Generative Recommendation | 本地 `literature/02_GenerativeRecommendation/` | 推荐生成模型推理加速 | 借鉴 register token/early reduction 思路 | P2 |
| 16 | 生成式推荐 | 论文 | SASRec: Self-Attentive Sequential Recommendation | [IEEE ICDM](https://ieeexplore.ieee.org/document/8594844) | 序列推荐 Transformer 早期基础 | 理解行为序列注意力的指标风险 | P3 |
| 17 | 生成式推荐 | 论文 | BERT4Rec: Sequential Recommendation with Bidirectional Encoder Representations | [arXiv](https://arxiv.org/abs/1904.06690) | 推荐序列建模代表作 | 判断双向/单向 attention 改动是否改变语义 | P3 |
| 18 | 生成式推荐 | 论文 | DIN: Deep Interest Network for Click-Through Rate Prediction | [KDD](https://dl.acm.org/doi/10.1145/3219819.3219823) | 广告 CTR 兴趣建模经典 | CTR 指标敏感性与特征交互背景 | P3 |
| 19 | 生成式推荐 | 论文 | DIEN: Deep Interest Evolution Network for CTR Prediction | [AAAI](https://ojs.aaai.org/index.php/AAAI/article/view/4455) | 用户兴趣演化建模 | 理解历史行为序列对 AUC 的敏感性 | P3 |
| 20 | 生成式推荐 | 论文 | MIND: Multi-Interest Network with Dynamic Routing | [CIKM](https://dl.acm.org/doi/10.1145/3357384.3357814) | 多兴趣路由与推荐表示 | 对比 SMoE/router 的语义风险 | P3 |
| 21 | Attention | 论文/会议 | FlashAttention: Fast and Memory-Efficient Exact Attention with IO-Awareness | [arXiv](https://arxiv.org/abs/2205.14135) / 本地 PDF | 精确 attention 的 IO 优化 | 若改 attention，优先等价 kernel 而非近似 | P1 |
| 22 | Attention | 论文/会议 | FlashAttention-2: Faster Attention with Better Parallelism and Work Partitioning | [arXiv](https://arxiv.org/abs/2307.08691) / 本地 PDF | 长序列/低 batch attention 并行 | varlen attention、减少非 matmul FLOP | P1 |
| 23 | Attention | 论文/会议 | FlashAttention-3: Fast and Accurate Attention with Asynchrony and Low-Precision | [arXiv](https://arxiv.org/abs/2407.08608) / 本地 PDF | 低精度 attention 和异步流水 | V100 不直接适配 H100 特性，但可借鉴低精度思路 | P3 |
| 24 | Attention | 开源项目 | Dao-AILab flash-attention | [GitHub](https://github.com/Dao-AILab/flash-attention) | FlashAttention 官方实现 | 检查 V100/SM70、PyTorch 版本和安装成本 | P1 |
| 25 | Attention | 文档 | PyTorch scaled_dot_product_attention | [PyTorch docs](https://pytorch.org/docs/stable/generated/torch.nn.functional.scaled_dot_product_attention.html) | PyTorch 原生 attention dispatcher | 评估能否用 SDPA 替换手写 attention | P1 |
| 26 | Attention | 论文 | Dynamic Sparse Flash Attention | 本地 `literature/07_SparseAttention/` | 稀疏 attention 加速 | 高风险，只适合作为开关实验 | P2 |
| 27 | Attention | 论文 | SeerAttention: Learning Intrinsic Sparse Attention | 本地 `literature/07_SparseAttention/` | 学习式稀疏 attention | 指标风险高，需独立验证 AUC/PCOC | P2 |
| 28 | Attention | 论文 | Loki: Low-rank Keys for Efficient Sparse Attention | 本地 `literature/07_SparseAttention/` | 低秩 key 选择和稀疏模式 | 可做中后层 token pruning 参考 | P2 |
| 29 | Attention | 论文 | Longformer: The Long-Document Transformer | [arXiv](https://arxiv.org/abs/2004.05150) | 滑窗/全局稀疏注意力 | 推荐序列若有局部性，可作为近似启发 | P3 |
| 30 | Attention | 论文 | BigBird: Transformers for Longer Sequences | [arXiv](https://arxiv.org/abs/2007.14062) | 块稀疏 attention | 高风险近似，适合理论对照 | P3 |
| 31 | Attention | 论文 | Reformer: The Efficient Transformer | [arXiv](https://arxiv.org/abs/2001.04451) | LSH attention 和可逆层 | 赛题短期落地价值低，作为 sparse 参考 | P3 |
| 32 | Attention | 论文 | Linformer: Self-Attention with Linear Complexity | [arXiv](https://arxiv.org/abs/2006.04768) | 低秩 attention | 可能破坏语义，不宜主线提交 | P3 |
| 33 | Attention | 论文 | Performer: FAVOR+ Linear Attention | [arXiv](https://arxiv.org/abs/2009.14794) | kernelized attention | 近似 attention 风险参考 | P3 |
| 34 | Attention | 论文 | xFormers: A modular and hackable Transformer modelling library | [GitHub](https://github.com/facebookresearch/xformers) | attention kernel 组件库 | 可参考 memory_efficient_attention API/限制 | P2 |
| 35 | Attention | 项目 | NVIDIA Transformer Engine | [GitHub](https://github.com/NVIDIA/TransformerEngine) | FP8/Transformer kernel 实践 | V100 不适配 FP8，作为低精度工程参考 | P3 |
| 36 | MoE | 论文/会议 | Toward Efficient Inference for Mixture of Experts | 本地 `literature/03_MoE_Optimization/` | MoE 推理效率和专家执行 | expert batching/grouped GEMM 主要依据 | P1 |
| 37 | MoE | 论文 | LExI: Layer-Adaptive Active Experts for Efficient MoE Inference | 本地 PDF | 层自适应专家选择 | 可做高风险 gating skip，但需守住 PCOC | P2 |
| 38 | MoE | 论文/会议 | MiLo: Efficient Quantized MoE Inference with Mixture of Low-Rank Compensators | 本地 PDF | MoE 量化误差补偿 | INT8/低秩补偿备选 | P2 |
| 39 | MoE | 论文 | Switch Transformers | [JMLR](https://www.jmlr.org/papers/v23/21-0998.html) | Top-1 expert routing 经典 | 解释 V10 Top-1 破坏 PCOC 的风险 | P2 |
| 40 | MoE | 论文 | GShard: Scaling Giant Models with Conditional Computation | [arXiv](https://arxiv.org/abs/2006.16668) | 条件计算和专家路由 | MoE load balance 背景 | P3 |
| 41 | MoE | 论文 | GLaM: Efficient Scaling of Language Models with MoE | [arXiv](https://arxiv.org/abs/2112.06905) | 稀疏激活专家 | 比较 top-k 路由指标风险 | P3 |
| 42 | MoE | 开源项目 | DeepSpeed-MoE | [Docs](https://www.deepspeed.ai/tutorials/mixture-of-experts/) | MoE 训练/推理工程 | expert parallel、capacity、routing 实现参考 | P2 |
| 43 | MoE | 开源项目 | MegaBlocks | [GitHub](https://github.com/databricks/megablocks) | block-sparse MoE 高效训练 | blocked expert GEMM 思路 | P3 |
| 44 | MoE | 开源项目 | Tutel | [GitHub](https://github.com/microsoft/tutel) | MoE kernel 和调度优化 | grouped expert dispatch 参考 | P2 |
| 45 | MoE | 开源项目 | MegaBlocks / dropless MoE paper | [arXiv](https://arxiv.org/abs/2211.15841) | dropless MoE 和 block sparse | 提醒避免 token drop 改语义 | P3 |
| 46 | MoE | 开源项目 | vLLM MoE kernels | [GitHub](https://github.com/vllm-project/vllm) | fused MoE kernel、专家并行 | 借鉴 fused_moe/gated matmul 结构 | P2 |
| 47 | MoE | 开源项目 | TensorRT-LLM MoE support | [GitHub](https://github.com/NVIDIA/TensorRT-LLM) | 高性能 MoE 推理 kernel | grouped GEMM 与 plugin 参考 | P2 |
| 48 | MoE | 论文 | FasterMoE: A Fast Mixture-of-Expert Training System | [arXiv](https://arxiv.org/abs/2203.13262) | MoE 通信/调度优化 | 单卡价值有限，但 dispatch 分析有用 | P3 |
| 49 | MoE | 论文 | FastMoE: A Fast Mixture-of-Expert Training System | [GitHub](https://github.com/laekov/fastmoe) | MoE 系统工程 | expert routing 数据结构参考 | P3 |
| 50 | MoE | 论文 | PreScope: Prefetching for MoE Inference | 本地 `literature/11_DataPrefetch/` | MoE expert 预取 | 当前 8 expert 常驻时价值有限 | P3 |
| 51 | 量化/低精度 | 论文/会议 | SmoothQuant: Accurate and Efficient PTQ for LLMs | [arXiv](https://arxiv.org/abs/2211.10438) / 本地 PDF | W8A8 平滑量化 | 线性层 INT8 量化候选 | P1 |
| 52 | 量化/低精度 | 论文 | INT-FlashAttention | 本地 `literature/06_Quantization/` | INT8 attention kernel | 高实现成本，作为后续低精度 attention | P2 |
| 53 | 量化/低精度 | 论文 | DQRM: Deep Quantized Recommendation Models | 本地 PDF | 推荐模型混合精度量化 | embedding INT8/INT4 风险评估 | P1 |
| 54 | 量化/低精度 | 论文/会议 | AWQ: Activation-aware Weight Quantization | [arXiv](https://arxiv.org/abs/2306.00978) | 低比特权重量化 | transformer linear 权重量化启发 | P2 |
| 55 | 量化/低精度 | 论文 | GPTQ: Accurate Post-Training Quantization | [arXiv](https://arxiv.org/abs/2210.17323) | 二阶 PTQ | 小模型可尝试离线量化，但实现复杂 | P3 |
| 56 | 量化/低精度 | 论文 | ZeroQuant: Efficient and Affordable Post-Training Quantization | [arXiv](https://arxiv.org/abs/2206.01861) | PTQ pipeline | 线性层量化校准参考 | P3 |
| 57 | 量化/低精度 | 论文 | LLM.int8() | [arXiv](https://arxiv.org/abs/2208.07339) | outlier-aware INT8 | 识别异常通道，保护 AUC/PCOC | P2 |
| 58 | 量化/低精度 | 开源项目 | bitsandbytes | [GitHub](https://github.com/bitsandbytes-foundation/bitsandbytes) | 8-bit/4-bit linear | 依赖和 V100 兼容性需谨慎 | P3 |
| 59 | 量化/低精度 | 开源项目 | AutoAWQ | [GitHub](https://github.com/casper-hansen/AutoAWQ) | AWQ 工具链 | 可参考量化校准流程 | P3 |
| 60 | 量化/低精度 | 开源项目 | AutoGPTQ | [GitHub](https://github.com/AutoGPTQ/AutoGPTQ) | GPTQ 工具链 | 可参考离线量化/打包格式 | P3 |
| 61 | 量化/低精度 | 开源项目 | PyTorch AO / torchao | [GitHub](https://github.com/pytorch/ao) | PyTorch 官方量化实验库 | 量化 API 与 kernel 支持检查 | P2 |
| 62 | 量化/低精度 | 官方文档 | PyTorch quantization | [Docs](https://pytorch.org/docs/stable/quantization.html) | 官方量化入口 | 校准、动态量化、模块替换参考 | P2 |
| 63 | 量化/低精度 | 官方文档 | NVIDIA TensorRT INT8 calibration | [Docs](https://docs.nvidia.com/deeplearning/tensorrt/developer-guide/index.html#int8-calibration) | INT8 校准工程 | 若走 TensorRT，需校准和 engine 构建 | P3 |
| 64 | 量化/低精度 | 论文 | FP8-LM: Training FP8 Large Language Models | [arXiv](https://arxiv.org/abs/2310.18313) | FP8 低精度背景 | V100 不支持 FP8，作为未来硬件参考 | P3 |
| 65 | 量化/低精度 | 论文 | Q-BERT: Hessian Based Ultra Low Precision Quantization | [AAAI](https://ojs.aaai.org/index.php/AAAI/article/view/6409) | Hessian 敏感度量化 | 层敏感度排序参考 | P3 |
| 66 | Embedding/推荐 | 论文/会议 | DLRM: Deep Learning Recommendation Model for Personalization and Recommendation Systems | [arXiv](https://arxiv.org/abs/1906.00091) | 推荐系统 embedding + dense 结构 | 对照 GRAB embedding 带宽瓶颈 | P1 |
| 67 | Embedding/推荐 | 开源项目 | Facebook DLRM | [GitHub](https://github.com/facebookresearch/dlrm) | DLRM 参考实现 | embedding bag、benchmark、数据布局 | P2 |
| 68 | Embedding/推荐 | 开源项目 | TorchRec | [GitHub](https://github.com/pytorch/torchrec) | 推荐模型 embedding/sharding 库 | EmbeddingBagCollection 和 jagged tensor 参考 | P1 |
| 69 | Embedding/推荐 | 开源项目 | FBGEMM | [GitHub](https://github.com/pytorch/FBGEMM) | embedding bag / quantized ops | 推荐 embedding 量化和 fused op 参考 | P1 |
| 70 | Embedding/推荐 | 论文/期刊 | Embedding Compression in Recommender Systems: A Survey | 本地 `literature/08_EmbeddingCompression/` | 推荐 embedding 压缩综述 | 系统梳理 hash、low-rank、quantization | P1 |
| 71 | Embedding/推荐 | 论文/会议 | CAFE: Compact Adaptive Fast Embedding | 本地 PDF | 热/冷特征差异压缩 | 频率感知 embedding 优化 | P2 |
| 72 | Embedding/推荐 | 论文 | TT-Rec: Tensor Train Compression for Deep Learning Recommendation Models | [arXiv](https://arxiv.org/abs/2101.11714) | embedding 表张量分解 | 可能降显存，但指标和查表开销风险 | P3 |
| 73 | Embedding/推荐 | 论文 | QR Trick: Compositional Embeddings Using Complementary Partitions | [arXiv](https://arxiv.org/abs/1909.02107) | quotient-remainder embedding 压缩 | 改 embedding 参数语义，需谨慎 | P3 |
| 74 | Embedding/推荐 | 论文 | Hash Embeddings for Efficient Word Representations | [arXiv](https://arxiv.org/abs/1709.03933) | hash embedding | 对 cold id 压缩有启发 | P3 |
| 75 | Embedding/推荐 | 论文 | Compositional Embeddings Using Complementary Partitions for Memory-Efficient Recommendation Systems | [KDD](https://dl.acm.org/doi/10.1145/3292500.3330829) | 推荐系统 compositional embedding | embedding 压缩参考 | P3 |
| 76 | Embedding/推荐 | 开源项目 | NVIDIA Merlin | [GitHub](https://github.com/NVIDIA-Merlin/Merlin) | 推荐系统端到端工程 | 数据加载、特征处理、GPU 推荐栈 | P2 |
| 77 | Embedding/推荐 | 开源项目 | HugeCTR | [GitHub](https://github.com/NVIDIA-Merlin/HugeCTR) | GPU CTR/recommender 训练推理 | embedding cache/HPS/fused embedding 参考 | P1 |
| 78 | Embedding/推荐 | 文档 | NVIDIA HugeCTR Hierarchical Parameter Server | [Docs](https://nvidia-merlin.github.io/HugeCTR/main/hierarchical_parameter_server/index.html) | 热点 embedding cache | 赛题单机可借鉴 cache 形态 | P2 |
| 79 | Embedding/推荐 | 论文 | Facebook deep learning recommendation model performance characterization | [arXiv](https://arxiv.org/abs/1906.03109) | 推荐模型硬件瓶颈分析 | embedding 内存带宽与 MLP 计算对照 | P2 |
| 80 | Embedding/推荐 | 论文 | Training and Inference of Recommendation Systems on GPUs | [NVIDIA Developer](https://developer.nvidia.com/blog/accelerating-recommender-systems-with-gpus/) | 推荐系统 GPU 实践 | 工业推荐系统优化参考 | P2 |
| 81 | Embedding/推荐 | 开源项目 | NVIDIA DeepLearningExamples DLRM | [GitHub](https://github.com/NVIDIA/DeepLearningExamples/tree/master/PyTorch/Recommendation/DLRM) | DLRM CUDA Graph/AMP benchmark | 小 batch 推荐推理优化参考 | P1 |
| 82 | Embedding/推荐 | 基准 | MLPerf Inference DLRM | [MLCommons](https://mlcommons.org/benchmarks/inference-datacenter/) | 推荐模型推理基准 | 参考 datacenter 推荐模型优化目标 | P2 |
| 83 | Embedding/推荐 | 论文 | RecSys Challenge papers | [ACM RecSys](https://recsys.acm.org/) | 推荐系统会议资料 | 查找广告推荐工业方法 | P3 |
| 84 | Embedding/推荐 | 项目 | Recommenders repository | [GitHub](https://github.com/recommenders-team/recommenders) | 推荐算法工程集合 | 背景资料，不直接改 infer | P3 |
| 85 | 系统工程 | 官方博客 | Accelerating PyTorch with CUDA Graphs | [PyTorch Blog](https://pytorch.org/blog/accelerating-pytorch-with-cuda-graphs/) | 小 batch 多 kernel launch 开销 | per-shape CUDA Graph、减少同步 | P1 |
| 86 | 系统工程 | 官方文档 | `torch.cuda.CUDAGraph` | [PyTorch Docs](https://pytorch.org/docs/stable/generated/torch.cuda.CUDAGraph.html) | CUDA Graph API | 捕获静态 shape 推理片段 | P1 |
| 87 | 系统工程 | 官方文档 | Torch-TensorRT CUDA Graphs | [Docs](https://docs.pytorch.org/TensorRT/contributors/cuda_graphs.html) | TRT 与 PyTorch 混合图捕获 | 若尝试 TRT 编译，可保留 graph replay | P2 |
| 88 | 系统工程 | 官方文档 | Torch-TensorRT performance tuning | [Docs](https://docs.pytorch.org/TensorRT/user_guide/performance_tuning.html) | TensorRT 编译/延迟优化 | 判断 engine 构建成本是否适合 20 分钟限制 | P2 |
| 89 | 系统工程 | 官方博客 | Introducing nvFuser | [PyTorch Blog](https://pytorch.org/blog/introducing-nvfuser-a-deep-learning-compiler-for-pytorch/) | PyTorch fusion 背景 | elementwise/layernorm/residual fusion 思路 | P2 |
| 90 | 系统工程 | 论文/会议 | PyTorch 2: Faster Machine Learning Through Dynamic Python Bytecode Transformation | [Paper PDF](https://pytorch.org/assets/pytorch2-2.pdf) | `torch.compile` / Inductor | reduce-overhead、graph break 风险分析 | P2 |
| 91 | 系统工程 | 官方文档 | `torch.compile` | [PyTorch Docs](https://pytorch.org/docs/stable/generated/torch.compile.html) | 编译模型入口 | 可对静态模块试 `mode="reduce-overhead"` | P2 |
| 92 | 系统工程 | 论文/会议 | PyGraph: Robust Compiler Support for CUDA Graphs in PyTorch | 本地 PDF | CUDA Graph 编译支持 | 当前 Graph 路线的理论依据 | P2 |
| 93 | 系统工程 | 论文/会议 | SIRIUS: Whole-Program Optimization for DNNs | 本地 `literature/10_KernelFusion/` | 全程序算子融合 | kernel fusion 长线参考 | P2 |
| 94 | 系统工程 | 论文/会议 | Inference Optimization of Foundation Models on AI Accelerators | 本地 PDF | 推理优化综述 | 系统层优化 checklist | P2 |
| 95 | 系统工程 | 开源项目 | Triton language | [GitHub](https://github.com/triton-lang/triton) | 自定义 GPU kernel | 适合 fused layernorm/linear/SMoE 原型 | P1 |
| 96 | 系统工程 | 官方教程 | Triton matrix multiplication tutorial | [Docs](https://triton-lang.org/main/getting-started/tutorials/03-matrix-multiplication.html) | matmul kernel 入门 | grouped expert matmul 实现参考 | P2 |
| 97 | 系统工程 | 官方教程 | Triton fused softmax tutorial | [Docs](https://triton-lang.org/main/getting-started/tutorials/02-fused-softmax.html) | softmax fusion | attention/route softmax kernel 参考 | P2 |
| 98 | 系统工程 | 开源项目 | OpenAI Triton kernels examples | [GitHub](https://github.com/triton-lang/triton/tree/main/python/tutorials) | kernel 教程集合 | 快速写 fused ops 的样例库 | P2 |
| 99 | 系统工程 | 官方文档 | NVIDIA CUDA C Programming Guide | [Docs](https://docs.nvidia.com/cuda/cuda-c-programming-guide/) | CUDA kernel 基础 | 判断 V100 SM70 约束 | P2 |
| 100 | 系统工程 | 官方文档 | NVIDIA CUDA Graphs programming guide | [Docs](https://docs.nvidia.com/cuda/cuda-c-programming-guide/index.html#cuda-graphs) | CUDA Graph 底层机制 | 理解 graph capture 限制 | P2 |
| 101 | 工程项目/帖子 | 博客 | vLLM torch.compile optimization | [vLLM Blog](https://blog.vllm.ai/) | 推理图编译、kernel fusion 实战 | 借鉴 graph break 定位和融合收益表达 | P2 |
| 102 | 工程项目/帖子 | 开源项目 | vLLM | [GitHub](https://github.com/vllm-project/vllm) | 高吞吐 LLM 推理系统 | attention、paged cache、MoE kernel 参考 | P2 |
| 103 | 工程项目/帖子 | 开源项目 | TensorRT-LLM | [GitHub](https://github.com/NVIDIA/TensorRT-LLM) | NVIDIA LLM 推理优化 | fused kernels、quantization plugin 参考 | P2 |
| 104 | 工程项目/帖子 | 商业/闭源产品 | NVIDIA TensorRT | [Product](https://developer.nvidia.com/tensorrt) | 闭源推理优化引擎 | engine 构建可能超时，适合思路参考 | P3 |
| 105 | 工程项目/帖子 | 商业/闭源产品 | NVIDIA Merlin / HugeCTR enterprise stack | [NVIDIA Merlin](https://developer.nvidia.com/nvidia-merlin) | 推荐系统商业栈 | embedding cache 和 GPU CTR 实践参考 | P2 |
| 106 | 工程项目/帖子 | 博客 | NVIDIA Accelerating Recommender Systems with GPUs | [NVIDIA Blog](https://developer.nvidia.com/blog/accelerating-recommender-systems-with-gpus/) | 推荐系统 GPU 化实践 | batch、embedding、DLRM 优化参考 | P2 |
| 107 | 工程项目/帖子 | 博客 | PyTorch BetterTransformer / SDPA introduction | [PyTorch Blog](https://pytorch.org/blog/out-of-the-box-acceleration/) | attention fast path | 检查是否可用原生 SDPA | P2 |
| 108 | 工程项目/帖子 | 博客/论坛 | PyTorch Forums: custom CUDA kernel overhead with torch.compile reduce-overhead | [Forum](https://discuss.pytorch.org/t/unexplained-overhead-when-using-custom-cuda-kernel-in-torch-compile-reduce-overhead-mode/195609) | 小 kernel + compile overhead 案例 | 提醒 compile 不是自动收益，需线上 A/B | P3 |
| 109 | 工程项目/帖子 | 博客 | PyTorch 1.10 CUDA Graphs release | [PyTorch Blog](https://pytorch.org/blog/pytorch-1-10-released/) | CUDA Graph API 发布与限制 | 版本兼容性检查 | P3 |
| 110 | 工程项目/帖子 | 博客 | Accelerating Llama3 FP8 Inference with Triton Kernels | [PyTorch Blog](https://pytorch.org/blog/accelerating-llama3/) | Triton + low precision + graph | 借鉴 profiling 和 kernel fusion 方法 | P3 |
| 111 | 工程项目/帖子 | 开源项目 | FasterTransformer | [GitHub](https://github.com/NVIDIA/FasterTransformer) | Transformer 推理 kernel | 老牌 CUDA kernel 参考，集成成本高 | P3 |
| 112 | 工程项目/帖子 | 开源项目 | ONNX Runtime | [GitHub](https://github.com/microsoft/onnxruntime) | 通用推理运行时 | 导出和 engine 构建可能不适合赛题时间 | P3 |
| 113 | 工程项目/帖子 | 开源项目 | TVM | [GitHub](https://github.com/apache/tvm) | 深度学习编译器 | 全图编译参考，短期落地成本高 | P3 |
| 114 | 工程项目/帖子 | 开源项目 | XLA / OpenXLA | [Website](https://openxla.org/) | 编译优化生态 | 理论参考，不适合当前提交路径 | P3 |
| 115 | 工程项目/帖子 | 博客 | Hugging Face BetterTransformer fastpath | [Docs](https://huggingface.co/docs/optimum/bettertransformer/overview) | Transformer fast path | 对照 PyTorch SDPA/BetterTransformer 限制 | P3 |
| 116 | 工程项目/帖子 | 博客 | Hugging Face Optimum / ONNX Runtime optimization | [Docs](https://huggingface.co/docs/optimum/index) | 模型导出和推理优化 | 集成成本高，作为备选 | P3 |
| 117 | 工程项目/帖子 | 开源项目 | NVIDIA CUTLASS | [GitHub](https://github.com/NVIDIA/cutlass) | GEMM kernel 模板库 | grouped GEMM / Tensor Core kernel 参考 | P2 |
| 118 | 工程项目/帖子 | 开源项目 | cuBLASLt samples | [Docs](https://docs.nvidia.com/cuda/cublas/) | matmul heuristic / grouped GEMM 背景 | 判断是否能用 batched GEMM 替代 Python 循环 | P2 |
| 119 | 工程项目/帖子 | 工具 | NVIDIA Nsight Systems | [Product](https://developer.nvidia.com/nsight-systems) | GPU timeline profiling | 云端若能跑，定位 kernel gap 和同步 | P2 |
| 120 | 工程项目/帖子 | 工具 | NVIDIA Nsight Compute | [Product](https://developer.nvidia.com/nsight-compute) | 单 kernel profiling | 判断 Tensor Core 利用率和 memory bottleneck | P2 |

## 按优先级整理的下一步线索

| 优先级 | 方向 | 依据 | 风险 |
|---|---|---|---|
| P0 | 继续以 `V16-FP16EMB-sxyq` 为基线 | 本地成绩最佳，AUC/PCOC 稳定 | 不要把多个新变量混入同一包 |
| P1 | FP16/Embedding 带宽继续细化 | V14/V16 已验证，资料包括 GRAB、DLRM、TorchRec、FBGEMM、HugeCTR | INT8/INT4 可能因反量化和指标风险回退 |
| P1 | CUDA Graph / reduce-overhead / 静态 shape graph replay | PyTorch CUDA Graphs、PyGraph、Torch-TensorRT 文档 | 动态 shape、指针复用、graph break 会让收益消失 |
| P1 | MoE grouped GEMM / expert batching | MoE 论文、vLLM/TensorRT-LLM/Tutel | 纯 PyTorch indexing 已在 V22 证明容易变慢 |
| P2 | FlashAttention-2 / SDPA 等价替换 | FlashAttention-1/2、PyTorch SDPA | 本地 V6 表明 attention 不是第一热点，需谨慎排优先级 |
| P2 | SmoothQuant / DQRM / AWQ | 量化论文和工具链 | V100 INT8/INT4 支持、校准成本、PCOC 波动 |
| P3 | 稀疏 attention、Top-1 专家、结构性 pruning | Loki、SeerAttention、Switch Transformer 等 | 可能改变模型语义，不能作为默认主线 |

