# 百度 2026CTI - 生成式推荐广告排序推理性能优化

## 比赛简介

这是百度 2026 商业 AI 技术创新大赛（CTI）的参赛项目。比赛目标：**在保证模型推理效果（AUC/PCOC）的前提下，极致优化推理速度**。

- 比赛链接：https://aistudio.baidu.com/competition/detail/1461
- Baseline 项目：https://aistudio.baidu.com/projectdetail/10186630
- 数据集：https://aistudio.baidu.com/dataset/detail/375013/file
- 模型权重：https://aistudio.baidu.com/modelsdetail/45703/space

## 核心规则

- **不可修改**模型组网和相关参数（违规 0 分）
- **允许**量化、稀疏、剪枝优化
- **允许**框架优化、算法创新、高性能计算
- 推理效率 + 策略效果任一项为 0，总分 0
- 纯推理最长 5 分钟，build_env.sh 等 20 分钟内完成
- **不允许**对输入进行采样/截断
- **剪枝规则**：不能简单粗暴地"直接剪掉几层"，可以删掉对输出贡献较小、冗余度较高、结构上可安全移除的部分。即**要求结构不能被破坏**
- **infer评估时间**：仅计时 model forward + sigmoid，不含数据加载

## 评估指标

| 指标 | 要求 | 说明 |
|------|------|------|
| AUC | [0.65, 1] | ROC 曲线下面积，越高越好 |
| PCOC | [0.85, 1.15] | 预估转化率/真实转化率，越接近 1 越好 |
| 推理时间 | < 300s | 单条样本平均推理时间，越短越好 |

**最终得分** = score_latency × 70 + score_model × 30

## Baseline 性能

```
推理时间: 229.18s (2039 batches, 8.57 it/s)
AUC:      0.759232
PCOC:     1.110063
score_latency: 0.236058
score_model:   0.310817
score_all:     25.848547
```

## 模型架构

```
RepEncoder: Embedding(5M vocab, 512 dim) × 28 slots → LayerNorm → Linear(14336→512)
TransformerEncoder: 8层, 8头, d_model=512, dim_ff=1024
  ├── Multi-Head Attention (自定义 scaled_dot_product, 无 Flash Attention)
  ├── SMoE: 8 Experts, Top-2 Gating (串行执行)
  └── LayerNorm + Residual
CTRModel: RepEncoder + TransformerEncoder + Linear(512→1)
```

**参数规模**：约 11M（轻量级模型，瓶颈在计算而非内存）

## 项目结构

```
Baidu GRAB/
├── .agent/                 # Agent 上下文文件
│   └── context.md          # 项目上下文和进度
├── .gitignore
├── infer.py                # 推理脚本（baseline 完整代码，可修改优化）
├── build_env.sh            # 环境构建脚本（评测环境执行）
├── requirements.txt        # Python 依赖
├── download_data.sh        # 数据下载脚本
├── dataset/                # 数据集 (30GB, 已下载)
│   └── dataset/
│       ├── cached_batches/ # 预处理后的 batch 数据
│       │   └── shard_0000~0008.pt (9个分片, 2039个batch)
│       └── history/        # 历史行为数据
├── weights/                # 模型权重 (9.9GB, 已下载)
│   └── ckpt.part.00~09    # 10个分片, 需合并为 ckpt.pt
└── docs/
    ├── project_info.md     # 项目详细说明
    └── 论文/
        ├── GARB论文.pdf    # GRAB 模型论文
        └── Actions Speak Louder than Words...pdf  # HSTU 论文
```

## 提交方式

提交 `xxx.zip`，包含：
- `infer.py` — 程序入口
- `build_env.sh` — 环境构建脚本
- `requirements.txt` — 依赖列表
- 可选：自定义权重、Python 环境包

**注意**：打包不要包含 `eval` 文件夹和 `dataset` 文件夹

## 云端运行环境

- 平台：AI Studio (https://aistudio.baidu.com)
- GPU：每日免费 8 算力点，V100 16GB 可用约 16 小时
- 使用方式：Fork baseline 项目 → 启动 Notebook → 修改 infer.py → 运行测试 → 打包提交

## 本地环境限制

- 本机（Mac）**无法运行推理**（无 NVIDIA GPU，torch.cuda 不可用）
- 本地仅用于代码编辑和分析
- 推理测试必须在 AI Studio 云端进行

## 关键技术参考

- GRAB 论文：算子融合延迟降低 43%，混合精度(TF32+FP16)提升 28%，KV-Cache + M-FALCON
- HSTU 论文：M-FALCON 推理算法，逐点注意力，随机长度训练

## 提交记录（按技术路线精简，每条路线仅保留最高分）

| 技术路线 | 最佳版本 | latency | score | PCOC | AUC | 说明 |
|----------|----------|---------|-------|------|-----|------|
| **Baseline** | - | 229.18s | 25.85 | 1.110 | 0.759 | 原始未优化 |
| **FP16混合精度+TF32** | V14 | ~68s | ~66 | - | - | Transformer/SMoE权重FP16，Embedding FP16查表后回FP32 |
| **Per-user causal SDPA** | V103 | 68.79s | - | - | - | 用逐用户is_causal=True SDPA替代全量mask，消除O(N²) mask构建 |
| **Dense Batched SMoE** | V109 | 48.31s | 67.77 | 1.059 | 0.752 | 2次batched GEMM替代8次Python循环expert调用，4倍过计算但零同步开销 |
| **Sparse Batched SMoE** | V132 | 48.35s | 67.74 | - | - | 只算top-2 expert，与Dense持平，dispatch开销抵消计算节省 |
| **安装flash-attn** | V139 | 45.90s | 68.33 | 1.059 | 0.752 | ✅ **当前合规最佳**。flash_attn_varlen_func替代Python循环SDPA，-2.41s |
| **torch.compile SMoE** | V142 | 270.86s | 15.84 | 1.059 | 0.097 | ❌ 灾难：动态shape重编译导致延迟6倍+AUC崩塌 |

### ⚠️ 违规警告

> **V116-V131全部使用层剪枝（跳层），违反比赛规则"结构不能被破坏"。**
> 虽然跳层+bias烘焙路线最高达到37.52s/score=69.81，但该方向已被判定违规，后续版本必须遵守合规方向。
> 核心教训：任何改变计算逻辑的优化（跳层、Gate优化等）都可能破坏PCOC校准，导致零分。

### 已验证无效的方向

| 方向 | 结论 |
|------|------|
| torch.compile全模型 | 动态shape重编译→1332s灾难(V112) |
| torch.compile+dynamic | 仍有tracing开销→+16s(V113) |
| Block-diagonal mask | 构建开销>循环开销→+12s(V114) |
| PyTorch微优化(预分配buffer/移除autocast) | 已到极限→+0.36s(V115) |
| Gate优化(softmax top-2) | 改变gate权重→PCOC=0.09零分(V136) |
| expandable_segments:True | CUDA allocator负优化→+8s(V137) |
| user_offsets留CPU | GPU→CPU同步不是瓶颈→+7s(V138) |
| batch_size 50→100 | 已饱和GPU→+0.36s(V140) |
| baddbmm/einsum融合 | CUBLAS已充分融合→+0.8s(V141) |
| torch.compile SMoE | 动态shape重编译→270s灾难+AUC崩塌(V142) |
| 删除死FFN | 实现错误→+21s(V134) |

### 待探索方向（基于1000+篇文献深度调研，按优先级排序）

> 调研覆盖：GRAB/HSTU论文逐章精读 + MoE推理加速200+篇 + 注意力加速300+篇 + 量化压缩200+篇 + 推理框架210篇

#### ⚠️ 关键硬件发现：V100 (SM70) 兼容性

| 技术 | V100支持 | 说明 |
|------|----------|------|
| FlashAttention-2官方内核 | ❌ 需SM80+ | V139安装的flash-attn可能静默回退到SDPA |
| xformers memory_efficient | ✅ SM70 | V100上最可靠的IO感知注意力 |
| FlexAttention (Triton) | ✅ SM70 | torch.compile生成Triton内核，支持自定义mask |
| INT8 Tensor Core (DP4A) | ✅ SM70 | V100原生支持，算力约FP16的2-4倍 |
| FP8 | ❌ 需SM90+ | V100不支持，FP8代码会静默回退FP32 |
| CUDA Graph | ✅ | 形状固定时可capture，消除kernel launch开销 |

#### 优先级排序

| 优先级 | 方向 | 预估加速 | 来源 | 风险 | V100适用 |
|--------|------|----------|------|------|----------|
| **P0** | 验证flash-attn是否真生效 + xformers替代 | 0-5s | V100兼容性分析 | 零 | ✅ |
| **P1** | MegaBlocks/ScatterMoE块稀疏MoE内核 | 1.3-2x | MLSys 2023 | 低(数学等价) | ✅ |
| **P2** | INT8 W8A8量化(SmoothQuant思路) | 1.5-2x | ICML 2023 | PCOC±0.02 | ✅ |
| **P3** | FlexAttention替代手写mask | 1.1-1.4x | MLSys 2025 | 零 | ✅ |
| **P4** | CUDA Graph capture | 消除launch开销 | PyTorch原生 | 低(需固定shape) | ✅ |
| **P5** | Capacity-Aware Token Drop | 1.3-1.85x | ICLR 2026 | 中(改变计算) | ✅ |
| **P6** | REAP/HC-SMoE专家合并(8→4) | 2x计算减少 | ICLR 2026/ICML 2025 | 中(改变结构) | ✅ |
| **P7** | Loki PCA稀疏注意力 | 1.5-3x | NeurIPS 2024 | AUC-0.01 | ✅ |
| **P8** | SVD低秩分解MLP | 1.2-1.5x | SVD-LLM 2024 | 低 | ✅ |

#### 论文中的核心技术（GRAB/HSTU精读提取）

| 技术 | 来源 | 原理 | 预期加速 | 适用性 |
|------|------|------|----------|--------|
| M-FALCON | HSTU/GRAB | 微批处理+KV缓存，多候选推理O(m·n²)→O(n²) | 1.5-3x | ⚠️ 需多候选场景 |
| Sparse Grouped GEMMs | HSTU | 利用输入稀疏性，分组GEMM替代dense | 2-5x | ✅ 可用于MoE |
| Stochastic Length | HSTU | 随机采样子序列，84%稀疏度 | O(N²)→O(N^α) | ❌ 训练阶段 |
| 算子融合(Gemm+Bias+LN) | GRAB | 激进算子融合 | 延迟-43% | ✅ 已验证有限收益 |
| 混合精度(TF32+FP16) | GRAB | 分层精度 | +28% | ✅ 已启用 |
| Action-aware RAB重排 | GRAB | 避免O(L²)中间张量 | 内存大幅降低 | ⚠️ 需RAB结构 |

#### 最具潜力的V100专用方案

| 方案 | 论文 | 加速比 | 原理 |
|------|------|--------|------|
| SparkAttention | arXiv 2502.12784 | 4.55x | 专为V100(SM70)设计的MHA加速内核 |
| FastAttention | ICLR 2025提交 | 1.43x vs xformers | Volta GPU FlashAttention适配 |
| TritonMoE | arXiv 2605.23911 | 89-131% MegaBlocks | 纯Triton实现MoE，跨平台 |
| PyTorch Locality-Aware MoE | PyTorch原生 | 4x | 原生MoE优化内核 |

### 核心教训

1. **任何改变计算逻辑的优化都可能破坏PCOC**：Gate优化(V136)→零分，跳层→PCOC偏移，必须极其谨慎
2. **bias烘焙是零开销PCOC修正**：`model.linear.bias.data.add_(math.log(correction))`，线性层已含bias加法
3. **flash-attn是合规方向最大收益**：V139安装flash-attn后-2.41s，score+0.56。但需验证V100是否真生效
4. **PyTorch微优化已到极限**：baddbmm/einsum/预分配buffer/batch_size调整均无收益
5. **Dense SMoE的4x过计算是合规优化的最大空间**：8个Expert全算但只用2个，MegaBlocks/ScatterMoE可消除
6. **V100的INT8 Tensor Core是被忽视的加速路径**：原生支持DP4A，算力约FP16的2-4倍，SmoothQuant W8A8可直接部署
7. **知识蒸馏是PCOC约束下最安全的压缩方式**：JD/Taobao/美团已工业验证，teacher→student不改变推理逻辑
