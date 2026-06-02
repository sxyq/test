# 项目上下文 - Agent 快速参考

> 本文件为无上下文的 Agent 提供项目全貌，请始终与本文件保持同步更新。

## 项目身份

- **名称**：百度 2026CTI 生成式推荐广告排序推理性能优化
- **类型**：AI 竞赛（推理优化赛道，非训练赛道）
- **目标**：在保证 AUC≥0.65 且 PCOC∈[0.85,1.15] 的前提下，最小化推理时间
- **平台**：AI Studio (https://aistudio.baidu.com)
- **本地路径**：`/Users/sunyiyang/Desktop/Project/Baidu  GRAB`

## 当前进度

### 已完成
- [x] 本地 Git 初始化
- [x] Baseline 代码获取（infer.py 完整源码，25KB）
- [x] 论文阅读（GRAB 论文 + HSTU 论文 + 23篇优化论文）
- [x] 文献库搭建（23篇论文 + 博文/行业报道 + 本地知识方案，按四梯队分层）
- [x] 项目结构搭建（README.md, .agent/, .gitignore 等）

### 待完成
- [ ] 实现 infer.py 推理优化（核心任务）
- [ ] 在 AI Studio 云端测试优化效果
- [ ] 打包提交至比赛平台

## 模型架构速查

```
输入: 28个 slot 的稀疏特征 (每个 slot 含 sign IDs)
  ↓
RepEncoder: Embedding(5M, 512) × 28 → segment_reduce sum → concat(14336) → LayerNorm → Linear(14336→512)
  ↓
TransformerEncoder × 8层:
  ├── LayerNorm → QKV Projection → 自定义 scaled_dot_product (无 Flash Attention)
  ├── Output Projection + Residual
  ├── LayerNorm → SMoE (8 Experts, Top-2 Gating, 串行执行)
  └── Residual
  ↓
Linear(512→1) → sigmoid → CTR 预测
```

**关键参数**：vocab_size=5M, emb_dim=512, slot_num=28, d_model=512, n_heads=8, num_layers=8, dim_ff=1024, num_experts=8, top_k=2

## Baseline 性能基线

```
推理时间: 229.18s (2039 batches, 8.57 it/s)
AUC:      0.759232
PCOC:     1.110063
score_latency: 0.236058
score_model:   0.310817
score_all:     25.848547
```

## 推理瓶颈分析

1. **自定义 scaled_dot_product** — 未使用 Flash Attention，手动计算 QKV，O(n²) 显存
2. **SMoE 串行执行** — 每层 for 循环遍历 8 个 Expert，无法并行
3. **全 FP32 推理** — 未使用半精度，计算量翻倍
4. **逐 batch 串行** — 2039 个 batch 逐个处理，无 overlap
5. **无 torch.compile** — 未利用 PyTorch 2.x 编译优化
6. **RepEncoder 逐 slot 循环** — 28 个 slot 逐个 segment_reduce

## 文献库索引（按预期效果四梯队分层）

> 23 篇论文 + 博文/行业报道 + 本地知识方案，详见 `literature/` 目录

| 梯队 | 文档 | 定位 | 预期加速 | AUC 风险 | 核心内容 |
|------|------|------|----------|---------|---------|
| Tier 1 | `literature/01_Tier1_核心必做优化.md` | 零风险高收益 | 4-6x | 零 | FP16、FlashAttention-2 varlen、SMoE向量化、torch.compile、CUDA Graph |
| Tier 2 | `literature/02_Tier2_量化与压缩优化.md` | 低风险中收益 | ×1.5-2.5x | 低 | SmoothQuant W8A8、INT-FlashAttention、DQRM Embedding INT4、CAFE热/冷压缩 |
| Tier 3 | `literature/03_Tier3_算法创新优化.md` | 中风险高收益 | ×1.5-3x | 中-高 | Loki PCA稀疏注意力、LExI层自适应Top-K、KV缓存复用、EARN序列压缩、SeerAttention |
| Tier 4 | `literature/04_Tier4_辅助与探索优化.md` | 零风险微收益 | ×1.1-1.3x | 零 | CPU-GPU双缓冲预取、channels_last、MiLo低秩补偿、知识蒸馏、博文精华 |

### 总体预期

```
Baseline: 229.18s (AUC=0.759, PCOC=1.110)
Tier 1 (必做):     4-6x      → 38-57s   AUC≈0.759 (零损失)
Tier 2 (推荐):     ×1.5-2.5x → 15-38s   AUC≈0.750 (轻微下降)
Tier 3 (进阶):     ×1.5-3x   → 5-25s    AUC≈0.730 (需验证)
Tier 4 (锦上添花): ×1.1-1.3x → 4-23s    AUC≈0.730 (零额外损失)
保守总加速: 8-12x → 推理时间 19-29s
乐观总加速: 15-25x → 推理时间 9-15s
```

### 论文-梯队映射

| 梯队 | 论文 | PDF 位置 |
|------|------|---------|
| **Tier 1** | #2 FlashAttention-2 ⭐ | `01_FlashAttention/FlashAttention-2.pdf` |
| **Tier 1** | #7 MoE Expert向量化 ⭐ | `03_MoE_Optimization/Toward_Efficient_Inference_for_MoE.pdf` |
| **Tier 1** | #11 PyGraph CUDA Graph | `04_InferenceOptimization/PyGraph.pdf` |
| **Tier 1** | #21 SIRIUS 算子融合 | `10_KernelFusion/SIRIUS.pdf` |
| **Tier 2** | #12 SmoothQuant ⭐ | `06_Quantization/SmoothQuant.pdf` |
| **Tier 2** | #13 INT-FlashAttention | `06_Quantization/INT-FlashAttention.pdf` |
| **Tier 2** | #14 DQRM Embedding量化 | `06_Quantization/DQRM.pdf` |
| **Tier 2** | #18 CAFE 热/冷压缩 | `08_EmbeddingCompression/CAFE.pdf` |
| **Tier 2** | #19 Embedding压缩综述 | `08_EmbeddingCompression/Embedding_Compression_Survey.pdf` |
| **Tier 3** | #17 Loki PCA稀疏注意力 | `07_SparseAttention/Loki.pdf` |
| **Tier 3** | #15 Dynamic Sparse FA | `07_SparseAttention/Dynamic_Sparse_Flash_Attention.pdf` |
| **Tier 3** | #16 SeerAttention | `07_SparseAttention/SeerAttention.pdf` |
| **Tier 3** | #9 LExI 层自适应Top-K | `03_MoE_Optimization/LExI.pdf` |
| **Tier 3** | #4 HSTU KV缓存复用 | `02_GenerativeRecommendation/HSTU.pdf` |
| **Tier 3** | #6 EARN Register Token | `02_GenerativeRecommendation/EARN.pdf` |
| **Tier 3** | #5 Jagged Attention | `02_GenerativeRecommendation/Scaling_GR_Context_Parallelism.pdf` |
| **Tier 4** | #1 FlashAttention-1 | `01_FlashAttention/FlashAttention-1.pdf` |
| **Tier 4** | #3 FlashAttention-3 | `01_FlashAttention/FlashAttention-3.pdf` |
| **Tier 4** | #8 MiLo 低秩补偿 | `03_MoE_Optimization/MiLo.pdf` |
| **Tier 4** | #10 推理优化综述 | `04_InferenceOptimization/Inference_Optimization_Survey.pdf` |
| **Tier 4** | #20 HetComp 知识蒸馏 | `09_KnowledgeDistillation/HetComp.pdf` |
| **Tier 4** | #22 PreScope 预取 | `11_DataPrefetch/PreScope.pdf` |
| **Tier 4** | #23 KV Cache预取 | `11_DataPrefetch/KV_Cache_Prefetching.pdf` |

### 博文与行业报道精华

| 来源 | 核心内容 | 对应梯队 |
|------|---------|---------|
| iQiyi CTR GPU优化 | 算子融合 kernel launch 数百→<10 | Tier 1 |
| JD 广告稀疏模型 | CPU-GPU异构流水线，Embedding I/O >30% | Tier 2/4 |
| TensorRT 广告CTR | Layer Fusion + INT8，30ms→6-8ms | Tier 2 |
| PyTorch 官方博客 | torch.compile + CUDA Graph 2.7x | Tier 1 |
| vLLM torch.compile | Attention+Quant融合7%，AllReduce+RMSNorm融合15% | Tier 1/4 |
| CSDN 41技巧 | torch.compile 1.2-2.5x，FP16+channels_last 1.4x | Tier 1/4 |

## 关键约束

- **本机无法运行推理**（Mac 无 NVIDIA GPU）
- **本地不保留模型权重和数据集**（已删除，节省 40GB 磁盘）
- 推理测试必须在 AI Studio 云端进行（V100 GPU）
- 提交格式：`xxx.zip`，包含 `infer.py` + `build_env.sh` + `requirements.txt`
- 不可修改模型组网和参数（量化/稀疏/剪枝除外）
- 纯推理最长 5 分钟，build_env.sh 等 20 分钟内

## .agent 文件清单

| 文件 | 说明 |
|------|------|
| `.agent/context.md` | 本文件 — 项目上下文和进度 |
| `.agent/papers.md` | 两篇参考论文的详细信息（GRAB + HSTU） |

## 项目文件说明

| 文件 | 说明 |
|------|------|
| `infer.py` | 推理脚本（可修改优化，这是主要工作文件） |
| `build_env.sh` | 环境构建脚本（评测环境执行，需 20 分钟内完成） |
| `requirements.txt` | Python 依赖列表 |
| `weights/` | 模型权重（**本地已删除，仅云端可用**） |
| `dataset/` | 数据集（**本地已删除，仅云端可用**） |
| `docs/论文/` | GRAB 论文 + HSTU 论文 |
| `literature/` | 23篇优化论文 + 四梯队索引文档 |

## AI Studio 访问

- Token: 使用本地环境变量或 AI Studio 登录态管理，不写入仓库（可用于 aistudio-sdk 下载模型/数据集）
- 项目代码（infer.py 等）无法通过 API 获取，只能通过浏览器访问
- 云端环境：V100 16GB GPU，每日 8 算力点
