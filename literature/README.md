# 文献库索引

> 百度 2026CTI 生成式推荐广告排序推理性能优化
> 共 23 篇论文 + 博文/行业报道 + 本地知识方案，按预期效果分为四个梯队

---

## 文档结构

| 文档 | 定位 | 预期加速 | AUC 风险 | 核心内容 |
|------|------|----------|---------|---------|
| [01_Tier1_核心必做优化](01_Tier1_核心必做优化.md) | 零风险高收益 | 4-6x | 零 | FP16、FlashAttention-2、SMoE向量化、torch.compile、CUDA Graph |
| [02_Tier2_量化与压缩优化](02_Tier2_量化与压缩优化.md) | 低风险中收益 | ×1.5-2.5x | 低 | SmoothQuant、INT-FA、DQRM、CAFE、Embedding压缩 |
| [03_Tier3_算法创新优化](03_Tier3_算法创新优化.md) | 中风险高收益 | ×1.5-3x | 中-高 | Loki稀疏注意力、LExI、KV缓存、EARN、SeerAttention |
| [04_Tier4_辅助与探索优化](04_Tier4_辅助与探索优化.md) | 零风险微收益 | ×1.1-1.3x | 零 | 预取、channels_last、低秩补偿、蒸馏、博文精华 |

---

## 总体预期

```
Baseline: 229.18s (AUC=0.759, PCOC=1.110)

Tier 1 (必做):   4-6x    → 38-57s   AUC≈0.759 (零损失)
Tier 2 (推荐):   ×1.5-2.5x → 15-38s  AUC≈0.750 (轻微下降)
Tier 3 (进阶):   ×1.5-3x   → 5-25s   AUC≈0.730 (需验证)
Tier 4 (锦上添花): ×1.1-1.3x → 4-23s  AUC≈0.730 (零额外损失)

保守总加速: 8-12x → 推理时间 19-29s
乐观总加速: 15-25x → 推理时间 9-15s
```

---

## 论文-梯队映射

| 论文 # | 论文简称 | 所属梯队 | PDF 位置 |
|--------|---------|---------|---------|
| #2 | FlashAttention-2 ⭐ | Tier 1 | `01_FlashAttention/FlashAttention-2.pdf` |
| #7 | MoE Expert向量化 ⭐ | Tier 1 | `03_MoE_Optimization/Toward_Efficient_Inference_for_MoE.pdf` |
| #11 | PyGraph CUDA Graph | Tier 1 | `04_InferenceOptimization/PyGraph.pdf` |
| #21 | SIRIUS 算子融合 | Tier 1 | `10_KernelFusion/SIRIUS.pdf` |
| #12 | SmoothQuant ⭐ | Tier 2 | `06_Quantization/SmoothQuant.pdf` |
| #13 | INT-FlashAttention | Tier 2 | `06_Quantization/INT-FlashAttention.pdf` |
| #14 | DQRM Embedding量化 | Tier 2 | `06_Quantization/DQRM.pdf` |
| #18 | CAFE 热/冷压缩 | Tier 2 | `08_EmbeddingCompression/CAFE.pdf` |
| #19 | Embedding压缩综述 | Tier 2 | `08_EmbeddingCompression/Embedding_Compression_Survey.pdf` |
| #17 | Loki PCA稀疏注意力 | Tier 3 | `07_SparseAttention/Loki.pdf` |
| #15 | Dynamic Sparse FA | Tier 3 | `07_SparseAttention/Dynamic_Sparse_Flash_Attention.pdf` |
| #16 | SeerAttention | Tier 3 | `07_SparseAttention/SeerAttention.pdf` |
| #9 | LExI 层自适应Top-K | Tier 3 | `03_MoE_Optimization/LExI.pdf` |
| #4 | HSTU KV缓存复用 | Tier 3 | `02_GenerativeRecommendation/HSTU.pdf` |
| #6 | EARN Register Token | Tier 3 | `02_GenerativeRecommendation/EARN.pdf` |
| #5 | Jagged Attention | Tier 3 | `02_GenerativeRecommendation/Scaling_GR_Context_Parallelism.pdf` |
| #1 | FlashAttention-1 | Tier 4 | `01_FlashAttention/FlashAttention-1.pdf` |
| #3 | FlashAttention-3 | Tier 4 | `01_FlashAttention/FlashAttention-3.pdf` |
| #8 | MiLo 低秩补偿 | Tier 4 | `03_MoE_Optimization/MiLo.pdf` |
| #10 | 推理优化综述 | Tier 4 | `04_InferenceOptimization/Inference_Optimization_Survey.pdf` |
| #20 | HetComp 知识蒸馏 | Tier 4 | `09_KnowledgeDistillation/HetComp.pdf` |
| #22 | PreScope 预取 | Tier 4 | `11_DataPrefetch/PreScope.pdf` |
| #23 | KV Cache预取 | Tier 4 | `11_DataPrefetch/KV_Cache_Prefetching.pdf` |

---

## 博文与行业报道索引

| 来源 | 核心内容 | 对应梯队 |
|------|---------|---------|
| iQiyi CTR GPU优化 | 算子融合 kernel launch 数百→<10 | Tier 1 |
| JD 广告稀疏模型 | CPU-GPU异构流水线，Embedding I/O >30% | Tier 2/4 |
| TensorRT 广告CTR | Layer Fusion + INT8，30ms→6-8ms | Tier 2 |
| PyTorch 官方博客 | torch.compile + CUDA Graph 2.7x | Tier 1 |
| vLLM torch.compile | Attention+Quant融合7%，AllReduce+RMSNorm融合15% | Tier 1/4 |
| CSDN 41技巧 | torch.compile 1.2-2.5x，FP16+channels_last 1.4x | Tier 1/4 |

---

## 建议阅读顺序

1. **[01_Tier1_核心必做优化](01_Tier1_核心必做优化.md)** — 立即实施，4-6x 加速
2. **[02_Tier2_量化与压缩优化](02_Tier2_量化与压缩优化.md)** — Tier 1 完成后实施
3. **[03_Tier3_算法创新优化](03_Tier3_算法创新优化.md)** — 指标有余量时尝试
4. **[04_Tier4_辅助与探索优化](04_Tier4_辅助与探索优化.md)** — 锦上添花
