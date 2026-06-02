# 第四梯队：辅助与探索优化

> **预期额外加速：1.1-1.3x** | **AUC 风险：零-低** | **边际收益但零风险**
>
> 这些优化单独收益较小，但几乎不影响模型精度，且与前三梯队正交互补。
> 适合作为"锦上添花"的最后一步优化。

---

## 1. CPU-GPU 双缓冲预取

| 项目 | 说明 |
|------|------|
| 预期加速 | 1.05-1.15x |
| AUC 风险 | 零 |
| 实施难度 | 低-中 |
| 来源论文 | #22 PreScope + #23 KV Cache Prefetching |
| PDF | `11_DataPrefetch/PreScope.pdf`, `11_DataPrefetch/KV_Cache_Prefetching.pdf` |

### 核心思路

在 GPU 计算当前 batch 时，CPU 异步准备下一个 batch 的数据，overlap 计算与数据传输。

### 关键技术细节

**来自 PreScope (#22)**：
- LLaPor 预测器：2 层 MLP（隐藏层 64 维），输入当前层 gate logits，输出下一层 Expert 激活概率
- 异步预取：双 CUDA Stream，计算与预取 overlap
- Cross-Layer Scheduling：全局优化预取调度

**来自 KV Cache Prefetching (#23)**：
- L2 Cache 预取：`cudaPrefetchAsync` 指令，在当前 token 计算时预取下一步 KV Cache
- 双 Stream 架构：Stream Compute + Stream Prefetch 并行
- 自适应预取窗口：短序列 W=1 page，长序列 W=4-8 pages

### 本项目实施要点

```python
# CPU-GPU 双缓冲
stream_compute = torch.cuda.Stream()
stream_prefetch = torch.cuda.Stream()

for i, batch in enumerate(dataloader):
    # 在 prefetch stream 上异步传输下一个 batch
    if i + 1 < len(dataloader):
        with torch.cuda.stream(stream_prefetch):
            next_batch = next_batch.to('cuda', non_blocking=True)

    # 在 compute stream 上计算当前 batch
    with torch.cuda.stream(stream_compute):
        output = model(batch)

    torch.cuda.current_stream().wait_stream(stream_compute)
```

- Expert 较小（8.4M/层），全部在 GPU 无需 offload
- 可借鉴 Gate 提前计算 + Expert 权重 L2 预取思路
- Embedding Lookup L2 预取、CPU-GPU 双缓冲数据传输

### 博文/实战佐证

- **JD 广告稀疏模型**：CPU-GPU 异构流水线，Embedding 查表在 CPU，模型计算在 GPU，双缓冲 overlap 通信
- **TensorRT**：内置异步数据传输和双缓冲机制

### 本地知识补充

- **PyTorch DataLoader**：`DataLoader(..., pin_memory=True, non_blocking=True)` 已内置 CPU-GPU 异步传输
- **CUDA Stream**：双 Stream 架构需注意同步点，避免 race condition
- **实际收益**：如果数据加载不是瓶颈（GPU 计算时间远大于数据传输），双缓冲收益有限

---

## 2. FlashAttention-1 与 FlashAttention-3 参考

### FlashAttention-1 (#1, NeurIPS 2022)

| 项目 | 说明 |
|------|------|
| 预期加速 | 1.5-2.0x（V100 上，已被 FA2 取代） |
| PDF | `01_FlashAttention/FlashAttention-1.pdf` |

**核心创新**：IO 感知 tiling + online softmax，精确注意力 HBM 写入从 O(N²) 降至 O(N)

**关键技术细节**：
- Tiling 参数：block size B_r = B_c = ceil(M / 4d)，本项目 d=64 时约 96 tokens/block
- Online Softmax：增量维护行最大值 m_i 和分母 l_i
- IO 复杂度：标准注意力 O(Nd + N²) → FlashAttention O(N²d²/M)，本项目约 14x IO 减少

**本项目适用性**：
- causal=True 支持因果 mask
- 本项目是分段块对角因果 mask（`user_offsets`），FA1 的全局下三角 mask 不完全匹配
- **已被 FlashAttention-2 取代**，FA2 的 varlen 接口完美匹配

### FlashAttention-3 (#3, NeurIPS 2024)

| 项目 | 说明 |
|------|------|
| 预期加速 | V100 上不可直接使用 |
| PDF | `01_FlashAttention/FlashAttention-3.pdf` |

**核心创新**：H100 GPU 三重异步流水线（WGMMA + TMA）+ FP8 低精度

**关键技术细节**：
- 三阶段流水线：TMA 加载 Q/K → WGMMA 异步 GEMM → Thread softmax
- FP8 混合精度：QK^T 在 FP8 → softmax 在 FP32 → PV 在 FP8
- Per-block scaling：对 Q/K 做逐块缩放避免 FP8 溢出

**本项目适用性**：
- V100 (SM70) 不支持 WGMMA、TMA、FP8
- 可借鉴 cp.async 预取思路（1.1-1.2x 额外加速）

---

## 3. MiLo 低秩补偿器

| 项目 | 说明 |
|------|------|
| 预期加速 | 配合 INT8 量化使用 |
| AUC 风险 | 低（仅补偿量化误差） |
| 来源论文 | #8 MiLo (MLSys 2025) |
| PDF | `03_MoE_Optimization/MiLo.pdf` |

### 核心创新

MoE 3-bit 量化 + 低秩补偿器混合（Mixture of Low-Rank Compensators）

### 关键技术细节

- **3-bit 量化**：per-group 量化（group_size=128），W3A16 方案
- **低秩补偿**：对量化误差矩阵做 SVD，取前 r 个奇异值（r=4-16），推理时加回
- **Mixture 机制**：每个 Expert 维护 K 个低秩补偿器（K=2-4），根据 routing decision 选择
- **融合 kernel**：3-bit 权重运行时解量化为 FP16，与 GEMM 融合，低秩补偿开销 <3%

### 本项目适用性

- V100 不支持 INT8 Tensor Core GEMM（需 Turing+），3-bit 更无硬件支持
- **低秩补偿思路可用于 INT8 量化后的精度恢复**
- AUC/PCOC 约束下量化风险较高

### 本地知识补充

- **低秩补偿的通用价值**：不仅限于 MoE，任何量化后的精度损失都可用低秩补偿恢复
- **实现方式**：量化后收集误差矩阵 `E = W - dequantize(quantize(W))`，对 E 做 SVD 取前 r 个奇异值，推理时加回
- **与 SmoothQuant 配合**：SmoothQuant W8A8 后如果 AUC 下降，可用低秩补偿恢复

---

## 4. HetComp 知识蒸馏

| 项目 | 说明 |
|------|------|
| 预期加速 | 1.5-2x（需重新训练） |
| AUC 风险 | 高 |
| 来源论文 | #20 HetComp (WWW 2023) |
| PDF | `09_KnowledgeDistillation/HetComp.pdf` |

### 核心创新

异构模型蒸馏 + Easy-to-Hard Curriculum + Adaptive Knowledge Transfer

### 关键技术细节

- **Multi-Expert Distillation**：Attention-based Fusion 动态决定各 Teacher 权重
- **Feature Alignment Module**：投影矩阵对齐异构中间表示
- **Adaptive Sample Weighting**：按 Teacher 置信度分配样本权重
- **Curriculum Distillation**：Temperature 从高到低退火

### 本项目适用性

- 推理优化赛道不可修改模型组网，蒸馏需训练新模型
- 启发价值：SMoE Expert 蒸馏（8→4）、Embedding 蒸馏（512→256）

### 本地知识补充

- **比赛规则考量**：如果允许训练新模型，蒸馏是最有效的模型压缩手段
- **简化蒸馏方案**：直接用原始模型做 Teacher，训练一个更小的 Student（少层/少 Expert/小 Embedding）
- **Expert 蒸馏**：8 Expert → 4 Expert，每个新 Expert 是 2 个旧 Expert 的加权平均

---

## 5. 推理优化全景参考

| 项目 | 说明 |
|------|------|
| 来源论文 | #10 Inference Optimization of Foundation Models (KDD 2024) |
| PDF | `04_InferenceOptimization/Inference_Optimization_Survey.pdf` |

### 核心内容

推理优化全栈技术体系（算法层→系统层→硬件层）+ 跨层协同优化框架

### 关键洞察

- **推荐系统模型内存带宽（而非计算）通常是瓶颈**
- 算法层：PTQ(GPTQ/AWQ/SmoothQuant)、KV Cache 优化(PagedAttention)、稀疏注意力
- 系统层：Continuous Batching、Speculative Decoding、Prefix Caching
- 硬件层：FlashAttention、Tensor Core 维度对齐（FP16 对齐 8，INT8 对齐 16）

### 本地知识补充

- **带宽 vs 计算**：本项目 Embedding(5M, 512) 约 10.24GB，每次推理需读取大量 Embedding，带宽确实是瓶颈
- **Tensor Core 对齐**：FP16 对齐 8（d_model=512 已满足），INT8 对齐 16（需注意）
- **全局优化原则**：减少内存访问 > 减少计算量 > 提高并行度

---

## 6. SIRIUS 多面体融合（torch.compile 参考）

| 项目 | 说明 |
|------|------|
| 预期加速 | 通过 torch.compile 间接实现 1.3-1.5x |
| 来源论文 | #21 SIRIUS (MLSys 2023) |
| PDF | `10_KernelFusion/SIRIUS.pdf` |

### 核心创新

多面体模型全程序分析 + 自动发现跨算子融合机会

### 融合模式参考

- **Elementwise 融合**：LN+Linear+ReLU → 单一 kernel
- **MatMul+Elementwise 融合**：QKV+Reshape+Transpose → 单一 kernel
- **Reduction+Broadcast 融合**：segment_reduce+concat → 单一 kernel

### 本项目融合机会

- RepEncoder：28 个 slot 的 segment_reduce + concat 可融合为单一 kernel
- TransformerEncoder：每层 10+ kernel → 2-3 kernel
- 实际方案：用 torch.compile 的 inductor 后端间接实现

---

## 工程实践补充：博文与行业报道精华

### iQiyi CTR GPU 推理优化

**来源**：iQiyi 技术博客

**核心经验**：
- 算子融合将 kernel launch 从数百次降至 <10 次
- MultiStream 实现计算与通信 overlap
- FP16 + 算子融合 + CUDA Graph 三板斧

**对本项目的启发**：
- 本项目每层 Transformer 有 10+ kernel launch，8 层共 80+ 次，融合后可降至 16-24 次
- MultiStream 可用于 Embedding 查表与 Transformer 计算的 overlap

### JD 广告稀疏模型 GPU 优化

**来源**：京东技术博客

**核心经验**：
- CPU-GPU 异构流水线：Embedding 查表在 CPU（利用大内存），模型计算在 GPU
- Embedding I/O 占推理时间 >30%
- 稀疏特征 GPU 查表优化：hash-based 索引 + GPU 缓存热特征

**对本项目的启发**：
- 本项目 Embedding(5M, 512) 是显存大户，CPU-GPU 异构方案值得考虑
- 热特征 GPU 缓存 + 冷特征 CPU 查表的策略与 CAFE 思路一致

### TensorRT 广告 CTR 推理优化

**来源**：NVIDIA 技术博客

**核心经验**：
- Layer Fusion + INT8 量化：延迟从 30ms 降至 6-8ms（3.75-5x）
- 稀疏特征处理优化：自定义 plugin 处理 Embedding 查表
- Dynamic Batching：多个请求合并为一个 batch 提高吞吐

**对本项目的启发**：
- TensorRT 的 layer fusion 思路可通过 torch.compile 实现
- INT8 量化 + 算子融合是推荐模型 GPU 推理的黄金组合

### PyTorch 推理加速 41 技巧

**来源**：CSDN 博文

**核心技巧（精选与本项目相关的）**：
1. `torch.compile(model, mode="max-autotune")` — 1.2-2.5x
2. CUDA Graph — 消除 kernel launch 开销
3. FP16 + `channels_last` 内存格式 — 额外 1.4x
4. `torch.no_grad()` + `model.eval()` — 基础但必须
5. 避免不必要的 `.item()` 调用 — 阻塞 GPU
6. `pin_memory=True` + `non_blocking=True` — CPU-GPU 异步传输
7. 预分配输出 tensor — 避免动态内存分配
8. `torch.set_float32_matmul_precision('high')` — 启用 Tensor Core

### vLLM torch.compile 集成经验

**来源**：vLLM GitHub / 技术博客

**核心经验**：
- 自定义 compiler pass 实现 Attention+Quant 融合（7% 加速）
- AllReduce+RMSNorm 融合（15% 加速）
- `torch.compile` 与 CUDA Graph 的配合是最佳实践

---

## 本地知识库：额外优化方案

### 1. channels_last 内存格式

```python
model = model.to(memory_format=torch.channels_last)
```

- V100 上 FP16 + channels_last 比 contiguous 格式快 1.2-1.4x
- 适用于卷积类操作，对 Transformer 的 GEMM 也有一定收益
- 注意：Embedding 查表不支持 channels_last，仅对 TransformerEncoder 和 SMoE 部分生效

### 2. 梯度检查点复用为推理缓存

- 训练时的 gradient checkpointing 思路可逆向用于推理：缓存中间激活避免重复计算
- 适用于 RepEncoder 的 28 个 slot Embedding 查表，如果多个请求共享部分特征

### 3. PyTorch 2.x SDPA 后端

```python
# PyTorch 2.0+ 内置 scaled_dot_product_attention
# 自动选择最优后端：FlashAttention > Memory-Efficient > Math
attn_output = torch.nn.functional.scaled_dot_product_attention(
    q, k, v, attn_mask=None, is_causal=True
)
```

- 如果 FlashAttention-2 安装困难，PyTorch 2.0+ 的 SDPA 是替代方案
- SDPA 的 varlen 支持不如 flash_attn 库完善

### 4. 算子级优化：自定义 CUDA Kernel

- **RepEncoder segment_reduce + concat**：28 个 slot 的循环可融合为单一 kernel
- **SMoE Gate + Scatter + Gather**：融合为单一 kernel，减少中间结果写回 HBM
- 实现成本高，但收益也高（1.2-1.5x）

### 5. 批处理优化

- 如果允许 batch 推理，将多个用户请求合并为一个 batch
- FlashAttention-2 varlen 接口天然支持不等长 batch
- batch_size 从 1 增大到 4-8，吞吐可提升 2-4x

---

## 第四梯队组合实施路线

### 实施顺序

```
Step 1: CPU-GPU 双缓冲预取（最简单，零风险）
  ↓
Step 2: channels_last 内存格式（简单，零风险）
  ↓
Step 3: PyTorch SDPA 后端（FlashAttention-2 的替代方案）
  ↓
Step 4: MiLo 低秩补偿（配合 SmoothQuant 使用）
  ↓
Step 5: 自定义 CUDA Kernel（高收益但高成本）
  ↓
Step 6: 知识蒸馏（需重新训练，最后考虑）
```

### 预期效果

| 优化 | 额外加速 | AUC 风险 |
|------|----------|---------|
| CPU-GPU 双缓冲 | 1.05-1.15x | 零 |
| channels_last | 1.1-1.2x | 零 |
| SDPA 后端 | 1.3-1.5x | 零 |
| 低秩补偿 | AUC 恢复 | 零（补偿用） |
| 自定义 CUDA Kernel | 1.2-1.5x | 零 |
| 知识蒸馏 | 1.5-2x | 高 |

---

## 全四梯队总览

| 梯队 | 核心优化 | 预期加速 | AUC 风险 | 推理时间 |
|------|----------|----------|---------|---------|
| **Tier 1** | FP16 + FA2 + SMoE向量化 + torch.compile | 4-6x | 零 | 38-57s |
| **Tier 2** | SmoothQuant + Embedding量化 + INT-FA | ×1.5-2.5x | 低 | 15-38s |
| **Tier 3** | Loki稀疏 + LExI + KV缓存 | ×1.5-3x | 中 | 5-25s |
| **Tier 4** | 预取 + channels_last + 自定义kernel | ×1.1-1.3x | 零 | 4-23s |

**保守总加速：8-12x → 推理时间 19-29s**
**乐观总加速：15-25x → 推理时间 9-15s**
