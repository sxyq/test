# 第三梯队：算法创新优化

> **预期额外加速：1.5-3x** | **AUC 风险：中-高** | **需充分验证指标**
>
> 这些优化涉及算法层面的修改（稀疏化、Expert 剪枝、KV 压缩等），
> 可能显著影响模型精度，需在验证集上严格测试 AUC/PCOC。
> 适合在第一、二梯队已实施且指标有余量时尝试。

---

## 1. Loki PCA 稀疏注意力

| 项目 | 说明 |
|------|------|
| 预期加速 | 1.5-3x（取决于 k 值选择） |
| AUC 风险 | 中 |
| 实施难度 | 中 |
| 来源论文 | #17 Loki (NeurIPS 2024) |
| PDF | `07_SparseAttention/Loki.pdf` |

### 核心创新

Key 矩阵 PCA 降维 → 低秩近似 Key' → top-k 选择重要 KV 对 → token-level 稀疏

### 关键技术细节

- **离线 PCA**：校准集上收集 K 矩阵，每头独立 PCA，取前 r=32~64 主成分，得投影矩阵 P ∈ R^{d×r}
- **在线流程**：`K' = K·P` → `S' = Q'·K'^T / sqrt(r)` → top-k 选择 → 精确注意力
- **P 矩阵存储**：8 层×8 头=64 个 P 矩阵，每个 P ∈ R^{64×32}，总参数仅 131KB
- **与 FA 兼容**：gather 操作将选中 KV 对连续排列后调用 FlashAttention

### 本项目适用性

- d_head=64，PCA 降至 r=16~32，Q'K'^T 计算量仅为完整注意力的 25-50%
- 推荐场景中少数历史行为与候选强相关，token-level 选择比 block-level 更精准
- 与 KV Cache 复用正交互补

### 博文/实战佐证

- **Quest**：类似思路的 KV Cache 压缩方法，在 LLM 推理中已验证有效性
- **Sparse Attention 工业界**：阿里 DMA 稀疏注意力在推荐场景实现 2-3x 加速

### 本地知识补充

- **推荐场景的特殊性**：CTR 推荐中用户近期行为权重远高于远期，注意力天然稀疏，Loki 的 top-k 选择非常契合
- **k 值选择策略**：建议从 k=N/2 开始（保留 50% token），逐步降低到 k=N/4，每步验证 AUC
- **与 FlashAttention-2 的配合**：Loki 的 gather 操作将选中 KV 对连续排列，可直接传入 `flash_attn_varlen_func`
- **PCA 校准数据**：用推理数据集的 1000-2000 个样本收集 K 矩阵做 PCA 即可

---

## 2. Dynamic Sparse Flash Attention

| 项目 | 说明 |
|------|------|
| 预期加速 | 1.2-2x |
| AUC 风险 | 中 |
| 实施难度 | 中 |
| 来源论文 | #15 Dynamic Sparse Flash Attention (NeurIPS 2023) |
| PDF | `07_SparseAttention/Dynamic_Sparse_Flash_Attention.pdf` |

### 核心创新

FlashAttention tiling 循环中嵌入动态稀疏掩码，跳过零值 block

### 关键技术细节

- **Block-sparse kernel**：按 block_size=64/128 分块，用 Q/K block 的 L2 范数乘积作为重要性分数
- **动态掩码**：在 FlashAttention outer loop 中即时生成，不预计算
- **Causal 稀疏**：因果 mask 已跳过上三角，Dynamic Sparse 进一步跳过远距离下三角 block
- **加速比**：≈ 1 / (1 - sparsity_ratio)，50% 稀疏度约 2x

### 本项目适用性

- CTR 推荐中近期行为远比远期重要，注意力分布天然稀疏
- 短序列（<128）收益有限，长序列（>256）收益显著

### 博文/实战佐证

- **Block Sparse Attention**：OpenAI 的 block sparse attention 在长序列上实现 2-5x 加速
- **Longformer**：局部窗口 + 全局 attention 的稀疏模式在推荐场景也有应用

### 本地知识补充

- **与 Loki 的对比**：Dynamic Sparse 是 block-level 稀疏（粗粒度），Loki 是 token-level 稀疏（细粒度）。推荐场景建议优先 Loki
- **实现复杂度**：Dynamic Sparse 需要修改 FlashAttention kernel，实现成本高于 Loki
- **建议**：如果实现 Loki 后 AUC 仍有余量，可进一步尝试 Dynamic Sparse 叠加

---

## 3. SeerAttention 可学习稀疏注意力

| 项目 | 说明 |
|------|------|
| 预期加速 | 1.3-2.5x |
| AUC 风险 | 中高 |
| 实施难度 | 中高 |
| 来源论文 | #16 SeerAttention |
| PDF | `07_SparseAttention/SeerAttention.pdf` |

### 核心创新

可学习 Gate 网络预测 block 级注意力稀疏性，数据自适应

### 关键技术细节

- **Gate 网络**：Q/K 在 seq 维度 avg-pooling → `G = sigmoid(Q_comp · K_comp^T / sqrt(d))` → 阈值二值化
- **训练策略**：联合训练 + 稀疏正则项 `L_sparse = mean(G)` + τ 退火调度
- **推理 kernel**：先计算 Gate 矩阵（O(⌈N/B⌉²×d)），非零 block 复用 FlashAttention

### 本项目适用性

- Gate 网络需联合训练，但比赛允许量化/稀疏/剪枝
- 推荐场景注意力模式规律（近期行为权重高），Gate 容易学习

### 本地知识补充

- **比赛规则考量**：如果比赛允许修改模型结构并重新训练，SeerAttention 是很好的选择
- **简化方案**：不训练 Gate 网络，直接用固定窗口（如最近 256 token）+ 全局 attention，实现更简单
- **与 Loki 的关系**：SeerAttention 是 block-level 稀疏（与 Dynamic Sparse 同类），Loki 是 token-level，两者可叠加

---

## 4. LExI 层自适应 Top-K

| 项目 | 说明 |
|------|------|
| 预期加速 | 1.1-1.3x |
| AUC 风险 | 中 |
| 实施难度 | 中 |
| 来源论文 | #9 LExI (arXiv 2025) |
| PDF | `03_MoE_Optimization/LExI.pdf` |

### 核心创新

逐层自适应调整活跃 Expert 数量，浅层减少 Expert 计算

### 关键技术细节

- **关键观察**：浅层 Expert 选择高度集中（大部分 token 只激活同一个 Expert），深层更分散
- **Layer-Adaptive Top-K**：浅层 k_l=1，深层 k_l=2，在验证集上逐层搜索最优 k_l
- **Expert 合并**：`output = g_1 * Expert_1(x) + g_2 * Expert_2(x)`，当 g_1 >> g_2 时近似为 Expert_1(x)

### 本项目适用性

- 可直接分析每层 Expert 激活集中度，对集中度高的层（top-1 score > 0.8）使用 k=1
- AUC=0.759 裕度不大，需验证减少 Expert 后指标是否达标

### 博文/实战佐证

- **MoE 推理优化实践**：浅层 Expert 路由集中是普遍现象，浅层 k=1 通常 AUC 损失 < 0.1%
- **Switch Transformer**：Google 的 Switch Transformer 已验证 Top-1 路由在多数层可行

### 本地知识补充

- **分析方法**：在验证集上统计每层每个 token 的 top-1 gate score，如果某层 > 80% 的 token 的 top-1 score > 0.7，则该层可安全使用 k=1
- **渐进策略**：先对所有层保持 k=2，然后逐层尝试 k=1，每步验证 AUC
- **与 Expert 向量化的配合**：k=1 时 Expert 向量化更简单（每个 token 只路由到一个 Expert）

---

## 5. KV 缓存复用

| 项目 | 说明 |
|------|------|
| 预期加速 | 1.2-1.5x |
| AUC 风险 | 中 |
| 实施难度 | 中 |
| 来源论文 | #4 HSTU / Actions Speak Louder than Words (ICML 2024) |
| PDF | `02_GenerativeRecommendation/HSTU.pdf` |

### 核心创新

M-FALCON 推理算法——逐点注意力 + 微批次候选评分 + 编码器级 KV 缓存

### 关键技术细节

- **逐点注意力**：去掉 Softmax 归一化，直接 `QK^T * V`，保留用户偏好强度信号
- **微批次推理**：m 个候选分为 ceil(m/b_m) 个微批次，block-diagonal mask 使候选互不可见
- **KV 缓存复用**：历史 token 的 K/V 只计算一次并缓存，每个微批次只计算候选 token 的 KV
- **门控残差**：`X' = X + f2(Norm(A(X)V(X)) * U(X))`，U(X) 为门控向量

### 本项目适用性

- KV 缓存复用：同一用户历史序列 KV 不变，可跨请求缓存（1.2-1.5x）
- 逐点注意力修改注意力机制，影响预训练权重，风险较高
- 微批次候选评分在单用户推理场景收益有限

### 博文/实战佐证

- **vLLM PagedAttention**：KV Cache 管理是 LLM 推理的核心优化，推荐模型同样适用
- **HSTU 论文**：KV 缓存复用在推荐场景实现 1.2-1.5x 加速

### 本地知识补充

- **KV 缓存的关键问题**：本项目中同一用户的历史序列是否在多次推理中不变？如果是，KV 缓存收益巨大
- **实现方式**：`torch.save` 缓存 KV 到磁盘或 GPU HBM，下次推理时 `torch.load` 直接使用
- **内存管理**：8 层 × 8 头 × d_head=64，每个用户的 KV Cache 约 8×8×64×2×2=16KB（FP16），可缓存大量用户

---

## 6. EARN Register Token 序列压缩

| 项目 | 说明 |
|------|------|
| 预期加速 | 1.3-1.8x（近似方案），2-4x（完整方案需训练） |
| AUC 风险 | 中高 |
| 实施难度 | 中高 |
| 来源论文 | #6 EARN (arXiv 2025) |
| PDF | `02_GenerativeRecommendation/EARN.pdf` |

### 核心创新

Register Token 将长用户历史序列 KV Cache 压缩为固定数量 token

### 关键技术细节

- **Register Token 机制**：在输入序列中插入 r 个可学习 Register Token（r=4-16），通过 attention 聚合历史信息
- **推理压缩流程**：第一遍完整计算 → 缓存 Register Token 的 KV（r 个）→ 评分候选时只与 r 个 Register Token 做 attention
- **序列长度从 N 压缩到 r**，r=8 时推荐场景接近无损

### 本项目适用性

- Register Token 是训练时引入的，推理时插入会导致权重不匹配
- 可用 mean pooling / strided sampling / segment-level pooling 近似实现（1.3-1.8x）
- 直接实现需要修改训练流程

### 本地知识补充

- **近似方案**：对用户历史序列做 strided sampling（每隔 k 个 token 取一个），将序列长度从 N 压缩到 N/k
- **Segment-level pooling**：将历史序列分为 k 段，每段做 mean pooling 得到一个代表 token，序列长度从 N 压缩到 k
- **风险提示**：这些近似方案会丢失信息，AUC 可能下降 0.01-0.03

---

## 7. Jagged Attention 序列打包

| 项目 | 说明 |
|------|------|
| 预期加速 | 1.05-1.15x |
| AUC 风险 | 零 |
| 实施难度 | 中高 |
| 来源论文 | #5 Scaling GR with Context Parallelism (RecSys 2025) |
| PDF | `02_GenerativeRecommendation/Scaling_GR_Context_Parallelism.pdf` |

### 核心创新

Jagged Attention（不等长序列无 padding tiling）+ Context Parallelism

### 关键技术细节

- **Jagged Attention**：用 offsets 数组记录每个序列起止位置，kernel 内部跳过 padding 区域
- **CP 通信与计算 overlap**：计算当前 segment 时异步发送下一个 segment 的 KV
- **负载均衡**：按用户序列长度排序后交替分配到不同 GPU

### 本项目适用性

- 序列打包消除 padding：1.05-1.15x
- 单 GPU 环境，分布式 CP 不适用
- Jagged Attention 需自定义 CUDA kernel，实现成本高

### 本地知识补充

- **FlashAttention-2 varlen 已解决**：`flash_attn_varlen_func` 的 `cu_seqlens` 参数本质上就是 Jagged Attention 的实现，无需额外实现
- **Padding 消除**：如果当前实现有 padding，切换到 varlen 接口即可消除

---

## 第三梯队组合实施路线

### 实施顺序（按风险从低到高）

```
Step 1: Loki PCA 稀疏注意力（离线 PCA，无需训练，风险可控）
  ↓
Step 2: LExI 层自适应 Top-K（分析激活集中度，渐进调整）
  ↓
Step 3: KV 缓存复用（如果用户历史序列跨请求不变）
  ↓
Step 4: Dynamic Sparse Flash Attention（与 Loki 叠加）
  ↓
Step 5: EARN 近似序列压缩（strided sampling / segment pooling）
  ↓
Step 6: SeerAttention（需训练 Gate 网络，风险最高）
```

### 预期效果

| 优化 | 额外加速 | AUC 预期变化 |
|------|----------|-------------|
| Loki PCA (k=N/2) | 1.5-2x | -0.003 ~ -0.01 |
| LExI Top-1 浅层 | 1.1-1.2x | -0.001 ~ -0.005 |
| KV 缓存复用 | 1.2-1.5x | 0（等价） |
| Dynamic Sparse | 1.2-1.5x | -0.005 ~ -0.02 |
| EARN 近似 | 1.3-1.8x | -0.01 ~ -0.03 |

### AUC 安全边界分析

- Baseline AUC = 0.759，下限 = 0.65，裕度 = 0.109
- 第三梯队所有优化最坏情况 AUC 下降约 0.05-0.07，仍有 0.04-0.06 裕度
- **建议**：先实施 Loki + LExI（AUC 下降 < 0.015），验证通过后再尝试更高风险优化

### 与第一、二梯队的叠加效果

```
第一梯队（4-6x）+ 第二梯队（1.5-2.5x）+ 第三梯队（1.5-3x）
= 总加速 9-45x（保守 9-15x）
推理时间：229s → 15-25s
```
