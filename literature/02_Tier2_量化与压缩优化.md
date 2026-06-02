# 第二梯队：量化与压缩优化

> **预期额外加速：1.5-2.5x** | **AUC 风险：低-中** | **需验证 AUC/PCOC 指标**
>
> 在第一梯队（4-6x）基础上，通过量化与压缩进一步加速。
> 这些优化会引入数值近似，需在验证集上确认 AUC≥0.65、PCOC∈[0.85,1.15]。

---

## 1. SmoothQuant W8A8 量化

| 项目 | 说明 |
|------|------|
| 预期加速 | 1.5-2x（V100 上 INT8 GEMM） |
| AUC 风险 | 低 |
| 实施难度 | 中 |
| 来源论文 | #12 SmoothQuant (ICML 2023) ⭐ |
| PDF | `06_Quantization/SmoothQuant.pdf` |

### 核心创新

数学等价的逐通道缩放，将激活量化难度迁移至权重端，实现 W8A8 无损 PTQ

### 关键技术细节

- **迁移因子 s**：`s_j = max(|X_j|)^α / max(|W_j|)^{1-α}`，α=0.5 为推荐值
- **等价变换**：`Y = (X·diag(s)^{-1})·(diag(s)·W)`，乘积不变，X_hat 的 outlier 被压制
- **离线预计算**：`W_hat = diag(s)·W` 可离线计算存储，运行时零额外开销
- **量化粒度**：权重 per-channel 对称量化，激活 per-tensor 对称量化，均为 INT8

### 本项目实施要点

```python
# Step 1: 校准 - 收集激活统计量
with torch.no_grad():
    for batch in calibration_dataloader:
        _ = model(batch)
        # 收集每层输入激活的 max(|X_j|)

# Step 2: 计算迁移因子
alpha = 0.5  # 推荐模型可能需 α=0.6~0.7
for layer in model.transformer_encoder.layers:
    s = (x_max ** alpha) / (w_max ** (1 - alpha))
    layer.weight = (s.unsqueeze(1) * layer.weight)  # W_hat
    # 保存 s 用于运行时缩放输入

# Step 3: 量化推理
# 权重已预量化为 INT8，运行时激活 INT8 量化
```

- 所有线性层（QKV/Out/FFN/Expert）完全适用
- 推荐模型特征分布与 LLM 不同，可能需 α=0.6~0.7（更激进地迁移到权重端）
- V100 INT8 Tensor Core 利用率不如 A100/H100，但仍有加速

### 博文/实战佐证

- **TensorRT INT8 实践**：广告 CTR 模型 INT8 量化后延迟从 30ms 降至 6-8ms（3.75-5x）
- **iQiyi CTR 优化**：INT8 量化 + 算子融合是推荐模型 GPU 推理的标配组合
- **SmoothQuant 官方**：OPT-175B 上 W8A8 实现无损量化，推荐模型权重更规则，量化更容易

### 本地知识补充

- **V100 INT8 限制**：V100 (SM70) 的 INT8 Tensor Core 仅支持 `dp4a` 指令（点积），不如 Turing/Ampere 的 `mma` 指令高效。实际加速比约 1.5-2x（而非理论 2-4x）
- **校准数据量**：SmoothQuant 校准通常只需 128-512 个样本，推荐模型建议用 512 个覆盖各特征分布
- **Embedding 量化**：SmoothQuant 主要针对线性层；Embedding 量化见 DQRM 方案
- **与 FP16 的关系**：先做 FP16 推理，再做 SmoothQuant W8A8，两者叠加

---

## 2. INT-FlashAttention

| 项目 | 说明 |
|------|------|
| 预期加速 | 1.5-2.5x（FA 的 IO 优化 + INT8 带宽节省） |
| AUC 风险 | 低 |
| 实施难度 | 中高 |
| 来源论文 | #13 INT-FlashAttention |
| PDF | `06_Quantization/INT-FlashAttention.pdf` |

### 核心创新

Q/K/V 全 INT8 的 FlashAttention，token-level 量化 + online softmax 融合

### 关键技术细节

- **INT8 online softmax**：QK^T 在 INT8 Tensor Core → 反量化 FP32 做 softmax → 再量化 INT8 → INT8 PV
- **Token-level 量化**：per-token scale（`scale_q: [seq_len, 1]`），比 per-tensor 更精确
- **HBM 带宽节省**：Q/K/V 以 INT8 存储，HBM 读取带宽减半

### 本项目实施要点

- 与 SmoothQuant 配合形成完整 INT8 推理流水线
- V100 INT8 GEMM 加速有限，但 HBM 带宽节省仍有效
- 需要自定义 CUDA kernel 或使用 `flash_attn` 的 INT8 变体

### 博文/实战佐证

- **FlashAttention-2 + INT8**：社区有 `flash_attn_with_kvcache` 的 INT8 变体实现
- **vLLM**：FP8/INT8 Attention + Quant 融合 pass 已在 vLLM 中实现

### 本地知识补充

- **V100 兼容性**：INT-FlashAttention 的 INT8 Tensor Core 操作在 V100 上需要 `dp4a` 指令，性能不如 Ampere+
- **替代方案**：如果 INT-FlashAttention 在 V100 上收益不大，可保持 FP16 FlashAttention-2，仅对线性层做 INT8
- **实现建议**：优先使用 FlashAttention-2 FP16 版本，INT8 作为第二阶段优化

---

## 3. DQRM Embedding INT4 量化

| 项目 | 说明 |
|------|------|
| 预期加速 | 1.3-1.8x（Embedding 带宽节省） |
| AUC 风险 | 中高 |
| 实施难度 | 中高 |
| 来源论文 | #14 DQRM (arXiv 2024) |
| PDF | `06_Quantization/DQRM.pdf` |

### 核心创新

推荐模型混合精度量化——Embedding INT4 + Attention INT8 + MLP INT4/INT8 自适应

### 关键技术细节

- **INT4 Embedding**：per-row + group_size=128 分组量化 + 非对称量化
- **AUC 保护**：敏感度分析（逐层量化测 AUC 变化）+ QAT 微调（~1000 步 STE）
- **混合精度**：QKV 投影 INT8（softmax 敏感），FFN/MLP INT4，Embedding INT4
- **Hash-based 分组**：相同 hash 桶的 embedding 行共享量化参数

### 本项目实施要点

- Embedding(5M, 512) 约 10.24GB，INT4 量化后降至 2.5GB，显存减 75%
- 28 个 slot 可按 DQRM 的敏感度分析分别选择 INT4/INT8
- V100 无 INT4 Tensor Core，需软件反量化，加速主要来自带宽节省

### 博文/实战佐证

- **JD 广告稀疏模型优化**：Embedding I/O 占推理时间 >30%，量化后带宽瓶颈显著缓解
- **DQRM 论文**：推荐模型 Embedding INT4 量化后 AUC 下降 < 0.5%，可接受

### 本地知识补充

- **Embedding 量化策略**：建议分步实施——先 FP16 Embedding（2x 带宽节省），验证 AUC 后再 INT8（4x），最后 INT4（8x）
- **V100 上的实际收益**：INT4 需软件反量化（`torch.dequantize`），计算开销可能抵消部分带宽收益
- **28 slot 差异化**：不同 slot 的 Embedding 对量化敏感度不同，建议逐 slot 测试

---

## 4. CAFE 热/冷差异化 Embedding 压缩

| 项目 | 说明 |
|------|------|
| 预期加速 | 1.1-1.2x（间接：显存释放后可增大 batch_size） |
| AUC 风险 | 低-中 |
| 实施难度 | 中 |
| 来源论文 | #18 CAFE (SIGMOD 2024) ⭐ Best Artifact Award |
| PDF | `08_EmbeddingCompression/CAFE.pdf` |

### 核心创新

HotSketch 实时识别热特征 + 热/冷差异化压缩策略

### 关键技术细节

- **HotSketch**：Count-Min Sketch 变体，O(1) 时间统计特征频率，阈值 θ 标记热特征
- **差异化策略**：热特征保留完整 512 维 FP32（GPU HBM），冷特征多表哈希压缩（4×64 维）
- **双层索引**：O(1) 判断热/冷 + O(1) 定位 Embedding
- **滑动窗口衰减**：热特征识别随数据分布漂移自适应

### 本项目实施要点

- Embedding(5M, 512) 约 10.24GB，CAFE 可降至约 3.5GB
- CTR 推荐特征访问长尾分布明显
- 冷特征哈希重建引入近似误差，需验证 AUC/PCOC

### 博文/实战佐证

- **JD 广告系统**：特征访问遵循 Zipf 分布，Top 10% 特征覆盖 80% 访问
- **阿里妈妈**：Embedding 分级存储（热 HBM / 温 DDR / 冷 SSD）是工业界标配

### 本地知识补充

- **简化实现**：比赛场景下可跳过 HotSketch，直接统计训练集中特征频率，离线标记热/冷
- **与量化组合**：热特征 FP16 + 冷特征 INT4/INT8，比 CAFE 的哈希压缩更简单且精度更好
- **显存释放的价值**：释放的显存可用于增大 batch_size 或缓存更多中间结果

---

## 5. Embedding 压缩综述参考

| 项目 | 说明 |
|------|------|
| 预期加速 | 1.1-1.2x（显存释放间接加速） |
| 来源论文 | #19 Embedding Compression Survey (CSUR 2024) |
| PDF | `08_EmbeddingCompression/Embedding_Compression_Survey.pdf` |

### 核心内容

系统梳理四大压缩范式（量化/蒸馏/哈希/低秩分解）+ 压缩比-精度 Pareto 前沿

### 推荐实施路径

```
FP16 Embedding → INT8 冷特征 → PQ 极冷特征
```

- **量化**：Uniform INT8 → Mixed-Precision → Product Quantization（256x 压缩比）
- **低秩分解**：SVD（V×d → V×k + k×d）→ Tensor Train
- **精度排序**：Mixed-Precision > PQ > Uniform INT8 > Hashing（相同压缩比下）

### 本地知识补充

- **SVD 低秩分解**：Embedding(5M, 512) → Embedding_A(5M, 64) × Embedding_B(64, 512)，参数量从 2.56B 降至 0.36B（7x 压缩），AUC 损失通常 < 0.5%
- **Product Quantization**：将 512 维分为 64 个子空间（每子空间 8 维），每子空间 256 个聚类中心，压缩比 64x

---

## 第二梯队组合实施路线

### 实施顺序

```
Step 1: SmoothQuant W8A8（线性层 INT8 量化）
  ↓
Step 2: Embedding FP16（最安全的 Embedding 优化）
  ↓
Step 3: INT-FlashAttention（注意力 INT8，与 SmoothQuant 配合）
  ↓
Step 4: Embedding INT8 冷特征（渐进式量化）
  ↓
Step 5: CAFE 热/冷差异化（显存优化）
  ↓
Step 6: DQRM INT4 Embedding（激进量化，需验证 AUC）
```

### 预期效果

| 优化 | 额外加速 | 累计推理时间 |
|------|----------|-------------|
| 第一梯队后 | - | 38-57s |
| +SmoothQuant W8A8 | 1.5-2x | 19-38s |
| +Embedding FP16 | 1.1x | 17-35s |
| +INT-FlashAttention | 1.1-1.2x | 15-32s |
| +Embedding INT8/INT4 | 1.2-1.5x | 10-27s |

**保守估计：第一梯队 + 第二梯队 = 6-10x 总加速，推理时间 23-38s**

### AUC 风险评估

| 优化 | AUC 预期变化 | PCOC 预期变化 | 风险等级 |
|------|-------------|--------------|---------|
| SmoothQuant W8A8 | -0.001 ~ -0.005 | ±0.02 | 低 |
| Embedding FP16 | < -0.001 | ±0.01 | 极低 |
| INT-FlashAttention | -0.001 ~ -0.003 | ±0.01 | 低 |
| Embedding INT8 | -0.003 ~ -0.01 | ±0.03 | 低-中 |
| CAFE 热/冷 | -0.005 ~ -0.02 | ±0.05 | 中 |
| DQRM INT4 | -0.01 ~ -0.03 | ±0.05 | 中高 |

> **关键策略**：每步优化后立即验证 AUC/PCOC，不达标则回退。AUC=0.759 有 0.109 的裕度（下限 0.65），空间较大。
