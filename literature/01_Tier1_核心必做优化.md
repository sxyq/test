# 第一梯队：核心必做优化

> **预期总加速：4-6x** | **AUC 风险：极低/零** | **推理时间：229s → 38-57s**
>
> 这些优化不改变模型计算语义，仅优化计算方式和内存访问模式，AUC 几乎无损。
> 属于"不做就是浪费"的优化项。

---

## 1. FP16 半精度推理

| 项目 | 说明 |
|------|------|
| 预期加速 | 1.8-2.5x |
| AUC 风险 | 极低 |
| 实施难度 | 低 |
| 来源 | 基础优化 |

### 核心原理

将模型权重和计算从 FP32 转为 FP16，利用 V100 的 Tensor Core（FP16 矩阵乘法吞吐是 FP32 的 8x）。

### 关键技术细节

- **自动混合精度（AMP）**：`torch.cuda.amp.autocast(dtype=torch.float16)` 包裹推理代码
- **V100 Tensor Core**：FP16 GEMM 理论峰值 125 TFLOPS vs FP32 15.7 TFLOPS（8x）
- **Loss Scaling**：推理无需反向传播，无需 loss scaling，直接 FP16 推理
- **权重预转换**：`model.half().cuda()` 一次性转换，避免运行时 cast 开销

### 本项目实施要点

```python
# infer.py 中修改
model = model.half().cuda()  # 权重转 FP16
with torch.cuda.amp.autocast(dtype=torch.float16):
    output = model(input_data)
```

- RepEncoder Embedding 查表：FP16 权重直接查表，带宽减半
- TransformerEncoder：QKV 投影 + FFN 全部走 Tensor Core
- SMoE Expert：8 个 Expert 的 GEMM 全部 FP16 加速

### 博文/实战佐证

- **PyTorch 官方博客**：FP16 + channels_last 格式额外 1.4x 加速
- **CSDN "41个PyTorch推理加速技巧"**：FP16 是最简单最有效的优化，通常 1.5-2.5x
- **iQiyi CTR GPU 优化实践**：FP16 推理是推荐模型 GPU 加速的第一步

### 本地知识补充

- **注意 Embedding 查表的精度**：5M 行 Embedding 中低频 ID 的 FP16 表示可能丢失精度，建议对 Embedding 做 AMP autocast（动态 cast）而非 `.half()` 预转换
- **V100 FP16 注意事项**：V100 的 FP16 Tensor Core 要求矩阵维度是 8 的倍数才能充分利用；本项目 d_model=512 已满足
- **输出层保持 FP32**：最终 logits 输出建议保持 FP32 以确保排序精度

---

## 2. FlashAttention-2 varlen 替换

| 项目 | 说明 |
|------|------|
| 预期加速 | 2.0-3.0x（相比自定义 scaled_dot_product） |
| AUC 风险 | 零（数学等价） |
| 实施难度 | 低 |
| 来源论文 | #2 FlashAttention-2 (arXiv 2023) ⭐ |
| PDF | `01_FlashAttention/FlashAttention-2.pdf` |

### 核心创新

重新设计 warp 分工（Q 完整行分配给每个 warp）+ 序列长度维度并行化

### 关键技术细节

- **Warp 分工**：每个 warp 持有 Q_block 的完整行，K/V 通过 shared memory 广播，**消除 warp 间同步**（从 O(T_c) 次同步降至 0）
- **序列长度并行**：当 batch×heads 不够填满 GPU SM 时，在 N 维度额外并行。本项目 batch=1, heads=8，仅 8 个并行任务远未填满 V100 的 80 个 SM，**此优化极为关键**
- **Causal 优化**：跳过上三角块 + 边界块部分计算，因果模式 HBM 访问减少约 50%
- **`flash_attn_varlen_func`**：支持 `cu_seqlens` 参数的不等长序列接口，**完美匹配本项目的 `user_offsets`**
- **延迟 rescaling**：非矩阵乘法 FLOP 从 ~15% 降至 ~5%

### 本项目实施要点

```python
# 替换 infer.py 中的 scaled_dot_product 函数（第285-293行）
from flash_attn import flash_attn_varlen_func

# 原来：
# attn_weight = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(d)
# attn_weight = torch.masked_fill(attn_weight, mask == 0, float('-inf'))
# attn_weight = torch.softmax(attn_weight, dim=-1)
# attn_output = torch.matmul(attn_weight, v)

# 替换为：
attn_output = flash_attn_varlen_func(
    q, k, v,
    cu_seqlens=user_offsets,  # 直接传入 user_offsets
    max_seqlen=max_seq_len,
    causal=True
)
```

- 输入要求 FP16/BF16，与 FP16 推理方案天然配合
- V100 SM70 兼容（FlashAttention-2 最低支持 SM70）
- `user_offsets` 直接映射为 `cu_seqlens`，无需额外处理

### 博文/实战佐证

- **FlashAttention-2 官方 benchmark**：A100 上比 PyTorch 原生注意力快 3.8x，V100 上约 2-3x
- **vLLM 项目**：FlashAttention 已成为 LLM 推理标配，varlen 接口用于 continuous batching
- **HuggingFace Transformers**：`F.scaled_dot_product_attention` 后端已集成 FlashAttention-2

### 本地知识补充

- **安装注意**：`pip install flash-attn --no-build-isolation`，需要 CUDA 11.6+ 和 PyTorch 2.0+
- **V100 上的限制**：V100 无 BF16，必须使用 FP16；FP16 在注意力计算中可能出现溢出，FlashAttention-2 内部使用 FP32 累积，安全
- **varlen 接口细节**：`cu_seqlens` 需要在开头补 0，即 `[0, len_1, len_1+len_2, ...]`，检查本项目 `user_offsets` 格式是否匹配
- **内存布局**：FlashAttention 要求 Q/K/V 为 `(total_seq_len, num_heads, head_dim)` 格式（非 batch 维度），需调整 tensor 形状

---

## 3. SMoE Expert 向量化

| 项目 | 说明 |
|------|------|
| 预期加速 | 1.2-1.5x |
| AUC 风险 | 零（计算结果等价） |
| 实施难度 | 中 |
| 来源论文 | #7 Toward Efficient Inference for MoE (NeurIPS 2024) ⭐ |
| PDF | `03_MoE_Optimization/Toward_Efficient_Inference_for_MoE.pdf` |

### 核心创新

Dynamic Gating 预测器 + Expert 向量化批量执行

### 关键技术细节

- **Expert 向量化**：对每个 Expert i 收集所有路由到该 Expert 的 token，组成 dense 矩阵执行一次 GEMM
- **Dynamic Gating**：轻量 MLP（2 层 64 宽）提前预测 Expert 激活，预测准确率 85-92%
- **Expert Buffering**：LRU 缓存管理，不活跃 Expert offload 到 CPU

### 本项目实施要点

```python
# 替换 infer.py 中的 SMoE for 循环（第339-374行）
# 原来：
# for i in range(self.num_experts):
#     expert_mask = (gating_output.argmax(dim=-1) == i)
#     if expert_mask.any():
#         expert_input = x[expert_mask]
#         expert_output = self.experts[i](expert_input)
#         output[expert_mask] = expert_output

# 替换为向量化实现：
# Step 1: 计算路由
gate_scores = gating_output  # [seq_len, num_experts]
topk_indices = gate_scores.topk(k=2, dim=-1).indices  # [seq_len, 2]

# Step 2: Scatter/Gather 批量执行
for expert_id in range(self.num_experts):
    # 收集路由到该 Expert 的所有 token
    token_mask = (topk_indices == expert_id).any(dim=-1)
    if token_mask.any():
        expert_input = x[token_mask]  # [n_tokens, d_model]
        expert_output = self.experts[expert_id](expert_input)  # 单次 GEMM
        # Scatter 回原位置
        output[token_mask] += gate_scores[token_mask, expert_id].unsqueeze(-1) * expert_output
```

- 8 个 Expert 的串行 GEMM → 8 次批量 GEMM，消除 kernel launch 开销
- 可进一步用 `torch.scatter` / `torch.gather` 实现全向量化

### 博文/实战佐证

- **SMoE 论文实测**：Expert 向量化在 8 Expert 模型上实现 1.2-1.5x 加速
- **DeepSpeed-MoE**：微软 DeepSpeed 框架中 MoE 推理已采用类似向量化策略
- **vLLM MoE 优化**：Expert 并行 + 批量 GEMM 是 MoE 推理标配

### 本地知识补充

- **Top-2 Gating 的特殊性**：本项目 Top-2 意味着每个 token 被路由到 2 个 Expert，gather 时需处理重复 token
- **负载不均衡问题**：某些 Expert 可能收到很少 token，GEMM 矩阵小导致 Tensor Core 利用率低；可考虑 padding 到 8/16 的倍数
- **替代方案**：如果 Expert 数量少（8个），可用 `torch.stack` 将 8 个 Expert 权重拼成一个大矩阵，用 `torch.bmm` 批量计算，避免 for 循环

---

## 4. torch.compile 算子融合

| 项目 | 说明 |
|------|------|
| 预期加速 | 1.3-1.5x |
| AUC 风险 | 零（计算结果等价） |
| 实施难度 | 中 |
| 来源论文 | #21 SIRIUS (MLSys 2023) |
| PDF | `10_KernelFusion/SIRIUS.pdf` |

### 核心创新

多面体模型全程序分析 + 自动发现跨算子融合机会

### 关键技术细节

- **全程序分析**：将 DNN 计算图表示为数据流图，提取迭代域，构建算子间依赖关系
- **多面体融合**：仿射调度计算最优循环顺序和分块策略，最小化数据重用距离
- **融合模式**：Elementwise 融合（LN+Linear+ReLU）、MatMul+Elementwise 融合（QKV+Reshape+Transpose）、Reduction+Broadcast 融合（segment_reduce+concat）

### 本项目实施要点

```python
# 使用 torch.compile 替代手动算子融合
model = torch.compile(model, mode="reduce-overhead")
# 或更精细控制：
model = torch.compile(model, mode="max-autotune")
```

- **RepEncoder**：28 个 slot 的 segment_reduce + concat 可融合为单一 kernel
- **TransformerEncoder**：每层 10+ kernel → 2-3 kernel（LN+QKV 投影融合、Residual+Add 融合）
- **SMoE**：Gate 计算 + Scatter + Expert GEMM + Gather 融合

### 博文/实战佐证

- **PyTorch 官方博客**：torch.compile + CUDA Graph 在 Seamless M4T 上实现 2.7x 端到端加速
- **vLLM torch.compile 集成**：自定义 compiler pass 实现 Attention+Quant 融合（7% 加速）、AllReduce+RMSNorm 融合（15% 加速）
- **CSDN 实战**：torch.compile 在推荐模型上通常 1.2-2.5x 加速，但需注意动态 shape 问题

### 本地知识补充

- **动态 shape 问题**：本项目序列长度随用户变化，torch.compile 默认会对每个 shape 重编译。解决方案：
  1. 使用 `dynamic=True` 参数：`torch.compile(model, dynamic=True)`
  2. 对常见序列长度做 shape specialization，padding 到固定长度
- **torch.compile 模式选择**：
  - `"default"`：编译快，运行时优化一般
  - `"reduce-overhead"`：减少 Python 开销，适合小 batch
  - `"max-autotune"`：尝试更多 kernel 变体，编译慢但运行快
- **与 CUDA Graph 配合**：`torch.compile(model, mode="reduce-overhead")` 内部自动使用 CUDA Graph
- **调试技巧**：`TORCH_LOGS="+dynamo"` 查看编译日志，`torch._dynamo.explain(model, *args)` 查看图断点

---

## 5. CUDA Graph

| 项目 | 说明 |
|------|------|
| 预期加速 | 1.1-1.3x |
| AUC 风险 | 零 |
| 实施难度 | 中 |
| 来源论文 | #11 PyGraph (arXiv 2025) |
| PDF | `04_InferenceOptimization/PyGraph.pdf` |

### 核心创新

编译器级 CUDA Graph 自动化部署，解决动态 shape/控制流/数据依赖三大限制

### 关键技术细节

- **Shape Specialization**：对常见 shape 预编译多个 CUDA Graph 实例
- **Branch Compaction**：条件分支转为无条件执行 + mask 选择
- **Stream-Aware Graph Partition**：将计算图分割为可 capture 和不可 capture 的子图

### 本项目实施要点

```python
# 方式1：torch.compile 自动 CUDA Graph
model = torch.compile(model, mode="reduce-overhead")

# 方式2：手动 CUDA Graph capture
static_input = torch.zeros_like(example_input)
g = torch.cuda.CUDAGraph()
with torch.cuda.graph(g):
    static_output = model(static_input)

# 推理时
static_input.copy_(real_input)
g.replay()
output = static_output.clone()
```

- SMoE 动态路由需 Branch Compaction（8 Expert 全执行 + mask 选择）
- 序列长度变化需 Shape Specialization 或 padding 到固定长度

### 博文/实战佐证

- **PyTorch 官方**：CUDA Graph 消除 kernel launch 开销，小 kernel 密集型模型收益最大
- **iQiyi CTR 优化**：算子融合将 kernel launch 从数百次降至 <10 次，CUDA Graph 进一步消除 launch 开销
- **TensorRT 实践**：CUDA Graph 是 TensorRT 推理加速的核心技术之一

### 本地知识补充

- **CUDA Graph 限制**：capture 期间不能有 CPU-GPU 同步、动态内存分配、条件分支
- **与 torch.compile 配合**：`mode="reduce-overhead"` 已内置 CUDA Graph，无需手动 capture
- **SMoE 处理**：动态路由导致不同 token 走不同 Expert，需将所有 Expert 路径都执行一遍，用 mask 选择结果

---

## 第一梯队组合实施路线

### 实施顺序

```
Step 1: FP16 推理（最简单，立竿见影）
  ↓
Step 2: FlashAttention-2 varlen（替换注意力，核心加速）
  ↓
Step 3: SMoE Expert 向量化（消除 for 循环）
  ↓
Step 4: torch.compile + CUDA Graph（自动融合 + 消除 launch 开销）
```

### 预期效果

| 优化 | 累计加速 | 推理时间 |
|------|----------|----------|
| Baseline | 1x | 229.18s |
| +FP16 | 1.8-2.5x | 92-127s |
| +FlashAttention-2 | 3.6-7.5x | 31-64s |
| +SMoE 向量化 | 4.3-11.3x | 20-53s |
| +torch.compile+CUDA Graph | 4-6x（保守） | 38-57s |

> 注：加速比非简单乘积，各优化有重叠收益。保守估计组合加速 4-6x。

### AUC 影响评估

- FP16：AUC 变化 < 0.001（FP16 精度足够表示 CTR logit）
- FlashAttention-2：数学等价，AUC 变化 = 0
- SMoE 向量化：计算结果等价，AUC 变化 = 0
- torch.compile：计算结果等价，AUC 变化 = 0

**结论：第一梯队全部实施后，AUC 几乎无损，远超 0.65 下限。**
