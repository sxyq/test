# 论文基本信息

---

## 论文 1: GRAB

- **标题**: GRAB: An LLM-Inspired Sequence-First Click-Through Rate Prediction Modeling Paradigm
- **文件**: `docs/论文/GARB论文.pdf`
- **来源**: 百度
- **与比赛关系**: 本比赛使用的模型即 GRAB 的简化版本

### 核心架构

```
用户行为日志 → 密集分词器(Dense Tokenizer) → CamA多通道因果注意力 → CTR预测
```

三阶段流水线：
1. **稀疏特征层** — 原始行为日志处理为时间有序事件序列，经 PSTable 哈希映射为稀疏 ID
2. **密集分词器** — 字段级 Embedding Lookup → GateMLP 融合 → 投影到 d_model 形成事件 token
3. **自回归式序列建模层** — CamA（Causal Action-aware Multi-channel Attention）

### 关键创新

| 创新 | 说明 |
|------|------|
| CamA 多通道注意力 | 历史行为和候选广告分通道独立编码，目标位置门控融合 |
| 异构 Token | Partial Token（历史，去静态字段）+ Full Token（候选，完整信息） |
| Action-aware RAB | 相对位置+相对时间+相对动作三种注意力偏置，query-aware |
| STS 训练 | Stage I 冻结嵌入训练 Transformer，Stage II 冻结 Transformer 训练嵌入 |
| 序列打包 | 同用户 token 合并，不同用户严格隔离，消除 padding 开销 |

### 模型参数规模

| 配置 | 参数量 |
|------|--------|
| GRAB 2l-2h-64d | 6.51M |
| GRAB 4l-4h-128d | 7.05M |
| GRAB 4l-4h-256d | 8.13M |
| GRAB 4l-4h-512d | 11.27M |

### 推理优化成果（论文实测）

| 优化 | 效果 |
|------|------|
| 算子融合（Gemm+Bias+LayerNorm） | 延迟降低 **43%** |
| 混合精度（TF32+FP16） | 性能提升 **28%**，精度损失可忽略 |
| KV-Cache 复用 + M-FALCON | 候选评分从 O(n²) 降至 O(n) |
| 分离式服务架构（UIC） | 异步计算和更新 KV-cache |
| 并行物料召回 | 用户编码与候选检索并行执行 |
| User Slice 存储 | 预聚合行为日志，替代时间分区 |
| Binary+LZ4 压缩 | 存储成本降 12%，解码延迟降 70% |

### 在线效果

- 部署于百度首页信息流广告，每日数十亿请求
- CTR 提升 3.49%，CPM 提升 3.05%
- 在线推理成本与先前 DLRM 模型持平

### 与其他模型对比

| 维度 | DLRMs | HSTU | GRAB |
|------|-------|------|------|
| 核心架构 | Embedding+MLP+Target Attn | 纯 Transformer | 端到端生成式框架 |
| 特征交互 | 手动交叉 | 标准自注意力 | CamA 多通道注意力 |
| 异构特征 | 拼接展平 | 同构 token | 异构 token |
| 动作语义 | 隐式 | 隐式 | Action-aware RAB（显式） |
| 训练策略 | 标准监督 | 自回归 | STS（解耦优化） |

---

## 论文 2: HSTU

- **标题**: Actions Speak Louder than Words: Trillion-Parameter Sequential Transducers for Generative Recommendations
- **文件**: `docs/论文/Actions Speak Louder than Words- Trillion-Parameter Sequential Transducers for Generative Recommendations.pdf`
- **来源**: Meta (Facebook)
- **与比赛关系**: 参考论文，提供推理优化思路（M-FALCON 算法）

### 核心架构

HSTU（Hierarchical Sequential Transduction Units）— 专为推荐系统设计的序列转导编码器。

### 关键创新

| 创新 | 说明 |
|------|------|
| 逐点注意力 | 去掉 Softmax，保留用户偏好强度信号 |
| 相对注意力偏置 (rab) | 基于位置和时间的相对偏置，替代绝对位置编码 |
| 门控残差 | `X' = X + f2(Norm(A(X)V(X)) ⊙ U(X))`，U(X) 为门控向量 |
| 简化层结构 | 中间激活 15d（标准 Transformer 33d），同显存可建 2x 深度网络 |
| 随机长度训练 | 幂律分布采样子序列，80% 稀疏率下指标下降 <0.2% |

### M-FALCON 推理算法（核心优化）

**Microbatched-Fast Attention Leveraging Cacheable OperatioNs**

三个关键思想：
1. **批量推理** — 修改注意力掩码，m 个候选 token 互不可见，单次前向传播处理 b_m 个候选
2. **微批次扩展** — 将 m 个候选分为 ⌈m/b_m⌉ 个微批次，每批 b_m 个
3. **编码器级 KV 缓存** — K(X) 和 V(X) 跨微批次和跨请求缓存

**效果**：GR 模型 FLOPs 是 DLRM 的 285 倍，但通过 M-FALCON 实现：
- 评分 1024 个候选：**1.50x QPS**
- 评分 16384 个候选：**2.99x QPS**

### 模型参数规模

- 最终部署：**1.5 万亿参数（1.5 Trillion）**
- 最大测试配置：8192 序列长度 × 1024 嵌入维度 × 24 层 HSTU
- DLRM 在约 2000 亿参数时饱和，GR 在三个数量级上持续幂律缩放

### 与传统推荐模型的区别

| 维度 | 传统 DLRM | 生成式推荐器 (GR) |
|------|----------|------------------|
| 建模方式 | 判别式 p(Φ_i \| 历史) | 生成式联合分布 |
| 特征工程 | ~1000 稠密 + ~50 稀疏特征 | 仅原始类别型交互特征 |
| 序列建模 | 仅建模用户交互过的物品 | 交错建模推荐内容和用户动作 |
| 架构 | 多模块组合 (DIN+DCN+MMoE) | 统一 HSTU 编码器 |

### 在线效果

- A/B 测试：相比生产 DLRM 基线，参与度指标提升 **12.4%**，消费指标提升 **4.4%**

---

## 对本比赛优化的启发

| 优化方向 | 来源 | 适用性 |
|----------|------|--------|
| FP16/BF16 半精度推理 | GRAB 论文验证 | ✅ 直接可用 |
| Flash Attention | HSTU 论文基于此优化 | ✅ 直接可用 |
| 算子融合 | GRAB 论文验证延迟降 43% | ⚠️ 需自定义 CUDA kernel |
| KV-Cache 复用 | 两篇论文都强调 | ⚠️ 需改造推理流程 |
| M-FALCON 微批次 | HSTU 论文 | ⚠️ 需改造注意力掩码 |
| torch.compile | PyTorch 2.x | ✅ 直接可用 |
| SMoE 向量化 | baseline 代码分析 | ✅ 直接可用 |
