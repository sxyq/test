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
- [x] 数据集下载（30GB，9个 shard，已存入 `dataset/dataset/cached_batches/`）
- [x] 模型权重下载（9.9GB，10个分片，已存入 `weights/ckpt.part.00~09`）
- [x] Baseline 代码获取（infer.py 完整源码，25KB）
- [x] 论文阅读（GRAB 论文 + HSTU 论文，关键优化点已提取）
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

## 已识别的优化方向（按优先级排序）

| # | 优化方向 | 预期加速 | 风险 | 论文依据 |
|---|----------|----------|------|----------|
| 1 | FP16/BF16 半精度推理 | 2-3x | 低 | GRAB 论文验证混合精度提升 28% |
| 2 | Flash Attention 2 | 1.5-2x | 低 | HSTU 论文基于 FlashAttention 优化 |
| 3 | torch.compile | 1.3-2x | 中 | PyTorch 2.x 自动算子融合 |
| 4 | SMoE 向量化（消除 for 循环） | 1.2-1.5x | 低 | 当前串行遍历 8 个 Expert |
| 5 | CUDA Graph | 1.2-1.5x | 中 | 减少 kernel launch 开销 |
| 6 | 数据预加载/prefetch | 1.1-1.3x | 低 | 当前逐 batch 从 CPU 加载 |
| 7 | 算子融合（自定义 CUDA kernel） | 1.3-1.5x | 高 | GRAB 论文验证延迟降低 43% |
| 8 | KV-Cache 复用 | 1.3-2x | 中 | 两篇论文都强调此优化 |
| 9 | 模型量化（INT8） | 2-4x | 高 | 可能影响 AUC/PCOC |

## 推理瓶颈分析

1. **自定义 scaled_dot_product** — 未使用 Flash Attention，手动计算 QKV，O(n²) 显存
2. **SMoE 串行执行** — 每层 for 循环遍历 8 个 Expert，无法并行
3. **全 FP32 推理** — 未使用半精度，计算量翻倍
4. **逐 batch 串行** — 2039 个 batch 逐个处理，无 overlap
5. **无 torch.compile** — 未利用 PyTorch 2.x 编译优化
6. **RepEncoder 逐 slot 循环** — 28 个 slot 逐个 segment_reduce

## 关键约束

- **本机无法运行推理**（Mac 无 NVIDIA GPU）
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
| `dataset/` | 数据集（30GB，已下载，不可修改） |
| `weights/` | 模型权重（9.9GB，已下载，分片需合并） |
| `docs/论文/` | GRAB 论文 + HSTU 论文 |

## AI Studio 访问

- Token: `d4606d269ac6e051277e42d1146eea723ec208b5`（可用于 aistudio-sdk 下载模型/数据集）
- 项目代码（infer.py 等）无法通过 API 获取，只能通过浏览器访问
- 云端环境：V100 16GB GPU，每日 8 算力点
