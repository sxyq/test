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
