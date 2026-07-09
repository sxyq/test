# submit1 selective PTQ 候选清单

更新时间：2026-07-09

## 1. 目标

这份清单只服务于当前总主线：

- `V201-TRT-MOE-UNQUANT-sxyq`

目标不是做 PyTorch eager INT8，也不是做 CPU dynamic quant，而是为后续真正的：

1. `Selective PTQ`
2. `ONNX / TensorRT Q-DQ`
3. `plugin 保持 FP16，外围 dense 子图量化`

先把可量化边界划清楚。

## 2. 当前策略

优先级固定如下：

1. `MoE plugin = FP16`，先不动
2. `MoE gate / routing logits = FP16`，先不动
3. 只先碰非 MoE dense 子图
4. 先看 `Linear`，不先碰 `Embedding`

## 3. 第一批候选

第一批“优先考虑 selective PTQ”的模块：

1. `rep_encoder.linear`
2. `linear`
3. `seq_encoder.qkv_proj.*`
4. `seq_encoder.out_proj.*`
5. `seq_encoder.ffn1.*`
6. `seq_encoder.ffn2.*`

这些模块的共同特征是：

1. 结构规则；
2. 不直接属于 TRT MoE plugin 打包专家权重的路径；
3. 更接近标准 Transformer dense 子图；
4. 更适合做 Q-DQ / PTQ 起步实验。

## 4. 暂时保持 FP16 的模块

### 4.1 plugin-sensitive

先不要碰：

1. `seq_encoder.moe.*.experts.*.fc1`
2. `seq_encoder.moe.*.experts.*.fc2`

原因：

这些权重会被 `prepare_trt_moe()` 打包进 TRT plugin 路径，当前不适合直接替换成量化线。

### 4.2 routing-sensitive

先不要碰：

1. `seq_encoder.moe.*.gate.w_g`

原因：

这条线直接控制 expert 选择，属于 routing-sensitive 模块。哪怕精度波动很小，也可能把路由决策和最终 PCOC 一起带偏。

## 5. 本地脚本

已新增脚本：

- [research/submit1_quant/selective_ptq_inventory.py](/Users/sunyiyang/Desktop/Project/Baidu%20%20GRAB/research/submit1_quant/selective_ptq_inventory.py)

用途：

1. 基于 `V201` 自动列出全部 `Linear` 模块；
2. 标记为：
   - `safe_ptq_candidate`
   - `routing_sensitive_hold_fp16`
   - `plugin_sensitive_hold_fp16`
   - `review_manually`
3. 输出模块名、shape、参数量。

## 6. 结论

`V203` 不是这条策略的真正实现。

`V203` 做的是：

1. CPU dynamic quant
2. PyTorch `quantize_dynamic`
3. `load_model()` 后一次性替换 `Linear`

而当前这份清单对应的才是你之前文档里要求的量化主线：

1. `V201` 为底座
2. 只量化非 MoE dense 子图
3. `plugin/gate` 先保持 FP16
4. 后续往 `ONNX / TensorRT / Q-DQ / PTQ` 接
