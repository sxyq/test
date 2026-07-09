# submit1 量化准备与未量化基线

更新时间：2026-07-09

## 1. 未量化可提交基线

已从 `/Users/sunyiyang/Downloads/submit (1).zip` 落出未量化提交基线：

- 目录：`submission/V201-TRT-MOE-UNQUANT-sxyq`
- 压缩包：`submission/V201-TRT-MOE-UNQUANT-sxyq.zip`

基于“离线批任务、吞吐优先”的新候选：

- 目录：`submission/V202-TRT-MOE-THROUGHPUT-sxyq`
- 压缩包：`submission/V202-TRT-MOE-THROUGHPUT-sxyq.zip`

基于“在线动态量化研究”的受限实现：

- 目录：`submission/V203-ONLINE-DYNQUANT-sxyq`
- 压缩包：`submission/V203-ONLINE-DYNQUANT-sxyq.zip`

基于“按文档策略收敛后的 selective PTQ prep 研究版（当前编号）”：

- 目录：`submission/V205-SELECTIVE-PTQ-PREP-sxyq`
- 压缩包：`submission/V205-SELECTIVE-PTQ-PREP-sxyq.zip`

保留上一编号副本：

- 目录：`submission/V204-SELECTIVE-QUANT-sxyq`
- 压缩包：`submission/V204-SELECTIVE-QUANT-sxyq.zip`

已完成检查：

1. `infer.py` 可通过 `python3 -m py_compile`
2. zip 根目录仅包含：
   - `infer.py`
   - `build_env.sh`
   - `requirements.txt`
   - `libbaidu_moe_top2_plugin.so`
   - `libnvinfer.so.10`

这版保持了 `submit (1)` 的原始未量化形态，不额外叠加本地 `bias / balance / prune / force rebuild` 变量。

线上结果已确认：

- `V201`: `73.93993 / PCOC=1.05895 / AUC=0.75261 / latency=21.85059s`
- `V202`: `71.57366 / PCOC=1.05932 / AUC=0.75267 / latency=31.99176s`
- `V203`: `73.84160 / PCOC=1.05895 / AUC=0.75261 / latency=22.27200s`
- `V204`: `73.19211 / PCOC=1.05895 / AUC=0.75261 / latency=25.05551s`
- `V205`: `73.84305 / PCOC=1.05895 / AUC=0.75261 / latency=22.26579s`
- `V206`: `73.64762 / PCOC=1.05895 / AUC=0.75261 / latency=23.10335s`
- `V207`: `74.03155 / PCOC=1.05895 / AUC=0.75260 / latency=21.45791s`
- `V208`: `72.59492 / PCOC=1.05932 / AUC=0.75267 / latency=27.61493s`

当前结论：

1. `V207` 已经成为当前总主线；
2. `V201` 仍然是最重要的原始未量化参考底座；
3. `V202` 虽然也有效，但没有超过 `V201`；
4. `V203` 虽然分数接近 `V201`，但这不表示当前 TRT 主线已经真正吃到了量化收益；
5. `V204` 已经进一步证明“接近文档但仍混有 runtime selective quant 语义”的版本，不仅没超过 `V201`，还比 `V203` 明显更慢；
6. `V205` 说明：严格按文档收敛后的 PTQ-prep 边界版，默认 scoring path 仍然基本贴着 `V201` 运行，没有引入新的明显回退；
7. `V206` 说明：把“全量 INT8 研究路径 + embedding INT8”激进叠加到当前主线上，并没有换来更好的线上结果；
8. `V207` 反而给出了最有价值的新结论：只保留 `RepEncoder embedding INT8` 这一条主变量，既超过了 `V201`，也超过了 `V205/V206`；
9. `V208` 则进一步证明：把“embedding INT8 + 吞吐默认项 + 更大叠加组合”一次性堆满，会明显破坏当前主线收益；
10. 因此后续量化、runtime、plugin 相关实验，应改成以 `V207` 为当前主线、以 `V201` 为原始未量化参考底座，并继续保留 `V205` 作为 PTQ-prep 分支。

## 2. 本地已准备好的量化/分析工具

本地虚拟环境：

- `./.venv-quant`

已安装：

- `onnx`
- `onnxruntime`
- `polygraphy`

说明：

1. 当前机器是 Apple M5，本地没有 CUDA / TensorRT / `trtexec` / `nsys`
2. 因此可以先做：
   - ONNX 图级分析
   - ONNX Runtime CPU 侧可执行性检查
   - Polygraphy 基础图检查
3. 不能在本机直接完成：
   - TensorRT engine 构建
   - TRT PTQ 真正校准
   - plugin 低精度 kernel 实测

## 3. 这条线的量化切入点

当前 `submit (1)` 路线不适合直接做全图量化。

更合理的顺序是：

1. 保持 `MoE plugin = FP16`
2. 优先研究 `Selective PTQ`
3. 只考虑非 MoE 子图：
   - `RepEncoder` 中可导出、可替换的 dense 子路径
   - attention / linear 等常规子图
4. 最后再判断：
   - `SmoothQuant / CLE`
   - `W8A8`
   - `FP8 mixed precision`

## 4. 当前最值得继续做的非量化单变量

在进入量化前，更建议优先验证这几个“代码已存在但默认未开”的点：

1. `BAIDU_FUSED_TOP2=1`
2. `BAIDU_REP_COUNT_MATMUL_ENABLE=1`
3. `BAIDU_REP_NORM_LINEAR_FUSE=1`
4. `pin_memory + non_blocking H2D`
5. `timing/profile cache`

原因：

- 这些改动更贴近当前 73 分路线的主收益来源
- 与当前 plugin/kernel 主线冲突更小
- 即使后续做量化，也需要先把未量化强基线吃满

## 5. V202 吞吐版默认叠加项

`V202` 不改模型结构，专门把当前 73 分路线改成更偏离线吞吐的默认配置：

1. 默认开启 `BAIDU_FUSED_TOP2`
2. 默认开启 `BAIDU_REP_COUNT_MATMUL_ENABLE`
3. 默认开启 `BAIDU_REP_NORM_LINEAR_FUSE`
4. 默认开启 `pin_memory + non_blocking H2D`
5. 默认对 grouped batches 做 `pin_memory`
6. 默认启用 `BAIDU_BATCH_GROUP_TOKEN_CAP_DEFAULT=300000`

这版的设计目标不是压首包 latency，而是在 `cached_batches -> grouped batches -> TRT-MoE plugin` 这条离线批链路上提高整体吞吐。

## 6. 在线量化思路的可行性结论

结论要分开看：

1. **对当前 CUDA + TRT-MoE plugin 主线，不可直接作为有效线上提速方案**
   - 当前实现不是 TensorRT engine build / Q-DQ graph 路线；
   - `MoE plugin`、`custom embedding kernels`、`FP16 seq path` 都依赖现有浮点/半精度执行链；
   - `load_model()` 后直接套 PyTorch dynamic quantization，不会把这条 CUDA/plugin 路径变成真正的 TRT INT8/FP8 推理。
   - `V203` 的线上结果也支持这个判断：它与 `V201` 的 `PCOC/AUC` 完全一致，且 latency 只小幅波动，更像 CUDA 路径下直接跳过 CPU-only dynamic quant，而不是实际执行了你文档里要的 selective PTQ / Q-DQ / TensorRT 量化。

2. **对 CPU / 无 TRT 的研究路径，可行**
   - `V203` 是严格基于 `V201` 制作，不继承 `V202` 的吞吐默认项；
   - 已在 `V203` 中实现受控开关：
     - `BAIDU_ONLINE_DYNAMIC_QUANT=1`
     - `BAIDU_ONLINE_DYNAMIC_QUANT_TARGET=all|seq_only|rep_seq_split`
     - `BAIDU_ONLINE_DYNAMIC_QUANT_DTYPE=qint8|float16`
   - 实现方式是：`load_model()` 后、真正推理前，对 `nn.Linear` 做一次性 `torch.ao.quantization.quantize_dynamic(...)`
   - 当前本机已实测跑通，默认会自动选择可用 quantized backend（本机为 `qnnpack`）
   - 这版不需要额外第三方 Python 包；依赖的是运行时 `torch` 自带的 `torch.ao.quantization`

3. **这条实现的定位**
   - 它是“在线量化研究版”，不是当前 TRT 提交主线；
   - 更适合做：
     - CPU fallback 对照
     - 非 TRT 子图量化可行性验证
     - 后续 selective PTQ / Q-DQ 重构前的模块边界摸底

## 7. V203 是否符合既定量化策略

不符合。

你之前文档里要求的量化主线是：

1. `V201` 作为唯一底座；
2. `MoE plugin = FP16` 保持不动；
3. 只对非 MoE 子图做 `Selective PTQ`；
4. 优先走 `ONNX / TensorRT / Q-DQ / PTQ` 路线；
5. PTQ 不够，再考虑混合精度或更激进量化。

而 `V203` 实际做的是：

1. `load_model()` 后的 PyTorch CPU dynamic quant；
2. 量化对象是 `nn.Linear`；
3. 只在 CPU / 非 TRT 路径生效；
4. 不是 TensorRT PTQ，也不是 Q-DQ graph。

所以 `V203` 的作用是：

- 验证“在线量化后再计算”这件事在代码层面能不能挂进去；
- 证明它不会自动等价于你要的 TRT selective PTQ 主线；
- 作为后续真正 `Selective PTQ` 的研究起点，而不是结果本身。

## 8. V205 的定位

`V205` 是严格按文档边界修正后的当前编号版本。

它的特点是：

1. 仍然严格基于 `V201`
2. 不继承 `V202` 的吞吐默认项
3. 默认推理路径不做 runtime quant
4. 不碰 `MoE plugin`
5. 不碰 `MoE gate / routing`
6. 只暴露非 MoE dense 子图的 selective PTQ prep 边界

当前已验证的 selective PTQ prep 边界：

1. 作为后续 PTQ 候选：
   - `rep_encoder.linear`
   - `seq_encoder.qkv_proj.*`
   - `seq_encoder.out_proj.*`
   - `seq_encoder.ffn1.*`
   - `seq_encoder.ffn2.*`
   - `linear`
2. 明确保持 FP16 / 浮点：
   - `seq_encoder.moe.*.gate.w_g`
   - `seq_encoder.moe.*.experts.*.fc1`
   - `seq_encoder.moe.*.experts.*.fc2`

所以 `V205` 不是“又一个 runtime 量化包”，而是一个和文档完全一致的 PTQ-prep 版本：

1. 不再偷偷做 eager/runtime quant；
2. 默认推理路径回到 `V201`；
3. 只保留后续 `ONNX / TensorRT / Q-DQ / PTQ` 需要的量化对象边界与清单导出能力。

`V204` 继续保留，作为上一编号的同内容副本，不删除、不覆盖。

## 9. V204 的最终位置

`V204` 现在可以定性为：

- 已线上验证；
- 但不是当前量化主线；
- 也不是文档严格对齐版。

它的实际结果是：

- `score_all=73.19211`
- `PCOC=1.05895`
- `AUC=0.75261`
- `latency=25.05551s`

相对 `V201`：

- 分数低 `-0.74782`
- latency 高 `+3.20492s`

相对 `V203`：

- 分数低 `-0.64949`
- latency 高 `+2.78351s`

所以 `V204` 的意义主要是：

1. 保留“从 runtime/selective quant 过渡到文档严格 PTQ-prep”这段版本演进痕迹；
2. 作为反证样本，说明只要量化语义没有严格收敛到文档要求，最终就很容易变成既没超过 `V201`，又没有形成真正可继续放大的 TRT PTQ 路线；
3. 后续真正的量化推进，应直接跳过 `V204` 语义，转到 `V205 -> selective PTQ manifest -> ONNX / TensorRT / Q-DQ` 这条线上。

## 10. 外部 embedding INT8 迭代包给出的信息

额外核对了外部包：

- `submit_iter_002_embedding_int8_20260709_162935.zip`
- 线上结果：`73.73159 / PCOC=1.05932 / AUC=0.75248 / latency=22.74346s`

源码层面，它默认新增的核心点不是“真正的 TRT selective PTQ”，而是：

1. `RepEncoder embedding INT8` 预量化缓存
2. 自定义 `embedding_gather_12_int8_strided`
3. 自定义 `embedding_bag_16_int8_strided`

同时它虽然也带了 `packed H2D` 代码，但默认不是开启态，因此不能把这版结果归因到 `packed H2D`。

对照结论：

- 相对 `V201`：
  - 分数低 `-0.20834`
  - latency 高 `+0.89287s`
  - AUC 低 `-0.00013`
- 相对 `V205`：
  - 分数低 `-0.11146`
  - latency 高 `+0.47767s`
  - AUC 低 `-0.00013`

因此它提供的信息很明确：

1. 单独做 `RepEncoder embedding INT8`，并不能自动超过 `V201`；
2. 这条 embedding INT8 路线至少在当前实现里，不是“白拿吞吐”的低风险项；
3. 它更像是一个有轻微精度扰动、但没有换来确定端到端收益的局部 kernel 实验；
4. 这反而进一步支持：后续量化主线应继续沿 `V205 -> 非 MoE dense 子图 selective PTQ` 推进，而不是优先把 embedding INT8 当默认主项。

## 11. V206 全量 INT8 研究版给出的信息

`V206-FULL-INT8-sxyq` 已线上验证：

- `73.64762 / PCOC=1.05895 / AUC=0.75261 / latency=23.10335s`

相对 `V201`：

- 分数低 `-0.29231`
- latency 高 `+1.25276s`

相对 `V205`：

- 分数低 `-0.19543`
- latency 高 `+0.83756s`

这版的关键含义是：

1. “激进全量 INT8 研究版”没有超过当前主线；
2. `AUC/PCOC` 仍完全一致，说明它没有引入明显校准漂移；
3. 但 latency 和总分都回退，说明这次 INT8 叠加并没有在当前 TRT/CUDA 主链路上兑现收益；
4. 结合实现方式，`V206` 更像是在验证“这类 eager/runtime INT8 + embedding INT8 组合能否直接迁移到当前主线”，结论是否定的。

因此，`V206` 应归为：

- 已验证的高风险反证样本；
- 不是后续量化主线；
- 也不应替代 `V205` 作为 PTQ 起点。

## 12. V207 embedding INT8-only 版给出的信息

`V207-EMB-INT8-ONLY-sxyq` 已线上验证：

- `74.03155 / PCOC=1.05895 / AUC=0.75260 / latency=21.45791s`

相对 `V201`：

- 分数高 `+0.09162`
- latency 低 `-0.39268s`
- AUC 仅低 `-0.00001`

相对 `V205`：

- 分数高 `+0.18850`
- latency 低 `-0.80788s`

相对 `V206`：

- 分数高 `+0.38393`
- latency 低 `-1.64544s`

这版的意义很明确：

1. 当前真正有效的不是“全量 INT8”，而是更收敛的 `embedding INT8` 单变量；
2. 它是目前第一条已经在线上实测打赢 `V201` 的量化相关分支；
3. `RepEncoder embedding INT8` 不应再被当成边缘试探项，而应升级为当前 TRT 主线上的正式方向；
4. 后续如果继续叠加其他变量，都应该先以 `V207` 为当前主线继续，而不是回退到 `V206`。

## 13. V208 最大叠加高风险版给出的信息

`V208-MAX-STACKED-HIGHRISK-sxyq` 已线上验证：

- `72.59492 / PCOC=1.05932 / AUC=0.75267 / latency=27.61493s`

相对 `V207`：

- 分数低 `-1.43663`
- latency 高 `+6.15702s`

相对 `V201`：

- 分数低 `-1.34501`
- latency 高 `+5.76434s`

这版的结论非常直接：

1. “尽可能多叠加”在当前主链路上是负收益；
2. `embedding INT8` 的正收益，一旦和大批吞吐默认项、搬运优化、grouping、warmup 一起默认叠加，就会被明显破坏；
3. 当前最该避免的，不是量化本身，而是把太多变量在一版里同时打开；
4. `V208` 应作为最大叠加失败样本保留，不应继续沿这条整包默认叠加路线推进。
