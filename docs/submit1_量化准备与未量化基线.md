# submit1 量化准备与未量化基线

更新时间：2026-07-09

## 1. 未量化可提交基线

已从 `/Users/sunyiyang/Downloads/submit (1).zip` 落出未量化提交基线：

- 目录：`submission/V201-TRT-MOE-UNQUANT-sxyq`
- 压缩包：`submission/V201-TRT-MOE-UNQUANT-sxyq.zip`

已完成检查：

1. `infer.py` 可通过 `python3 -m py_compile`
2. zip 根目录仅包含：
   - `infer.py`
   - `build_env.sh`
   - `requirements.txt`
   - `libbaidu_moe_top2_plugin.so`
   - `libnvinfer.so.10`

这版保持了 `submit (1)` 的原始未量化形态，不额外叠加本地 `bias / balance / prune / force rebuild` 变量。

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
