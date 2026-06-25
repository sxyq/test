#!/bin/bash
# 百度生成式推荐广告排序推理性能优化
# 环境构建脚本
# 注意：此脚本在评测环境中执行，需在20分钟内完成

set -e

echo "=========================================="
echo "开始构建环境..."
echo "=========================================="

# infer.py 会优先从 /home/aistudio/libraries 加载依赖；评测时当前目录通常是 /home/aistudio/code。
LIB_DIR="../libraries"
mkdir -p "${LIB_DIR}"

# 方式1：使用预编译 tar 包，避免在线安装 torch/paddlepaddle-gpu 超时或版本不匹配。
if [ ! -d "${LIB_DIR}/torch" ]; then
    if [ ! -f "libraries.tar" ]; then
        wget https://studio-package.bj.bcebos.com/2026-shangye-python-package/external-libraries.tar -O libraries.tar
    fi
    tar -xf libraries.tar --strip-components=1 -C "${LIB_DIR}"
fi

# 方式2：补齐 requirements 中预编译包未覆盖的轻量依赖。
if [ -x "${LIB_DIR}/bin/uv" ]; then
    "${LIB_DIR}/bin/uv" pip install -r requirements.txt --target "${LIB_DIR}" -i https://mirrors.aliyun.com/pypi/simple/ || true
else
    pip install uv -i https://mirrors.aliyun.com/pypi/simple/ || true
    uv pip install -r requirements.txt --target "${LIB_DIR}" -i https://mirrors.aliyun.com/pypi/simple/ || true
fi

# V139: 安装 flash-attn 以启用 flash_attn_varlen_func 路径
# 这将消除 per-user SDPA Python 循环，用单次 kernel 调用替代
echo "=========================================="
echo "V139: 安装 flash-attn (可选，加速注意力计算)..."
echo "=========================================="

# 先安装 ninja 加速编译
pip install ninja -i https://mirrors.aliyun.com/pypi/simple/ 2>/dev/null || true

# 安装 flash-attn，设置 PYTHONPATH 让编译时找到 PyTorch
export PYTHONPATH="${LIB_DIR}:${PYTHONPATH}"
pip install flash-attn --no-build-isolation --target "${LIB_DIR}" -i https://mirrors.aliyun.com/pypi/simple/ 2>/dev/null || {
    echo "[WARNING] flash-attn 安装失败，将使用 fallback SDPA 路径"
}

# V156: 安装 xformers 以启用 memory_efficient_attention 路径
# xformers 是 V100 (SM70) 上最可靠的注意力加速后端，明确支持 SM70+
# 不像 flash-attn 可能在 SM70 上静默回退到 SDPA
echo "=========================================="
echo "V156: 安装 xformers (V100/SM70 原生支持注意力加速)..."
echo "=========================================="
pip install xformers --target "${LIB_DIR}" -i https://mirrors.aliyun.com/pypi/simple/ 2>/dev/null || {
    echo "[WARNING] xformers 安装失败，将使用 flash-attn 或 SDPA fallback 路径"
}

echo "=========================================="
echo "环境构建完成！"
echo "=========================================="
