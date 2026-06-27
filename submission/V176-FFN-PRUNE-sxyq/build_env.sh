#!/bin/bash
set -e

echo "=========================================="
echo "开始构建环境..."
echo "=========================================="

LIB_DIR="../libraries"
mkdir -p "${LIB_DIR}"

if [ ! -d "${LIB_DIR}/torch" ]; then
    if [ ! -f "libraries.tar" ]; then
        wget https://studio-package.bj.bcebos.com/2026-shangye-python-package/external-libraries.tar -O libraries.tar
    fi
    tar -xf libraries.tar --strip-components=1 -C "${LIB_DIR}"
fi

if [ -x "${LIB_DIR}/bin/uv" ]; then
    "${LIB_DIR}/bin/uv" pip install -r requirements.txt --target "${LIB_DIR}" -i https://mirrors.aliyun.com/pypi/simple/ || true
else
    pip install uv -i https://mirrors.aliyun.com/pypi/simple/ || true
    uv pip install -r requirements.txt --target "${LIB_DIR}" -i https://mirrors.aliyun.com/pypi/simple/ || true
fi

echo "Attempting flash-attn precompiled wheel (300s timeout)..."
# V138: Increase timeout to 300s for flash-attn wheel download.
# flash-attn provides varlen API that replaces per-user SDPA loop,
# saving ~15s latency (160 kernel launches -> 1 fused kernel per batch).
timeout 300 pip install flash-attn --no-build-isolation -f https://flash-attn.github.io/whl/ 2>/dev/null || echo "flash-attn skipped (will use SDPA fallback)"

echo "=========================================="
echo "环境构建完成！"
echo "=========================================="
