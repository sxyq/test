#!/bin/bash
set -e

echo "=========================================="
echo "开始构建环境..."
echo "=========================================="

LIB_DIR="../libraries"
mkdir -p "${LIB_DIR}"

# 方式1：使用预编译 tar 包
if [ ! -d "${LIB_DIR}/torch" ]; then
    if [ ! -f "libraries.tar" ]; then
        wget https://studio-package.bj.bcebos.com/2026-shangye-python-package/external-libraries.tar -O libraries.tar
    fi
    tar -xf libraries.tar --strip-components=1 -C "${LIB_DIR}"
fi

# 方式2：补齐轻量依赖
if [ -x "${LIB_DIR}/bin/uv" ]; then
    "${LIB_DIR}/bin/uv" pip install -r requirements.txt --target "${LIB_DIR}" -i https://mirrors.aliyun.com/pypi/simple/ || true
else
    pip install uv -i https://mirrors.aliyun.com/pypi/simple/ || true
    uv pip install -r requirements.txt --target "${LIB_DIR}" -i https://mirrors.aliyun.com/pypi/simple/ || true
fi

# V117: 尝试安装 flash-attn 预编译 wheel（带超时保护）
# 如果有匹配当前 CUDA/PyTorch 版本的预编译 wheel，安装只需几秒
# 如果需要从源码编译，90秒超时后会自动跳过
echo "Attempting flash-attn precompiled wheel installation (90s timeout)..."
timeout 90 pip install flash-attn --no-build-isolation -f https://flash-attn.github.io/whl/ 2>/dev/null || echo "flash-attn installation skipped (no precompiled wheel or timeout)"

echo "=========================================="
echo "环境构建完成！"
echo "=========================================="
