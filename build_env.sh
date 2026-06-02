#!/bin/bash
# 百度生成式推荐广告排序推理性能优化
# 环境构建脚本
# 注意：此脚本在评测环境中执行，需在20分钟内完成

set -e

echo "=========================================="
echo "开始构建环境..."
echo "=========================================="

# 方式1：使用预编译的tar包（推荐，更快）
# 下载预编译依赖包
# wget https://studio-package.bj.bcebos.com/2026-shangye-python-package/external-libraries.tar -O libraries.tar
# tar -xf libraries.tar --strip-components=1 -C libraries

# 方式2：自行安装依赖
pip install uv
uv pip install -r requirements.txt --target ./libraries -i https://mirrors.aliyun.com/pypi/simple/

echo "=========================================="
echo "环境构建完成！"
echo "=========================================="
