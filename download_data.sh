#!/bin/bash
# 百度生成式推荐广告排序推理性能优化
# 数据集和模型下载脚本

set -e

echo "=========================================="
echo "开始下载数据集和模型..."
echo "=========================================="

# 检查是否安装了 aistudio-sdk
if ! command -v aistudio &> /dev/null; then
    echo "正在安装 aistudio-sdk..."
    pip install --upgrade aistudio-sdk
fi

# 创建目录
mkdir -p dataset weights

# 下载数据集
echo "正在下载数据集..."
aistudio download --dataset gump/2026_cti_data --local_dir ./dataset

# 下载模型权重
echo "正在下载模型权重..."
aistudio download --model gump/2026_cti_model --local_dir ./weights

echo "=========================================="
echo "下载完成！"
echo "数据集位置: ./dataset"
echo "模型权重位置: ./weights"
echo "=========================================="
