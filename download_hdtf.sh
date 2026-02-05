#!/bin/bash
# HDTF 数据集下载脚本 (Linux/Mac)
# 使用方法: 
#   bash download_hdtf.sh          # 直接下载
#   bash download_hdtf.sh mirror   # 使用镜像加速

echo "========================================"
echo "HDTF Dataset Downloader"
echo "========================================"

# 检查 Python
if ! command -v python &> /dev/null; then
    echo "Error: Python not found"
    exit 1
fi

# 安装依赖
echo "Installing dependencies..."
pip install huggingface_hub tqdm -q

# 检查是否使用镜像
if [ "$1" == "mirror" ]; then
    echo "Using HuggingFace mirror..."
    python scripts/download_hdtf.py --output_dir ./dataset/HDTF/source --mirror --workers 4
else
    python scripts/download_hdtf.py --output_dir ./dataset/HDTF/source --workers 4
fi

if [ $? -ne 0 ]; then
    echo "Download failed!"
    exit 1
fi

echo ""
echo "========================================"
echo "Download complete!"
echo "Next step: Run preprocessing"
echo "  python scripts/preprocess.py --config configs/training/preprocess.yaml"
echo "========================================"
