@echo off
REM HDTF 数据集下载脚本 (Windows)
REM 使用方法: download_hdtf.bat [mirror]

echo ========================================
echo HDTF Dataset Downloader
echo ========================================

REM 检查 Python
python --version >nul 2>&1
if errorlevel 1 (
    echo Error: Python not found
    exit /b 1
)

REM 安装依赖
echo Installing dependencies...
pip install huggingface_hub tqdm -q

REM 检查是否使用镜像
if "%1"=="mirror" (
    echo Using HuggingFace mirror...
    python scripts/download_hdtf.py --output_dir ./dataset/HDTF/source --mirror --workers 4
) else (
    python scripts/download_hdtf.py --output_dir ./dataset/HDTF/source --workers 4
)

if errorlevel 1 (
    echo Download failed!
    exit /b 1
)

echo.
echo ========================================
echo Download complete!
echo Next step: Run preprocessing
echo   python scripts/preprocess.py --config configs/training/preprocess.yaml
echo ========================================
pause
