"""
HDTF Dataset Download Script
============================
从 Hugging Face 下载 HDTF 数据集并准备用于 MuseTalk 训练

使用方法:
    python scripts/download_hdtf.py --output_dir ./dataset/HDTF/source
    
    # 使用镜像站加速 (国内推荐)
    python scripts/download_hdtf.py --output_dir ./dataset/HDTF/source --mirror
    
    # 只下载部分数据 (测试用)
    python scripts/download_hdtf.py --output_dir ./dataset/HDTF/source --max_files 10
"""

import os
import sys
import argparse
import shutil
from pathlib import Path
from typing import Optional, List
from concurrent.futures import ThreadPoolExecutor, as_completed
from tqdm import tqdm

def check_dependencies():
    """检查必要的依赖"""
    missing = []
    try:
        from huggingface_hub import HfApi, hf_hub_download, snapshot_download
    except ImportError:
        missing.append("huggingface_hub")
    
    if missing:
        print(f"缺少依赖: {', '.join(missing)}")
        print(f"请运行: pip install {' '.join(missing)}")
        sys.exit(1)

check_dependencies()

from huggingface_hub import HfApi, hf_hub_download, snapshot_download, list_repo_files

# 数据集信息
DATASET_REPO = "global-optima-research/HDTF"
HF_MIRROR = "https://hf-mirror.com"


def setup_mirror(use_mirror: bool = False):
    """设置 HuggingFace 镜像"""
    if use_mirror:
        os.environ["HF_ENDPOINT"] = HF_MIRROR
        print(f"使用镜像站: {HF_MIRROR}")


def get_file_list(repo_id: str) -> List[str]:
    """获取数据集文件列表"""
    api = HfApi()
    try:
        files = list_repo_files(repo_id, repo_type="dataset")
        # 只保留视频文件
        video_files = [f for f in files if f.endswith(('.mp4', '.avi', '.mov', '.mkv'))]
        return video_files
    except Exception as e:
        print(f"获取文件列表失败: {e}")
        return []


def download_single_file(
    repo_id: str,
    filename: str,
    output_dir: str,
    token: Optional[str] = None
) -> bool:
    """下载单个文件"""
    try:
        local_path = hf_hub_download(
            repo_id=repo_id,
            filename=filename,
            repo_type="dataset",
            local_dir=output_dir,
            token=token,
        )
        return True
    except Exception as e:
        print(f"下载失败 {filename}: {e}")
        return False


def download_dataset_snapshot(
    output_dir: str,
    token: Optional[str] = None,
    max_workers: int = 4
) -> bool:
    """使用 snapshot_download 下载整个数据集"""
    print(f"开始下载数据集到: {output_dir}")
    
    try:
        snapshot_download(
            repo_id=DATASET_REPO,
            repo_type="dataset",
            local_dir=output_dir,
            token=token,
            max_workers=max_workers,
            ignore_patterns=["*.md", "*.txt", "*.json"],  # 忽略非视频文件
        )
        print("数据集下载完成!")
        return True
    except Exception as e:
        print(f"snapshot_download 失败: {e}")
        return False


def download_dataset_selective(
    output_dir: str,
    max_files: Optional[int] = None,
    token: Optional[str] = None,
    max_workers: int = 4
) -> bool:
    """选择性下载数据集文件"""
    print("获取文件列表...")
    files = get_file_list(DATASET_REPO)
    
    if not files:
        print("未找到视频文件，尝试完整下载...")
        return download_dataset_snapshot(output_dir, token, max_workers)
    
    print(f"找到 {len(files)} 个视频文件")
    
    if max_files:
        files = files[:max_files]
        print(f"限制下载前 {max_files} 个文件")
    
    os.makedirs(output_dir, exist_ok=True)
    
    success_count = 0
    failed_files = []
    
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = {
            executor.submit(download_single_file, DATASET_REPO, f, output_dir, token): f
            for f in files
        }
        
        with tqdm(total=len(files), desc="下载进度") as pbar:
            for future in as_completed(futures):
                filename = futures[future]
                if future.result():
                    success_count += 1
                else:
                    failed_files.append(filename)
                pbar.update(1)
    
    print(f"\n下载完成: {success_count}/{len(files)} 成功")
    
    if failed_files:
        print(f"失败文件 ({len(failed_files)}):")
        for f in failed_files[:10]:
            print(f"  - {f}")
        if len(failed_files) > 10:
            print(f"  ... 还有 {len(failed_files) - 10} 个")
    
    return success_count > 0


def reorganize_files(output_dir: str):
    """重新组织文件结构，确保所有视频在 source 目录"""
    output_path = Path(output_dir)
    
    # 查找所有视频文件
    video_extensions = {'.mp4', '.avi', '.mov', '.mkv'}
    video_files = []
    
    for ext in video_extensions:
        video_files.extend(output_path.rglob(f"*{ext}"))
    
    if not video_files:
        print("未找到视频文件")
        return
    
    # 移动到根目录
    moved_count = 0
    for video_file in video_files:
        if video_file.parent != output_path:
            dest = output_path / video_file.name
            if not dest.exists():
                shutil.move(str(video_file), str(dest))
                moved_count += 1
    
    if moved_count > 0:
        print(f"重新组织了 {moved_count} 个文件")
    
    # 清理空目录
    for dirpath, dirnames, filenames in os.walk(output_path, topdown=False):
        if dirpath != str(output_path) and not os.listdir(dirpath):
            os.rmdir(dirpath)


def verify_dataset(output_dir: str) -> dict:
    """验证下载的数据集"""
    output_path = Path(output_dir)
    
    video_extensions = {'.mp4', '.avi', '.mov', '.mkv'}
    video_files = []
    
    for ext in video_extensions:
        video_files.extend(output_path.glob(f"*{ext}"))
    
    total_size = sum(f.stat().st_size for f in video_files)
    
    stats = {
        "video_count": len(video_files),
        "total_size_gb": total_size / (1024**3),
        "output_dir": str(output_path),
    }
    
    print("\n=== 数据集统计 ===")
    print(f"视频数量: {stats['video_count']}")
    print(f"总大小: {stats['total_size_gb']:.2f} GB")
    print(f"保存位置: {stats['output_dir']}")
    
    return stats


def main():
    parser = argparse.ArgumentParser(
        description="下载 HDTF 数据集",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例:
    # 基本下载
    python scripts/download_hdtf.py --output_dir ./dataset/HDTF/source
    
    # 使用镜像加速 (国内推荐)
    python scripts/download_hdtf.py --output_dir ./dataset/HDTF/source --mirror
    
    # 测试下载 (只下载10个文件)
    python scripts/download_hdtf.py --output_dir ./dataset/HDTF/source --max_files 10
    
    # 使用 HF token (私有数据集)
    python scripts/download_hdtf.py --output_dir ./dataset/HDTF/source --token YOUR_TOKEN
        """
    )
    
    parser.add_argument(
        "--output_dir", 
        type=str, 
        default="./dataset/HDTF/source",
        help="数据集保存目录 (默认: ./dataset/HDTF/source)"
    )
    parser.add_argument(
        "--mirror", 
        action="store_true",
        help="使用 HuggingFace 镜像站 (国内加速)"
    )
    parser.add_argument(
        "--token", 
        type=str, 
        default=None,
        help="HuggingFace API token (可选)"
    )
    parser.add_argument(
        "--max_files", 
        type=int, 
        default=None,
        help="最大下载文件数 (用于测试)"
    )
    parser.add_argument(
        "--workers", 
        type=int, 
        default=4,
        help="并行下载线程数 (默认: 4)"
    )
    parser.add_argument(
        "--method",
        type=str,
        choices=["snapshot", "selective"],
        default="snapshot",
        help="下载方式: snapshot (完整下载) 或 selective (选择性下载)"
    )
    
    args = parser.parse_args()
    
    # 设置镜像
    setup_mirror(args.mirror)
    
    # 创建输出目录
    os.makedirs(args.output_dir, exist_ok=True)
    
    print(f"HDTF 数据集下载器")
    print(f"================")
    print(f"数据集: {DATASET_REPO}")
    print(f"输出目录: {args.output_dir}")
    print(f"下载方式: {args.method}")
    print()
    
    # 下载数据集
    if args.method == "snapshot" and args.max_files is None:
        success = download_dataset_snapshot(
            args.output_dir,
            token=args.token,
            max_workers=args.workers
        )
    else:
        success = download_dataset_selective(
            args.output_dir,
            max_files=args.max_files,
            token=args.token,
            max_workers=args.workers
        )
    
    if success:
        # 重新组织文件
        reorganize_files(args.output_dir)
        
        # 验证数据集
        verify_dataset(args.output_dir)
        
        print("\n下一步:")
        print("  运行预处理: python scripts/preprocess.py --config configs/training/preprocess.yaml")
    else:
        print("\n下载失败，请检查网络连接或尝试使用 --mirror 参数")
        sys.exit(1)


if __name__ == "__main__":
    main()
