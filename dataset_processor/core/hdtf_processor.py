# f:\python\Musetalk\dataset_processor\core\hdtf_processor.py
from pathlib import Path  # 导入路径模块
from typing import Any, Dict, List, Optional  # 导入类型注解

import cv2  # 导入 OpenCV
from omegaconf import OmegaConf  # 导入配置解析

from core.base_processor import BaseProcessor, ProcessorStoppedException  # 导入基类
from core.validator import DatasetValidator  # 导入验证器

DEFAULT_VAL_IDS = [  # 与 configs/training/preprocess.yaml 保持一致
    "RD_Radio7_000",
    "RD_Radio8_000",
    "RD_Radio9_000",
    "WDA_TinaSmith_000",
    "WDA_TomCarper_000",
    "WDA_TomPerez_000",
    "WDA_TomUdall_000",
    "WDA_VeronicaEscobar0_000",
    "WDA_VeronicaEscobar1_000",
    "WDA_WhipJimClyburn_000",
    "WDA_XavierBecerra_000",
    "WDA_XavierBecerra_001",
    "WDA_XavierBecerra_002",
    "WDA_ZoeLofgren_000",
    "WRA_SteveScalise1_000",
    "WRA_TimScott_000",
    "WRA_ToddYoung_000",
    "WRA_TomCotton_000",
    "WRA_TomPrice_000",
    "WRA_VickyHartzler_000",
]


class HDTFProcessor(BaseProcessor):  # HDTF 处理器
    """HDTF 数据集处理器"""  # 类说明

    def __init__(self, progress_cb, log_cb, config: Dict[str, Any]) -> None:  # 初始化函数
        super().__init__(progress_cb, log_cb, config)  # 调用父类初始化
        self.validator = DatasetValidator()  # 创建验证器
        self._init_preprocess_funcs()  # 初始化预处理函数

    def _init_preprocess_funcs(self) -> None:  # 初始化预处理函数
        try:  # 尝试导入现有脚本
            from scripts.preprocess import (  # 导入预处理函数
                convert_video,  # 视频转换
                segment_video,  # 视频分割
                extract_audio,  # 音频提取
                analyze_video,  # 视频分析
            )  # 结束导入
            self.convert_video = convert_video  # 绑定转换函数
            self.segment_video = segment_video  # 绑定分割函数
            self.extract_audio = extract_audio  # 绑定音频函数
            self.analyze_video = analyze_video  # 绑定分析函数
        except Exception:  # 导入失败
            self.convert_video = None  # 置空转换函数
            self.segment_video = None  # 置空分割函数
            self.extract_audio = None  # 置空音频函数
            self.analyze_video = None  # 置空分析函数

    def _resolve_source_root(self, source_dir: str) -> Path:  # 解析实际源目录
        root = Path(source_dir)  # 构造输入路径
        nested_source = root / "source"  # 组合嵌套 source 路径
        if nested_source.exists() and nested_source.is_dir():  # 判断是否为 datasets/HDTF 形式
            return nested_source  # 返回嵌套 source 目录
        return root  # 返回原始路径

    def _list_mp4_files(self, folder: Path) -> List[str]:  # 列出目录下 MP4 文件
        if not folder.exists() or not folder.is_dir():  # 判断目录有效性
            return []  # 返回空列表
        return sorted([f.name for f in folder.iterdir() if f.is_file() and f.suffix.lower() == ".mp4"])  # 返回排序后文件名

    def _resolve_nested_dir(self, folder: Path) -> Path:  # 解析同名嵌套目录
        nested = folder / folder.name  # 组合同名嵌套目录
        if nested.exists() and nested.is_dir():  # 判断嵌套目录存在
            return nested  # 返回嵌套目录
        return folder  # 返回原目录

    def _load_val_ids(self) -> List[str]:  # 读取验证集 ID 列表
        val_ids_cfg = self.config.get("val_list_hdtf", [])
        if isinstance(val_ids_cfg, list) and len(val_ids_cfg) > 0:
            return [str(v).strip() for v in val_ids_cfg if str(v).strip()]

        preprocess_yaml = str(self.config.get("preprocess_yaml", "./configs/training/preprocess.yaml"))
        cfg_path = Path(preprocess_yaml)
        if cfg_path.exists():
            try:
                cfg = OmegaConf.load(str(cfg_path))
                val_ids = cfg.get("val_list_hdtf", [])
                if val_ids:
                    return [str(v).strip() for v in val_ids if str(v).strip()]
            except Exception as exc:
                self.log_cb("WARNING", f"读取 {cfg_path} 失败，使用默认 val_list_hdtf: {exc}")

        return list(DEFAULT_VAL_IDS)

    def _write_split_file(self, path: Path, names: List[str]) -> None:  # 写入 train/val 列表文件
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w", encoding="utf-8") as f:
            f.write("file_name\n")
            for name in names:
                f.write(f"{name}\n")

    def _build_train_val_lists(self, meta_dir: Path, output_root: Path) -> bool:  # 根据 meta 生成 train/val 列表
        if not meta_dir.exists():
            self.log_cb("WARNING", f"未找到 meta 目录，无法生成 train/val: {meta_dir}")
            return False

        meta_files = sorted([f.name for f in meta_dir.glob("*.json") if f.is_file()])
        if not meta_files:
            self.log_cb("WARNING", f"meta 目录为空，无法生成 train/val: {meta_dir}")
            return False

        val_ids = self._load_val_ids()
        val_files = [name for name in meta_files if any(val_id in name for val_id in val_ids)]
        val_set = set(val_files)
        train_files = [name for name in meta_files if name not in val_set]

        train_path = output_root / "train.txt"
        val_path = output_root / "val.txt"
        self._write_split_file(train_path, train_files)
        self._write_split_file(val_path, val_files)
        self.log_cb("INFO", f"已生成 train/val 列表: train={len(train_files)}, val={len(val_files)}")
        self.log_cb("INFO", f"train: {train_path}")
        self.log_cb("INFO", f"val: {val_path}")
        return True

    def _detect_layout(self, source_dir: str) -> Dict[str, Any]:  # 检测 HDTF 数据布局
        source_root = self._resolve_source_root(source_dir)  # 解析源目录
        raw_videos = self._list_mp4_files(source_root)  # 检测根目录原始视频
        clips_dir = self._resolve_nested_dir(source_root / "clips")  # 解析 clips 目录（兼容 clips/clips）
        clip_videos = self._list_mp4_files(clips_dir)  # 检测 clips 目录视频
        if raw_videos:  # 若根目录直接有 mp4
            mode = "raw_videos"  # 标记原始视频模式
            check_dir = source_root  # 完整性检查目录为根目录
            check_files = raw_videos  # 完整性检查文件为原始视频
        elif clip_videos:  # 若仅有 clips 子目录
            mode = "prebuilt_clips"  # 标记已分段模式
            check_dir = clips_dir  # 完整性检查目录为 clips
            check_files = clip_videos  # 完整性检查文件为 clips 视频
        else:  # 两种模式都不匹配
            mode = "empty"  # 标记为空目录模式
            check_dir = source_root  # 默认检查目录
            check_files = []  # 默认空列表
        return {  # 返回布局信息
            "mode": mode,  # 返回检测模式
            "source_root": source_root,  # 返回解析后的源目录
            "raw_dir": source_root,  # 返回原始视频目录
            "raw_videos": raw_videos,  # 返回原始视频列表
            "clips_dir": clips_dir,  # 返回 clips 目录
            "clip_videos": clip_videos,  # 返回 clips 视频列表
            "check_dir": check_dir,  # 返回完整性检查目录
            "check_files": check_files,  # 返回完整性检查文件列表
        }  # 结束字典

    def check_integrity(self, source_dir: str) -> Dict[str, Any]:  # 完整性检查
        layout = self._detect_layout(source_dir)  # 检测目录布局
        self.log_cb("INFO", f"开始检查 HDTF 数据集: {layout['source_root']} (mode={layout['mode']})")  # 记录日志
        video_dir: Path = layout["check_dir"]  # 获取检查目录
        video_names: List[str] = layout["check_files"]  # 获取检查文件列表
        video_files = [video_dir / name for name in video_names]  # 构造完整路径列表
        total_videos = len(video_files)  # 统计数量
        total_size = sum(f.stat().st_size for f in video_files if f.exists()) / (1024 ** 3)  # 统计大小
        corrupt_files = []  # 初始化损坏列表
        for video_file in video_files:  # 遍历视频
            try:  # 尝试打开视频
                cap = cv2.VideoCapture(str(video_file))  # 打开视频
                if not cap.isOpened():  # 判断可读性
                    corrupt_files.append(video_file.name)  # 记录损坏
                cap.release()  # 释放资源
            except Exception:  # 捕获异常
                corrupt_files.append(video_file.name)  # 记录异常文件
        stats = {  # 统计结果
            "mode": layout["mode"],  # 布局模式
            "source_root": str(layout["source_root"]),  # 解析后源目录
            "video_dir": str(video_dir),  # 实际检查目录
            "total_videos": total_videos,  # 总数量
            "total_size_gb": round(total_size, 2),  # 总大小
            "corrupt_files": corrupt_files,  # 损坏文件
            "intact_videos": total_videos - len(corrupt_files),  # 完整数量
        }  # 结束字典
        self.log_cb("SUCCESS", f"完整性检查完成: {stats['intact_videos']}/{total_videos}")  # 记录日志
        if corrupt_files:  # 判断是否有损坏
            self.log_cb("WARNING", f"发现 {len(corrupt_files)} 个损坏文件")  # 记录警告
        return stats  # 返回统计

    def preprocess(self, source_dir: str, output_dir: str, steps: Dict[str, bool], batch_size: int = 8) -> bool:  # 预处理
        try:  # 总体异常捕获
            layout = self._detect_layout(source_dir)  # 检测目录布局
            source_root: Path = layout["source_root"]  # 获取解析后的源目录
            raw_videos: List[str] = layout["raw_videos"]  # 获取原始视频列表
            prebuilt_clip_videos: List[str] = layout["clip_videos"]  # 获取已有 clips 列表
            output_root = Path(output_dir)  # 构造输出根目录
            output_root.mkdir(parents=True, exist_ok=True)  # 创建输出根目录
            meta_dir = output_root / "meta"  # 统一 meta 目录
            work_video_dir = source_root  # 初始化当前工作视频目录
            work_video_list = list(raw_videos)  # 初始化当前工作视频列表
            clip_video_dir: Optional[Path] = None  # 初始化 clip 目录
            clip_video_list: List[str] = []  # 初始化 clip 文件列表

            if work_video_list:  # 若存在原始视频
                if steps.get("convert"):  # 若启用转换步骤
                    if self.convert_video:  # 若转换函数可用
                        self.log_cb("INFO", "步骤 1/5: 转换视频为 25fps")  # 记录日志
                        self.progress_cb.add_step("转换", "running", 0)  # 添加进度
                        convert_dir = output_root / "25fps"  # 构造转换输出目录
                        convert_dir.mkdir(parents=True, exist_ok=True)  # 创建目录
                        total = len(work_video_list)  # 统计数量
                        for batch_idx in range(0, total, batch_size):  # 批处理循环
                            self._check_interrupts()  # 检查中断
                            batch = work_video_list[batch_idx : batch_idx + batch_size]  # 获取批次
                            self.convert_video(str(work_video_dir), str(convert_dir), batch)  # 执行转换
                            progress = int((batch_idx + len(batch)) / max(1, total) * 100)  # 计算进度
                            self.progress_cb.update_step("转换", "running", progress)  # 更新进度
                        self.progress_cb.update_step("转换", "completed", 100)  # 完成步骤
                        work_video_dir = convert_dir  # 更新工作目录
                        work_video_list = self._list_mp4_files(convert_dir)  # 更新视频列表
                    else:  # 转换函数不可用
                        self.log_cb("WARNING", "跳过转换步骤: 未成功导入 scripts.preprocess.convert_video")  # 记录警告

                if steps.get("segment"):  # 若启用分段步骤
                    if self.segment_video:  # 若分段函数可用
                        self.log_cb("INFO", "步骤 2/5: 分割视频为 30 秒片段")  # 记录日志
                        self.progress_cb.add_step("分割", "running", 0)  # 添加进度
                        clips_dir = output_root / "clips"  # 构造 clips 输出目录
                        clips_dir.mkdir(parents=True, exist_ok=True)  # 创建目录
                        self.segment_video(str(work_video_dir), str(clips_dir), work_video_list)  # 执行分段
                        self.progress_cb.update_step("分割", "completed", 100)  # 完成步骤
                        clip_video_dir = clips_dir  # 更新 clip 目录
                        clip_video_list = self._list_mp4_files(clips_dir)  # 更新 clip 列表
                    else:  # 分段函数不可用
                        self.log_cb("WARNING", "跳过分段步骤: 未成功导入 scripts.preprocess.segment_video")  # 记录警告
                        clip_video_dir = work_video_dir  # 回退使用当前目录
                        clip_video_list = list(work_video_list)  # 回退使用当前列表
                else:  # 未启用分段步骤
                    clip_video_dir = work_video_dir  # 直接使用当前目录
                    clip_video_list = list(work_video_list)  # 直接使用当前列表
            elif prebuilt_clip_videos:  # 若不存在原始视频但存在预构建 clips
                clip_video_dir = layout["clips_dir"]  # 使用已有 clips 目录
                clip_video_list = list(prebuilt_clip_videos)  # 使用已有 clips 列表
                if steps.get("convert") or steps.get("segment"):  # 若用户勾选了前置步骤
                    self.log_cb("INFO", "检测到 source/clips 结构，已自动跳过 convert/segment，直接使用 clips")  # 记录提示
            else:  # 两种视频来源都不存在
                self.log_cb("ERROR", f"未在 {source_root} 或 {source_root / 'clips'} 找到 mp4 文件")  # 记录错误
                return False  # 返回失败

            if not clip_video_dir or not clip_video_list:  # 判断 clip 结果是否有效
                self.log_cb("ERROR", "未获得可用于后续处理的 clip 视频列表")  # 记录错误
                return False  # 返回失败

            if steps.get("audio"):  # 提取音频步骤
                if self.extract_audio:  # 若提取函数可用
                    self.log_cb("INFO", "步骤 3/5: 提取 16kHz 音频")  # 记录日志
                    self.progress_cb.add_step("音频", "running", 0)  # 添加进度
                    # 为兼容 scripts.preprocess.analyze_video 中的 wav_path 生成逻辑，
                    # 音频直接输出到 clip 同目录（xxx.mp4 -> xxx.wav）。
                    self.extract_audio(str(clip_video_dir), str(clip_video_dir), clip_video_list)  # 执行提取
                    self.progress_cb.update_step("音频", "completed", 100)  # 完成步骤
                else:  # 提取函数不可用
                    self.log_cb("WARNING", "跳过音频步骤: 未成功导入 scripts.preprocess.extract_audio")  # 记录警告

            if steps.get("analyze"):  # 人脸分析步骤
                if self.analyze_video:  # 若分析函数可用
                    self.log_cb("INFO", "步骤 4/5: 人脸检测和 landmark 提取")  # 记录日志
                    self.progress_cb.add_step("分析", "running", 0)  # 添加进度
                    meta_dir.mkdir(parents=True, exist_ok=True)  # 创建目录
                    self.analyze_video(str(clip_video_dir), str(meta_dir), clip_video_list)  # 执行分析
                    self.progress_cb.update_step("分析", "completed", 100)  # 完成步骤
                else:  # 分析函数不可用
                    self.log_cb("WARNING", "跳过分析步骤: 未成功导入 scripts.preprocess.analyze_video")  # 记录警告

            if steps.get("metadata"):  # 生成元数据
                self.log_cb("INFO", "步骤 5/5: 生成元数据")  # 记录日志
                self.progress_cb.add_step("元数据", "running", 0)  # 添加进度
                self.progress_cb.update_step("元数据", "completed", 100)  # 简化完成
            if steps.get("splits"):  # 生成训练划分文件
                self.log_cb("INFO", "生成 train/val 列表")  # 记录提示
                ok = self._build_train_val_lists(meta_dir=meta_dir, output_root=output_root)
                if not ok:
                    self.log_cb("WARNING", "train/val 列表生成失败，请检查 meta 目录")
            self.log_cb("SUCCESS", "HDTF 预处理完成")  # 记录成功
            return True  # 返回成功
        except ProcessorStoppedException:  # 捕获停止
            self.log_cb("WARNING", "HDTF 预处理已停止")  # 记录日志
            return False  # 返回失败
        except Exception as e:  # 捕获异常
            self.log_cb("ERROR", f"预处理失败: {str(e)}")  # 记录错误
            return False  # 返回失败

    def validate_quality(self, meta_dir: str) -> Dict[str, Any]:  # 质量验证
        self.log_cb("INFO", "开始验证数据质量")  # 记录日志
        issues = self.validator.validate_meta_dir(meta_dir)  # 使用验证器
        self.log_cb("SUCCESS", "质量验证完成")  # 记录成功
        return issues  # 返回问题
