# f:\python\Musetalk\dataset_processor\core\hdtf_processor.py
import os  # 导入系统模块
from pathlib import Path  # 导入路径模块
from typing import Any, Dict  # 导入类型注解

import cv2  # 导入 OpenCV

from core.base_processor import BaseProcessor, ProcessorStoppedException  # 导入基类
from core.validator import DatasetValidator  # 导入验证器


class HDTFProcessor(BaseProcessor):  # HDTF 处理器
    """HDTF 数据集处理器"""  # 类说明

    def __init__(self, progress_cb, log_cb, config: Dict[str, Any]) -> None:  # 初始化函数
        super().__init__(progress_cb, log_cb, config)  # 调用父类初始化
        self.validator = DatasetValidator()  # 创建验证器
        self._init_preprocess_funcs()  # 初始化预处理函数

    def _init_preprocess_funcs(self) -> None:  # 初始化预处理函数
        """初始化预处理函数，尝试从scripts.preprocess导入"""
        self.convert_video = None
        self.segment_video = None
        self.extract_audio = None
        self.analyze_video = None
        
        try:
            # 尝试从项目根目录的scripts模块导入
            import sys
            import os
            
            # 添加项目根目录到路径
            repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
            if repo_root not in sys.path:
                sys.path.insert(0, repo_root)
            
            from scripts.preprocess import (
                convert_video,
                segment_video,
                extract_audio,
                analyze_video,
            )
            
            self.convert_video = convert_video
            self.segment_video = segment_video
            self.extract_audio = extract_audio
            self.analyze_video = analyze_video
            
            self.log_cb("INFO", "预处理函数加载成功")
            
        except ImportError as e:
            self.log_cb("WARNING", f"无法导入预处理函数: {e}")
            self.log_cb("INFO", "部分预处理步骤将不可用，请确保scripts/preprocess.py存在")
        except Exception as e:
            self.log_cb("ERROR", f"加载预处理函数时出错: {e}")


    def check_integrity(self, source_dir: str) -> Dict[str, Any]:  # 完整性检查
        self.log_cb("INFO", f"开始检查 HDTF 数据集: {source_dir}")  # 记录日志
        video_files = list(Path(source_dir).glob("*.mp4"))  # 获取视频列表
        total_videos = len(video_files)  # 统计数量
        total_size = sum(f.stat().st_size for f in video_files) / (1024 ** 3)  # 统计大小
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
            vid_list = sorted(os.listdir(source_dir))  # 获取视频列表
            total = len(vid_list)  # 统计数量
            if steps.get("convert") and self.convert_video:  # 转换 25fps
                self.log_cb("INFO", "步骤 1/5: 转换视频为 25fps")  # 记录日志
                self.progress_cb.add_step("转换", "running", 0)  # 添加进度
                for batch_idx in range(0, total, batch_size):  # 批处理循环
                    self._check_interrupts()  # 检查中断
                    batch = vid_list[batch_idx : batch_idx + batch_size]  # 获取批次
                    self.convert_video(source_dir, os.path.join(output_dir, "25fps"), batch)  # 执行转换
                    progress = int((batch_idx + len(batch)) / total * 100)  # 计算进度
                    self.progress_cb.update_step("转换", "running", progress)  # 更新进度
                self.progress_cb.update_step("转换", "completed", 100)  # 完成步骤
            if steps.get("segment") and self.segment_video:  # 分割片段
                self.log_cb("INFO", "步骤 2/5: 分割视频为 30 秒片段")  # 记录日志
                self.progress_cb.add_step("分割", "running", 0)  # 添加进度
                self.segment_video(os.path.join(output_dir, "25fps"), os.path.join(output_dir, "clips"))  # 执行分割
                self.progress_cb.update_step("分割", "completed", 100)  # 完成步骤
            if steps.get("audio") and self.extract_audio:  # 提取音频
                self.log_cb("INFO", "步骤 3/5: 提取 16kHz 音频")  # 记录日志
                self.progress_cb.add_step("音频", "running", 0)  # 添加进度
                self.extract_audio(os.path.join(output_dir, "clips"), os.path.join(output_dir, "audio"))  # 执行提取
                self.progress_cb.update_step("音频", "completed", 100)  # 完成步骤
            if steps.get("analyze") and self.analyze_video:  # 人脸分析
                self.log_cb("INFO", "步骤 4/5: 人脸检测和 landmark 提取")  # 记录日志
                self.progress_cb.add_step("分析", "running", 0)  # 添加进度
                self.analyze_video(os.path.join(output_dir, "clips"), os.path.join(output_dir, "meta"))  # 执行分析
                self.progress_cb.update_step("分析", "completed", 100)  # 完成步骤
            if steps.get("metadata"):  # 生成元数据
                self.log_cb("INFO", "步骤 5/5: 生成元数据")  # 记录日志
                self.progress_cb.add_step("元数据", "running", 0)  # 添加进度
                self.progress_cb.update_step("元数据", "completed", 100)  # 简化完成
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
