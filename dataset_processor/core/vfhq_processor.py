# f:\python\Musetalk\dataset_processor\core\vfhq_processor.py
import importlib.util  # 导入动态加载模块
import os  # 导入系统模块
import subprocess  # 导入子进程模块
from pathlib import Path  # 导入路径模块
from typing import Any, Dict, List  # 导入类型注解

from core.base_processor import BaseProcessor, ProcessorStoppedException  # 导入基类
from core.validator import DatasetValidator  # 导入验证器
from utils.ffmpeg_wrapper import run_ffmpeg  # 导入 FFmpeg 封装


class VFHQProcessor(BaseProcessor):  # VFHQ 处理器
    """VFHQ 数据集处理器（支持对应匹配）"""  # 类说明

    def __init__(self, progress_cb, log_cb, config: Dict[str, Any]) -> None:  # 初始化函数
        super().__init__(progress_cb, log_cb, config)  # 调用父类初始化
        self.validator = DatasetValidator()  # 创建验证器
        self.dataset_manager = None  # 初始化管理器
        self._init_dataset_manager()  # 初始化管理器

    def _init_dataset_manager(self) -> None:  # 初始化管理器
        try:  # 尝试导入管理器
            from datasets.VFHQ_Test.dataset_manager import DatasetManager  # type: ignore  # 导入类
            self.dataset_manager = DatasetManager  # 绑定类
        except Exception:  # 导入失败
            try:  # 再次尝试路径
                from datasets.VFHQTest.dataset_manager import DatasetManager  # type: ignore  # 导入类
                self.dataset_manager = DatasetManager  # 绑定类
            except Exception:  # 继续尝试文件加载
                repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))  # 仓库根目录
                dm_path = os.path.join(repo_root, "datasets", "VFHQ-Test", "dataset_manager.py")  # 管理器路径
                if os.path.exists(dm_path):  # 判断文件存在
                    spec = importlib.util.spec_from_file_location("vfhq_dataset_manager", dm_path)  # 创建模块规范
                    if spec and spec.loader:  # 判断加载器
                        module = importlib.util.module_from_spec(spec)  # 创建模块
                        spec.loader.exec_module(module)  # 执行模块
                        self.dataset_manager = getattr(module, "DatasetManager", None)  # 获取类
                if not self.dataset_manager:  # 仍然失败
                    self.dataset_manager = None  # 设置为空

    def check_integrity(self, base_path: str) -> Dict[str, Any]:  # 完整性检查
        if not self.dataset_manager:  # 判断管理器是否可用
            self.log_cb("ERROR", "未找到 VFHQ DatasetManager")  # 记录错误
            return {}  # 返回空结果
        self.log_cb("INFO", "检查 VFHQ 数据集结构")  # 记录日志
        manager = self.dataset_manager(base_path)  # 创建管理器
        report = manager.generate_report(output_file=None)  # 获取报告
        self.log_cb("INFO", f"总 label 文件: {report.get('total_labels', 0)}")  # 记录信息
        self.log_cb("INFO", f"完全匹配: {report.get('matched', 0)}")  # 记录信息
        self.log_cb("WARNING", f"GT 缺失: {report.get('missing_gt', 0)}")  # 记录警告
        self.log_cb("WARNING", f"Blind-LR 缺失: {report.get('missing_blind_lr', 0)}")  # 记录警告
        return report  # 返回报告

    def _get_complete_samples(self, base_path: str) -> List[str]:  # 获取完整样本
        if not self.dataset_manager:  # 判断管理器是否可用
            return []  # 返回空列表
        manager = self.dataset_manager(base_path)  # 创建管理器
        label_path = Path(base_path) / "label"  # 标签路径
        complete_samples: List[str] = []  # 完整样本列表
        for label_file in label_path.glob("*.txt"):  # 遍历标签
            clip_name = label_file.stem  # 获取样本名
            gt_folder = manager.get_clip_folder_path(clip_name, "GT")  # 获取 GT
            lr_folder = manager.get_clip_folder_path(clip_name, "Blind-LR")  # 获取 LR
            has_gt = bool(gt_folder and gt_folder.exists())  # 判断 GT
            has_lr = bool(lr_folder and lr_folder.exists())  # 判断 LR
            if has_gt and has_lr:  # 完整样本
                complete_samples.append(clip_name)  # 添加样本
            else:  # 不完整样本
                self.log_cb("INFO", f"跳过不完整样本: {clip_name} (GT: {'✓' if has_gt else '✗'}, LR: {'✓' if has_lr else '✗'})")  # 记录日志
        return complete_samples  # 返回列表

    def preprocess(self, base_path: str, output_dir: str, steps: Dict[str, bool], batch_size: int = 4, strategy: str = "complete_only") -> bool:  # 预处理
        try:  # 捕获异常
            if strategy == "complete_only":  # 仅完整样本
                samples = self._get_complete_samples(base_path)  # 获取样本
                self.log_cb("INFO", f"找到 {len(samples)} 个完整样本")  # 记录日志
            else:  # 处理所有
                label_path = Path(base_path) / "label"  # 标签路径
                samples = [f.stem for f in label_path.glob("*.txt")]  # 获取样本
                self.log_cb("INFO", f"处理所有可用样本: {len(samples)} 个")  # 记录日志
            if steps.get("convert_to_video"):  # 图像序列转视频
                self.progress_cb.add_step("转换视频", "running", 0)  # 添加进度
                temp_dir = Path(output_dir) / "temp_videos"  # 临时目录
                temp_dir.mkdir(parents=True, exist_ok=True)  # 创建目录
                for idx, clip_name in enumerate(samples):  # 遍历样本
                    self._check_interrupts()  # 检查中断
                    self._convert_sequence_to_video(base_path, clip_name, temp_dir)  # 执行转换
                    progress = int((idx + 1) / max(1, len(samples)) * 100)  # 计算进度
                    self.progress_cb.update_step("转换视频", "running", progress)  # 更新进度
                self.progress_cb.update_step("转换视频", "completed", 100)  # 完成步骤
            self.log_cb("SUCCESS", "VFHQ 预处理完成")  # 记录成功
            return True  # 返回成功
        except ProcessorStoppedException:  # 捕获停止
            self.log_cb("WARNING", "VFHQ 预处理已停止")  # 记录日志
            return False  # 返回失败
        except Exception as e:  # 捕获异常
            self.log_cb("ERROR", f"VFHQ 预处理失败: {str(e)}")  # 记录错误
            return False  # 返回失败

    def validate_quality(self, meta_dir: str) -> Dict[str, Any]:  # 质量验证
        self.log_cb("INFO", "开始验证数据质量")  # 记录日志
        issues = self.validator.validate_meta_dir(meta_dir)  # 调用验证器
        self.log_cb("SUCCESS", "质量验证完成")  # 记录成功
        return issues  # 返回结果

    def _convert_sequence_to_video(self, base_path: str, clip_name: str, output_dir: Path) -> bool:  # 转换序列
        if not self.dataset_manager:  # 判断管理器是否可用
            return False  # 返回失败
        manager = self.dataset_manager(base_path)  # 创建管理器
        gt_folder = manager.get_clip_folder_path(clip_name, "GT")  # 获取 GT 目录
        if not gt_folder or not gt_folder.exists():  # 判断目录
            return False  # 返回失败
        images = sorted(gt_folder.glob("*.png"))  # 查找 PNG
        if not images:  # 为空则查找 JPG
            images = sorted(gt_folder.glob("*.jpg"))  # 查找 JPG
        if not images:  # 仍为空
            return False  # 返回失败
        output_file = output_dir / f"{clip_name}.mp4"  # 输出文件
        cmd = [  # FFmpeg 命令
            "ffmpeg", "-framerate", "25",  # 帧率参数
            "-i", str(gt_folder / "%05d.png"),  # 输入序列
            "-c:v", "libx264", "-crf", "15",  # 编码参数
            "-pix_fmt", "yuv420p", "-y", str(output_file),  # 像素格式
        ]  # 结束命令
        ok, err = run_ffmpeg(cmd)  # 执行命令
        if not ok:  # 判断失败
            self.log_cb("WARNING", f"{clip_name} 转换失败: {err}")  # 记录警告
        return ok  # 返回结果
