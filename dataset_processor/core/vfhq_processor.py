# f:\python\Musetalk\dataset_processor\core\vfhq_processor.py
import importlib.util  # 导入动态加载模块
import os  # 导入系统模块
from pathlib import Path  # 导入路径模块
from typing import Any, Dict, List, Optional, Set  # 导入类型注解

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

    def _get_vfhq_root(self, base_path: str) -> Optional[Path]:  # 获取 VFHQ-512-new 根目录
        root = Path(base_path) / "VFHQ-512-new"  # 组合目录路径
        if root.exists() and root.is_dir():  # 判断目录存在
            return root  # 返回目录对象
        return None  # 返回空值

    def _collect_image_files(self, folder: Path) -> List[Path]:  # 收集图像文件
        images = sorted(folder.glob("*.png"))  # 优先收集 PNG
        if not images:  # 若 PNG 为空
            images = sorted(folder.glob("*.jpg"))  # 回退收集 JPG
        if not images:  # 若 JPG 仍为空
            images = sorted(folder.glob("*.jpeg"))  # 回退收集 JPEG
        return images  # 返回图像列表

    def _parse_label_file(self, label_file: Path) -> Dict[str, Any]:  # 解析标签文件
        row_count = 0  # 初始化有效帧行计数
        delete_count = 0  # 初始化 DELETE 行计数
        try:  # 捕获解析异常
            with open(label_file, "r", encoding="utf-8") as fp:  # 打开标签文件
                for raw_line in fp:  # 按行遍历
                    line = raw_line.strip()  # 去除首尾空白
                    if not line:  # 跳过空行
                        continue  # 继续下一行
                    if line.startswith(("Video ID", "H:", "W:", "FPS:", "FRAME", "CROP_BBOX")):  # 跳过头信息
                        continue  # 继续下一行
                    parts = line.split()  # 按空白切分
                    if len(parts) < 2:  # 至少应包含原始帧号与映射字段
                        continue  # 跳过异常行
                    if not parts[0].isdigit():  # 第一列不是数字视作无效
                        continue  # 跳过异常行
                    row_count += 1  # 统计有效帧行
                    if parts[1].upper() == "DELETE":  # 判断删除标记
                        delete_count += 1  # 统计删除行
        except Exception as exc:  # 捕获异常
            self.log_cb("WARNING", f"解析 label 失败: {label_file.name} -> {exc}")  # 记录警告
        return {  # 返回解析结果
            "row_count": row_count,  # 返回有效帧行数
            "delete_count": delete_count,  # 返回删除行数
        }  # 结束字典

    def _get_source_folder(self, base_path: str, clip_name: str) -> Optional[Path]:  # 获取样本源目录
        vfhq_root = self._get_vfhq_root(base_path)  # 获取 VFHQ-512-new 根目录
        if vfhq_root is not None:  # 若目录存在
            folder = vfhq_root / clip_name  # 组合候选目录
            if folder.exists() and folder.is_dir():  # 判断候选目录有效
                return folder  # 优先返回 VFHQ-512-new 目录
        if self.dataset_manager is None:  # 若无兼容管理器
            return None  # 返回空值
        manager = self.dataset_manager(base_path)  # 构造兼容管理器
        gt_folder = manager.get_clip_folder_path(clip_name, "GT")  # 获取 GT 目录
        if gt_folder and gt_folder.exists():  # 判断 GT 目录是否可用
            return gt_folder  # 回退返回 GT 目录
        return None  # 返回空值

    def check_integrity(self, base_path: str) -> Dict[str, Any]:  # 完整性检查
        self.log_cb("INFO", "检查 VFHQ 数据集结构")  # 记录日志
        label_path = Path(base_path) / "label"  # 获取标签目录
        if not label_path.exists():  # 判断目录是否存在
            self.log_cb("ERROR", f"label 目录不存在: {label_path}")  # 记录错误
            return {}  # 返回空结果
        label_files = sorted(label_path.glob("*.txt"))  # 获取全部标签文件
        label_names = {f.stem for f in label_files}  # 构建标签样本集合
        vfhq_root = self._get_vfhq_root(base_path)  # 获取 VFHQ-512-new 根目录
        if vfhq_root is None and not self.dataset_manager:  # 两种方式都不可用
            self.log_cb("ERROR", "未找到 VFHQ-512-new，也未找到兼容 DatasetManager")  # 记录错误
            return {}  # 返回空结果

        checked = 0  # 初始化已检查计数
        frame_count_mismatch = 0  # 初始化帧数不匹配计数
        sequence_mismatch = 0  # 初始化序列不匹配计数
        total_delete_rows = 0  # 初始化删除行总数
        details: List[Dict[str, Any]] = []  # 初始化详情列表

        source_clips: Set[str] = set()  # 初始化源样本集合
        if vfhq_root is not None:  # VFHQ-512-new 可用时
            source_clips = {d.name for d in vfhq_root.iterdir() if d.is_dir()}  # 收集源目录名
            matched_names = sorted(label_names & source_clips)  # 获取名称交集
            labels_without_source = len(label_names - source_clips)  # 统计仅标签样本
            source_without_label = len(source_clips - label_names)  # 统计仅源目录样本
            for clip_name in matched_names:  # 遍历交集样本
                self._check_interrupts()  # 检查中断
                checked += 1  # 增加已检查数
                label_info = self._parse_label_file(label_path / f"{clip_name}.txt")  # 解析标签信息
                total_delete_rows += int(label_info.get("delete_count", 0))  # 累加删除行
                source_folder = vfhq_root / clip_name  # 组合源目录
                images = self._collect_image_files(source_folder)  # 收集图像文件
                actual_count = len(images)  # 统计实际图像帧数
                expected_count = int(label_info.get("row_count", 0))  # 获取期望帧数
                if expected_count != actual_count:  # 判断帧数差异
                    frame_count_mismatch += 1  # 累加差异计数
                    details.append({  # 记录差异详情
                        "clip": clip_name,  # 样本名
                        "reason": "frame_count_mismatch",  # 差异类型
                        "expected": expected_count,  # 期望帧数
                        "actual": actual_count,  # 实际帧数
                    })  # 结束详情
                numeric_stems = [img.stem for img in images if img.stem.isdigit()]  # 收集数字文件名
                stem_lengths = {len(stem) for stem in numeric_stems}  # 收集位宽集合
                if numeric_stems and len(numeric_stems) == actual_count and len(stem_lengths) == 1:  # 满足序列检查条件
                    width = next(iter(stem_lengths))  # 获取位宽
                    expected_names = {f"{idx:0{width}d}" for idx in range(expected_count)}  # 生成期望文件名集合
                    actual_names = set(numeric_stems)  # 获取实际文件名集合
                    if expected_names != actual_names:  # 判断文件名序列是否一致
                        sequence_mismatch += 1  # 累加序列差异计数
                        details.append({  # 记录差异详情
                            "clip": clip_name,  # 样本名
                            "reason": "filename_sequence_mismatch",  # 差异类型
                            "expected_count": len(expected_names),  # 期望名称数量
                            "actual_count": len(actual_names),  # 实际名称数量
                        })  # 结束详情

            report = {  # 构建返回报告
                "mode": "vfhq_512_new",  # 校验模式
                "total_labels": len(label_files),  # 标签总数
                "total_source_clips": len(source_clips),  # 源目录总数
                "matched_by_name": len(matched_names),  # 名称交集数量
                "labels_without_source": labels_without_source,  # 仅标签样本数量
                "source_without_label": source_without_label,  # 仅源目录样本数量
                "checked": checked,  # 实际检查数量
                "frame_count_mismatch": frame_count_mismatch,  # 帧数差异数量
                "filename_sequence_mismatch": sequence_mismatch,  # 文件序列差异数量
                "total_delete_rows": total_delete_rows,  # 删除行总数
                "details": details[:200],  # 截断详情，避免结果过大
            }  # 结束报告
            self.log_cb("INFO", f"总 label 文件: {report['total_labels']}")  # 记录信息
            self.log_cb("INFO", f"VFHQ-512-new 文件夹: {report['total_source_clips']}")  # 记录信息
            self.log_cb("INFO", f"名称匹配数量: {report['matched_by_name']}")  # 记录信息
            self.log_cb("WARNING", f"仅 label 样本: {report['labels_without_source']}")  # 记录警告
            self.log_cb("WARNING", f"帧数不匹配: {report['frame_count_mismatch']}")  # 记录警告
            self.log_cb("WARNING", f"序列命名不匹配: {report['filename_sequence_mismatch']}")  # 记录警告
            return report  # 返回报告

        manager = self.dataset_manager(base_path)  # 创建兼容管理器
        report = manager.generate_report(output_file=None)  # 获取旧版报告
        self.log_cb("INFO", f"总 label 文件: {report.get('total_labels', 0)}")  # 记录信息
        self.log_cb("INFO", f"完全匹配: {report.get('matched', 0)}")  # 记录信息
        self.log_cb("WARNING", f"GT 缺失: {report.get('missing_gt', 0)}")  # 记录警告
        self.log_cb("WARNING", f"Blind-LR 缺失: {report.get('missing_blind_lr', 0)}")  # 记录警告
        return report  # 返回报告

    def _get_complete_samples(self, base_path: str) -> List[str]:  # 获取完整样本
        label_path = Path(base_path) / "label"  # 标签路径
        vfhq_root = self._get_vfhq_root(base_path)  # 获取 VFHQ-512-new 根目录
        if vfhq_root is not None and label_path.exists():  # 优先使用 VFHQ-512-new 结构
            source_names = {d.name for d in vfhq_root.iterdir() if d.is_dir()}  # 收集源目录名
            complete_samples = []  # 初始化样本列表
            missing_samples = []  # 记录未匹配样本（仅用于日志汇总）
            for label_file in label_path.glob("*.txt"):  # 遍历标签文件
                clip_name = label_file.stem  # 获取样本名
                if clip_name in source_names:  # 判断是否存在同名目录
                    complete_samples.append(clip_name)  # 添加完整样本
                else:  # 不存在同名目录
                    missing_samples.append(clip_name)  # 暂存缺失样本
            if missing_samples:  # 输出精简日志，避免海量打印
                preview = ", ".join(missing_samples[:10])  # 仅展示前 10 个示例
                self.log_cb("INFO", f"跳过未匹配样本: {len(missing_samples)} 个（示例: {preview}）")  # 汇总日志
            return complete_samples  # 返回完整样本

        if not self.dataset_manager:  # 判断管理器是否可用
            return []  # 返回空列表
        manager = self.dataset_manager(base_path)  # 创建管理器
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
            label_path = Path(base_path) / "label"  # 标签路径
            vfhq_root = self._get_vfhq_root(base_path)  # 获取 VFHQ-512-new 根目录
            if strategy == "complete_only":  # 仅完整样本
                samples = self._get_complete_samples(base_path)  # 获取样本
                self.log_cb("INFO", f"找到 {len(samples)} 个完整样本")  # 记录日志
            elif vfhq_root is not None:  # 存在 VFHQ-512-new 时
                samples = sorted([d.name for d in vfhq_root.iterdir() if d.is_dir()])  # 直接处理所有可用文件夹
                self.log_cb("INFO", f"处理 VFHQ-512-new 中可用样本: {len(samples)} 个")  # 记录日志
            else:  # 回退到标签驱动
                samples = [f.stem for f in label_path.glob("*.txt")]  # 获取样本
                self.log_cb("INFO", f"处理所有可用样本: {len(samples)} 个")  # 记录日志
            if steps.get("convert_to_video"):  # 图像序列转视频
                self.progress_cb.add_step("转换视频", "running", 0)  # 添加进度
                temp_dir = Path(output_dir) / "temp_videos"  # 临时目录
                temp_dir.mkdir(parents=True, exist_ok=True)  # 创建目录
                success_count = 0  # 初始化成功数量
                for idx, clip_name in enumerate(samples):  # 遍历样本
                    self._check_interrupts()  # 检查中断
                    if self._convert_sequence_to_video(base_path, clip_name, temp_dir):  # 执行转换
                        success_count += 1  # 统计成功转换
                    progress = int((idx + 1) / max(1, len(samples)) * 100)  # 计算进度
                    self.progress_cb.update_step("转换视频", "running", progress)  # 更新进度
                self.progress_cb.update_step("转换视频", "completed", 100)  # 完成步骤
                self.log_cb("INFO", f"视频转换完成: {success_count}/{len(samples)}")  # 记录转换结果
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
        source_folder = self._get_source_folder(base_path, clip_name)  # 获取源图像目录
        if not source_folder or not source_folder.exists():  # 判断目录
            self.log_cb("WARNING", f"缺少源目录，跳过: {clip_name}")  # 记录警告
            return False  # 返回失败
        images = self._collect_image_files(source_folder)  # 收集图像文件
        if not images:  # 仍为空
            self.log_cb("WARNING", f"源目录无图像，跳过: {clip_name}")  # 记录警告
            return False  # 返回失败
        output_file = output_dir / f"{clip_name}.mp4"  # 输出文件
        fps = str(self.config.get("fps", 25))  # 获取帧率配置
        codec = str(self.config.get("codec", "libx264"))  # 获取编码配置
        crf = str(self.config.get("crf", 15))  # 获取压缩质量配置
        numeric_names = [img.stem for img in images if img.stem.isdigit()]  # 收集数字命名帧
        one_suffix = len({img.suffix.lower() for img in images}) == 1  # 判断扩展名是否一致
        if one_suffix and len(numeric_names) == len(images):  # 可走数字序列模式
            sorted_nums = sorted(int(name) for name in numeric_names)  # 数值排序
            expected_nums = list(range(sorted_nums[0], sorted_nums[0] + len(sorted_nums)))  # 期望连续序列
            if sorted_nums == expected_nums:  # 判断是否连续
                digits = len(numeric_names[0])  # 获取数字位宽
                suffix = images[0].suffix.lower()  # 获取文件后缀
                pattern = source_folder / f"%0{digits}d{suffix}"  # 构造序列输入模板
                cmd = [  # 构造 FFmpeg 命令
                    "ffmpeg", "-hide_banner", "-y",  # 隐藏横幅并覆盖输出
                    "-framerate", fps,  # 设置输入帧率
                    "-start_number", str(sorted_nums[0]),  # 设置起始帧号
                    "-i", str(pattern),  # 设置输入模板
                    "-c:v", codec, "-crf", crf,  # 设置编码参数
                    "-pix_fmt", "yuv420p", str(output_file),  # 设置像素格式与输出
                ]  # 结束命令
                ok, err = run_ffmpeg(cmd)  # 执行命令
                if not ok:  # 判断失败
                    self.log_cb("WARNING", f"{clip_name} 转换失败: {err}")  # 记录警告
                return ok  # 返回结果

        list_file = output_dir / f".{clip_name}.frames.txt"  # 构造回退列表文件路径
        try:  # 捕获回退模式异常
            with open(list_file, "w", encoding="utf-8") as fp:  # 打开列表文件
                for img in images:  # 遍历图像列表
                    safe_path = img.resolve().as_posix().replace("'", r"'\''")  # 转换为安全路径
                    fp.write(f"file '{safe_path}'\n")  # 写入 concat 列表项
            cmd = [  # 构造 concat 模式命令
                "ffmpeg", "-hide_banner", "-y",  # 隐藏横幅并覆盖输出
                "-f", "concat", "-safe", "0",  # 启用 concat 模式
                "-r", fps, "-i", str(list_file),  # 设置帧率并指定列表
                "-c:v", codec, "-crf", crf,  # 设置编码参数
                "-pix_fmt", "yuv420p", str(output_file),  # 设置像素格式与输出
            ]  # 结束命令
            ok, err = run_ffmpeg(cmd)  # 执行命令
            if not ok:  # 判断失败
                self.log_cb("WARNING", f"{clip_name} 回退转换失败: {err}")  # 记录警告
            return ok  # 返回结果
        finally:  # 始终尝试清理临时文件
            if list_file.exists():  # 判断文件存在
                try:  # 捕获删除异常
                    list_file.unlink()  # 删除临时列表
                except Exception:  # 删除失败时忽略
                    pass  # 保持流程继续
