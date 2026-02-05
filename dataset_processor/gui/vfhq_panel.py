# f:\\python\\Musetalk\\dataset_processor\\gui\\vfhq_panel.py
import os  # 导入系统模块
import tkinter as tk  # 导入 tkinter
from tkinter import ttk, filedialog, messagebox  # 导入 ttk 与对话框

from utils.thread_manager import ThreadManager  # 导入线程管理


class VFHQPanel(ttk.Frame):  # VFHQ 面板
    """VFHQ 数据集面板"""  # 类说明

    def __init__(self, master, app) -> None:  # 初始化函数
        super().__init__(master)  # 调用父类初始化
        self.app = app  # 保存主窗口引用
        self.thread_manager = ThreadManager()  # 创建线程管理器
        self.processor = None  # 处理器占位
        self._build_ui()  # 构建界面

    def _build_ui(self) -> None:  # 构建界面
        pad = {"padx": 6, "pady": 4}  # 通用内边距
        ttk.Label(self, text="VFHQ 数据集").pack(anchor=tk.W, **pad)  # 标题
        path_frame = ttk.Frame(self)  # 路径框
        path_frame.pack(fill=tk.X, **pad)  # 布局
        ttk.Label(path_frame, text="基础目录:").pack(side=tk.LEFT)  # 标签
        self.base_dir_var = tk.StringVar(value=self.app.config_data.get("vfhq", {}).get("base_path", ""))  # 默认路径
        ttk.Entry(path_frame, textvariable=self.base_dir_var).pack(side=tk.LEFT, fill=tk.X, expand=True, padx=4)  # 输入框
        ttk.Button(path_frame, text="浏览...", command=self._browse_dir).pack(side=tk.LEFT)  # 浏览按钮
        strategy_frame = ttk.LabelFrame(self, text="处理策略")  # 策略框
        strategy_frame.pack(fill=tk.X, **pad)  # 布局
        self.strategy_var = tk.StringVar(value="complete_only")  # 默认策略
        ttk.Radiobutton(strategy_frame, text="只处理完全匹配的数据", value="complete_only", variable=self.strategy_var).pack(anchor=tk.W)  # 单选
        ttk.Radiobutton(strategy_frame, text="处理所有可用数据", value="all_available", variable=self.strategy_var).pack(anchor=tk.W)  # 单选
        options_frame = ttk.LabelFrame(self, text="图像序列转视频选项")  # 选项框
        options_frame.pack(fill=tk.X, **pad)  # 布局
        self.fps_var = tk.IntVar(value=self.app.config_data.get("vfhq", {}).get("fps", 25))  # FPS
        self.crf_var = tk.IntVar(value=self.app.config_data.get("vfhq", {}).get("crf", 15))  # CRF
        ttk.Label(options_frame, text="帧率:").pack(side=tk.LEFT)  # 标签
        ttk.Entry(options_frame, textvariable=self.fps_var, width=6).pack(side=tk.LEFT, padx=4)  # 输入框
        ttk.Label(options_frame, text="CRF:").pack(side=tk.LEFT)  # 标签
        ttk.Entry(options_frame, textvariable=self.crf_var, width=6).pack(side=tk.LEFT, padx=4)  # 输入框
        batch_frame = ttk.Frame(self)  # 批次框
        batch_frame.pack(fill=tk.X, **pad)  # 布局
        ttk.Label(batch_frame, text="批次大小:").pack(side=tk.LEFT)  # 标签
        self.batch_var = tk.IntVar(value=self.app.config_data.get("vfhq", {}).get("batch_size", 4))  # 默认批次
        ttk.Entry(batch_frame, textvariable=self.batch_var, width=6).pack(side=tk.LEFT, padx=4)  # 输入框
        btn_frame = ttk.Frame(self)  # 按钮框
        btn_frame.pack(fill=tk.X, **pad)  # 布局
        ttk.Button(btn_frame, text="开始处理", command=self.start).pack(side=tk.LEFT)  # 开始按钮
        ttk.Button(btn_frame, text="暂停", command=self.pause).pack(side=tk.LEFT, padx=4)  # 暂停按钮
        ttk.Button(btn_frame, text="停止", command=self.stop).pack(side=tk.LEFT)  # 停止按钮
        self.status_var = tk.StringVar(value="状态: 就绪")  # 状态文本
        ttk.Label(self, textvariable=self.status_var).pack(anchor=tk.W, **pad)  # 状态标签

    def _browse_dir(self) -> None:  # 浏览目录
        path = filedialog.askdirectory(title="选择 VFHQ 基础目录")  # 选择目录
        if path:  # 判断选择
            self.base_dir_var.set(path)  # 更新路径

    def start(self) -> None:  # 开始处理
        if self.thread_manager.is_running():  # 判断是否运行中
            messagebox.showwarning("提示", "已有任务在运行")  # 提示警告
            return  # 返回
        self.processor = self.app.create_vfhq_processor()  # 创建处理器
        self.status_var.set("状态: 处理中")  # 更新状态
        self.thread_manager.start(self._run_tasks)  # 启动后台线程

    def pause(self) -> None:  # 暂停处理
        if self.processor:  # 判断处理器存在
            self.processor.pause()  # 暂停处理

    def stop(self) -> None:  # 停止处理
        if self.processor:  # 判断处理器存在
            self.processor.stop()  # 停止处理

    def _run_tasks(self) -> None:  # 执行任务
        base_path = self.base_dir_var.get()  # 获取基础路径
        output_dir = self.app.config_data.get("vfhq", {}).get("output_dir", os.path.join(base_path, "processed"))  # 输出目录
        self.app.progress_panel.set_current_task("完整性检查")  # 更新当前任务
        self.processor.check_integrity(base_path)  # 执行检查
        self.app.progress_panel.set_current_task("数据预处理")  # 更新当前任务
        steps = self.app.config_data.get("vfhq", {}).get("steps", {})  # 获取步骤
        self.processor.preprocess(base_path, output_dir, steps, self.batch_var.get(), self.strategy_var.get())  # 执行预处理
        self.app.progress_panel.set_current_task("质量验证")  # 更新当前任务
        meta_dir = os.path.join(output_dir, "meta")  # 元数据目录
        self.processor.validate_quality(meta_dir)  # 执行验证
        self.status_var.set("状态: 完成")  # 更新状态
