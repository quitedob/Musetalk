# f:\\python\\Musetalk\\dataset_processor\\gui\\hdtf_panel.py
import os  # 导入系统模块
import tkinter as tk  # 导入 tkinter
from tkinter import ttk, filedialog, messagebox  # 导入 ttk 与对话框

from utils.thread_manager import ThreadManager  # 导入线程管理


class HDTFPanel(ttk.Frame):  # HDTF 面板
    """HDTF 数据集面板"""  # 类说明

    def __init__(self, master, app) -> None:  # 初始化函数
        super().__init__(master)  # 调用父类初始化
        self.app = app  # 保存主窗口引用
        self.thread_manager = ThreadManager()  # 创建线程管理器
        self.processor = None  # 处理器占位
        self._build_ui()  # 构建界面

    def _build_ui(self) -> None:  # 构建界面
        pad = {"padx": 6, "pady": 4}  # 通用内边距
        ttk.Label(self, text="HDTF 数据集").pack(anchor=tk.W, **pad)  # 标题
        path_frame = ttk.Frame(self)  # 路径框
        path_frame.pack(fill=tk.X, **pad)  # 布局
        ttk.Label(path_frame, text="数据目录:").pack(side=tk.LEFT)  # 标签
        self.data_dir_var = tk.StringVar(value=self.app.config_data.get("hdtf", {}).get("source_dir", ""))  # 默认路径
        ttk.Entry(path_frame, textvariable=self.data_dir_var).pack(side=tk.LEFT, fill=tk.X, expand=True, padx=4)  # 输入框
        ttk.Button(path_frame, text="浏览...", command=self._browse_dir).pack(side=tk.LEFT)  # 浏览按钮
        tasks = ttk.LabelFrame(self, text="任务选择")  # 任务框
        tasks.pack(fill=tk.X, **pad)  # 布局
        self.task_integrity = tk.BooleanVar(value=True)  # 完整性检查
        self.task_preprocess = tk.BooleanVar(value=True)  # 预处理
        self.task_quality = tk.BooleanVar(value=True)  # 质量验证
        self.task_metadata = tk.BooleanVar(value=True)  # 元数据管理
        ttk.Checkbutton(tasks, text="完整性检查", variable=self.task_integrity).pack(anchor=tk.W)  # 复选框
        ttk.Checkbutton(tasks, text="数据预处理", variable=self.task_preprocess).pack(anchor=tk.W)  # 复选框
        ttk.Checkbutton(tasks, text="质量验证", variable=self.task_quality).pack(anchor=tk.W)  # 复选框
        ttk.Checkbutton(tasks, text="元数据管理", variable=self.task_metadata).pack(anchor=tk.W)  # 复选框
        batch_frame = ttk.Frame(self)  # 批次框
        batch_frame.pack(fill=tk.X, **pad)  # 布局
        ttk.Label(batch_frame, text="批次大小:").pack(side=tk.LEFT)  # 标签
        self.batch_var = tk.IntVar(value=self.app.config_data.get("hdtf", {}).get("batch_size", 8))  # 默认批次
        ttk.Entry(batch_frame, textvariable=self.batch_var, width=6).pack(side=tk.LEFT, padx=4)  # 输入框
        btn_frame = ttk.Frame(self)  # 按钮框
        btn_frame.pack(fill=tk.X, **pad)  # 布局
        ttk.Button(btn_frame, text="开始处理", command=self.start).pack(side=tk.LEFT)  # 开始按钮
        ttk.Button(btn_frame, text="暂停", command=self.pause).pack(side=tk.LEFT, padx=4)  # 暂停按钮
        ttk.Button(btn_frame, text="停止", command=self.stop).pack(side=tk.LEFT)  # 停止按钮
        self.status_var = tk.StringVar(value="状态: 就绪")  # 状态文本
        ttk.Label(self, textvariable=self.status_var).pack(anchor=tk.W, **pad)  # 状态标签

    def _browse_dir(self) -> None:  # 浏览目录
        path = filedialog.askdirectory(title="选择 HDTF 数据目录")  # 选择目录
        if path:  # 判断选择
            self.data_dir_var.set(path)  # 更新路径

    def start(self) -> None:  # 开始处理
        if self.thread_manager.is_running():  # 判断是否运行中
            messagebox.showwarning("提示", "已有任务在运行")  # 提示警告
            return  # 返回
        self.processor = self.app.create_hdtf_processor()  # 创建处理器
        self.status_var.set("状态: 处理中")  # 更新状态
        self.thread_manager.start(self._run_tasks)  # 启动后台线程

    def pause(self) -> None:  # 暂停处理
        if self.processor:  # 判断处理器存在
            self.processor.pause()  # 暂停处理

    def stop(self) -> None:  # 停止处理
        if self.processor:  # 判断处理器存在
            self.processor.stop()  # 停止处理

    def _run_tasks(self) -> None:  # 执行任务
        source_dir = self.data_dir_var.get()  # 获取目录
        output_dir = self.app.config_data.get("hdtf", {}).get("output_dir", os.path.join(source_dir, "processed"))  # 输出目录
        if self.task_integrity.get():  # 完整性检查
            self.app.progress_panel.set_current_task("完整性检查")  # 更新当前任务
            self.processor.check_integrity(source_dir)  # 执行检查
        if self.task_preprocess.get():  # 预处理
            self.app.progress_panel.set_current_task("数据预处理")  # 更新当前任务
            steps = self.app.config_data.get("hdtf", {}).get("steps", {})  # 获取步骤
            self.processor.preprocess(source_dir, output_dir, steps, self.batch_var.get())  # 执行预处理
        if self.task_quality.get():  # 质量验证
            self.app.progress_panel.set_current_task("质量验证")  # 更新当前任务
            meta_dir = os.path.join(output_dir, "meta")  # 元数据目录
            self.processor.validate_quality(meta_dir)  # 执行验证
        self.status_var.set("状态: 完成")  # 更新状态
