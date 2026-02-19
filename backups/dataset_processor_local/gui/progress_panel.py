# f:\\python\\Musetalk\\dataset_processor\\gui\\progress_panel.py
import tkinter as tk  # 导入 tkinter
from tkinter import ttk  # 导入 ttk
from typing import Dict  # 导入类型注解

from utils.memory_monitor import MemoryMonitor  # 导入内存监控


class ProgressPanel(ttk.Frame):  # 进度面板
    """进度显示面板"""  # 类说明

    def __init__(self, master, memory_monitor: MemoryMonitor) -> None:  # 初始化函数
        super().__init__(master)  # 调用父类初始化
        self.memory_monitor = memory_monitor  # 保存监控器
        self.steps: Dict[str, Dict[str, int]] = {}  # 步骤状态表
        self._build_ui()  # 构建界面
        self._schedule_memory_update()  # 启动定时更新

    def _build_ui(self) -> None:  # 构建界面
        ttk.Label(self, text="进度监控").pack(anchor=tk.W, padx=6, pady=4)  # 标题
        self.total_var = tk.IntVar(value=0)  # 总进度变量
        self.total_bar = ttk.Progressbar(self, maximum=100, variable=self.total_var)  # 总进度条
        self.total_bar.pack(fill=tk.X, padx=6)  # 布局
        self.current_task = ttk.Label(self, text="当前任务: -")  # 当前任务标签
        self.current_task.pack(anchor=tk.W, padx=6, pady=2)  # 布局
        self.tree = ttk.Treeview(self, columns=("status", "percent"), show="headings", height=6)  # 步骤树
        self.tree.heading("status", text="状态")  # 列标题
        self.tree.heading("percent", text="进度")  # 列标题
        self.tree.pack(fill=tk.X, padx=6, pady=4)  # 布局
        self.ram_var = tk.StringVar(value="RAM: 0.0/0.0 GB (0%)")  # RAM 文本
        self.vram_var = tk.StringVar(value="VRAM: 0.0/0.0 GB (0%)")  # VRAM 文本
        ttk.Label(self, textvariable=self.ram_var).pack(anchor=tk.W, padx=6)  # RAM 标签
        ttk.Label(self, textvariable=self.vram_var).pack(anchor=tk.W, padx=6)  # VRAM 标签

    def add_step(self, name: str, status: str, percent: int) -> None:  # 添加步骤
        self.after(0, self._add_step_ui, name, status, percent)  # 调度到主线程

    def update_step(self, name: str, status: str, percent: int) -> None:  # 更新步骤
        self.after(0, self._update_step_ui, name, status, percent)  # 调度到主线程

    def set_current_task(self, text: str) -> None:  # 设置当前任务
        self.after(0, self.current_task.config, {"text": f"当前任务: {text}"})  # 调度更新

    def set_total_progress(self, percent: int) -> None:  # 设置总进度
        self.after(0, self.total_var.set, percent)  # 调度更新

    def _render_steps(self) -> None:  # 渲染步骤树
        for item in self.tree.get_children():  # 清空已有项目
            self.tree.delete(item)  # 删除项目
        for name, info in self.steps.items():  # 渲染新项目
            self.tree.insert("", tk.END, values=(f"{name} {info['status']}", f"{info['percent']}%"))  # 插入行

    def _add_step_ui(self, name: str, status: str, percent: int) -> None:  # 主线程添加步骤
        self.steps[name] = {"status": status, "percent": percent}  # 保存步骤
        self._render_steps()  # 刷新显示

    def _update_step_ui(self, name: str, status: str, percent: int) -> None:  # 主线程更新步骤
        if name not in self.steps:  # 判断是否存在
            self.steps[name] = {"status": status, "percent": percent}  # 添加步骤
        else:  # 存在则更新
            self.steps[name]["status"] = status  # 更新状态
            self.steps[name]["percent"] = percent  # 更新进度
        self._render_steps()  # 刷新显示

    def _schedule_memory_update(self) -> None:  # 定时更新内存
        self._update_memory_usage()  # 更新一次
        self.after(1000, self._schedule_memory_update)  # 定时调用

    def _update_memory_usage(self) -> None:  # 更新内存显示
        ram = self.memory_monitor.get_ram_usage()  # 获取 RAM
        vram = self.memory_monitor.get_vram_usage()  # 获取 VRAM
        self.ram_var.set(f"RAM: {ram['used_gb']:.1f}/{self.memory_monitor.ram_limit_gb} GB ({ram['percent']:.0f}%)")  # 更新 RAM
        self.vram_var.set(f"VRAM: {vram['used_gb']:.1f}/{self.memory_monitor.vram_limit_gb} GB ({vram['percent']:.0f}%)")  # 更新 VRAM
