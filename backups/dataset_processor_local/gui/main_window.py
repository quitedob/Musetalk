# f:\\python\\Musetalk\\dataset_processor\\gui\\main_window.py
import os  # 导入系统模块
import tkinter as tk  # 导入 tkinter
from tkinter import ttk, messagebox  # 导入 ttk 和消息框

from core.hdtf_processor import HDTFProcessor  # 导入 HDTF 处理器
from core.vfhq_processor import VFHQProcessor  # 导入 VFHQ 处理器
from gui.hdtf_panel import HDTFPanel  # 导入 HDTF 面板
from gui.log_panel import LogPanel  # 导入日志面板
from gui.metadata_editor import MetadataEditor  # 导入元数据编辑器
from gui.progress_panel import ProgressPanel  # 导入进度面板
from gui.quality_report import QualityReportWindow  # 导入质量报告
from gui.vfhq_panel import VFHQPanel  # 导入 VFHQ 面板
from utils.config_manager import ConfigManager  # 导入配置管理
from utils.memory_monitor import MemoryMonitor  # 导入内存监控


class MainWindow(ttk.Frame):  # 主窗口
    """主窗口组件"""  # 类说明

    def __init__(self, master) -> None:  # 初始化函数
        super().__init__(master)  # 调用父类初始化
        self.config_path = os.path.join(os.path.dirname(__file__), "..", "config", "default_config.yaml")  # 配置路径
        self.config_path = os.path.abspath(self.config_path)  # 绝对路径
        self.config_manager = ConfigManager(self.config_path)  # 配置管理器
        self.config_data = self._load_config_safe()  # 加载配置
        self.memory_monitor = MemoryMonitor(  # 创建内存监控
            ram_limit_gb=self.config_data.get("system", {}).get("ram_limit", 64),  # RAM 限制
            vram_limit_gb=self.config_data.get("system", {}).get("gpu_memory_limit", 16),  # VRAM 限制
        )  # 结束创建
        self._build_ui()  # 构建界面

    def _load_config_safe(self) -> dict:  # 安全加载配置
        try:  # 尝试加载配置
            return self.config_manager.load()  # 返回配置
        except Exception as e:  # 捕获异常
            messagebox.showwarning("配置加载失败", str(e))  # 弹出警告
            return {}  # 返回空配置

    def _build_ui(self) -> None:  # 构建界面
        self._build_menu()  # 构建菜单
        self.notebook = ttk.Notebook(self)  # 创建 Notebook
        self.notebook.pack(fill=tk.BOTH, expand=True)  # 布局
        self.progress_panel = ProgressPanel(self.notebook, self.memory_monitor)  # 进度面板
        self.log_panel = LogPanel(self.notebook)  # 日志面板
        self.hdtf_panel = HDTFPanel(self.notebook, self)  # HDTF 面板
        self.vfhq_panel = VFHQPanel(self.notebook, self)  # VFHQ 面板
        self.notebook.add(self.hdtf_panel, text="HDTF")  # 添加 HDTF 标签
        self.notebook.add(self.vfhq_panel, text="VFHQ")  # 添加 VFHQ 标签
        self.notebook.add(self.progress_panel, text="进度监控")  # 添加进度标签
        self.notebook.add(self.log_panel, text="日志")  # 添加日志标签

    def _build_menu(self) -> None:  # 构建菜单栏
        menubar = tk.Menu(self)  # 创建菜单栏
        file_menu = tk.Menu(menubar, tearoff=0)  # 文件菜单
        file_menu.add_command(label="退出", command=self.master.destroy)  # 退出命令
        menubar.add_cascade(label="文件", menu=file_menu)  # 添加文件菜单
        tools_menu = tk.Menu(menubar, tearoff=0)  # 工具菜单
        tools_menu.add_command(label="元数据编辑器", command=self._open_metadata_editor)  # 打开元数据
        tools_menu.add_command(label="质量报告窗口", command=self._open_quality_report)  # 打开质量报告
        menubar.add_cascade(label="工具", menu=tools_menu)  # 添加工具菜单
        help_menu = tk.Menu(menubar, tearoff=0)  # 帮助菜单
        help_menu.add_command(label="关于", command=self._show_about)  # 关于命令
        menubar.add_cascade(label="帮助", menu=help_menu)  # 添加帮助菜单
        self.master.config(menu=menubar)  # 设置菜单栏

    def _open_metadata_editor(self) -> None:  # 打开元数据编辑器
        MetadataEditor(self.master)  # 创建窗口

    def _open_quality_report(self) -> None:  # 打开质量报告窗口
        QualityReportWindow(self.master, {"face_detection_failed": [], "audio_missing": [], "frame_mismatch": [], "small_face": []})  # 空报告

    def _show_about(self) -> None:  # 显示关于信息
        messagebox.showinfo("关于", "MuseTalk 数据集处理工具\n基于 tkinter 的数据集管理界面")  # 显示对话框

    def create_hdtf_processor(self) -> HDTFProcessor:  # 创建 HDTF 处理器
        return HDTFProcessor(self.progress_panel, self.log_panel.add_log, self.config_data.get("hdtf", {}))  # 返回处理器

    def create_vfhq_processor(self) -> VFHQProcessor:  # 创建 VFHQ 处理器
        return VFHQProcessor(self.progress_panel, self.log_panel.add_log, self.config_data.get("vfhq", {}))  # 返回处理器
