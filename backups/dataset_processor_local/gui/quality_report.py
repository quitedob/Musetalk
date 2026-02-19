# f:\\python\\Musetalk\\dataset_processor\\gui\\quality_report.py
import tkinter as tk  # 导入 tkinter
from tkinter import ttk  # 导入 ttk
from typing import Dict, List  # 导入类型注解


class QualityReportWindow(tk.Toplevel):  # 质量报告窗口
    """数据质量验证报告窗口"""  # 类说明

    def __init__(self, master, report: Dict[str, List[str]]) -> None:  # 初始化函数
        super().__init__(master)  # 调用父类初始化
        self.title("数据质量验证报告")  # 设置标题
        self.geometry("760x520")  # 设置尺寸
        self.report = report  # 保存报告
        self._build_ui()  # 构建界面

    def _build_ui(self) -> None:  # 构建界面
        summary = ttk.LabelFrame(self, text="总体统计")  # 总体统计框
        summary.pack(fill=tk.X, padx=6, pady=4)  # 布局
        total = sum(len(v) for v in self.report.values())  # 统计问题数
        ttk.Label(summary, text=f"问题总数: {total}").pack(anchor=tk.W, padx=6, pady=2)  # 显示问题数
        detail = ttk.LabelFrame(self, text="问题分类")  # 分类框
        detail.pack(fill=tk.BOTH, expand=True, padx=6, pady=4)  # 布局
        self.tree = ttk.Treeview(detail, columns=("count",), show="headings")  # 树形视图
        self.tree.heading("count", text="数量")  # 设置标题
        self.tree.pack(fill=tk.BOTH, expand=True, padx=6, pady=4)  # 布局
        for key, items in self.report.items():  # 遍历报告
            self.tree.insert("", tk.END, values=(f"{key}: {len(items)}",))  # 插入行
