# d:\python\Musetalk\dataset_processor\gui\quality_report.py
"""
数据质量验证报告窗口

显示数据集验证结果的详细报告。
"""

import tkinter as tk
from tkinter import ttk, filedialog, messagebox
from typing import Dict, List


class QualityReportWindow(tk.Toplevel):
    """数据质量验证报告窗口"""

    def __init__(self, master, report: Dict[str, List[str]]) -> None:
        super().__init__(master)
        self.title("数据质量验证报告")
        self.geometry("860x620")
        self.report = report
        self._build_ui()

    def _build_ui(self) -> None:
        # 总体统计框
        summary = ttk.LabelFrame(self, text="总体统计")
        summary.pack(fill=tk.X, padx=6, pady=4)
        
        total = sum(len(v) for v in self.report.values())
        categories = len([k for k, v in self.report.items() if v])
        
        stats_frame = ttk.Frame(summary)
        stats_frame.pack(fill=tk.X, padx=6, pady=4)
        
        ttk.Label(stats_frame, text=f"问题总数: {total}").pack(side=tk.LEFT, padx=10)
        ttk.Label(stats_frame, text=f"问题类别: {categories}").pack(side=tk.LEFT, padx=10)
        
        # 状态指示
        if total == 0:
            status_text = "✅ 数据集质量良好，未发现问题"
            status_color = "green"
        elif total < 10:
            status_text = "⚠️ 发现少量问题，建议检查"
            status_color = "orange"
        else:
            status_text = "❌ 发现较多问题，需要处理"
            status_color = "red"
        status_label = ttk.Label(stats_frame, text=status_text)
        status_label.pack(side=tk.RIGHT, padx=10)
        
        # 分类详情框
        detail = ttk.LabelFrame(self, text="问题分类详情")
        detail.pack(fill=tk.BOTH, expand=True, padx=6, pady=4)
        
        # 左侧：类别列表
        left_frame = ttk.Frame(detail)
        left_frame.pack(side=tk.LEFT, fill=tk.Y, padx=4, pady=4)
        
        ttk.Label(left_frame, text="问题类别:").pack(anchor=tk.W)
        
        self.category_list = tk.Listbox(left_frame, width=25, height=15)
        self.category_list.pack(fill=tk.Y, expand=True, pady=4)
        self.category_list.bind("<<ListboxSelect>>", self._on_category_select)
        
        # 填充类别列表
        category_names = {
            "face_detection_failed": "人脸检测失败",
            "audio_missing": "音频缺失",
            "frame_mismatch": "帧数不一致/JSON错误",
            "small_face": "人脸过小",
            "invalid_json": "JSON格式无效",
            "error": "其他错误",
        }
        
        for key, items in self.report.items():
            display_name = category_names.get(key, key)
            count = len(items)
            self.category_list.insert(tk.END, f"{display_name} ({count})")
        
        # 右侧：文件列表
        right_frame = ttk.Frame(detail)
        right_frame.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=4, pady=4)
        
        ttk.Label(right_frame, text="问题文件列表:").pack(anchor=tk.W)
        
        # 添加滚动条
        file_frame = ttk.Frame(right_frame)
        file_frame.pack(fill=tk.BOTH, expand=True, pady=4)
        
        scrollbar = ttk.Scrollbar(file_frame)
        scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
        
        self.file_text = tk.Text(file_frame, wrap=tk.NONE, height=15, yscrollcommand=scrollbar.set)
        self.file_text.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        scrollbar.config(command=self.file_text.yview)
        
        # 底部按钮
        btn_frame = ttk.Frame(self)
        btn_frame.pack(fill=tk.X, padx=6, pady=4)
        
        ttk.Button(btn_frame, text="导出报告", command=self._export_report).pack(side=tk.LEFT, padx=4)
        ttk.Button(btn_frame, text="关闭", command=self.destroy).pack(side=tk.RIGHT, padx=4)
    
    def _on_category_select(self, event) -> None:
        """处理类别选择事件"""
        selection = self.category_list.curselection()
        if not selection:
            return
            
        idx = selection[0]
        keys = list(self.report.keys())
        if idx >= len(keys):
            return
            
        selected_key = keys[idx]
        items = self.report.get(selected_key, [])
        
        # 更新文件列表
        self.file_text.delete("1.0", tk.END)
        if items:
            for item in items:
                self.file_text.insert(tk.END, f"• {item}\n")
        else:
            self.file_text.insert(tk.END, "（该类别无问题文件）")
    
    def _export_report(self) -> None:
        """导出报告到文件"""
        path = filedialog.asksaveasfilename(
            title="导出验证报告",
            defaultextension=".txt",
            filetypes=[("Text", "*.txt"), ("Markdown", "*.md")]
        )
        if not path:
            return
            
        try:
            with open(path, "w", encoding="utf-8") as f:
                f.write("# 数据集质量验证报告\n\n")
                
                total = sum(len(v) for v in self.report.values())
                f.write(f"## 总体统计\n\n")
                f.write(f"- 问题总数: {total}\n")
                f.write(f"- 问题类别: {len([k for k, v in self.report.items() if v])}\n\n")
                
                f.write("## 问题详情\n\n")
                
                category_names = {
                    "face_detection_failed": "人脸检测失败",
                    "audio_missing": "音频缺失",
                    "frame_mismatch": "帧数不一致/JSON错误",
                    "small_face": "人脸过小",
                    "invalid_json": "JSON格式无效",
                    "error": "其他错误",
                }
                
                for key, items in self.report.items():
                    display_name = category_names.get(key, key)
                    f.write(f"### {display_name} ({len(items)} 个)\n\n")
                    if items:
                        for item in items:
                            f.write(f"- {item}\n")
                    else:
                        f.write("（无问题）\n")
                    f.write("\n")
                    
            messagebox.showinfo("导出成功", f"报告已保存到:\n{path}")
        except Exception as e:
            messagebox.showerror("导出失败", str(e))
