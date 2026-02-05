# f:\python\Musetalk\dataset_processor\gui\log_panel.py
import datetime  # 导入时间模块
import tkinter as tk  # 导入 tkinter
from tkinter import ttk, filedialog  # 导入 ttk 与文件对话框


class LogPanel(ttk.Frame):  # 日志面板
    """日志输出面板"""  # 类说明

    def __init__(self, master) -> None:  # 初始化函数
        super().__init__(master)  # 调用父类初始化
        self._auto_scroll = tk.BooleanVar(value=True)  # 自动滚动开关
        self._level_filters = {  # 日志级别过滤
            "INFO": tk.BooleanVar(value=True),  # INFO 过滤
            "SUCCESS": tk.BooleanVar(value=True),  # SUCCESS 过滤
            "WARNING": tk.BooleanVar(value=True),  # WARNING 过滤
            "ERROR": tk.BooleanVar(value=True),  # ERROR 过滤
        }  # 结束过滤字典
        self._build_ui()  # 构建界面

    def _build_ui(self) -> None:  # 构建界面
        header = ttk.Frame(self)  # 顶部工具栏
        header.pack(fill=tk.X, padx=6, pady=4)  # 布局
        ttk.Label(header, text="日志输出").pack(side=tk.LEFT)  # 标题标签
        ttk.Button(header, text="清除", command=self.clear).pack(side=tk.RIGHT, padx=4)  # 清除按钮
        ttk.Button(header, text="保存", command=self.save).pack(side=tk.RIGHT)  # 保存按钮
        filter_frame = ttk.Frame(self)  # 过滤框
        filter_frame.pack(fill=tk.X, padx=6, pady=2)  # 布局
        for level, var in self._level_filters.items():  # 创建过滤选项
            ttk.Checkbutton(filter_frame, text=level, variable=var).pack(side=tk.LEFT, padx=4)  # 复选框
        ttk.Checkbutton(filter_frame, text="自动滚动", variable=self._auto_scroll).pack(side=tk.RIGHT)  # 自动滚动
        self.text = tk.Text(self, height=10, wrap=tk.NONE)  # 日志文本框
        self.text.pack(fill=tk.BOTH, expand=True, padx=6, pady=4)  # 布局
        self.text.tag_configure("INFO", foreground="black")  # INFO 颜色
        self.text.tag_configure("SUCCESS", foreground="green")  # SUCCESS 颜色
        self.text.tag_configure("WARNING", foreground="orange")  # WARNING 颜色
        self.text.tag_configure("ERROR", foreground="red")  # ERROR 颜色

    def add_log(self, level: str, message: str) -> None:  # 添加日志
        self.after(0, self._add_log_ui, level, message)  # 调度到主线程

    def _add_log_ui(self, level: str, message: str) -> None:  # 主线程写日志
        if not self._level_filters.get(level, tk.BooleanVar(value=True)).get():  # 过滤判断
            return  # 过滤掉不显示的日志
        ts = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")  # 时间戳
        line = f"[{ts}] {level}: {message}\n"  # 日志内容
        self.text.insert(tk.END, line, level)  # 插入文本
        if self._auto_scroll.get():  # 判断自动滚动
            self.text.see(tk.END)  # 滚动到底部

    def clear(self) -> None:  # 清除日志
        self.text.delete("1.0", tk.END)  # 清空文本

    def save(self) -> None:  # 保存日志
        path = filedialog.asksaveasfilename(  # 选择保存路径
            title="保存日志", defaultextension=".log", filetypes=[("Log", "*.log"), ("Text", "*.txt")]  # 文件类型
        )  # 结束对话框
        if not path:  # 判断是否取消
            return  # 直接返回
        content = self.text.get("1.0", tk.END)  # 获取内容
        with open(path, "w", encoding="utf-8") as f:  # 写入文件
            f.write(content)  # 写入内容
