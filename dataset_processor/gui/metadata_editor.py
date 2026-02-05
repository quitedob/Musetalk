# f:\\python\\Musetalk\\dataset_processor\\gui\\metadata_editor.py
import json  # 导入 JSON 模块
import tkinter as tk  # 导入 tkinter
from tkinter import ttk, filedialog, messagebox  # 导入 ttk 与对话框


class MetadataEditor(tk.Toplevel):  # 元数据编辑器
    """元数据查看器/编辑器"""  # 类说明

    def __init__(self, master) -> None:  # 初始化函数
        super().__init__(master)  # 调用父类初始化
        self.title("元数据查看器/编辑器")  # 设置标题
        self.geometry("760x520")  # 设置窗口大小
        self._build_ui()  # 构建界面

    def _build_ui(self) -> None:  # 构建界面
        path_frame = ttk.Frame(self)  # 路径框
        path_frame.pack(fill=tk.X, padx=6, pady=4)  # 布局
        ttk.Label(path_frame, text="选择文件:").pack(side=tk.LEFT)  # 标签
        self.path_var = tk.StringVar(value="")  # 文件路径变量
        ttk.Entry(path_frame, textvariable=self.path_var).pack(side=tk.LEFT, fill=tk.X, expand=True, padx=4)  # 输入框
        ttk.Button(path_frame, text="浏览...", command=self._browse_file).pack(side=tk.LEFT)  # 浏览按钮
        self.text = tk.Text(self, wrap=tk.NONE)  # 文本框
        self.text.pack(fill=tk.BOTH, expand=True, padx=6, pady=4)  # 布局
        btn_frame = ttk.Frame(self)  # 按钮框
        btn_frame.pack(fill=tk.X, padx=6, pady=4)  # 布局
        ttk.Button(btn_frame, text="保存", command=self._save).pack(side=tk.LEFT)  # 保存按钮
        ttk.Button(btn_frame, text="重新加载", command=self._load).pack(side=tk.LEFT, padx=4)  # 重新加载按钮

    def _browse_file(self) -> None:  # 浏览文件
        path = filedialog.askopenfilename(title="选择元数据文件", filetypes=[("JSON", "*.json")])  # 选择文件
        if path:  # 判断选择
            self.path_var.set(path)  # 更新路径
            self._load()  # 加载文件

    def _load(self) -> None:  # 加载文件
        path = self.path_var.get()  # 获取路径
        if not path:  # 判断路径
            return  # 直接返回
        try:  # 尝试读取
            with open(path, "r", encoding="utf-8") as f:  # 使用with确保关闭
                data = json.load(f)  # 读取数据
            self.text.delete("1.0", tk.END)  # 清空文本
            self.text.insert(tk.END, json.dumps(data, ensure_ascii=False, indent=2))  # 写入文本
        except json.JSONDecodeError as e:  # JSON解析错误
            messagebox.showerror("加载失败", f"JSON格式错误: {e}")  # 显示错误
        except Exception as e:  # 捕获异常
            messagebox.showerror("加载失败", str(e))  # 显示错误


    def _save(self) -> None:  # 保存文件
        path = self.path_var.get()  # 获取路径
        if not path:  # 判断路径
            return  # 直接返回
        try:  # 尝试保存
            data = json.loads(self.text.get("1.0", tk.END))  # 解析 JSON
            with open(path, "w", encoding="utf-8") as f:  # 打开文件
                json.dump(data, f, ensure_ascii=False, indent=2)  # 写入文件
            messagebox.showinfo("保存成功", "元数据已保存")  # 提示成功
        except Exception as e:  # 捕获异常
            messagebox.showerror("保存失败", str(e))  # 显示错误
