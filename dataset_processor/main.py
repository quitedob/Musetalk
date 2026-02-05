# f:\python\Musetalk\dataset_processor\main.py
import os  # 导入系统路径模块
import tkinter as tk  # 导入主窗口模块
from gui.main_window import MainWindow  # 导入主窗口类


def _set_windows_dpi_awareness() -> None:  # 设置 Windows DPI 感知
    try:  # 尝试调用系统 API
        import ctypes  # 导入 Windows API 调用模块
        ctypes.windll.shcore.SetProcessDpiAwareness(1)  # 设置系统 DPI 感知
    except Exception:  # 失败时忽略
        return  # 直接返回


def main() -> None:  # 程序入口函数
    _set_windows_dpi_awareness()  # 启用 DPI 感知
    os.environ.setdefault("TK_SILENCE_DEPRECATION", "1")  # 设置环境变量
    root = tk.Tk()  # 创建主窗口
    root.title("MuseTalk 数据集处理工具")  # 设置窗口标题
    root.geometry("1100x780")  # 设置默认窗口大小
    root.minsize(980, 700)  # 设置最小窗口大小
    app = MainWindow(root)  # 初始化主窗口组件
    app.pack(fill=tk.BOTH, expand=True)  # 填充布局
    root.mainloop()  # 进入事件循环


if __name__ == "__main__":  # 主程序入口判断
    main()  # 调用入口函数
