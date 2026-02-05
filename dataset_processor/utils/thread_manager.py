# f:\python\Musetalk\dataset_processor\utils\thread_manager.py
import threading  # 导入线程模块
from typing import Callable, Optional  # 导入类型注解


class ThreadManager:  # 线程管理器
    """后台任务线程管理"""  # 类说明

    def __init__(self) -> None:  # 初始化函数
        self._thread: Optional[threading.Thread] = None  # 线程对象

    def start(self, target: Callable[[], None]) -> None:  # 启动后台线程
        if self._thread and self._thread.is_alive():  # 判断是否已有任务
            return  # 已有任务时直接返回
        self._thread = threading.Thread(target=target, daemon=True)  # 创建线程
        self._thread.start()  # 启动线程

    def is_running(self) -> bool:  # 判断是否运行中
        return bool(self._thread and self._thread.is_alive())  # 返回运行状态
