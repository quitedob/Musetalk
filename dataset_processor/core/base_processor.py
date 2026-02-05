# f:\python\Musetalk\dataset_processor\core\base_processor.py
import threading  # 导入线程模块
import time  # 导入时间模块
from abc import ABC, abstractmethod  # 导入抽象基类
from typing import Any, Callable, Dict  # 导入类型注解


class ProcessorStoppedException(Exception):  # 停止异常
    """处理器停止异常"""  # 类说明


class BaseProcessor(ABC):  # 处理器基类
    """数据集处理器基类"""  # 类说明

    def __init__(self, progress_callback, log_callback, config: Dict[str, Any]) -> None:  # 初始化函数
        self.progress_cb = progress_callback  # 进度回调
        self.log_cb = log_callback  # 日志回调
        self.config = config  # 配置字典
        self._stop_event = threading.Event()  # 停止事件
        self._pause_event = threading.Event()  # 暂停事件

    @abstractmethod
    def check_integrity(self, **kwargs) -> Dict[str, Any]:  # 完整性检查接口
        pass  # 抽象方法占位

    @abstractmethod
    def preprocess(self, **kwargs) -> bool:  # 预处理接口
        pass  # 抽象方法占位

    @abstractmethod
    def validate_quality(self, **kwargs) -> Dict[str, Any]:  # 质量验证接口
        pass  # 抽象方法占位

    def stop(self) -> None:  # 停止处理
        self._stop_event.set()  # 触发停止
        self.log_cb("INFO", "处理已停止")  # 记录日志

    def pause(self) -> None:  # 暂停处理
        self._pause_event.set()  # 触发暂停
        self.log_cb("INFO", "处理已暂停")  # 记录日志

    def resume(self) -> None:  # 恢复处理
        self._pause_event.clear()  # 清除暂停
        self.log_cb("INFO", "处理已恢复")  # 记录日志

    def _check_interrupts(self) -> None:  # 中断检查
        if self._stop_event.is_set():  # 判断停止
            raise ProcessorStoppedException()  # 抛出异常
        while self._pause_event.is_set():  # 暂停循环
            time.sleep(0.1)  # 轻量休眠
