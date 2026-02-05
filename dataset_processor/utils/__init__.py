# d:\python\Musetalk\dataset_processor\utils\__init__.py
"""
工具模块初始化

提供配置管理、内存监控、线程管理和FFmpeg封装功能。
"""

# 使用延迟导入以避免循环依赖
__all__ = [
    "ConfigManager",
    "MemoryMonitor", 
    "ThreadManager",
    "run_ffmpeg",
    "check_ffmpeg_available",
]


def __getattr__(name):
    """延迟加载模块成员"""
    if name == "ConfigManager":
        from utils.config_manager import ConfigManager
        return ConfigManager
    elif name == "MemoryMonitor":
        from utils.memory_monitor import MemoryMonitor
        return MemoryMonitor
    elif name == "ThreadManager":
        from utils.thread_manager import ThreadManager
        return ThreadManager
    elif name == "run_ffmpeg":
        from utils.ffmpeg_wrapper import run_ffmpeg
        return run_ffmpeg
    elif name == "check_ffmpeg_available":
        from utils.ffmpeg_wrapper import check_ffmpeg_available
        return check_ffmpeg_available
    raise AttributeError(f"module 'utils' has no attribute '{name}'")
