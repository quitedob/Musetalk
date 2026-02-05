# f:\python\Musetalk\dataset_processor\utils\memory_monitor.py
from typing import Dict  # 导入类型注解

import psutil  # 导入系统监控模块


class MemoryMonitor:  # 内存监控类
    """RAM 与 VRAM 监控"""  # 类说明

    def __init__(self, ram_limit_gb: int = 64, vram_limit_gb: int = 16) -> None:  # 初始化函数
        self.ram_limit_gb = ram_limit_gb  # RAM 限制值
        self.vram_limit_gb = vram_limit_gb  # VRAM 限制值

    def get_ram_usage(self) -> Dict[str, float]:  # 获取 RAM 使用情况
        process = psutil.Process()  # 获取进程对象
        mem_info = process.memory_info()  # 获取内存信息
        used_gb = mem_info.rss / (1024 ** 3)  # 计算使用量
        percent = (used_gb / self.ram_limit_gb) * 100 if self.ram_limit_gb else 0  # 计算百分比
        return {"used_gb": used_gb, "percent": percent}  # 返回结果

    def get_vram_usage(self) -> Dict[str, float]:  # 获取 VRAM 使用情况
        try:  # 尝试导入 torch
            import torch  # 导入 torch
            if not torch.cuda.is_available():  # 判断 CUDA 可用性
                return {"used_gb": 0.0, "percent": 0.0}  # 无 GPU 则返回 0
            allocated = torch.cuda.memory_allocated()  # 获取分配显存
            total = torch.cuda.get_device_properties(0).total_memory  # 获取显存总量
            used_gb = allocated / (1024 ** 3)  # 计算使用量
            percent = (allocated / total) * 100 if total else 0  # 计算百分比
            return {"used_gb": used_gb, "percent": percent}  # 返回结果
        except Exception:  # 发生异常
            return {"used_gb": 0.0, "percent": 0.0}  # 返回默认值

    def suggest_batch_size(self, current_batch: int, current_vram_gb: float) -> int:  # 建议批次大小
        if current_vram_gb > (self.vram_limit_gb * 0.9):  # 接近限制
            return max(1, current_batch // 2)  # 减小批次
        if current_vram_gb < (self.vram_limit_gb * 0.6):  # 有余量
            return min(32, current_batch * 2)  # 增大批次
        return current_batch  # 保持不变

    def is_safe(self) -> bool:  # 判断是否安全
        ram = self.get_ram_usage()  # 获取 RAM
        vram = self.get_vram_usage()  # 获取 VRAM
        return ram["percent"] < 90 and vram["percent"] < 90  # 返回安全性
