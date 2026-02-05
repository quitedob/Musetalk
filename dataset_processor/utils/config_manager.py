# f:\python\Musetalk\dataset_processor\utils\config_manager.py
import os  # 导入系统路径模块
from typing import Any, Dict  # 导入类型注解

import yaml  # 导入 YAML 解析模块


class ConfigManager:  # 配置管理器
    """配置文件读取与保存"""  # 类说明

    def __init__(self, config_path: str) -> None:  # 初始化函数
        self.config_path = config_path  # 保存配置路径
        self.data: Dict[str, Any] = {}  # 初始化配置字典

    def load(self) -> Dict[str, Any]:  # 加载配置
        if not os.path.exists(self.config_path):  # 判断配置是否存在
            raise FileNotFoundError(f"配置文件不存在: {self.config_path}")  # 抛出异常
        with open(self.config_path, "r", encoding="utf-8") as f:  # 打开配置文件
            self.data = yaml.safe_load(f) or {}  # 读取 YAML 数据
        return self.data  # 返回配置内容

    def save(self, data: Dict[str, Any]) -> None:  # 保存配置
        self.data = data  # 更新内部数据
        os.makedirs(os.path.dirname(self.config_path), exist_ok=True)  # 创建目录
        with open(self.config_path, "w", encoding="utf-8") as f:  # 打开配置文件
            yaml.safe_dump(self.data, f, allow_unicode=True, sort_keys=False)  # 写入 YAML

    def get(self, key: str, default: Any = None) -> Any:  # 获取配置项
        return self.data.get(key, default)  # 返回配置值
