"""统一音频编码器抽象接口。"""

from abc import ABC, abstractmethod
from typing import Tuple

import torch
import torch.nn as nn


class BaseAudioEncoder(ABC):
    """音频编码器基类，约定统一输出格式。"""

    def __init__(self, device: str = "cuda", feature_dim: int = 384) -> None:
        # 保存运行设备。  
        self.device = device
        # 保存输出特征维度。  
        self.feature_dim = feature_dim
        # 保存底层模型实例。  
        self.model = None

    @abstractmethod
    def encode(self, audio_path: str) -> Tuple[torch.Tensor, int]:
        """提取帧级特征并返回 (features, audio_num_samples)。"""

    def project_to_target_dim(self, features: torch.Tensor, target_dim: int = 384) -> torch.Tensor:
        """将任意维度特征映射到目标维度。"""
        # 维度一致时直接返回。  
        if features.shape[-1] == target_dim:
            return features
        # 低维特征右侧补零。  
        if features.shape[-1] < target_dim:
            padding = torch.zeros(
                *features.shape[:-1],
                target_dim - features.shape[-1],
                device=features.device,
                dtype=features.dtype,
            )
            return torch.cat([features, padding], dim=-1)
        # 高维特征按前 target_dim 截断。  
        return features[..., :target_dim]

    @property
    def param_count(self) -> int:
        """返回可训练参数量。"""
        # 模型为空时返回 0。  
        if self.model is None:
            return 0
        # 累加可训练参数总数。  
        return sum(p.numel() for p in self.model.parameters() if p.requires_grad)


class LearnableFeatureAdapter(nn.Module):
    """可学习特征适配器，将输入维度映射到统一 384 维。"""

    def __init__(self, input_dim: int, output_dim: int = 384, hidden_dim: int = 256) -> None:
        # 初始化父类。  
        super().__init__()
        # 使用轻量 MLP 进行维度映射。  
        self.adapter = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, output_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """前向映射特征。"""
        # 返回适配后的特征。  
        return self.adapter(x)
