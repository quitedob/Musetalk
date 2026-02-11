"""SenseVoice 编码器包装。"""

from typing import Tuple

import librosa
import torch

from musetalk.audio.base_audio_encoder import BaseAudioEncoder
from musetalk.audio.sensevoice_processor import SenseVoiceAudioProcessor


class SenseVoiceAudioEncoder(BaseAudioEncoder):
    """将现有 SenseVoiceAudioProcessor 适配到统一编码器接口。"""

    def __init__(self, device: str = "cuda", model_name: str = "FunAudioLLM/SenseVoiceSmall", use_vad: bool = True) -> None:
        # 初始化基类。  
        super().__init__(device=device, feature_dim=384)
        # 创建底层处理器。  
        self.processor = SenseVoiceAudioProcessor(model_name=model_name, device=device, use_vad=use_vad)
        # 同步后端模型引用。  
        self.model = self.processor.model

    @torch.no_grad()
    def encode(self, audio_path: str) -> Tuple[torch.Tensor, int]:
        """提取帧级特征并返回音频样本长度。"""
        # 使用已有处理器提取特征。  
        features, _ = self.processor.get_audio_feature(audio_path)
        # 再次确保输出维度为 384。  
        features = self.project_to_target_dim(features, target_dim=384)
        # 读取音频样本长度用于统一时间对齐。  
        audio, _ = librosa.load(audio_path, sr=16000)
        return features.to(self.device), len(audio)
