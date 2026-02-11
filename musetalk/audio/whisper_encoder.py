"""Whisper 音频编码器包装。"""

from typing import Tuple

import torch
from transformers import AutoFeatureExtractor

from musetalk.audio.base_audio_encoder import BaseAudioEncoder


class WhisperAudioEncoder(BaseAudioEncoder):
    """Whisper 编码器，保持与现有 MuseTalk 行为一致。"""

    def __init__(self, model_path: str = "./models/whisper", device: str = "cuda", use_float16: bool = False) -> None:
        # 初始化基类。  
        super().__init__(device=device, feature_dim=384)
        # 保存模型目录供后续调用。  
        self.model_path = model_path
        # 保存精度设置。  
        self.use_float16 = use_float16
        # 仅使用特征提取器准备 mel 输入。  
        self.feature_extractor = AutoFeatureExtractor.from_pretrained(model_path)

    @torch.no_grad()
    def encode(self, audio_path: str) -> Tuple[torch.Tensor, int]:
        """读取 Whisper encoder embeddings 并返回帧级 384 维特征。"""
        # 延迟导入，避免无关路径初始化失败。  
        from musetalk.whisper.audio2feature import Audio2Feature
        # 创建原始 Whisper 特征提取器。  
        processor = Audio2Feature(model_path=f"{self.model_path}/tiny.pt")
        # 提取原始特征数组，形状约为 (T, 384)。  
        features = processor.audio2feat(audio_path)
        # 转为张量并迁移到目标设备。  
        feature_tensor = torch.from_numpy(features).float().to(self.device)
        # 读取音频样本长度用于后续对齐。  
        import librosa
        audio, _ = librosa.load(audio_path, sr=16000)
        return feature_tensor, len(audio)
