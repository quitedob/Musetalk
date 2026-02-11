"""Cam++ 轻量音频编码器实现。"""

from typing import Tuple

import librosa
import numpy as np
import torch

from musetalk.audio.base_audio_encoder import BaseAudioEncoder, LearnableFeatureAdapter


class CamPlusAudioEncoder(BaseAudioEncoder):
    """Cam++ 编码器，优先使用 FunASR，失败时回退到 Mel 特征。"""

    def __init__(
        self,
        model_name: str = "iic/speech_campplus_sv_zh_en_16k-common_advanced",
        device: str = "cuda",
        use_learnable_adapter: bool = True,
    ) -> None:
        # 初始化基类。  
        super().__init__(device=device, feature_dim=384)
        # 保存模型名称用于日志。  
        self.model_name = model_name
        # 标记是否启用学习型适配。  
        self.use_learnable_adapter = use_learnable_adapter
        # 默认后端为 mel 回退。  
        self.backend = "mel_fallback"
        # 初始化适配器为空。  
        self.feature_adapter = None
        # 尝试加载 Cam++。  
        try:
            from funasr import AutoModel

            self.model = AutoModel(model=model_name, device=device)
            self.backend = "campplus"
        except Exception:
            self.model = None
            self.backend = "mel_fallback"

        # 创建可学习适配器以统一到 384 维。  
        if use_learnable_adapter:
            self.feature_adapter = LearnableFeatureAdapter(input_dim=192, output_dim=384).to(device)

    def _load_audio(self, audio_path: str, sr: int = 16000) -> Tuple[np.ndarray, int]:
        """加载音频并重采样。"""
        audio, sample_rate = librosa.load(audio_path, sr=sr)
        return audio, sample_rate

    def _extract_mel_features(self, audio: np.ndarray, sr: int = 16000) -> torch.Tensor:
        """提取标准 log-mel 帧特征作为回退方案。"""
        mel = librosa.feature.melspectrogram(
            y=audio,
            sr=sr,
            n_fft=400,
            hop_length=160,
            n_mels=80,
            fmin=0,
            fmax=8000,
        )
        mel = np.log10(np.maximum(mel, 1e-10))
        mel = (mel - mel.mean()) / (mel.std() + 1e-6)
        return torch.from_numpy(mel.T).float()

    @torch.no_grad()
    def encode(self, audio_path: str) -> Tuple[torch.Tensor, int]:
        """提取帧级特征并返回样本长度。"""
        audio, sample_rate = self._load_audio(audio_path, sr=16000)
        audio_num_samples = len(audio)

        # Cam++ 返回段级说话人向量，这里仅做弱时序展开以保持接口可用。  
        if self.backend == "campplus" and self.model is not None:
            try:
                result = self.model.generate(input=audio_path)
                embedding = None
                if isinstance(result, list) and len(result) > 0:
                    candidate = result[0]
                    if isinstance(candidate, dict):
                        embedding = candidate.get("spk_embedding", candidate.get("embedding"))
                    else:
                        embedding = candidate
                if embedding is not None:
                    vector = torch.tensor(embedding).float()
                    if vector.ndim == 1:
                        vector = vector.unsqueeze(0)
                    # 将段级向量扩展到近似帧级（按 50fps）。  
                    target_len = max(1, int(audio_num_samples / 16000 * 50))
                    frame_features = vector.repeat(target_len, 1)
                    if self.feature_adapter is not None:
                        frame_features = self.feature_adapter(frame_features)
                    else:
                        frame_features = self.project_to_target_dim(frame_features, target_dim=384)
                    return frame_features.to(self.device), audio_num_samples
            except Exception:
                pass

        # 回退到 log-mel，并升维到 384。  
        mel_features = self._extract_mel_features(audio, sr=sample_rate)
        if self.feature_adapter is not None:
            # 当适配器输入维度不匹配时退化为补零映射。  
            if mel_features.shape[-1] == 192:
                mel_features = self.feature_adapter(mel_features)
            else:
                mel_features = self.project_to_target_dim(mel_features, target_dim=384)
        else:
            mel_features = self.project_to_target_dim(mel_features, target_dim=384)
        return mel_features.to(self.device), audio_num_samples
