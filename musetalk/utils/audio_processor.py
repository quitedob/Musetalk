import math
import os
from typing import List, Optional, Tuple

import librosa
import torch
from einops import rearrange
from transformers import AutoFeatureExtractor

from musetalk.audio.campplus_encoder import CamPlusAudioEncoder
from musetalk.audio.feature_adapter import build_musetalk_audio_prompts
from musetalk.audio.sensevoice_encoder import SenseVoiceAudioEncoder
from musetalk.audio.whisper_encoder import WhisperAudioEncoder


class AudioProcessor:
    """统一音频处理器，支持 whisper/campplus/sensevoice 三种后端。"""

    def __init__(
        self,
        feature_extractor_path: str = "openai/whisper-tiny/",
        encoder_type: str = "whisper",
        device: str = "cuda",
        use_float16: bool = False,
    ) -> None:
        # 保存公共配置。  
        self.encoder_type = encoder_type
        self.device = device
        self.use_float16 = use_float16
        self.feature_extractor_path = feature_extractor_path

        # 默认仅在 whisper 路径使用 AutoFeatureExtractor。  
        self.feature_extractor: Optional[AutoFeatureExtractor] = None
        # 初始化统一编码器对象。  
        self.audio_encoder = None

        # 根据类型创建编码器。  
        if encoder_type == "whisper":
            self.feature_extractor = AutoFeatureExtractor.from_pretrained(feature_extractor_path)
            self.audio_encoder = WhisperAudioEncoder(
                model_path=feature_extractor_path,
                device=device,
                use_float16=use_float16,
            )
        elif encoder_type == "campplus":
            self.audio_encoder = CamPlusAudioEncoder(device=device)
        elif encoder_type == "sensevoice":
            self.audio_encoder = SenseVoiceAudioEncoder(device=device)
        else:
            raise ValueError(f"Unknown encoder type: {encoder_type}")

        # 打印初始化信息便于排障。  
        print(f"Initialized audio encoder: {self.encoder_type}")
        print(f"Encoder trainable params: {self.audio_encoder.param_count:,}")

    def get_audio_feature(self, wav_path: str, start_index: int = 0, weight_dtype=None):
        """提取音频特征，返回 (features_list, librosa_length)。"""
        # 音频不存在时返回空结果。  
        if not os.path.exists(wav_path):
            return None
        # whisper 路径保持原实现，避免行为漂移。  
        if self.encoder_type == "whisper":
            return self._get_whisper_audio_feature(wav_path=wav_path, weight_dtype=weight_dtype)

        # 新编码器路径输出统一为 list[Tensor] 以兼容旧调用。  
        features, librosa_length = self.audio_encoder.encode(wav_path)
        # 按需转换精度。  
        if weight_dtype is not None:
            features = features.to(dtype=weight_dtype)
        return [features], librosa_length

    def _get_whisper_audio_feature(self, wav_path: str, weight_dtype=None) -> Tuple[List[torch.Tensor], int]:
        """保留原 Whisper 特征提取流程。"""
        # 加载并重采样音频。  
        librosa_output, sampling_rate = librosa.load(wav_path, sr=16000)
        assert sampling_rate == 16000
        # 将长音频切分为 30 秒片段。  
        segment_length = 30 * sampling_rate
        segments = [librosa_output[i : i + segment_length] for i in range(0, len(librosa_output), segment_length)]
        # 逐段提取 mel 特征。  
        features = []
        for segment in segments:
            audio_feature = self.feature_extractor(
                segment,
                return_tensors="pt",
                sampling_rate=sampling_rate,
            ).input_features
            if weight_dtype is not None:
                audio_feature = audio_feature.to(dtype=weight_dtype)
            features.append(audio_feature)
        return features, len(librosa_output)

    def get_whisper_chunk(
        self,
        whisper_input_features,
        device,
        weight_dtype,
        whisper,
        librosa_length,
        fps=25,
        audio_padding_length_left=2,
        audio_padding_length_right=2,
    ):
        """获取模型条件音频块，输出形状固定为 [T, 50, 384]。"""
        # whisper 路径沿用历史逻辑。  
        if self.encoder_type == "whisper":
            return self._get_whisper_chunk_original(
                whisper_input_features=whisper_input_features,
                device=device,
                weight_dtype=weight_dtype,
                whisper=whisper,
                librosa_length=librosa_length,
                fps=fps,
                audio_padding_length_left=audio_padding_length_left,
                audio_padding_length_right=audio_padding_length_right,
            )

        # 非 whisper 路径使用统一特征适配。  
        return self._get_generic_encoder_chunk(
            encoder_features=whisper_input_features,
            device=device,
            librosa_length=librosa_length,
            fps=fps,
            audio_padding_length_left=audio_padding_length_left,
            audio_padding_length_right=audio_padding_length_right,
            weight_dtype=weight_dtype,
        )

    def _get_whisper_chunk_original(
        self,
        whisper_input_features,
        device,
        weight_dtype,
        whisper,
        librosa_length,
        fps=25,
        audio_padding_length_left=2,
        audio_padding_length_right=2,
    ):
        """原始 Whisper chunk 逻辑（保持兼容）。"""
        audio_feature_length_per_frame = 2 * (audio_padding_length_left + audio_padding_length_right + 1)
        whisper_feature = []
        # 处理多个 30 秒 mel 输入。  
        for input_feature in whisper_input_features:
            input_feature = input_feature.to(device).to(weight_dtype)
            audio_feats = whisper.encoder(input_feature, output_hidden_states=True).hidden_states
            audio_feats = torch.stack(audio_feats, dim=2)
            whisper_feature.append(audio_feats)

        whisper_feature = torch.cat(whisper_feature, dim=1)
        # 按真实音频长度裁剪。  
        sr = 16000
        audio_fps = 50
        fps = int(fps)
        whisper_idx_multiplier = audio_fps / fps
        num_frames = math.floor((librosa_length / sr) * fps)
        actual_length = math.floor((librosa_length / sr) * audio_fps)
        whisper_feature = whisper_feature[:, :actual_length, ...]

        # 构建边界 padding。  
        padding_nums = math.ceil(whisper_idx_multiplier)
        whisper_feature = torch.cat(
            [
                torch.zeros_like(whisper_feature[:, : padding_nums * audio_padding_length_left]),
                whisper_feature,
                torch.zeros_like(whisper_feature[:, : padding_nums * 3 * audio_padding_length_right]),
            ],
            1,
        )

        # 逐视频帧截取音频上下文片段。  
        audio_prompts = []
        for frame_index in range(num_frames):
            try:
                audio_index = math.floor(frame_index * whisper_idx_multiplier)
                audio_clip = whisper_feature[:, audio_index : audio_index + audio_feature_length_per_frame]
                assert audio_clip.shape[1] == audio_feature_length_per_frame
                audio_prompts.append(audio_clip)
            except Exception as e:
                print(f"Error occurred: {e}")
                print(f"whisper_feature.shape: {whisper_feature.shape}")
                print(f"audio_clip.shape: {audio_clip.shape}")
                print(f"num frames: {num_frames}, fps: {fps}, whisper_idx_multiplier: {whisper_idx_multiplier}")
                print(f"frame_index: {frame_index}, audio_index: {audio_index}-{audio_index + audio_feature_length_per_frame}")
                raise

        # 转换到 UNet 期望形状 [T, 50, 384]。  
        audio_prompts = torch.cat(audio_prompts, dim=0)
        audio_prompts = rearrange(audio_prompts, "b c h w -> b (c h) w")
        return audio_prompts

    def _get_generic_encoder_chunk(
        self,
        encoder_features,
        device,
        librosa_length: int,
        fps: int,
        audio_padding_length_left: int,
        audio_padding_length_right: int,
        weight_dtype=None,
    ) -> torch.Tensor:
        """将非 Whisper 帧特征转换为 MuseTalk 的音频条件输入。"""
        # 兼容 list 输入格式。  
        if isinstance(encoder_features, list):
            frame_features = encoder_features[0]
        else:
            frame_features = encoder_features

        # 处理多余维度，统一到 [T, D]。  
        if frame_features.ndim == 3:
            frame_features = frame_features.squeeze(0)
        if frame_features.ndim != 2:
            raise ValueError(f"Unexpected encoder feature shape: {tuple(frame_features.shape)}")

        # 确保维度为 384。  
        frame_features = self.audio_encoder.project_to_target_dim(frame_features, target_dim=384)
        # 将帧特征适配为 [T, 50, 384]。  
        audio_prompts = build_musetalk_audio_prompts(
            frame_features=frame_features.to(device),
            librosa_length=librosa_length,
            fps=fps,
            audio_padding_length_left=audio_padding_length_left,
            audio_padding_length_right=audio_padding_length_right,
        )
        # 对齐精度。  
        if weight_dtype is not None:
            audio_prompts = audio_prompts.to(dtype=weight_dtype)
        return audio_prompts

